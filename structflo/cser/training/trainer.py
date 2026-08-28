"""D-FINE detector training (replaces the AGPL ultralytics trainer).

Plain PyTorch loop around ``transformers.DFineForObjectDetection`` (Apache-2.0):
AdamW with a lower backbone LR, linear warmup + cosine decay, bf16 autocast,
EMA weights (evaluated + saved), gradient clipping, checkpoint selection on the
YOLO-style fitness (0.1·mAP50 + 0.9·mAP50-95) of the val split, early stopping.

Data is described by the same YAML the YOLO trainer used::

    path: /abs/root
    train: train/images
    val: real_val/images

Usage::

    sf-train --data config/data.yaml --init ustc-community/dfine-large-coco \
             --epochs 12 --batch 8 --imgsz 1280 --name dfine_l_synth
    sf-train --data data/finetune/plus/yolo/data.yaml \
             --init runs/labels_detect/dfine_l_synth/weights/best.safetensors \
             --epochs 30 --lr 5e-5 --name dfine_l_plus
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from structflo.cser.inference.dfine import DEFAULT_INIT, DFineDetector, decode_outputs
from structflo.cser.inference.metrics import (
    ImageEval,
    evaluate,
    fitness,
    load_yolo_labels,
)
from structflo.cser.training.dataset import (
    YoloDetectionDataset,
    collate,
    labels_dir_for,
)

_PROJECT_ROOT = Path(__file__).parents[3]
RUNS_DIR = _PROJECT_ROOT / "runs" / "labels_detect"
CLASS_IDS = [0, 1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class ModelEMA:
    """Exponential moving average of weights (decay ramps up over ``tau`` updates)."""

    def __init__(
        self, model: torch.nn.Module, decay: float = 0.9999, tau: int = 2000
    ) -> None:
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def _resolve_split(data_yaml: Path, key: str) -> Path:
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path", data_yaml.parent))
    p = Path(cfg[key])
    return p if p.is_absolute() else root / p


def _param_groups(
    model: torch.nn.Module, lr: float, backbone_lr: float, weight_decay: float
):
    groups = {"bb_decay": [], "bb_no_decay": [], "decay": [], "no_decay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = ".backbone." in name
        no_decay = p.ndim <= 1 or name.endswith(".bias")
        groups[("bb_" if is_bb else "") + ("no_decay" if no_decay else "decay")].append(
            p
        )
    return [
        {"params": groups["bb_decay"], "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": groups["bb_no_decay"], "lr": backbone_lr, "weight_decay": 0.0},
        {"params": groups["decay"], "lr": lr, "weight_decay": weight_decay},
        {"params": groups["no_decay"], "lr": lr, "weight_decay": 0.0},
    ]


def _to_device(labels: list[dict], device: torch.device) -> list[dict]:
    return [
        {k: v.to(device, non_blocking=True) for k, v in lab.items()} for lab in labels
    ]


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    labels_dir: Path,
    device: torch.device,
    *,
    imgsz: int,
    op_conf: float,
    amp: bool,
    conf_floor: float = 0.001,
) -> dict:
    """Batched inference on the val split → metrics dict (per class + all)."""
    model.eval()
    images: list[ImageEval] = []
    for batch in loader:
        x = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(pixel_values=x)
        for i, m in enumerate(batch["meta"]):
            dets = decode_outputs(
                out.logits[i],
                out.pred_boxes[i],
                imgsz,
                m["scale"],
                m["dx"],
                m["dy"],
                m["orig_w"],
                m["orig_h"],
                conf_floor,
            )
            gt_boxes, gt_cls = load_yolo_labels(
                labels_dir / f"{m['stem']}.txt", m["orig_w"], m["orig_h"]
            )
            images.append(
                ImageEval(
                    pred_boxes=np.array(
                        [d["bbox"] for d in dets], dtype=np.float64
                    ).reshape(-1, 4),
                    pred_scores=np.array([d["conf"] for d in dets], dtype=np.float64),
                    pred_classes=np.array(
                        [d["class_id"] for d in dets], dtype=np.int64
                    ),
                    gt_boxes=gt_boxes,
                    gt_classes=gt_cls,
                )
            )
    return evaluate(images, CLASS_IDS, op_conf=op_conf)


def _metrics_row(res: dict) -> dict:
    row = {
        "mAP50": res["all"]["mAP50"],
        "mAP50-95": res["all"]["mAP50-95"],
        "P": res["all"]["precision"],
        "R": res["all"]["recall"],
    }
    for c in CLASS_IDS:
        r = res[c]
        row[f"c{c}_mAP50"] = r.ap50
        row[f"c{c}_mAP50-95"] = r.ap
        row[f"c{c}_P"] = r.precision
        row[f"c{c}_R"] = r.recall
    return row


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"

    run_dir = Path(args.project) / args.name
    weights_dir = run_dir / "weights"
    if run_dir.exists() and not args.resume and not args.exist_ok:
        sys.exit(f"run dir {run_dir} exists (use --exist-ok or a new --name)")
    weights_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    data_yaml = Path(args.data).resolve()
    train_images = _resolve_split(data_yaml, "train")
    val_images = _resolve_split(data_yaml, "val")
    train_ds = YoloDetectionDataset(
        train_images,
        imgsz=args.imgsz,
        augment=not args.no_augment,
        scale_jitter=args.scale_jitter,
        brightness=args.brightness,
        downscale_aug=args.downscale_aug,
        photometric_aug=args.photometric_aug,
        limit=args.max_train_images,
    )
    val_ds = YoloDetectionDataset(
        val_images, imgsz=args.imgsz, augment=False, limit=args.max_val_images
    )
    from structflo.cser.training.photometric import fixed_variant

    val_variant_loaders = {}
    for name in args.val_variants:
        vds = YoloDetectionDataset(
            val_images,
            imgsz=args.imgsz,
            augment=False,
            limit=args.max_val_images,
            transform=fixed_variant(name),
        )
        val_variant_loaders[name] = DataLoader(
            vds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=max(1, args.workers // 2),
            collate_fn=collate,
            pin_memory=True,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        collate_fn=collate,
        pin_memory=True,
    )
    print(
        f"train: {len(train_ds)} images ({train_images})\nval:   {len(val_ds)} images ({val_images})"
    )

    # -- model ----------------------------------------------------------------
    init = args.init
    if Path(init).exists():
        det = DFineDetector.from_file(init, device=device, imgsz=args.imgsz)
    else:
        det = DFineDetector.from_hub(
            init, num_labels=len(CLASS_IDS), device=device, imgsz=args.imgsz
        )
    model = det.model.to(device)
    # Contrastive denoising (DN) groups are DISABLED by default. With DN groups on, transformers'
    # D-FINE implementation (5.16.1, sdpa and eager alike) lets the GT-derived DN queries influence
    # the normal queries during training, so the model learns confidences that only exist when
    # labels are fed in and eval-mode scores collapse. Measured on one batch after 250 steps:
    # train+labels max score 0.85 vs eval 0.28; with num_denoising=0 both 0.98 (val images 0.99).
    # The DN attention mask is built and passed to the decoder self-attention as expected, so the
    # leak path is not pinned down; disabling DN removes it and costs nothing in this regime.
    model.config.num_denoising = args.num_denoising
    print(
        f"model: D-FINE {det.num_parameters / 1e6:.1f}M params, init={init}, "
        f"num_denoising={model.config.num_denoising}"
    )

    optimizer = torch.optim.AdamW(
        _param_groups(model, args.lr, args.backbone_lr, args.weight_decay),
        betas=(0.9, 0.999),
    )
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return args.lrf + (1 - args.lrf) * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = ModelEMA(model, decay=args.ema_decay)

    start_epoch, best_fit, best_epoch = 0, -1.0, -1
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.ema.load_state_dict(ck["ema"])
        ema.updates = ck["ema_updates"]
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch, best_fit, best_epoch = (
            ck["epoch"] + 1,
            ck["best_fit"],
            ck["best_epoch"],
        )
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    results_csv = run_dir / "results.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "lr",
        "img_s",
        "fitness",
        "fitness_plain",
        "mAP50",
        "mAP50-95",
        "P",
        "R",
        *[f"c{c}_{k}" for c in CLASS_IDS for k in ("mAP50", "mAP50-95", "P", "R")],
    ]
    if not results_csv.exists() or not args.resume:
        with open(results_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.perf_counter()
        loss_sum, n_img = 0.0, 0
        for it, batch in enumerate(train_loader):
            x = batch["pixel_values"].to(device, non_blocking=True)
            labels = _to_device(batch["labels"], device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(pixel_values=x, labels=labels)
            loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            step += 1
            loss_sum += loss.item() * x.shape[0]
            n_img += x.shape[0]
            if (it + 1) % args.log_every == 0 or it + 1 == steps_per_epoch:
                el = time.perf_counter() - t0
                print(
                    f"epoch {epoch + 1}/{args.epochs}  it {it + 1}/{steps_per_epoch}  "
                    f"loss {loss_sum / n_img:.3f}  lr {scheduler.get_last_lr()[2]:.2e}  "
                    f"{n_img / el:.1f} img/s",
                    flush=True,
                )
        train_loss = loss_sum / max(n_img, 1)
        img_s = n_img / (time.perf_counter() - t0)

        # -- validate (EMA weights) --------------------------------------------
        res = validate(
            ema.ema,
            val_loader,
            labels_dir_for(val_images),
            device,
            imgsz=args.imgsz,
            op_conf=args.conf,
            amp=amp,
        )
        row = _metrics_row(res)
        fit = fitness(res["all"])
        if val_variant_loaders:
            # checkpoint selection on the mean fitness over the plain val split and its
            # deterministic photometric variants (e.g. inverted) so robustness counts
            fits = [fit]
            for name, vl in val_variant_loaders.items():
                vres = validate(
                    ema.ema,
                    vl,
                    labels_dir_for(val_images),
                    device,
                    imgsz=args.imgsz,
                    op_conf=args.conf,
                    amp=amp,
                )
                vf = fitness(vres["all"])
                fits.append(vf)
                print(
                    f"[val:{name}] epoch {epoch + 1}: fitness {vf:.4f}  mAP50 {vres['all']['mAP50']:.4f}  "
                    f"struct R {vres[0].recall:.3f}  label R {vres[1].recall:.3f}",
                    flush=True,
                )
            plain_fit = fits[0]
            fit = (
                1.0 - args.val_variant_weight
            ) * plain_fit + args.val_variant_weight * float(
                sum(fits[1:]) / len(fits[1:])
            )
            row["fitness_plain"] = plain_fit
        row.setdefault("fitness_plain", fitness(res["all"]))
        row.update(
            epoch=epoch + 1,
            train_loss=train_loss,
            lr=scheduler.get_last_lr()[2],
            img_s=img_s,
            fitness=fit,
        )
        with open(results_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(
                {k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in row.items()}
            )
        print(
            f"[val] epoch {epoch + 1}: fitness {fit:.4f}  mAP50 {row['mAP50']:.4f}  mAP50-95 {row['mAP50-95']:.4f}  "
            f"P {row['P']:.3f} R {row['R']:.3f}  | struct mAP50 {row['c0_mAP50']:.3f} R {row['c0_R']:.3f} "
            f"| label mAP50 {row['c1_mAP50']:.3f} R {row['c1_R']:.3f}",
            flush=True,
        )

        # -- checkpoints -------------------------------------------------------
        ema_det = DFineDetector(ema.ema, device=device, imgsz=args.imgsz)
        ema_det.save(
            weights_dir / "last.safetensors",
            epoch=epoch + 1,
            fitness=f"{fit:.5f}",
            data=str(data_yaml),
            init=init,
            seed=args.seed,
        )
        if fit > best_fit:
            best_fit, best_epoch = fit, epoch + 1
            ema_det.save(
                weights_dir / "best.safetensors",
                epoch=epoch + 1,
                fitness=f"{fit:.5f}",
                data=str(data_yaml),
                init=init,
                seed=args.seed,
            )
            print(
                f"  new best (epoch {epoch + 1}, fitness {fit:.4f}) → {weights_dir / 'best.safetensors'}"
            )
        if args.save_period and (epoch + 1) % args.save_period == 0:
            ema_det.save(weights_dir / f"epoch{epoch + 1}.safetensors", epoch=epoch + 1)
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.ema.state_dict(),
                "ema_updates": ema.updates,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_fit": best_fit,
                "best_epoch": best_epoch,
                "args": vars(args),
            },
            run_dir / "last_state.pt",
        )
        if args.patience and epoch + 1 - best_epoch >= args.patience:
            print(
                f"early stop: no improvement for {args.patience} epochs (best epoch {best_epoch})"
            )
            break

    print(
        f"\nbest fitness {best_fit:.4f} at epoch {best_epoch}\nweights: {weights_dir / 'best.safetensors'}"
    )
    (run_dir / "DONE").write_text(
        f"best_epoch={best_epoch}\nbest_fitness={best_fit:.5f}\n"
    )
    return weights_dir / "best.safetensors"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the D-FINE structure/label detector")
    p.add_argument(
        "--data",
        default=str(_PROJECT_ROOT / "config" / "data.yaml"),
        help="YOLO-style data yaml",
    )
    p.add_argument(
        "--init",
        default=DEFAULT_INIT,
        help="HF Hub model id or a .safetensors checkpoint to start from",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument(
        "--lr", type=float, default=1e-4, help="LR for encoder/decoder/heads"
    )
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--lrf", type=float, default=0.01, help="final LR fraction (cosine)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument(
        "--num-denoising",
        type=int,
        default=0,
        help="contrastive-denoising query groups (0 = off; ON leaks GT into eval-mode scores "
        "in transformers' D-FINE implementation — see trainer.py)",
    )
    p.add_argument(
        "--conf", type=float, default=0.4, help="operating point for reported P/R"
    )
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help="early-stop epochs without val improvement (0=off)",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default="dfine_l_panels")
    p.add_argument("--project", default=str(RUNS_DIR))
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--resume", default=None, help="path to <run>/last_state.pt")
    p.add_argument("--save-period", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--scale-jitter", type=float, default=0.3)
    p.add_argument("--brightness", type=float, default=0.1)
    p.add_argument(
        "--downscale-aug",
        type=float,
        default=0.0,
        help="probability of a random 0.4-1.0x pre-downscale (random interpolation) before the "
        "letterbox — robustness to lower-DPI renders / other rasterisers",
    )
    p.add_argument(
        "--photometric-aug",
        type=float,
        default=0.0,
        help="probability of a sampled luminance/polarity scenario (inversion, regional "
        "inversion, background/ink contrast, gradients) per training page — see "
        "training/photometric.py",
    )
    p.add_argument(
        "--val-variant-weight",
        type=float,
        default=0.5,
        help="selection fitness = (1-w)*plain val + w*mean(variants); plain fitness is logged too",
    )
    p.add_argument(
        "--val-variants",
        nargs="*",
        default=[],
        help="deterministic photometric variants of the val split (e.g. invert invert_grey) "
        "added to checkpoint selection: fitness = mean over plain val + variants",
    )
    p.add_argument(
        "--max-train-images", type=int, default=None, help="debug: cap training set"
    )
    p.add_argument(
        "--max-val-images", type=int, default=None, help="debug: cap val set"
    )
    return p


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
