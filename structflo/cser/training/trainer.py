"""YOLO11l training wrapper."""

import argparse
from pathlib import Path

from ultralytics import YOLO

# Paths are resolved relative to this file's location in the installed package
_PROJECT_ROOT = Path(__file__).parents[3]
DATA_YAML = _PROJECT_ROOT / "config" / "data.yaml"
RUNS_DIR = _PROJECT_ROOT / "runs" / "labels_detect"


def train(
    weights: str = "yolo11l.pt",
    imgsz: int = 1280,
    batch: int = 8,
    epochs: int = 50,
    resume: str | None = None,
    seed: int = 42,
    name: str = "yolo11l_panels",
) -> None:
    model = YOLO(resume if resume else weights)

    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        patience=10,
        batch=batch,
        imgsz=imgsz,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=5,
        cos_lr=True,
        # Document augmentation — grayscale training data, disable colour ops
        hsv_h=0.0,  # no hue shift (grayscale images, meaningless)
        hsv_s=0.0,  # no saturation shift (grayscale)
        hsv_v=0.1,  # slight brightness jitter still useful
        degrees=3.0,  # slight rotation (scanned page tilt)
        translate=0.1,
        scale=0.3,  # bridges train (1280) vs inference (1536 tile) scale gap
        shear=1.0,
        flipud=0.0,  # never: documents don't appear upside down
        fliplr=0.0,  # never: chemical handedness matters
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        cache="disk",  # preprocess once, reuse across epochs
        workers=8,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=False,
        seed=seed,  # vary across reproducibility seeds; data splits stay fixed
        plots=True,
        save_period=25,  # checkpoint every 25 epochs as backup
        resume=bool(resume),
    )

    best = RUNS_DIR / name / "weights" / "best.pt"
    if best.exists():
        m = YOLO(str(best)).val(data=str(DATA_YAML))
        print("\n--- Best model metrics ---")
        print(f"mAP50:     {m.box.map50:.4f}   (target > 0.95)")
        print(f"mAP50-95:  {m.box.map:.4f}")
        print(f"Precision: {m.box.mp:.4f}")
        print(f"Recall:    {m.box.mr:.4f}")
        print(f"\nWeights: {best}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train YOLO11l compound panel detector")
    p.add_argument(
        "--weights",
        default="yolo11l.pt",
        help="Pretrained weights (default: yolo11l.pt)",
    )
    p.add_argument("--imgsz", type=int, default=1280, help="Training image size")
    p.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (8 is safe for A6000 48GB at imgsz=1280)",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--resume", default=None, help="Path to last.pt to resume an interrupted run"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training RNG seed (init/shuffle/augmentation). Does NOT change data splits.",
    )
    p.add_argument(
        "--name",
        default="yolo11l_panels",
        help="Run name under runs/labels_detect/ (vary per seed, e.g. yolo11l_panels_s43)",
    )
    args = p.parse_args()

    if args.imgsz > 1280 and args.batch > 8:
        print(
            f"[warn] imgsz={args.imgsz} with batch={args.batch} may OOM. "
            f"Consider --batch 8."
        )

    train(
        args.weights,
        args.imgsz,
        args.batch,
        args.epochs,
        args.resume,
        seed=args.seed,
        name=args.name,
    )


if __name__ == "__main__":
    main()
