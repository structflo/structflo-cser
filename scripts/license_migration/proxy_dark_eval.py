"""Proxy evaluation for dark / low-contrast / gradient pages.

The annotated real corpus has no dark-background pages, so robustness to polarity and
luminance changes is measured on deterministic transformed copies of a real split
(``structflo.cser.training.photometric.fixed_variant``). Reports detection metrics and the
parameter-free Hungarian end-to-end pairing F1 per variant, for one checkpoint.

Usage:
    uv run python scripts/license_migration/proxy_dark_eval.py --weights <ckpt.safetensors> \
        --split real_test --conf 0.4 --out runs/license_migration/eval/proxy_<tag>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from structflo.cser.inference.dfine import DFineDetector
from structflo.cser.inference.metrics import ImageEval, evaluate, load_yolo_labels
from structflo.cser.pipeline.matcher import HungarianMatcher
from structflo.cser.pipeline.models import Detection
from structflo.cser.training.photometric import VARIANT_NAMES, fixed_variant

GT_DIR = Path(
    "/net-fs-ins/shared-docker-vols/structflo-cser-annotate/data/ground_truth"
)


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _pairs_correct(pairs, entries):
    tp = 0
    for p in pairs:
        ps, pl = p.structure.bbox.as_list(), p.label.bbox.as_list()
        bi, bv = -1, 0.0
        for i, e in enumerate(entries):
            v = _iou(ps, e["struct_bbox"])
            if v > bv:
                bi, bv = i, v
        if bi < 0:
            continue
        gl = entries[bi].get("label_bbox")
        if gl is None or bv < 0.5:
            continue
        cx, cy = (gl[0] + gl[2]) / 2, (gl[1] + gl[3]) / 2
        if pl[0] <= cx <= pl[2] and pl[1] <= cy <= pl[3]:
            tp += 1
    return tp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="real_test", choices=["real_test", "real_val"])
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--variants", nargs="*", default=["original", *VARIANT_NAMES])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    det = DFineDetector.from_file(args.weights)
    img_dir = Path(f"data/finetune/yolo/{args.split}/images")
    lbl_dir = Path(f"data/finetune/yolo/{args.split}/labels")
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg"))
    matcher = HungarianMatcher()
    results = {}
    print(f"{args.weights}  split={args.split}  conf={args.conf}")
    print(
        f"{'variant':14s} | mAP50  | mAP50-95 | struct R | label R | label P | e2e Hungarian P / R / F1"
    )
    for name in args.variants:
        f = None if name == "original" else fixed_variant(name)
        ims, tp, npred, ngt = [], 0, 0, 0
        for p in files:
            im = Image.open(p).convert("L").convert("RGB")
            arr = np.array(im)
            gtb, gtc = load_yolo_labels(lbl_dir / f"{p.stem}.txt", im.width, im.height)
            if f is not None:
                arr = f(arr, gtb[gtc == 0])
            dets = det.predict(arr, conf=0.001)
            ims.append(
                ImageEval(
                    np.array([d["bbox"] for d in dets]).reshape(-1, 4),
                    np.array([d["conf"] for d in dets]),
                    np.array([d["class_id"] for d in dets], dtype=int),
                    gtb,
                    gtc,
                )
            )
            entries = json.loads((GT_DIR / f"{p.stem}.json").read_text())
            ngt += sum(1 for e in entries if e.get("label_bbox") is not None)
            pairs = matcher.match(
                [Detection.from_dict(d) for d in dets if d["conf"] >= args.conf]
            )
            npred += len(pairs)
            tp += _pairs_correct(pairs, entries)
        r = evaluate(ims, [0, 1], op_conf=args.conf)
        P, R = tp / max(npred, 1), tp / max(ngt, 1)
        F = 2 * P * R / (P + R) if P + R else 0.0
        results[name] = {
            "mAP50": r["all"]["mAP50"],
            "mAP50-95": r["all"]["mAP50-95"],
            "struct_R": r[0].recall,
            "label_R": r[1].recall,
            "label_P": r[1].precision,
            "e2e_P": P,
            "e2e_R": R,
            "e2e_F1": F,
        }
        print(
            f"{name:14s} | {r['all']['mAP50']:.3f}  | {r['all']['mAP50-95']:.3f}    | {r[0].recall:.3f}    | {r[1].recall:.3f}   | {r[1].precision:.3f}   | {P:.3f} / {R:.3f} / {F:.3f}",
            flush=True,
        )
    dark = [v for k, v in results.items() if k != "original"]
    summary = {
        k: float(np.mean([d[k] for d in dark])) for k in ("mAP50", "label_R", "e2e_F1")
    }
    print(
        f"mean over {len(dark)} variants: mAP50 {summary['mAP50']:.3f}  label R {summary['label_R']:.3f}  e2e F1 {summary['e2e_F1']:.3f}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "weights": args.weights,
                    "split": args.split,
                    "conf": args.conf,
                    "variants": results,
                    "mean_variants": summary,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
