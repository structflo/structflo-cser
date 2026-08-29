"""Label-detection quality vs confidence threshold (real TEST set).

One detector pass per page at a low floor conf; then, for each threshold, count label
detections that are true-positive (match a GT label at IoU>=0.5) vs false-positive, and
derive label precision/recall. Aggregated mean +/- std over the given detector seeds.
Counts/metrics only — no page content is exposed.

Detector backend: D-FINE via ``structflo.cser.inference.detector`` (full-image
inference at ``--imgsz``; per-seed weights are ``.safetensors`` files).

Usage:
    uv run python scripts/repro/label_conf_sweep.py --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

REAL = Path("/net-fs-ins/shared-docker-vols/structflo-cser-annotate/data")
SPLIT = Path("data/finetune/real_split.json")
DET = Path("runs/repro/detector")
CONFS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70]
FLOOR = 0.05


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def sweep_detector(weights: Path, stems: list[str], imgsz: int) -> dict:
    from structflo.cser.inference.detector import detect_full, load_detector

    model = load_detector(weights, imgsz=imgsz)
    acc = {c: {"tp": 0, "fp": 0} for c in CONFS}
    gt_total = 0
    pages = 0
    for stem in stems:
        gtf = REAL / "ground_truth" / f"{stem}.json"
        ip = REAL / "images" / f"{stem}.jpg"
        if not ip.exists():
            ip = REAL / "images" / f"{stem}.png"
        if not gtf.exists() or not ip.exists():
            continue
        entries = json.loads(gtf.read_text())
        gt_labels = [e["label_bbox"] for e in entries if e.get("label_bbox")]
        gt_total += len(gt_labels)
        img = np.array(Image.open(ip).convert("L").convert("RGB"))
        # class_id 1 == compound_label
        dets = [
            (float(d["conf"]), [float(v) for v in d["bbox"]])
            for d in detect_full(model, img, conf=FLOOR, imgsz=imgsz)
            if d["class_id"] == 1
        ]
        for c in CONFS:
            kept = sorted([(cf, bx) for cf, bx in dets if cf >= c], key=lambda x: -x[0])
            used: set[int] = set()
            tp = 0
            for _, box in kept:
                best, bv = -1, 0.5
                for gi, g in enumerate(gt_labels):
                    if gi in used:
                        continue
                    v = _iou(box, g)
                    if v >= bv:
                        best, bv = gi, v
                if best >= 0:
                    used.add(best)
                    tp += 1
            acc[c]["tp"] += tp
            acc[c]["fp"] += len(kept) - tp
        pages += 1
    out = {}
    for c in CONFS:
        tp, fp = acc[c]["tp"], acc[c]["fp"]
        out[c] = {
            "labels_per_page": (tp + fp) / pages,
            "tp_per_page": tp / pages,
            "fp_per_page": fp / pages,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / gt_total if gt_total else 0.0,
        }
    out["_meta"] = {"pages": pages, "gt_labels": gt_total}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    stems = json.loads(SPLIT.read_text())["test"]
    runs = []
    for s in args.seeds:
        w = DET / f"finetuned_s{s}" / "best.safetensors"
        if w.exists():
            print(
                f"[seed {s}] sweeping {w} over {len(stems)} test pages ...", flush=True
            )
            runs.append(sweep_detector(w, stems, args.imgsz))

    meta = runs[0]["_meta"]
    print(
        f"\nReal TEST: {meta['pages']} pages, {meta['gt_labels']} GT labels "
        f"(mean +/- std over {len(runs)} detector seeds)\n"
    )
    hdr = f"{'conf':>5} | {'labels/pg':>9} | {'correct/pg':>10} | {'wrong/pg':>9} | {'precision':>9} | {'recall':>7}"
    print(hdr)
    print("-" * len(hdr))

    def cell(c, key, dec=2):
        vals = [r[c][key] for r in runs]
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{m:.{dec}f}±{sd:.{dec}f}"

    for c in CONFS:
        print(
            f"{c:>5.2f} | {cell(c, 'labels_per_page', 1):>9} | {cell(c, 'tp_per_page', 1):>10} | "
            f"{cell(c, 'fp_per_page', 1):>9} | {cell(c, 'precision', 3):>9} | {cell(c, 'recall', 3):>7}"
        )


if __name__ == "__main__":
    main()
