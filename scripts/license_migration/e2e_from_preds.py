"""End-to-end pairing F1 from a predictions JSON (Part B of eval_compare_all.py, detector-agnostic).

Protocol (unchanged from the paper scripts): detections at struct conf >= --conf and label
conf >= --label-conf are paired by each matcher; a predicted pair is correct when its
structure box has IoU >= 0.5 with a GT structure AND the GT label's centroid lies inside the
predicted label box. P = correct/predicted, R = correct/GT-labelled-pairs.

Usage:
    uv run python scripts/license_migration/e2e_from_preds.py \
        --preds runs/license_migration/preds/yolo_v0.4/real_test.json --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from structflo.cser.lps.matcher import LearnedMatcher
from structflo.cser.pipeline.matcher import HungarianMatcher
from structflo.cser.pipeline.models import Detection
from structflo.cser.relmatch.matcher import RelationalMatcher
from structflo.cser.weights import resolve_weights


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _cent(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _inside(pt, box):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument(
        "--manifest", type=Path, default=Path("data/finetune/real_split.json")
    )
    ap.add_argument(
        "--gt-dir",
        type=Path,
        default=Path(
            "/net-fs-ins/shared-docker-vols/structflo-cser-annotate/data/ground_truth"
        ),
    )
    ap.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="default: data/finetune/yolo/real_<split>/images",
    )
    ap.add_argument(
        "--lps", default=None, help="LPS weights (default: registry latest)"
    )
    ap.add_argument(
        "--relmatch", default=None, help="relational weights (default: registry latest)"
    )
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--label-conf", type=float, default=None)
    ap.add_argument(
        "--lps-pin-conf",
        action="store_true",
        help="feed the LPS matcher conf=1.0 for every detection (its training-time value; "
        "features 12/13 were constant 1.0 in GT training) instead of live detector scores",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    label_conf = args.label_conf if args.label_conf is not None else args.conf
    images_dir = args.images_dir or Path(f"data/finetune/yolo/real_{args.split}/images")

    stems = set(json.loads(args.manifest.read_text())[args.split])
    preds = json.loads(args.preds.read_text())
    missing = stems - set(preds["images"])
    if missing:
        raise SystemExit(
            f"{len(missing)} split stems missing from preds (e.g. {sorted(missing)[:3]})"
        )

    matchers = {
        "Hungarian": HungarianMatcher(),
        "LPS": LearnedMatcher(
            weights=str(args.lps or resolve_weights("cser-lps")), min_score=0.5
        ),
        "Relational": RelationalMatcher(
            weights=str(args.relmatch or resolve_weights("cser-relmatcher"))
        ),
    }
    matchers["Relational"].dustbin_margin = args.margin
    B = {n: dict(tp=0, npred=0) for n in matchers}
    gt_pairs = 0
    n_struct = n_label = 0
    for stem in sorted(stems):
        entries = json.loads((args.gt_dir / f"{stem}.json").read_text())
        labelled = [e for e in entries if e.get("label_bbox") is not None]
        gt_pairs += len(labelled)
        rec = preds["images"][stem]
        dets = [
            Detection.from_dict(d)
            for d in rec["dets"]
            if d["conf"] >= (label_conf if d["class_id"] == 1 else args.conf)
        ]
        n_struct += sum(d.class_id == 0 for d in dets)
        n_label += sum(d.class_id == 1 for d in dets)
        img_path = next(p for p in images_dir.iterdir() if p.stem == stem)
        img_l = np.array(Image.open(img_path).convert("L"))
        pinned = [Detection(bbox=d.bbox, conf=1.0, class_id=d.class_id) for d in dets]
        for name, m in matchers.items():
            if name == "Hungarian":
                pairs = m.match(dets)
            elif name == "LPS" and args.lps_pin_conf:
                pairs = m.match(pinned, image=img_l)
            else:
                pairs = m.match(dets, image=img_l)
            B[name]["npred"] += len(pairs)
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
                if gl is not None and bv >= 0.5 and _inside(_cent(gl), pl):
                    B[name]["tp"] += 1

    print(
        f"{args.preds} [{args.split}: {len(stems)} pages, GT pairs={gt_pairs}; dets kept: {n_struct} struct, {n_label} label @ conf {args.conf}/{label_conf}]"
    )
    print(f"  {'matcher':>11} | {'P':>7} {'R':>7} {'F1':>7}")
    result = {
        "preds": str(args.preds),
        "split": args.split,
        "conf": args.conf,
        "label_conf": label_conf,
        "lps_pin_conf": args.lps_pin_conf,
        "gt_pairs": gt_pairs,
        "matchers": {},
    }
    for n in matchers:
        b = B[n]
        P = b["tp"] / max(b["npred"], 1)
        R = b["tp"] / max(gt_pairs, 1)
        F = 2 * P * R / (P + R) if (P + R) else 0.0
        print(f"  {n:>11} | {P:7.3f} {R:7.3f} {F:7.3f}")
        result["matchers"][n] = {
            "P": P,
            "R": R,
            "F1": F,
            "tp": b["tp"],
            "npred": b["npred"],
        }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
