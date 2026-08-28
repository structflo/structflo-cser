"""Score a predictions JSON (from dump_yolo_preds.py / dump_preds.py) against YOLO-txt labels.

Backend-agnostic: the same code scores the retired YOLO detector and the D-FINE
replacement, so the comparison is apples-to-apples (ultralytics' own mAP code is
not used). Optional pycocotools cross-check with --cocoeval.

Usage:
    uv run python scripts/license_migration/eval_preds.py \
        --preds runs/license_migration/preds/yolo_v0.4/real_test.json \
        --labels data/finetune/yolo/real_test/labels --conf 0.3 [--cocoeval] [--out x.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from structflo.cser.inference.metrics import (
    ImageEval,
    evaluate,
    fitness,
    load_yolo_labels,
)

NAMES = {0: "structure", 1: "label"}


def load_images(preds_path: Path, labels_dir: Path) -> list[ImageEval]:
    d = json.loads(preds_path.read_text())
    out = []
    for stem, rec in d["images"].items():
        gt_boxes, gt_cls = load_yolo_labels(
            labels_dir / f"{stem}.txt", rec["width"], rec["height"]
        )
        out.append(
            ImageEval(
                pred_boxes=np.array(
                    [x["bbox"] for x in rec["dets"]], dtype=np.float64
                ).reshape(-1, 4),
                pred_scores=np.array(
                    [x["conf"] for x in rec["dets"]], dtype=np.float64
                ),
                pred_classes=np.array(
                    [x["class_id"] for x in rec["dets"]], dtype=np.int64
                ),
                gt_boxes=gt_boxes,
                gt_classes=gt_cls,
            )
        )
    return out, d["meta"]


def coco_crosscheck(images: list[ImageEval]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": c, "name": n} for c, n in NAMES.items()],
    }
    dt = []
    aid = 1
    for i, im in enumerate(images):
        gt["images"].append({"id": i, "width": 100000, "height": 100000})
        for b, c in zip(im.gt_boxes, im.gt_classes):
            w, h = b[2] - b[0], b[3] - b[1]
            gt["annotations"].append(
                {
                    "id": aid,
                    "image_id": i,
                    "category_id": int(c),
                    "bbox": [b[0], b[1], w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            aid += 1
        for b, s, c in zip(im.pred_boxes, im.pred_scores, im.pred_classes):
            dt.append(
                {
                    "image_id": i,
                    "category_id": int(c),
                    "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                    "score": float(s),
                }
            )
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        cdt = coco.loadRes(dt)
        ev = COCOeval(coco, cdt, "bbox")
        ev.params.maxDets = [1, 10, 1000]
        ev.evaluate()
        ev.accumulate()
    # precision: [T iou, R recall, K class, A area, M maxDets]; read area=all, maxDets=1000 directly
    # (COCOeval.summarize() reports -1 for AP@[.5:.95] when maxDets is customised).
    prec = ev.eval["precision"][:, :, :, 0, 2]
    res = {}
    for k, c in enumerate(ev.params.catIds):

        def _ap(t):
            v = prec[t, :, k]
            return float(v[v > -1].mean()) if (v > -1).any() else 0.0

        res[int(c)] = (_ap(0), float(np.mean([_ap(t) for t in range(prec.shape[0])])))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--cocoeval", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    images, meta = load_images(args.preds, args.labels)
    res = evaluate(images, list(NAMES), op_conf=args.conf)
    print(
        f"{args.preds}  ({len(images)} images; backend={meta.get('backend')} imgsz={meta.get('imgsz')})"
    )
    print(
        f"  {'class':10s} {'n_gt':>5s} {'P@' + str(args.conf):>8s} {'R@' + str(args.conf):>8s} {'FP':>5s} {'mAP50':>7s} {'mAP50-95':>9s}"
    )
    for c, n in NAMES.items():
        r = res[c]
        print(
            f"  {n:10s} {r.n_gt:5d} {r.precision:8.3f} {r.recall:8.3f} {r.fp:5d} {r.ap50:7.4f} {r.ap:9.4f}"
        )
    a = res["all"]
    print(
        f"  {'all':10s} {'':5s} {a['precision']:8.3f} {a['recall']:8.3f} {'':5s} {a['mAP50']:7.4f} {a['mAP50-95']:9.4f}   fitness {fitness(a):.4f}"
    )
    payload = {
        "preds": str(args.preds),
        "meta": meta,
        "conf": args.conf,
        "n_images": len(images),
        "all": a,
        "fitness": fitness(a),
        "per_class": {
            n: {
                "n_gt": res[c].n_gt,
                "P": res[c].precision,
                "R": res[c].recall,
                "TP": res[c].tp,
                "FP": res[c].fp,
                "mAP50": res[c].ap50,
                "mAP50-95": res[c].ap,
            }
            for c, n in NAMES.items()
        },
    }
    if args.cocoeval:
        cc = coco_crosscheck(images)
        print(
            "  pycocotools cross-check:",
            {
                NAMES[c]: {"AP50": round(v[0], 4), "AP50-95": round(v[1], 4)}
                for c, v in cc.items()
            },
        )
        payload["pycocotools"] = {
            NAMES[c]: {"mAP50": v[0], "mAP50-95": v[1]} for c, v in cc.items()
        }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
