"""Evaluate a detector on an on-disk split (images/ + YOLO-txt labels/).

Backend-neutral replacement for ``YOLO(weights).val(data=yaml)``: runs the
detector at a low confidence floor, then scores with
:mod:`structflo.cser.inference.metrics` (COCO-style AP, operating-point P/R).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from structflo.cser.inference.dfine import DFineDetector
from structflo.cser.inference.metrics import (
    ImageEval,
    evaluate,
    fitness,
    load_yolo_labels,
)
from structflo.cser.training.dataset import IMAGE_EXTS, labels_dir_for

CLASS_NAMES = {0: "chemical_structure", 1: "compound_label"}


def split_from_yaml(data_yaml: str | Path, key: str = "val") -> tuple[Path, Path]:
    """``(images_dir, labels_dir)`` for the ``train``/``val`` entry of a YOLO data yaml."""
    import yaml

    data_yaml = Path(data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path", data_yaml.parent))
    p = Path(cfg[key])
    images = p if p.is_absolute() else root / p
    return images, labels_dir_for(images)


def predict_split(
    model: DFineDetector,
    images_dir: str | Path,
    *,
    imgsz: int = 1280,
    conf_floor: float = 0.001,
    grayscale: bool = True,
    scale: float = 1.0,
    max_det: int = 300,
) -> dict[str, dict]:
    """Raw detections per image stem: ``{stem: {"file","width","height","dets":[...]}}``.

    Boxes are in ORIGINAL image pixels even when ``scale`` pre-downscales the
    image (used to emulate lower-DPI renders).
    """
    images_dir = Path(images_dir)
    out: dict[str, dict] = {}
    for p in sorted(x for x in images_dir.iterdir() if x.suffix.lower() in IMAGE_EXTS):
        im = Image.open(p).convert("RGB")
        if grayscale:
            im = im.convert("L").convert("RGB")
        run = im
        if scale != 1.0:
            run = im.resize(
                (max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                Image.Resampling.LANCZOS,
            )
        dets = model.predict(
            np.array(run), conf=conf_floor, imgsz=imgsz, max_det=max_det
        )
        if scale != 1.0:
            sx, sy = im.width / run.width, im.height / run.height
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                d["bbox"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        out[p.stem] = {
            "file": p.name,
            "width": im.width,
            "height": im.height,
            "dets": dets,
        }
    return out


def score_predictions(
    preds: dict[str, dict], labels_dir: str | Path, *, op_conf: float = 0.3
) -> dict:
    """Score ``predict_split`` output against YOLO-txt labels.

    Returns ``{"all": {P, R, mAP50, mAP50-95, fitness}, "chemical_structure": {...},
    "compound_label": {...}, "n_images": N}`` — the same shape the retired
    ``eval_detector.py`` emitted from ultralytics ``model.val()``.
    """
    labels_dir = Path(labels_dir)
    images = []
    for stem, rec in preds.items():
        gt_boxes, gt_cls = load_yolo_labels(
            labels_dir / f"{stem}.txt", rec["width"], rec["height"]
        )
        images.append(
            ImageEval(
                pred_boxes=np.array(
                    [d["bbox"] for d in rec["dets"]], dtype=np.float64
                ).reshape(-1, 4),
                pred_scores=np.array(
                    [d["conf"] for d in rec["dets"]], dtype=np.float64
                ),
                pred_classes=np.array(
                    [d["class_id"] for d in rec["dets"]], dtype=np.int64
                ),
                gt_boxes=gt_boxes,
                gt_classes=gt_cls,
            )
        )
    res = evaluate(images, list(CLASS_NAMES), op_conf=op_conf)
    out = {
        "all": {
            "P": res["all"]["precision"],
            "R": res["all"]["recall"],
            "mAP50": res["all"]["mAP50"],
            "mAP50-95": res["all"]["mAP50-95"],
            "fitness": fitness(res["all"]),
        },
        "n_images": len(images),
        "op_conf": op_conf,
    }
    for c, name in CLASS_NAMES.items():
        r = res[c]
        out[name] = {
            "P": r.precision,
            "R": r.recall,
            "mAP50": r.ap50,
            "mAP50-95": r.ap,
            "n_gt": r.n_gt,
            "TP": r.tp,
            "FP": r.fp,
        }
    return out


def evaluate_detector_on_split(
    model: DFineDetector,
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    *,
    imgsz: int = 1280,
    op_conf: float = 0.3,
    conf_floor: float = 0.001,
    grayscale: bool = True,
    scale: float = 1.0,
) -> dict:
    """One-call replacement for ``YOLO(w).val(data=..., imgsz=...)``."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir else labels_dir_for(images_dir)
    preds = predict_split(
        model,
        images_dir,
        imgsz=imgsz,
        conf_floor=conf_floor,
        grayscale=grayscale,
        scale=scale,
    )
    return score_predictions(preds, labels_dir, op_conf=op_conf)


def evaluate_detector_on_yaml(
    model: DFineDetector, data_yaml: str | Path, *, key: str = "val", **kw
) -> dict:
    images, labels = split_from_yaml(data_yaml, key)
    return evaluate_detector_on_split(model, images, labels, **kw)
