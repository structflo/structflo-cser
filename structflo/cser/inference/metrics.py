"""Backend-agnostic detection metrics (COCO-style AP, operating-point P/R).

Deliberately dependency-free so the same code scores predictions from any
detector backend (the retired YOLO dumps and the D-FINE detector alike) and
can run inside the training loop for checkpoint selection.

AP follows the COCO convention: greedy per-class matching of confidence-sorted
predictions to the unmatched ground-truth box of highest IoU, precision
envelope, 101-point recall interpolation, averaged over IoU 0.50:0.95.
``mAP50`` / ``mAP50-95`` here agree with pycocotools to ~1e-3 (crowd handling
and area ranges are not modelled — we have neither).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between ``a`` (N,4) and ``b`` (M,4), both xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


@dataclass
class ImageEval:
    """Predictions and ground truth for one image, xyxy pixel boxes."""

    pred_boxes: np.ndarray  # (P,4)
    pred_scores: np.ndarray  # (P,)
    pred_classes: np.ndarray  # (P,)
    gt_boxes: np.ndarray  # (G,4)
    gt_classes: np.ndarray  # (G,)


@dataclass
class ClassResult:
    n_gt: int
    ap50: float
    ap: float  # mean over IoU 0.50:0.95
    precision: float  # at the operating point (conf >= op_conf, IoU >= 0.5)
    recall: float
    tp: int
    fp: int
    ap_per_iou: list[float] = field(default_factory=list)


def _match_image(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thrs: np.ndarray,
) -> np.ndarray:
    """Greedy COCO matching for one class in one image.

    Returns a bool array (len(iou_thrs), P) — whether each prediction (sorted by
    descending score) is a true positive at each IoU threshold.
    """
    order = np.argsort(-pred_scores, kind="stable")
    pred_boxes = pred_boxes[order]
    tp = np.zeros((len(iou_thrs), len(pred_boxes)), dtype=bool)
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return tp, order
    ious = box_iou(pred_boxes, gt_boxes)
    for t, thr in enumerate(iou_thrs):
        taken = np.zeros(len(gt_boxes), dtype=bool)
        for p in range(len(pred_boxes)):
            cand = np.where(~taken & (ious[p] >= thr))[0]
            if len(cand) == 0:
                continue
            g = cand[np.argmax(ious[p, cand])]
            taken[g] = True
            tp[t, p] = True
    return tp, order


def _average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """COCO 101-point interpolated AP for one IoU threshold."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # precision envelope (monotone non-increasing from the right)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    interp = np.where(
        idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0
    )
    return float(interp.mean())


def evaluate(
    images: list[ImageEval],
    class_ids: list[int],
    op_conf: float = 0.3,
    iou_thrs: np.ndarray = IOU_THRS,
) -> dict:
    """Score a set of images. Returns per-class ``ClassResult`` plus ``all`` means.

    ``op_conf`` is the deployment operating point at which precision/recall are
    reported (IoU 0.5); AP is threshold-free.
    """
    out: dict = {}
    for c in class_ids:
        tps, scores, n_gt = [], [], 0
        op_tp = op_fp = 0
        for im in images:
            pm = im.pred_classes == c
            gm = im.gt_classes == c
            n_gt += int(gm.sum())
            tp, order = _match_image(
                im.pred_boxes[pm], im.pred_scores[pm], im.gt_boxes[gm], iou_thrs
            )
            s = im.pred_scores[pm][order]
            tps.append(tp)
            scores.append(s)
            keep = s >= op_conf
            op_tp += int(tp[0][keep].sum())
            op_fp += int((~tp[0][keep]).sum())
        tp_all = (
            np.concatenate(tps, axis=1) if tps else np.zeros((len(iou_thrs), 0), bool)
        )
        sc_all = np.concatenate(scores) if scores else np.zeros(0)
        aps = [
            _average_precision(tp_all[t], sc_all, n_gt) for t in range(len(iou_thrs))
        ]
        out[c] = ClassResult(
            n_gt=n_gt,
            ap50=aps[0],
            ap=float(np.nanmean(aps)) if n_gt else float("nan"),
            precision=op_tp / max(op_tp + op_fp, 1),
            recall=op_tp / max(n_gt, 1),
            tp=op_tp,
            fp=op_fp,
            ap_per_iou=aps,
        )
    valid = [r for r in out.values() if r.n_gt > 0]
    out["all"] = {
        "mAP50": float(np.mean([r.ap50 for r in valid])) if valid else float("nan"),
        "mAP50-95": float(np.mean([r.ap for r in valid])) if valid else float("nan"),
        "precision": float(np.mean([r.precision for r in valid]))
        if valid
        else float("nan"),
        "recall": float(np.mean([r.recall for r in valid])) if valid else float("nan"),
    }
    return out


def fitness(summary: dict) -> float:
    """Checkpoint-selection score: 0.1·mAP50 + 0.9·mAP50-95 (the YOLO convention,
    kept so selection behaviour is comparable with the retired detector)."""
    return 0.1 * summary["mAP50"] + 0.9 * summary["mAP50-95"]


def load_yolo_labels(path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO txt file (``cls cx cy w h`` normalised) into xyxy pixels + classes."""
    boxes, classes = [], []
    try:
        lines = open(path).read().strip().splitlines()
    except FileNotFoundError:
        lines = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, w, h = int(float(parts[0])), *map(float, parts[1:5])
        boxes.append(
            [
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ]
        )
        classes.append(c)
    return (
        np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        np.asarray(classes, dtype=np.int64),
    )
