"""Backend-agnostic detection metrics + letterbox geometry."""

from __future__ import annotations

import numpy as np
import pytest

from structflo.cser.inference.metrics import ImageEval, box_iou, evaluate, fitness
from structflo.cser.inference.preprocess import letterbox


def _img(preds, gts):
    preds = np.array(preds, dtype=float).reshape(-1, 6)  # x1 y1 x2 y2 score cls
    gts = np.array(gts, dtype=float).reshape(-1, 5)  # x1 y1 x2 y2 cls
    return ImageEval(
        pred_boxes=preds[:, :4],
        pred_scores=preds[:, 4],
        pred_classes=preds[:, 5].astype(int),
        gt_boxes=gts[:, :4],
        gt_classes=gts[:, 4].astype(int),
    )


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=float)
    np.testing.assert_allclose(box_iou(a, b)[0], [1.0, 25 / 175, 0.0])


def test_perfect_predictions_score_one():
    im = _img(
        [[0, 0, 10, 10, 0.9, 0], [20, 20, 30, 40, 0.8, 1]],
        [[0, 0, 10, 10, 0], [20, 20, 30, 40, 1]],
    )
    res = evaluate([im], [0, 1], op_conf=0.3)
    assert res["all"]["mAP50"] == pytest.approx(1.0)
    assert res["all"]["mAP50-95"] == pytest.approx(1.0)
    assert res[0].precision == 1.0 and res[0].recall == 1.0 and res[0].fp == 0
    assert fitness(res["all"]) == pytest.approx(1.0)


def test_false_positive_and_miss_are_counted_at_the_operating_point():
    # one correct struct, one FP struct above conf, one missed label, one low-conf label hit
    im = _img(
        [[0, 0, 10, 10, 0.9, 0], [50, 50, 60, 60, 0.5, 0], [20, 20, 30, 40, 0.1, 1]],
        [[0, 0, 10, 10, 0], [20, 20, 30, 40, 1]],
    )
    res = evaluate([im], [0, 1], op_conf=0.3)
    assert (
        res[0].tp == 1
        and res[0].fp == 1
        and res[0].precision == 0.5
        and res[0].recall == 1.0
    )
    assert res[1].tp == 0 and res[1].recall == 0.0  # below op point → miss at conf 0.3
    assert res[1].ap50 == pytest.approx(1.0)  # ...but AP is threshold-free
    assert res[0].ap50 == pytest.approx(
        1.0
    )  # FP ranks below the TP → precision envelope 1.0


def test_duplicate_detection_is_a_false_positive():
    im = _img([[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 11, 0.8, 0]], [[0, 0, 10, 10, 0]])
    res = evaluate([im], [0], op_conf=0.3)
    assert res[0].tp == 1 and res[0].fp == 1
    assert res[0].ap50 == pytest.approx(1.0)


def test_class_with_no_gt_is_excluded_from_means():
    im = _img([[0, 0, 10, 10, 0.9, 0]], [[0, 0, 10, 10, 0]])
    res = evaluate([im], [0, 1], op_conf=0.3)
    assert res[1].n_gt == 0 and np.isnan(res[1].ap50)
    assert res["all"]["mAP50"] == pytest.approx(1.0)


def test_letterbox_geometry_round_trips():
    img = np.full((300, 600, 3), 200, dtype=np.uint8)  # landscape 2:1
    canvas, scale, dx, dy = letterbox(img, 256)
    assert canvas.shape == (256, 256, 3)
    assert scale == pytest.approx(256 / 600)
    assert (dx, dy) == (0, (256 - 128) // 2)
    # image region carries the image value, padding carries the pad value
    assert canvas[dy + 5, 10].tolist() == [200, 200, 200]
    assert canvas[5, 10].tolist() == [114, 114, 114]
    # map a canvas point back to original pixels
    x_orig = (dx + 128 - dx) / scale
    assert x_orig == pytest.approx(300.0)


def test_letterbox_offset_crops_overhang():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    canvas, scale, dx, dy = letterbox(img, 64, scale=1.0, offset=(-20, -30))
    assert canvas.shape == (64, 64, 3)
    assert (dx, dy) == (-20, -30)
    assert canvas.max() == 0  # fully covered by the (cropped) image, no padding visible


def test_decode_outputs_undoes_letterbox_and_thresholds():
    import torch

    from structflo.cser.inference.dfine import decode_outputs

    # canvas 256, image 600x300 letterboxed: scale 256/600, dx 0, dy 64
    scale, dx, dy, W, H = 256 / 600, 0, 64, 600, 300
    # query 0: strong structure at canvas box (0,64)-(128,192) → original (0,0)-(300,300)
    # query 1: weak label (below conf) ; query 2: label above conf, partly outside → clipped
    logits = torch.tensor([[4.0, -4.0], [-4.0, -2.0], [-4.0, 2.0]])
    boxes = torch.tensor(
        [
            [64 / 256, 128 / 256, 128 / 256, 128 / 256],
            [0.5, 0.5, 0.1, 0.1],
            [0.98, 0.5, 0.1, 0.1],
        ]
    )
    dets = decode_outputs(logits, boxes, 256, scale, dx, dy, W, H, conf=0.3)
    assert [d["class_id"] for d in dets] == [0, 1]
    assert dets[0]["bbox"] == pytest.approx([0.0, 0.0, 300.0, 300.0], abs=1e-3)
    assert dets[0]["conf"] == pytest.approx(torch.sigmoid(torch.tensor(4.0)).item())
    x2 = dets[1]["bbox"][2]
    assert x2 == W  # clipped to the image width
    assert dets[1]["bbox"][0] < x2
