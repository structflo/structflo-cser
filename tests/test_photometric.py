"""Photometric augmentation: shapes, ranges, determinism, coverage of every family."""

from __future__ import annotations

import random

import numpy as np
import pytest

from structflo.cser.training import photometric as ph


def _page(h=120, w=200):
    x = np.full((h, w, 3), 255, dtype=np.uint8)
    x[30:60, 40:120] = 0  # some "ink"
    return x


@pytest.mark.parametrize("fn", [ph.full_inversion, ph.gradient, ph.luminance_contrast])
def test_families_keep_shape_dtype_range(fn):
    x = _page()
    y = fn(x, random.Random(1))
    assert y.shape == x.shape and y.dtype == np.uint8
    assert (y[..., 0] == y[..., 1]).all() and (
        y[..., 0] == y[..., 2]
    ).all()  # stays grayscale-RGB


def test_full_inversion_makes_background_dark_and_ink_light():
    y = ph.full_inversion(_page(), random.Random(0))
    assert y[0, 0, 0] <= 60 and y[45, 80, 0] >= 180


@pytest.mark.parametrize("kind", ph.REGION_KINDS)
def test_every_region_kind_inverts_only_inside_its_mask(kind):
    x = _page()
    boxes = np.array([[40, 30, 120, 60]], dtype=float)
    y = ph.regional_inversion(x, random.Random(3), boxes=boxes, kind=kind)
    changed = y[..., 0] != x[..., 0]
    assert changed.any() and not changed.all()


@pytest.mark.parametrize("kind", ph.LUM_KINDS)
def test_every_luminance_kind_runs(kind):
    y = ph.luminance_contrast(_page(), random.Random(5), kind=kind)
    assert y.dtype == np.uint8 and y.shape == (120, 200, 3)


def test_scenarios_cover_all_four_families_and_never_touch_boxes():
    ops_seen = set()
    boxes = np.array([[40, 30, 120, 60]], dtype=float)
    for seed in range(300):
        y, ops = ph.photometric_augment(_page(), random.Random(seed), boxes)
        ops_seen.update(ops)
        assert y.shape == (120, 200, 3) and y.dtype == np.uint8
    assert ops_seen == {"invert", "regional", "lum", "gradient"}
    assert boxes.tolist() == [[40, 30, 120, 60]]  # augmentation is purely photometric


def test_scenario_probabilities_sum_to_one():
    assert abs(sum(p for p, _ in ph.SCENARIOS) - 1.0) < 1e-9


@pytest.mark.parametrize("name", ph.VARIANT_NAMES)
def test_fixed_variants_are_deterministic(name):
    f = ph.fixed_variant(name)
    boxes = np.array([[40, 30, 120, 60]], dtype=float)
    a, b = f(_page(), boxes), f(_page(), boxes)
    assert np.array_equal(a, b) and a.dtype == np.uint8 and a.shape == (120, 200, 3)


def test_dataset_transform_and_photometric_leave_labels_unchanged(tmp_path):
    from PIL import Image

    from structflo.cser.training.dataset import YoloDetectionDataset

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    Image.fromarray(_page()).save(tmp_path / "images" / "p.png")
    (tmp_path / "labels" / "p.txt").write_text(
        "0 0.4 0.375 0.4 0.25\n1 0.8 0.8 0.1 0.05\n"
    )
    plain = YoloDetectionDataset(tmp_path / "images", imgsz=128, augment=False)[0]
    inv = YoloDetectionDataset(
        tmp_path / "images",
        imgsz=128,
        augment=False,
        transform=ph.fixed_variant("invert"),
    )[0]
    assert np.allclose(plain["labels"]["boxes"].numpy(), inv["labels"]["boxes"].numpy())
    assert not np.array_equal(
        plain["pixel_values"].numpy(), inv["pixel_values"].numpy()
    )
    random.seed(0)
    aug = YoloDetectionDataset(
        tmp_path / "images",
        imgsz=128,
        augment=True,
        scale_jitter=0.0,
        brightness=0.0,
        photometric_aug=1.0,
    )[0]
    assert aug["labels"]["class_labels"].tolist() == [0, 1]
