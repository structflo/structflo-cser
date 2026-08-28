"""Photometric augmentation: shapes, ranges, determinism, coverage of every family, box-awareness."""

from __future__ import annotations

import random

import numpy as np
import pytest

from structflo.cser.training import photometric as ph


def _page(h=120, w=200):
    x = np.full((h, w, 3), 255, dtype=np.uint8)
    x[30:60, 40:120] = 0  # "structure" ink
    x[70:80, 40:80] = 0  # "label" ink
    x[95:110, 150:190] = 128  # a mid-tone patch so gamma / contrast variants are not identities
    return x


BOXES = np.array([[40, 30, 120, 60], [40, 70, 80, 80]], dtype=float)
CLASSES = np.array([0, 1])


@pytest.mark.parametrize(
    "fn", [ph.full_inversion, ph.gradient, ph.luminance_contrast, ph.overlay]
)
def test_families_keep_shape_dtype_range(fn):
    x = _page()
    y = fn(x, random.Random(1))
    assert y.shape == x.shape and y.dtype == np.uint8
    assert (y[..., 0] == y[..., 1]).all() and (y[..., 0] == y[..., 2]).all()


def test_full_inversion_makes_background_dark_and_ink_light():
    for seed in range(20):
        y = ph.full_inversion(_page(), random.Random(seed))
        bg, ink = int(np.median(y[..., 0])), int(np.median(y[35:55, 50:110, 0]))
        assert bg <= 110 + 50 and ink - bg >= 80 - 50  # texture may add ±50


@pytest.mark.parametrize("kind", ph.REGION_KINDS)
def test_every_region_kind_inverts_only_inside_its_mask(kind):
    x = _page()
    y = ph.regional_inversion(
        x, random.Random(3), boxes=BOXES, classes=CLASSES, kind=kind
    )
    changed = y[..., 0] != x[..., 0]
    assert changed.any() and not changed.all()


@pytest.mark.parametrize(
    "kind", ("rects", "title_band", "footer_band", "sidebar", "rows")
)
def test_region_seams_never_cut_through_a_gt_box(kind):
    for seed in range(30):
        rng = random.Random(seed)
        m = ph._region_mask(120, 200, kind, BOXES, CLASSES, rng)
        for x1, y1, x2, y2 in BOXES.astype(int):
            inside = m[y1:y2, x1:x2]
            assert inside.all() or not inside.any(), (kind, seed)


def test_cards_are_anchored_on_structures_and_do_not_slice_labels():
    hits = 0
    for seed in range(60):
        rng = random.Random(seed)
        m = ph._card_mask(120, 200, BOXES, CLASSES, rng)
        lab = m[70:80, 40:80]
        assert lab.all() or not lab.any(), (
            seed
        )  # label fully in or fully out of the card
        hits += int(m[30:60, 40:120].any())
    assert hits > 0


@pytest.mark.parametrize("kind", ph.LUM_KINDS)
def test_every_luminance_kind_runs(kind):
    y = ph.luminance_contrast(_page(), random.Random(5), kind=kind)
    assert y.dtype == np.uint8 and y.shape == (120, 200, 3)


def test_luminance_never_lightens_ink_on_a_dark_page():
    dark = ph.fixed_variant("invert")(_page())
    for seed in range(40):
        y = ph.luminance_contrast(dark, random.Random(seed))
        assert np.median(y[..., 0]) < 200  # ink_lighten is excluded on dark pages


def test_ink_attenuate_pulls_box_ink_towards_background_only_inside_boxes():
    x = _page()
    y = ph.ink_attenuate(x, random.Random(0), BOXES, CLASSES, alpha=0.4)
    assert y[45, 80, 0] > 100  # ink inside a box lifted towards white
    outside = y[..., 0] != x[..., 0]
    outside[30:60, 40:120] = False
    outside[70:80, 40:80] = False
    assert not outside.any()


def test_card_tint_shifts_background_but_not_ink():
    x = _page()
    y = ph.card_tint(x, random.Random(2), BOXES, CLASSES)
    assert (y[..., 0] != x[..., 0]).any()
    assert y[45, 80, 0] <= 5  # ink stays black


def test_scenarios_cover_every_family_and_never_touch_boxes():
    ops_seen = set()
    boxes = BOXES.copy()
    for seed in range(600):
        y, ops = ph.photometric_augment(_page(), random.Random(seed), boxes, CLASSES)
        ops_seen.update(ops)
        assert y.shape == (120, 200, 3) and y.dtype == np.uint8
    assert ops_seen == set(ph.OPS)
    assert np.array_equal(boxes, BOXES)


def test_scenario_probabilities_sum_to_one():
    assert abs(sum(p for p, _ in ph.SCENARIOS) - 1.0) < 1e-9


@pytest.mark.parametrize("name", ph.VARIANT_NAMES + ph.HELDOUT_NAMES)
def test_fixed_variants_are_deterministic(name):
    f = ph.fixed_variant(name)
    a, b = f(_page(), BOXES, CLASSES), f(_page(), BOXES, CLASSES)
    assert np.array_equal(a, b) and a.dtype == np.uint8 and a.shape == (120, 200, 3)
    assert not np.array_equal(a, _page())


def test_fixed_cards_invert_overlapping_cards_exactly_once():
    boxes = np.array(
        [[40, 30, 120, 60], [50, 55, 90, 75]], dtype=float
    )  # overlapping cards
    y = ph.fixed_variant("cards")(_page(), boxes, np.array([0, 1]))
    x = _page()
    overlap = (slice(55, 60), slice(50, 90))
    assert np.array_equal(y[overlap][..., 0], 255 - x[overlap][..., 0])


def test_dataset_transform_and_photometric_leave_labels_unchanged(tmp_path):
    from PIL import Image

    from structflo.cser.training.dataset import YoloDetectionDataset

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    Image.fromarray(_page()).save(tmp_path / "images" / "p.png")
    (tmp_path / "labels" / "p.txt").write_text(
        "0 0.4 0.375 0.4 0.25\n1 0.3 0.625 0.2 0.0833\n"
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
