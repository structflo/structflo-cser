"""Photometric (luminance / polarity) augmentation for document pages.

The detector sees grayscale, so colour is irrelevant; what real slide decks vary is
*polarity* (light text and bonds on dark backgrounds), *regional* polarity (dark
title bars, sidebars, highlighted compound cards, table header rows), *background
luminance / ink contrast* (grey or tinted backgrounds, coloured text that greys to
mid-tones), *gradients* (slide templates, vignettes), non-inverting *tinted cards*,
and translucent *overlays / watermarks*. None of these move a bounding box, so the
annotations are reused verbatim.

Ops (each takes/returns ``uint8`` ``(H, W, 3)`` with identical channels):

* ``full_inversion``     — whole page inverted; dark base lo ∈ [0, 110], light ink hi ∈ [max(lo+80,150), 255]
* ``regional_inversion`` — rects / title & footer bands / sidebars / box-aligned rows / cards
                           (cards are anchored on structure boxes and either union or avoid the
                           neighbouring label); region seams never cut through a GT box
* ``luminance_contrast`` — background darkening, ink lightening, gamma, contrast, offset
                           (kinds conditioned on page polarity)
* ``gradient``           — linear (any angle) or radial luminance ramp
* ``ink_attenuate``      — per-box ink contrast towards the local background (coloured
                           labels / bonds greying to mid-tones, on either polarity)
* ``card_tint``          — non-inverting tinted cards, highlight boxes, zebra rows
* ``overlay``            — translucent rectangles and rotated watermark text
* ``texture``            — low-frequency luminance texture (template art) inside dark regions

``photometric_augment`` samples a scenario from ``SCENARIOS`` and composes ops.
``fixed_variant`` provides deterministic transforms for validation / proxy evaluation
(``VARIANT_NAMES`` = in-distribution stress, ``HELDOUT_NAMES`` = differ in kind from training).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import cv2
import numpy as np

Array = np.ndarray

# scenario → (probability, ops in order)
SCENARIOS: list[tuple[float, tuple[str, ...]]] = [
    (0.20, ("invert",)),
    (0.10, ("invert", "gradient")),
    (0.08, ("invert", "lum")),
    (0.10, ("invert", "regional")),  # light islands on a dark page
    (0.12, ("regional",)),
    (0.08, ("regional", "lum")),
    (0.10, ("card_tint",)),
    (0.05, ("ink_attenuate",)),
    (0.07, ("lum",)),
    (0.05, ("gradient",)),
    (0.05, ("overlay",)),
]
P_INK_ATTENUATE_AFTER_INVERT = 0.3

# The simpler first mix (commit 21540ef): the shipped cser-detector v1.0 checkpoint
# (dfine_l_plus_photo) was trained with it at --photometric-aug 0.3. Kept for reproducibility;
# select with photometric_augment(..., mix="v1") / sf-train --photometric-mix v1.
SCENARIOS_V1: list[tuple[float, tuple[str, ...]]] = [
    (0.22, ("invert",)),
    (0.10, ("invert", "gradient")),
    (0.08, ("invert", "lum")),
    (0.15, ("regional",)),
    (0.10, ("regional", "lum")),
    (0.15, ("lum",)),
    (0.05, ("lum", "lum")),
    (0.10, ("gradient",)),
    (0.05, ("gradient", "lum")),
]
MIXES = {"v2": SCENARIOS, "v1": SCENARIOS_V1}
P_TEXTURE_IN_DARK = 0.3


def _gray(x: Array) -> np.ndarray:
    return x[..., 0].astype(np.float32) if x.ndim == 3 else x.astype(np.float32)


def _pack(y: np.ndarray, like: Array) -> Array:
    y = np.clip(np.rint(y), 0, 255).astype(np.uint8)
    return np.repeat(y[..., None], 3, axis=2) if like.ndim == 3 else y


def _is_dark(g: np.ndarray) -> bool:
    return float(np.median(g)) < 128.0


def _invert_values(v: np.ndarray, rng: random.Random) -> np.ndarray:
    """Invert and re-range: white background → dark base ``lo``, black ink → light ``hi``."""
    lo = rng.uniform(0.0, 110.0)
    hi = rng.uniform(max(lo + 80.0, 150.0), 255.0)
    return lo + (255.0 - v) * (hi - lo) / 255.0


def _texture_field(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Low-frequency luminance field (template art / photo backdrop), amplitude ±[15, 50]."""
    gh, gw = rng.randint(8, 32), rng.randint(8, 32)
    amp = rng.uniform(15.0, 50.0)
    small = np.array(
        [[rng.uniform(-1.0, 1.0) for _ in range(gw)] for _ in range(gh)],
        dtype=np.float32,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amp


def _box_slices(box, h: int, w: int, pad_x: float = 0.0, pad_y: float = 0.0):
    x1, y1, x2, y2 = box
    return (
        slice(max(0, int(y1 - pad_y)), min(h, int(math.ceil(y2 + pad_y)))),
        slice(max(0, int(x1 - pad_x)), min(w, int(math.ceil(x2 + pad_x)))),
    )


def _snap_mask_to_boxes(
    m: np.ndarray, boxes: Array | None, rng: random.Random
) -> np.ndarray:
    """No polarity seam through a GT box: each box the mask boundary crosses is either fully
    covered (+2–5 % margin) or fully excluded (+margin)."""
    if boxes is None or len(boxes) == 0:
        return m
    h, w = m.shape
    for box in boxes:
        ys, xs = _box_slices(box, h, w)
        inside = m[ys, xs]
        if inside.size == 0 or inside.all() or not inside.any():
            continue
        mx = (box[2] - box[0]) * rng.uniform(0.02, 0.05)
        my = (box[3] - box[1]) * rng.uniform(0.02, 0.05)
        ys, xs = _box_slices(box, h, w, mx, my)
        m[ys, xs] = rng.random() < 0.5
    return m


# ---------------------------------------------------------------------------
# 1. full inversion
# ---------------------------------------------------------------------------


def full_inversion(x: Array, rng: random.Random) -> Array:
    g = _invert_values(_gray(x), rng)
    if rng.random() < P_TEXTURE_IN_DARK:
        g = g + _texture_field(g.shape[0], g.shape[1], rng)
    return _pack(g, x)


# ---------------------------------------------------------------------------
# 2. regional inversion (box-aware)
# ---------------------------------------------------------------------------

REGION_KINDS = ("rects", "title_band", "footer_band", "sidebar", "rows", "cards")
REGION_WEIGHTS = (0.28, 0.15, 0.08, 0.15, 0.12, 0.22)


def _boxes_intersect(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _contains(a, b) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _card_mask(
    h: int, w: int, boxes: Array, classes: Array | None, rng: random.Random
) -> np.ndarray:
    """Highlighted compound cards: anchored on structure boxes; a neighbouring label is either
    unioned into the card (p=0.6) or the card stops short of it (p=0.4). Label-only pills at
    ~25 % of card draws."""
    m = np.zeros((h, w), dtype=bool)
    classes = (
        np.zeros(len(boxes), dtype=int) if classes is None else np.asarray(classes)
    )
    structs = [b for b, c in zip(boxes, classes) if c == 0]
    labels = [b for b, c in zip(boxes, classes) if c == 1]
    if rng.random() < 0.25 and labels:  # label pills
        k = rng.randint(1, min(4, len(labels)))
        for b in rng.sample(labels, k):
            bw, bh = b[2] - b[0], b[3] - b[1]
            ys, xs = _box_slices(
                b, h, w, bw * rng.uniform(0.15, 0.4), bh * rng.uniform(0.15, 0.5)
            )
            m[ys, xs] = True
        return m
    if not structs:
        return m
    k = len(structs) if rng.random() < 0.4 else rng.randint(1, min(4, len(structs)))
    for b in rng.sample(structs, k):
        card = list(b)
        pad = rng.uniform(0.03, 0.20)
        px, py = (card[2] - card[0]) * pad, (card[3] - card[1]) * pad
        padded = [card[0] - px, card[1] - py, card[2] + px, card[3] + py]
        for lb in labels:
            if _boxes_intersect(padded, lb) and not _contains(padded, lb):
                if rng.random() < 0.6:  # card includes the label
                    padded = [
                        min(padded[0], lb[0]),
                        min(padded[1], lb[1]),
                        max(padded[2], lb[2]),
                        max(padded[3], lb[3]),
                    ]
                    padded = [
                        padded[0] - px * 0.5,
                        padded[1] - py * 0.5,
                        padded[2] + px * 0.5,
                        padded[3] + py * 0.5,
                    ]
                else:  # card stops short of the label (2 % of the label size)
                    gx, gy = (lb[2] - lb[0]) * 0.02, (lb[3] - lb[1]) * 0.02
                    if lb[1] >= card[3]:
                        padded[3] = min(padded[3], lb[1] - gy)
                    elif lb[3] <= card[1]:
                        padded[1] = max(padded[1], lb[3] + gy)
                    elif lb[0] >= card[2]:
                        padded[2] = min(padded[2], lb[0] - gx)
                    elif lb[2] <= card[0]:
                        padded[0] = max(padded[0], lb[2] + gx)
        ys, xs = _box_slices(padded, h, w)
        m[ys, xs] = True
    return m


def _region_mask(
    h: int,
    w: int,
    kind: str,
    boxes: Array | None,
    classes: Array | None,
    rng: random.Random,
) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    if kind == "rects":
        for _ in range(rng.randint(1, 3)):
            area = rng.uniform(0.05, 0.5)
            ar = math.exp(rng.uniform(math.log(0.4), math.log(2.5)))
            rw = min(w, int(math.sqrt(area * w * h * ar)))
            rh = min(h, int(math.sqrt(area * w * h / ar)))
            x0 = rng.randint(0, max(0, w - rw))
            y0 = rng.randint(0, max(0, h - rh))
            m[y0 : y0 + rh, x0 : x0 + rw] = True
    elif kind == "title_band":
        m[: int(h * rng.uniform(0.06, 0.22))] = True
    elif kind == "footer_band":
        m[h - int(h * rng.uniform(0.04, 0.15)) :] = True
    elif kind == "sidebar":
        cw = int(w * rng.uniform(0.12, 0.35))
        if rng.random() < 0.5:
            m[:, :cw] = True
        else:
            m[:, w - cw :] = True
    elif kind == "rows":  # table header / zebra rows aligned to a box's vertical extent
        if boxes is None or len(boxes) == 0:
            return _region_mask(h, w, "title_band", boxes, classes, rng)
        for b in rng.sample(list(boxes), rng.randint(1, min(3, len(boxes)))):
            my = (b[3] - b[1]) * rng.uniform(0.05, 0.4)
            m[max(0, int(b[1] - my)) : min(h, int(math.ceil(b[3] + my)))] = True
    elif kind == "cards":
        if boxes is None or len(boxes) == 0:
            return _region_mask(h, w, "rects", boxes, classes, rng)
        m = _card_mask(h, w, boxes, classes, rng)
    if kind != "cards":
        m = _snap_mask_to_boxes(m, boxes, rng)
    return m


def regional_inversion(
    x: Array,
    rng: random.Random,
    boxes: Array | None = None,
    classes: Array | None = None,
    kind: str | None = None,
) -> Array:
    """Invert one or more regions (box-aware seams); ``boxes``/``classes`` enable rows and cards."""
    h, w = x.shape[:2]
    kind = kind or rng.choices(REGION_KINDS, weights=REGION_WEIGHTS)[0]
    m = _region_mask(h, w, kind, boxes, classes, rng)
    g = _gray(x)
    if m.any():
        g[m] = _invert_values(g[m], rng)
        if rng.random() < P_TEXTURE_IN_DARK:
            g[m] = g[m] + _texture_field(h, w, rng)[m]
    return _pack(g, x)


# ---------------------------------------------------------------------------
# 3. luminance / contrast (polarity-aware)
# ---------------------------------------------------------------------------

LUM_KINDS = ("bg_darken", "ink_lighten", "gamma", "contrast", "offset")
LUM_WEIGHTS_LIGHT = (0.30, 0.25, 0.15, 0.15, 0.15)
LUM_WEIGHTS_DARK = (
    0.30,
    0.00,
    0.25,
    0.25,
    0.20,
)  # ink_lighten would lift a dark page to grey


def luminance_contrast(x: Array, rng: random.Random, kind: str | None = None) -> Array:
    g = _gray(x)
    if kind is None:
        weights = LUM_WEIGHTS_DARK if _is_dark(g) else LUM_WEIGHTS_LIGHT
        kind = rng.choices(LUM_KINDS, weights=weights)[0]
    if kind == "bg_darken":  # grey / tinted background, ink follows
        g = g * rng.uniform(0.45, 0.95)
    elif (
        kind == "ink_lighten"
    ):  # coloured text/bonds greying to mid-tones on a white page
        g = 255.0 - (255.0 - g) * rng.uniform(0.35, 0.9)
    elif kind == "gamma":
        g = 255.0 * (g / 255.0) ** rng.uniform(0.5, 1.8)
    elif kind == "contrast":
        g = 128.0 + (g - 128.0) * rng.uniform(0.5, 1.25)
    elif kind == "offset":
        g = g + rng.uniform(-40.0, 40.0)
    return _pack(g, x)


# ---------------------------------------------------------------------------
# 4. gradients
# ---------------------------------------------------------------------------


def gradient_field(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Multiplicative luminance field in ``[lo, 1]``: linear ramp at a random angle or radial."""
    lo = rng.uniform(0.35, 0.85)
    xx = np.arange(w, dtype=np.float32)[None, :]
    yy = np.arange(h, dtype=np.float32)[:, None]
    if rng.random() < 0.65:
        theta = rng.uniform(0, 2 * math.pi)
        t = (xx / max(w - 1, 1)) * math.cos(theta) + (yy / max(h - 1, 1)) * math.sin(
            theta
        )
        t = (t - t.min()) / max(float(t.max() - t.min()), 1e-6)
    else:
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        t = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        t = t / max(float(t.max()), 1e-6)
        if rng.random() < 0.5:
            t = 1.0 - t  # bright rim, dark centre
    return lo + (1.0 - lo) * (1.0 - t)


def gradient(x: Array, rng: random.Random) -> Array:
    return _pack(_gray(x) * gradient_field(x.shape[0], x.shape[1], rng), x)


# ---------------------------------------------------------------------------
# 5. per-box ink attenuation (coloured labels / bonds greying to mid-tones)
# ---------------------------------------------------------------------------


def _local_background(g: np.ndarray, box, h: int, w: int) -> float:
    """Median luminance of a 4–8 px ring just outside the box (falls back to the page median)."""
    ys, xs = _box_slices(box, h, w, 8, 8)
    outer = g[ys, xs]
    iy, ix = _box_slices(box, h, w)
    inner = np.zeros_like(outer, dtype=bool)
    inner[
        (iy.start - ys.start) : (iy.stop - ys.start),
        (ix.start - xs.start) : (ix.stop - xs.start),
    ] = True
    ring = outer[~inner]
    return float(np.median(ring)) if ring.size else float(np.median(g))


def ink_attenuate(
    x: Array,
    rng: random.Random,
    boxes: Array | None = None,
    classes: Array | None = None,
    alpha: float | None = None,
) -> Array:
    """Pull the ink inside chosen GT boxes towards the local background: ``g = bg + (g - bg)·a``,
    a ∈ [0.15, 0.8] (a < 0.3 ≈ yellow-on-white / dark-red-on-black)."""
    if boxes is None or len(boxes) == 0:
        return x
    g = _gray(x)
    h, w = g.shape
    classes = (
        np.zeros(len(boxes), dtype=int) if classes is None else np.asarray(classes)
    )
    target = 1 if rng.random() < 0.6 else 0
    idx = [i for i, c in enumerate(classes) if c == target] or list(range(len(boxes)))
    shared = rng.uniform(0.15, 0.8) if rng.random() < 0.7 else None
    for i in idx:
        a = (
            alpha
            if alpha is not None
            else (shared if shared is not None else rng.uniform(0.15, 0.8))
        )
        bg = _local_background(g, boxes[i], h, w)
        ys, xs = _box_slices(boxes[i], h, w)
        g[ys, xs] = bg + (g[ys, xs] - bg) * a
    return _pack(g, x)


# ---------------------------------------------------------------------------
# 6. non-inverting tinted cards / highlight boxes / zebra rows
# ---------------------------------------------------------------------------


def card_tint(
    x: Array,
    rng: random.Random,
    boxes: Array | None = None,
    classes: Array | None = None,
) -> Array:
    """Shift the *background* inside card regions (light page: darker card; dark page: lighter
    card); ink is left alone. Optional 1–3 px border."""
    if boxes is None or len(boxes) == 0:
        return x
    g = _gray(x)
    h, w = g.shape
    dark = _is_dark(g)
    classes = (
        np.zeros(len(boxes), dtype=int) if classes is None else np.asarray(classes)
    )
    structs = [b for b, c in zip(boxes, classes) if c == 0] or list(boxes)
    labels = [b for b, c in zip(boxes, classes) if c == 1]
    regions = []
    for b in rng.sample(structs, rng.randint(1, min(4, len(structs)))):
        card = list(b)
        r = rng.random()
        if r < 0.6 and labels:  # struct ∪ nearest label
            near = min(
                labels,
                key=lambda lb: (
                    (lb[0] + lb[2] - b[0] - b[2]) ** 2
                    + (lb[1] + lb[3] - b[1] - b[3]) ** 2
                ),
            )
            card = [
                min(b[0], near[0]),
                min(b[1], near[1]),
                max(b[2], near[2]),
                max(b[3], near[3]),
            ]
        pad = rng.uniform(0.03, 0.25)
        px, py = (card[2] - card[0]) * pad, (card[3] - card[1]) * pad
        if rng.random() < 0.2:  # zebra row: full width, the card's vertical extent
            regions.append([0, card[1] - py, w, card[3] + py])
        else:
            regions.append([card[0] - px, card[1] - py, card[2] + px, card[3] + py])
    for reg in regions:
        ys, xs = _box_slices(reg, h, w)
        patch = g[ys, xs]
        if dark:
            bgmask = patch < 128
            patch[bgmask] = patch[bgmask] + rng.uniform(20.0, 90.0)
        else:
            t = rng.uniform(0.05, 0.45)
            bgmask = patch > 128
            patch[bgmask] = (
                patch[bgmask] - 255.0 * t
            )  # white 255 → 140..242, ink untouched
        g[ys, xs] = patch
        if rng.random() < 0.5:  # border
            bw_ = rng.randint(1, 3)
            delta = rng.uniform(40.0, 120.0) * (1.0 if dark else -1.0)
            g[ys.start : ys.start + bw_, xs] += delta
            g[max(ys.start, ys.stop - bw_) : ys.stop, xs] += delta
            g[ys, xs.start : xs.start + bw_] += delta
            g[ys, max(xs.start, xs.stop - bw_) : xs.stop] += delta
    return _pack(g, x)


# ---------------------------------------------------------------------------
# 7. overlays / watermarks
# ---------------------------------------------------------------------------

_WORDS = (
    "CONFIDENTIAL",
    "DRAFT",
    "INTERNAL",
    "DO NOT SHARE",
    "PRELIMINARY",
    "PROPRIETARY",
    "FOR REVIEW",
    "SAMPLE",
)


def overlay(x: Array, rng: random.Random, kind: str | None = None) -> Array:
    """Translucent rectangles (``kind="rect"``) or a rotated watermark word (``kind="text"``)."""
    g = _gray(x)
    h, w = g.shape
    kind = kind or ("rect" if rng.random() < 0.5 else "text")
    if kind == "rect":
        for _ in range(rng.randint(1, 2)):
            area = rng.uniform(0.05, 0.4)
            rw = min(w, max(1, int(math.sqrt(area * w * h * rng.uniform(0.5, 2.0)))))
            rh = min(h, max(1, int(area * w * h / rw)))
            x0, y0 = rng.randint(0, max(0, w - rw)), rng.randint(0, max(0, h - rh))
            a = rng.uniform(0.25, 0.7)
            c = (
                rng.uniform(0.0, 60.0)
                if rng.random() < 0.5
                else rng.uniform(200.0, 255.0)
            )
            g[y0 : y0 + rh, x0 : x0 + rw] = (1 - a) * g[
                y0 : y0 + rh, x0 : x0 + rw
            ] + a * c
    else:
        canvas = np.zeros((h, w), dtype=np.uint8)  # OpenCV text drawing needs uint8
        word = rng.choice(_WORDS)
        scale = max(
            0.5, h * rng.uniform(0.4, 1.2) / 30.0
        )  # Hershey glyph height ≈ 30 px at scale 1
        thick = max(2, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        org = (max(0, (w - tw) // 2), min(h - 1, (h + th) // 2))
        cv2.putText(
            canvas, word, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick, cv2.LINE_AA
        )
        ang = rng.uniform(20.0, 45.0) * (1 if rng.random() < 0.5 else -1)
        rot = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        mask = cv2.warpAffine(canvas, rot, (w, h)).astype(np.float32) / 255.0
        a = rng.uniform(0.10, 0.35)
        c = 255.0 if _is_dark(g) else 0.0
        g = g * (1 - a * mask) + a * mask * c
    return _pack(g, x)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

OPS: dict[str, Callable[..., Array]] = {
    "invert": lambda x, rng, b, c: full_inversion(x, rng),
    "regional": lambda x, rng, b, c: regional_inversion(x, rng, b, c),
    "lum": lambda x, rng, b, c: luminance_contrast(x, rng),
    "gradient": lambda x, rng, b, c: gradient(x, rng),
    "ink_attenuate": lambda x, rng, b, c: ink_attenuate(x, rng, b, c),
    "card_tint": lambda x, rng, b, c: card_tint(x, rng, b, c),
    "overlay": lambda x, rng, b, c: overlay(x, rng),
}


def photometric_augment(
    x: Array,
    rng: random.Random,
    boxes: Array | None = None,
    classes: Array | None = None,
    mix: str = "v2",
) -> tuple[Array, tuple[str, ...]]:
    """Apply one sampled scenario from ``MIXES[mix]``. Returns ``(image, ops)``; boxes untouched."""
    scenarios = MIXES[mix]
    ops = list(
        rng.choices([o for _, o in scenarios], weights=[p for p, _ in scenarios])[0]
    )
    if mix == "v2" and "invert" in ops and rng.random() < P_INK_ATTENUATE_AFTER_INVERT:
        ops.append("ink_attenuate")
    for op in ops:
        x = OPS[op](x, rng, boxes, classes)
    return x, tuple(ops)


# ---------------------------------------------------------------------------
# deterministic variants (validation / proxy evaluation)
# ---------------------------------------------------------------------------

Transform = Callable[[Array, Array | None, Array | None], Array]


def _fixed_invert(x: Array, lo: float = 0.0, hi: float = 255.0) -> Array:
    return _pack(lo + (255.0 - _gray(x)) * (hi - lo) / 255.0, x)


def _fixed_cards(x: Array, boxes: Array | None, pad: float = 0.2) -> Array:
    """Invert every GT box (structures and labels) padded by ``pad`` — one OR-mask, so
    overlapping cards are inverted exactly once."""
    g = _gray(x)
    if boxes is not None and len(boxes):
        h, w = g.shape
        m = np.zeros((h, w), dtype=bool)
        for x1, y1, x2, y2 in boxes:
            ys, xs = _box_slices(
                (x1, y1, x2, y2), h, w, (x2 - x1) * pad, (y2 - y1) * pad
            )
            m[ys, xs] = True
        g[m] = 255.0 - g[m]
    return _pack(g, x)


def _jpeg(x: Array, q: int) -> Array:
    ok, buf = cv2.imencode(".jpg", x[..., 0], [cv2.IMWRITE_JPEG_QUALITY, q])
    y = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return np.repeat(y[..., None], 3, axis=2)


def fixed_variant(name: str) -> Transform:
    """Deterministic page transforms ``f(arr, boxes_xyxy_px, classes) -> arr``."""
    # --- in-distribution stress (interior points of the training ranges) ---
    if name == "invert":
        return lambda x, b=None, c=None: _fixed_invert(x)
    if name == "invert_grey":
        return lambda x, b=None, c=None: _fixed_invert(x, 40.0, 230.0)
    if name == "invert_slate":
        return lambda x, b=None, c=None: _fixed_invert(x, 90.0, 235.0)
    if name == "title_band":

        def f(x, b=None, c=None):
            g = _gray(x)
            k = int(x.shape[0] * 0.15)
            g[:k] = 255.0 - g[:k]
            return _pack(g, x)

        return f
    if name == "sidebar":

        def f(x, b=None, c=None):
            g = _gray(x)
            k = int(x.shape[1] * 0.25)
            g[:, :k] = 255.0 - g[:, :k]
            return _pack(g, x)

        return f
    if name == "cards":
        return lambda x, b=None, c=None: _fixed_cards(x, b)
    if name == "bg150":
        return lambda x, b=None, c=None: _pack(_gray(x) * (150.0 / 255.0), x)
    if name == "ink_light":
        return lambda x, b=None, c=None: _pack(255.0 - (255.0 - _gray(x)) * 0.5, x)
    if name == "gamma06":
        return lambda x, b=None, c=None: _pack(255.0 * (_gray(x) / 255.0) ** 0.6, x)
    if name == "gamma16":
        return lambda x, b=None, c=None: _pack(255.0 * (_gray(x) / 255.0) ** 1.6, x)
    if name == "grad_linear":

        def f(x, b=None, c=None):
            t = np.linspace(1.0, 0.4, x.shape[0], dtype=np.float32)[:, None]
            return _pack(_gray(x) * t, x)

        return f
    if name == "grad_radial":

        def f(x, b=None, c=None):
            h, w = x.shape[:2]
            xx = np.arange(w, dtype=np.float32)[None, :]
            yy = np.arange(h, dtype=np.float32)[:, None]
            t = np.sqrt(
                ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
            ) / math.sqrt(2)
            return _pack(_gray(x) * (1.0 - 0.6 * t), x)

        return f
    if name == "invert_grad":
        lin = fixed_variant("grad_linear")
        return lambda x, b=None, c=None: lin(_fixed_invert(x), b, c)
    # --- held-out: differ in KIND from anything the training sampler produces ---
    if (
        name == "ho_invert_jpeg50"
    ):  # inverted page then JPEG re-encoded (ringing on light strokes)
        return lambda x, b=None, c=None: _jpeg(_fixed_invert(x), 50)
    if (
        name == "ho_invert_lowcontrast"
    ):  # contrast 110 < the trained minimum of 80..(hi-lo)
        return lambda x, b=None, c=None: _fixed_invert(x, 20.0, 130.0)
    if (
        name == "ho_invert_dim_labels"
    ):  # inverted page, label ink pulled to mid-grey (a=0.45)

        def f(x, b=None, c=None):
            y = _fixed_invert(x)
            if b is None or c is None:
                return y
            lb = np.asarray(b)[np.asarray(c) == 1]
            return (
                ink_attenuate(
                    y, random.Random(0), lb, np.ones(len(lb), dtype=int), alpha=0.45
                )
                if len(lb)
                else y
            )

        return f
    if name == "ho_invert_inset":  # dark page with a light inset panel (light island)

        def f(x, b=None, c=None):
            y = _fixed_invert(x)
            g = _gray(y)
            h, w = g.shape
            ys, xs = (
                slice(int(h * 0.25), int(h * 0.85)),
                slice(int(w * 0.3), int(w * 0.95)),
            )
            g[ys, xs] = 255.0 - g[ys, xs]
            return _pack(g, y)

        return f
    if name == "ho_invert_noise":  # inverted + Gaussian noise σ=8

        def f(x, b=None, c=None):
            g = _gray(_fixed_invert(x))
            g = g + np.random.default_rng(0).normal(0.0, 8.0, g.shape).astype(
                np.float32
            )
            return _pack(g, x)

        return f
    if (
        name == "ho_grid_invert"
    ):  # 3x3 grid, alternate cells inverted — not aligned to GT boxes

        def f(x, b=None, c=None):
            g = _gray(x)
            h, w = g.shape
            for i in range(3):
                for j in range(3):
                    if (i + j) % 2 == 0:
                        ys, xs = (
                            slice(i * h // 3, (i + 1) * h // 3),
                            slice(j * w // 3, (j + 1) * w // 3),
                        )
                        g[ys, xs] = 255.0 - g[ys, xs]
            return _pack(g, x)

        return f
    raise KeyError(name)


VARIANT_GROUPS: dict[str, tuple[str, ...]] = {
    "polarity": ("invert", "invert_grey", "invert_slate", "invert_grad"),
    "regional": ("cards", "sidebar", "title_band"),
    "luminance": (
        "bg150",
        "ink_light",
        "gamma06",
        "gamma16",
        "grad_linear",
        "grad_radial",
    ),
    "heldout": (
        "ho_invert_jpeg50",
        "ho_invert_lowcontrast",
        "ho_invert_dim_labels",
        "ho_invert_inset",
        "ho_invert_noise",
        "ho_grid_invert",
    ),
}
VARIANT_NAMES = (
    VARIANT_GROUPS["polarity"]
    + VARIANT_GROUPS["regional"]
    + VARIANT_GROUPS["luminance"]
)
HELDOUT_NAMES = VARIANT_GROUPS["heldout"]
