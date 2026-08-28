"""Photometric (luminance / polarity) augmentation for document pages.

The detector sees grayscale, so colour is irrelevant; what real slides vary is
*polarity* (light text and bonds on dark backgrounds), *regional* polarity
(dark title bars, sidebars, highlighted compound cards, table header rows),
*background luminance / ink contrast* (grey or tinted backgrounds, coloured text
that greys to mid-tones) and *gradients* (slide templates, vignettes). None of
these move a bounding box, so the annotations are reused verbatim.

Four families, each with its own parameter ranges:

1. ``full_inversion``       — whole page inverted, dark base in [0, 60], light ink in [180, 255]
2. ``regional_inversion``   — rectangles, title/footer bands, sidebars, header stripes,
                              or "cards" around ground-truth boxes
3. ``luminance_contrast``   — background darkening, ink lightening, gamma, contrast, offset
4. ``gradient``             — linear (any angle) or radial luminance ramp

``photometric_augment`` samples a scenario that composes them (see ``SCENARIOS``).
All functions take/return ``uint8`` arrays ``(H, W, 3)`` with identical channels
and operate in float32 internally.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import numpy as np

Array = np.ndarray

# scenario → (probability, ops in order); ops may repeat "lum" for stacking
SCENARIOS: list[tuple[float, tuple[str, ...]]] = [
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


def _gray(x: Array) -> np.ndarray:
    return x[..., 0].astype(np.float32) if x.ndim == 3 else x.astype(np.float32)


def _pack(y: np.ndarray, like: Array) -> Array:
    y = np.clip(np.rint(y), 0, 255).astype(np.uint8)
    return np.repeat(y[..., None], 3, axis=2) if like.ndim == 3 else y


def _invert_values(
    v: np.ndarray, rng: random.Random, lo_max: float = 60.0, hi_min: float = 180.0
) -> np.ndarray:
    """Invert and re-range: white background → dark base ``lo``, black ink → light ``hi``."""
    lo = rng.uniform(0.0, lo_max)
    hi = rng.uniform(hi_min, 255.0)
    return lo + (255.0 - v) * (hi - lo) / 255.0


# ---------------------------------------------------------------------------
# 1. full inversion
# ---------------------------------------------------------------------------


def full_inversion(x: Array, rng: random.Random) -> Array:
    return _pack(_invert_values(_gray(x), rng), x)


# ---------------------------------------------------------------------------
# 2. regional inversion
# ---------------------------------------------------------------------------

REGION_KINDS = ("rects", "title_band", "footer_band", "sidebar", "stripes", "cards")
REGION_WEIGHTS = (0.30, 0.15, 0.08, 0.15, 0.10, 0.22)


def _region_mask(
    h: int, w: int, kind: str, boxes: Array | None, rng: random.Random
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
    elif kind == "stripes":
        for _ in range(rng.randint(2, 6)):
            sh = max(2, int(h * rng.uniform(0.02, 0.06)))
            y0 = rng.randint(0, max(0, h - sh))
            m[y0 : y0 + sh] = True
    elif kind == "cards":
        if boxes is None or len(boxes) == 0:
            return _region_mask(h, w, "rects", boxes, rng)
        n = len(boxes)
        k = n if rng.random() < 0.4 else rng.randint(1, min(4, n))
        for i in rng.sample(range(n), k):
            x1, y1, x2, y2 = boxes[i]
            px = (x2 - x1) * rng.uniform(0.05, 0.4)
            py = (y2 - y1) * rng.uniform(0.05, 0.4)
            m[
                max(0, int(y1 - py)) : min(h, int(y2 + py) + 1),
                max(0, int(x1 - px)) : min(w, int(x2 + px) + 1),
            ] = True
    return m


def regional_inversion(
    x: Array, rng: random.Random, boxes: Array | None = None, kind: str | None = None
) -> Array:
    """Invert one or more regions; ``boxes`` (xyxy pixels) enable the "cards" kind."""
    h, w = x.shape[:2]
    kind = kind or rng.choices(REGION_KINDS, weights=REGION_WEIGHTS)[0]
    m = _region_mask(h, w, kind, boxes, rng)
    g = _gray(x)
    if m.any():
        g[m] = _invert_values(g[m], rng)
    return _pack(g, x)


# ---------------------------------------------------------------------------
# 3. luminance / contrast
# ---------------------------------------------------------------------------

LUM_KINDS = ("bg_darken", "ink_lighten", "gamma", "contrast", "offset")
LUM_WEIGHTS = (0.30, 0.25, 0.15, 0.15, 0.15)


def luminance_contrast(x: Array, rng: random.Random, kind: str | None = None) -> Array:
    g = _gray(x)
    kind = kind or rng.choices(LUM_KINDS, weights=LUM_WEIGHTS)[0]
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
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if rng.random() < 0.65:
        theta = rng.uniform(0, 2 * math.pi)
        t = (xx / max(w - 1, 1)) * math.cos(theta) + (yy / max(h - 1, 1)) * math.sin(
            theta
        )
        t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
    else:
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        t = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        t = t / max(t.max(), 1e-6)
        if rng.random() < 0.5:
            t = 1.0 - t  # bright rim, dark centre
    return lo + (1.0 - lo) * (1.0 - t)


def gradient(x: Array, rng: random.Random) -> Array:
    return _pack(_gray(x) * gradient_field(x.shape[0], x.shape[1], rng), x)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

OPS: dict[str, Callable[..., Array]] = {
    "invert": lambda x, rng, boxes: full_inversion(x, rng),
    "regional": lambda x, rng, boxes: regional_inversion(x, rng, boxes),
    "lum": lambda x, rng, boxes: luminance_contrast(x, rng),
    "gradient": lambda x, rng, boxes: gradient(x, rng),
}


def photometric_augment(
    x: Array, rng: random.Random, boxes: Array | None = None
) -> tuple[Array, tuple[str, ...]]:
    """Apply one sampled scenario. Returns ``(image, ops_applied)``; boxes are untouched."""
    probs = [p for p, _ in SCENARIOS]
    ops = rng.choices([o for _, o in SCENARIOS], weights=probs)[0]
    for op in ops:
        x = OPS[op](x, rng, boxes)
    return x, ops


# ---------------------------------------------------------------------------
# deterministic variants (validation / proxy evaluation)
# ---------------------------------------------------------------------------


def _fixed_invert(x: Array, lo: float = 0.0, hi: float = 255.0) -> Array:
    return _pack(lo + (255.0 - _gray(x)) * (hi - lo) / 255.0, x)


def fixed_variant(name: str) -> Callable[[Array, Array | None], Array]:
    """Deterministic page transforms by name (no randomness) — for val selection and proxy eval."""
    if name == "invert":
        return lambda x, b=None: _fixed_invert(x)
    if name == "invert_grey":
        return lambda x, b=None: _fixed_invert(x, 40.0, 230.0)
    if name == "title_band":

        def f(x, b=None):
            g = _gray(x)
            h = x.shape[0]
            g[: int(h * 0.15)] = 255.0 - g[: int(h * 0.15)]
            return _pack(g, x)

        return f
    if name == "sidebar":

        def f(x, b=None):
            g = _gray(x)
            w = x.shape[1]
            g[:, : int(w * 0.25)] = 255.0 - g[:, : int(w * 0.25)]
            return _pack(g, x)

        return f
    if name == "cards":

        def f(x, b=None):
            g = _gray(x)
            if b is not None:
                for x1, y1, x2, y2 in b:
                    px, py = (x2 - x1) * 0.2, (y2 - y1) * 0.2
                    ys, ye = max(0, int(y1 - py)), min(x.shape[0], int(y2 + py) + 1)
                    xs, xe = max(0, int(x1 - px)), min(x.shape[1], int(x2 + px) + 1)
                    g[ys:ye, xs:xe] = 255.0 - g[ys:ye, xs:xe]
            return _pack(g, x)

        return f
    if name == "bg150":
        return lambda x, b=None: _pack(_gray(x) * (150.0 / 255.0), x)
    if name == "ink_light":
        return lambda x, b=None: _pack(255.0 - (255.0 - _gray(x)) * 0.5, x)
    if name == "gamma06":
        return lambda x, b=None: _pack(255.0 * (_gray(x) / 255.0) ** 0.6, x)
    if name == "gamma16":
        return lambda x, b=None: _pack(255.0 * (_gray(x) / 255.0) ** 1.6, x)
    if name == "grad_linear":

        def f(x, b=None):
            h = x.shape[0]
            t = np.linspace(1.0, 0.4, h, dtype=np.float32)[:, None]
            return _pack(_gray(x) * t, x)

        return f
    if name == "grad_radial":

        def f(x, b=None):
            h, w = x.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            t = np.sqrt(
                ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
            ) / math.sqrt(2)
            return _pack(_gray(x) * (1.0 - 0.6 * t), x)

        return f
    if name == "invert_grad":
        lin = fixed_variant("grad_linear")
        return lambda x, b=None: lin(_fixed_invert(x), b)
    raise KeyError(name)


VARIANT_NAMES = (
    "invert",
    "invert_grey",
    "title_band",
    "sidebar",
    "cards",
    "bg150",
    "ink_light",
    "gamma06",
    "gamma16",
    "grad_linear",
    "grad_radial",
    "invert_grad",
)
