"""Letterbox preprocessing shared by training and inference.

The detector is fully convolutional, but training and inference must agree on
how a page is fitted into the ``imgsz``×``imgsz`` canvas. We keep aspect ratio
(A4 portrait pages and 16:9 slides both occur), scale the long side to
``imgsz``, centre the image and pad with ``PAD_VALUE`` — the same geometry the
retired YOLO detector used, so box scale statistics downstream are unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np

PAD_VALUE = 114


def letterbox(
    img: np.ndarray,
    size: int,
    *,
    scale: float | None = None,
    offset: tuple[int, int] | None = None,
    pad_value: int = PAD_VALUE,
) -> tuple[np.ndarray, float, int, int]:
    """Fit ``img`` (H,W,3 uint8) into a ``size``×``size`` canvas.

    Args:
        scale:  Resize factor. ``None`` → long side becomes ``size``.
        offset: (dx, dy) top-left of the resized image on the canvas.
                ``None`` → centred. The resized image is cropped where it
                overhangs the canvas (used by scale-jitter augmentation).

    Returns:
        (canvas, scale, dx, dy) — map canvas coords back with
        ``(x - dx) / scale``.
    """
    h, w = img.shape[:2]
    if scale is None:
        scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    if offset is None:
        dx, dy = (size - new_w) // 2, (size - new_h) // 2
    else:
        dx, dy = offset
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    # visible window of the resized image on the canvas
    sx1, sy1 = max(0, -dx), max(0, -dy)
    cx1, cy1 = max(0, dx), max(0, dy)
    vis_w = min(new_w - sx1, size - cx1)
    vis_h = min(new_h - sy1, size - cy1)
    if vis_w > 0 and vis_h > 0:
        canvas[cy1 : cy1 + vis_h, cx1 : cx1 + vis_w] = resized[
            sy1 : sy1 + vis_h, sx1 : sx1 + vis_w
        ]
    return canvas, scale, dx, dy
