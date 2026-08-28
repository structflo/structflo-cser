"""YOLO-txt detection dataset for the D-FINE trainer.

Reads the same on-disk layout the retired YOLO trainer used
(``<root>/images/*.jpg`` + ``<root>/labels/*.txt``, ``cls cx cy w h`` normalised)
so every existing data.yaml, synthetic corpus and fine-tune split works
unchanged. Produces letterboxed ``imgsz``×``imgsz`` tensors and HF-style
targets (normalised cxcywh on the canvas).

Augmentation (train only) is the document-appropriate subset of the old YOLO
recipe: scale jitter, random placement (translate), brightness jitter. No
flips (chemical handedness), no hue/saturation (grayscale input), no mosaic.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from structflo.cser.inference.metrics import load_yolo_labels
from structflo.cser.inference.preprocess import PAD_VALUE, letterbox

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def labels_dir_for(images_dir: Path) -> Path:
    """``.../images`` → ``.../labels`` (YOLO convention)."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels"


class YoloDetectionDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path | None = None,
        *,
        imgsz: int = 1280,
        augment: bool = False,
        scale_jitter: float = 0.3,
        brightness: float = 0.1,
        downscale_aug: float = 0.0,
        grayscale: bool = True,
        pad_value: int = PAD_VALUE,
        min_box_px: float = 2.0,
        min_visible_frac: float = 0.25,
        limit: int | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = (
            Path(labels_dir) if labels_dir else labels_dir_for(self.images_dir)
        )
        self.files = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"no images in {self.images_dir}")
        self.imgsz = imgsz
        self.augment = augment
        self.scale_jitter = scale_jitter
        self.brightness = brightness
        # Probability of pre-downscaling the decoded page by a random factor in [0.4, 1] with a
        # random interpolation before the letterbox. Mimics lower-DPI renders / different
        # rasterisation chains (deployment renders at 144-150 dpi, annotation pages are 300 dpi)
        # so the detector is not tied to one resampling pipeline.
        self.downscale_aug = downscale_aug
        self.grayscale = grayscale
        self.pad_value = pad_value
        self.min_box_px = min_box_px
        self.min_visible_frac = min_visible_frac

    def __len__(self) -> int:
        return len(self.files)

    # -- image decode (JPEG DCT-domain downscale keeps the loader GPU-bound) --

    def _load(self, path: Path, target_long: float) -> tuple[np.ndarray, int, int]:
        im = Image.open(path)
        orig_w, orig_h = im.size
        if im.format == "JPEG":
            s = target_long / max(orig_w, orig_h)
            if s < 1.0:
                im.draft(
                    "L" if self.grayscale else "RGB",
                    (math.ceil(orig_w * s), math.ceil(orig_h * s)),
                )
        im = im.convert("L") if self.grayscale else im.convert("RGB")
        arr = np.asarray(im)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        return arr, orig_w, orig_h

    def __getitem__(self, i: int) -> dict:
        path = self.files[i]
        S = self.imgsz
        jitter = 1.0
        if self.augment and self.scale_jitter > 0:
            jitter = random.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter)
        arr, orig_w, orig_h = self._load(path, S * jitter)
        if (
            self.augment
            and self.downscale_aug > 0
            and random.random() < self.downscale_aug
        ):
            f = random.uniform(0.4, 1.0)
            interp = random.choice(
                [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
            )
            arr = cv2.resize(
                arr,
                (max(1, round(arr.shape[1] * f)), max(1, round(arr.shape[0] * f))),
                interpolation=interp,
            )
        h, w = arr.shape[:2]
        boxes, classes = load_yolo_labels(self.labels_dir / f"{path.stem}.txt", w, h)

        if self.augment and self.brightness > 0:
            f = random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
            arr = np.clip(arr.astype(np.float32) * f, 0, 255).astype(np.uint8)

        scale = S / max(h, w) * jitter
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        if self.augment:
            dx = (
                random.randint(0, S - new_w)
                if new_w <= S
                else -random.randint(0, new_w - S)
            )
            dy = (
                random.randint(0, S - new_h)
                if new_h <= S
                else -random.randint(0, new_h - S)
            )
            offset = (dx, dy)
        else:
            offset = None
        canvas, scale, dx, dy = letterbox(
            arr, S, scale=scale, offset=offset, pad_value=self.pad_value
        )

        # boxes → canvas pixels, clip, drop what the crop removed
        if len(boxes):
            b = boxes * scale
            b[:, [0, 2]] += dx
            b[:, [1, 3]] += dy
            area0 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            b = np.clip(b, 0, S)
            bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
            keep = (
                (bw >= self.min_box_px)
                & (bh >= self.min_box_px)
                & (bw * bh >= self.min_visible_frac * np.maximum(area0, 1e-6))
            )
            b, classes = b[keep], classes[keep]
            cxcywh = (
                np.stack(
                    [
                        (b[:, 0] + b[:, 2]) / 2,
                        (b[:, 1] + b[:, 3]) / 2,
                        b[:, 2] - b[:, 0],
                        b[:, 3] - b[:, 1],
                    ],
                    axis=1,
                )
                / S
            )
        else:
            cxcywh = np.zeros((0, 4), dtype=np.float64)

        pixel_values = (
            torch.from_numpy(np.ascontiguousarray(canvas))
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        return {
            "pixel_values": pixel_values,
            "labels": {
                "class_labels": torch.as_tensor(classes, dtype=torch.long),
                "boxes": torch.as_tensor(cxcywh, dtype=torch.float32),
            },
            "meta": {
                "stem": path.stem,
                "file": path.name,
                "orig_w": orig_w,
                "orig_h": orig_h,
                # combined map: original pixels → canvas pixels
                "scale": scale * (w / orig_w),
                "dx": dx,
                "dy": dy,
            },
        }


def collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": [b["labels"] for b in batch],
        "meta": [b["meta"] for b in batch],
    }
