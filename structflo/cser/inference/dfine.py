"""D-FINE detector — Apache-2.0 replacement for the retired (AGPL) YOLO detector.

Uses ``DFineForObjectDetection`` from HuggingFace ``transformers`` (Apache-2.0)
with a HGNet-V2 backbone. NMS-free: the decoder emits 300 queries, each with a
per-class sigmoid score, so ``conf`` is the only post-processing knob.

Weights are stored as ONE ``.safetensors`` file whose metadata carries the
model config, so the existing HF-Hub weights registry (one file per version)
keeps working and loading never involves pickle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from structflo.cser.inference.preprocess import letterbox

CLASS_NAMES = {0: "chemical_structure", 1: "compound_label"}
WEIGHTS_FORMAT = "structflo-cser-dfine-v1"
DEFAULT_INIT = "ustc-community/dfine-large-coco"  # Apache-2.0, COCO-pretrained


def decode_outputs(
    logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    canvas_size: int,
    scale: float,
    dx: int,
    dy: int,
    width: int,
    height: int,
    conf: float,
    max_det: int = 300,
) -> list[dict]:
    """Turn one image's raw decoder outputs into pipeline-format detections.

    Mirrors ``RTDetrImageProcessor.post_process_object_detection`` (sigmoid
    scores, top-k over the flattened (query, class) grid), then undoes the
    letterbox so boxes land in original-image pixels.
    """
    logits = logits.float()
    pred_boxes = pred_boxes.float()
    num_queries, num_classes = logits.shape
    scores = logits.sigmoid().flatten()
    k = min(max_det, num_queries, scores.numel())
    scores, index = scores.topk(k)
    keep = scores >= conf
    scores, index = scores[keep], index[keep]
    classes = index % num_classes
    boxes = pred_boxes[index // num_classes]  # normalised cxcywh on the canvas
    cx, cy, w, h = boxes.unbind(-1)
    xyxy = (
        torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        * canvas_size
    )
    xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - dx) / scale).clamp(0, width)
    xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - dy) / scale).clamp(0, height)
    out = []
    for b, s, c in zip(xyxy.tolist(), scores.tolist(), classes.tolist()):
        if b[2] - b[0] <= 0 or b[3] - b[1] <= 0:
            continue
        out.append(
            {"bbox": [float(v) for v in b], "conf": float(s), "class_id": int(c)}
        )
    return out


class DFineDetector:
    """Thin inference wrapper: ``predict(img_rgb_uint8) -> list[dict]``.

    The returned dicts (``bbox`` xyxy pixels, ``conf``, ``class_id``) are the
    same shape ``Detection.from_dict`` consumes, so the rest of the pipeline is
    untouched by the backend swap.
    """

    def __init__(
        self,
        model,
        *,
        device: str | torch.device | None = None,
        imgsz: int = 1280,
        amp: bool = False,
        meta: dict | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device).eval()
        self.imgsz = imgsz
        self.amp = amp and self.device.type == "cuda"
        self.meta = dict(meta or {})

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_hub(
        cls, model_id: str = DEFAULT_INIT, num_labels: int = 2, **kw
    ) -> DFineDetector:
        """COCO-pretrained backbone+decoder with a fresh ``num_labels`` head (training init)."""
        from transformers import DFineForObjectDetection

        model = DFineForObjectDetection.from_pretrained(
            model_id,
            id2label={i: CLASS_NAMES.get(i, str(i)) for i in range(num_labels)},
            label2id={CLASS_NAMES.get(i, str(i)): i for i in range(num_labels)},
            ignore_mismatched_sizes=True,
        )
        return cls(model, meta={"init": model_id}, **kw)

    @classmethod
    def from_file(cls, path: str | Path, **kw) -> DFineDetector:
        """Load a single-file ``.safetensors`` checkpoint written by :meth:`save`."""
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers import DFineConfig, DFineForObjectDetection

        path = Path(path)
        if path.suffix == ".pt":
            raise ValueError(
                f"{path} is a legacy Ultralytics YOLO checkpoint (cser-detector <= v0.4, "
                "AGPL-3.0). Those weights are retired; pass a D-FINE .safetensors "
                "checkpoint (cser-detector >= v1.0) instead."
            )
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
        if meta.get("format") != WEIGHTS_FORMAT:
            raise ValueError(
                f"{path} is not a structflo-cser D-FINE checkpoint "
                f"(format={meta.get('format')!r}, expected {WEIGHTS_FORMAT!r})"
            )
        config = DFineConfig.from_dict(json.loads(meta["config"]))
        model = DFineForObjectDetection(config)
        state = load_file(str(path))
        # Denoising groups are training-only; a checkpoint trained with them off (our default)
        # may still carry the unused embedding if the model was built before the config change.
        state = {
            k: v
            for k, v in state.items()
            if "denoising_class_embed" not in k or config.num_denoising > 0
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if "denoising_class_embed" not in k]
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint/config mismatch in {path}: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        kw.setdefault("imgsz", int(meta.get("imgsz", 1280)))
        return cls(model, meta=meta, **kw)

    def save(self, path: str | Path, **extra: object) -> Path:
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            k: v.detach().cpu().clone().contiguous()
            for k, v in self.model.state_dict().items()
        }
        meta = {
            "format": WEIGHTS_FORMAT,
            "config": json.dumps(self.model.config.to_dict()),
            "imgsz": str(self.imgsz),
            "architecture": "D-FINE (transformers DFineForObjectDetection, HGNet-V2)",
            "license": "Apache-2.0",
        }
        meta.update({k: str(v) for k, v in extra.items()})
        save_file(state, str(path), metadata=meta)
        return path

    # -- inference ------------------------------------------------------------

    def preprocess(self, img: np.ndarray, imgsz: int | None = None):
        imgsz = imgsz or self.imgsz
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        canvas, scale, dx, dy = letterbox(img, imgsz)
        x = torch.from_numpy(canvas).permute(2, 0, 1).float().div_(255.0)
        return x, scale, dx, dy

    @torch.inference_mode()
    def predict(
        self,
        img: np.ndarray,
        *,
        conf: float = 0.5,
        imgsz: int | None = None,
        max_det: int = 300,
    ) -> list[dict]:
        """Detect on one RGB uint8 image (H,W,3). Boxes returned in its pixel coords."""
        imgsz = imgsz or self.imgsz
        h, w = img.shape[:2]
        x, scale, dx, dy = self.preprocess(img, imgsz)
        x = x.unsqueeze(0).to(self.device, non_blocking=True)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            out = self.model(pixel_values=x)
        return decode_outputs(
            out.logits[0], out.pred_boxes[0], imgsz, scale, dx, dy, w, h, conf, max_det
        )

    def __call__(self, img: np.ndarray, **kw) -> list[dict]:
        return self.predict(img, **kw)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
