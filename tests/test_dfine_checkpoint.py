"""DFineDetector single-file checkpoint round-trip (no network: model built from config)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def small_detector():
    from transformers import DFineConfig, DFineForObjectDetection

    from structflo.cser.inference.dfine import CLASS_NAMES, DFineDetector

    # Tiny D-FINE so the test runs in seconds on CPU; num_denoising=100 at construction
    # (the HF default) so the model owns a denoising_class_embed, as a hub-initialised
    # model does before the trainer switches denoising off.
    cfg = DFineConfig(
        id2label=CLASS_NAMES,
        label2id={v: k for k, v in CLASS_NAMES.items()},
        d_model=32,
        encoder_hidden_dim=32,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        decoder_layers=2,
        num_queries=20,
        num_denoising=100,
        decoder_in_channels=[32, 32, 32],
    )
    try:
        model = DFineForObjectDetection(cfg)
    except (
        Exception
    ) as e:  # config shape not constructible on this transformers version
        pytest.skip(f"cannot build a tiny D-FINE here: {e}")
    return DFineDetector(model, device="cpu", imgsz=256)


def test_round_trip_with_denoising_disabled(tmp_path, small_detector):
    from structflo.cser.inference.dfine import DFineDetector

    det = small_detector
    assert hasattr(det.model.model, "denoising_class_embed")
    det.model.config.num_denoising = 0  # what the trainer does after construction
    path = det.save(tmp_path / "best.safetensors", epoch=1)
    loaded = DFineDetector.from_file(path, device="cpu")
    assert loaded.model.config.num_denoising == 0
    assert loaded.imgsz == 256
    assert loaded.meta["format"] == "structflo-cser-dfine-v1"
    # weights identical for every shared parameter
    a, b = det.model.state_dict(), loaded.model.state_dict()
    for k, v in b.items():
        assert torch.equal(v, a[k]), k


def test_predict_returns_pipeline_dicts(small_detector):
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    dets = small_detector.predict(img, conf=0.0, max_det=5)
    assert len(dets) <= 5
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        assert 0 <= x1 <= x2 <= 400 and 0 <= y1 <= y2 <= 300
        assert 0.0 <= d["conf"] <= 1.0 and d["class_id"] in (0, 1)


def test_legacy_pt_is_refused(tmp_path):
    from structflo.cser.inference.dfine import DFineDetector

    p = tmp_path / "best.pt"
    p.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="legacy Ultralytics"):
        DFineDetector.from_file(p)
