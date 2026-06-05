"""Tests for weight introspection (`weight_info`) and `ChemPipeline.version`.

All hermetic — no weight downloads, no torch models loaded.
"""

from structflo.cser.pipeline import ChemPipeline
from structflo.cser.pipeline.matcher import HungarianMatcher
from structflo.cser.pipeline.ocr import NullOCR
from structflo.cser.pipeline.smiles_extractor import NullSmilesExtractor
from structflo.cser.pipeline.version import PipelineInfo
from structflo.cser.weights import weight_info


def test_weight_info_registry():
    info = weight_info("cser-detector")  # latest
    assert info["source"] == "registry"
    assert info["repo_id"] == "sidxz/structflo-cser-detector"
    assert info["version"] and info["revision"] and info["sha256"]
    assert info["requires"]


def test_weight_info_local_path():
    info = weight_info("cser-detector", "/tmp/whatever/best.pt")
    assert info["source"] == "local"
    assert info["version"] == "/tmp/whatever/best.pt"


def test_weight_info_unknown_tag():
    info = weight_info("cser-detector", "v999.0")
    assert info["source"] == "unknown"
    assert info["version"] == "v999.0"


def test_hungarian_weight_descriptor_is_none():
    # Parameter-free matcher has no learned weights to describe.
    assert HungarianMatcher().weight_descriptor() is None


def _hermetic_pipeline():
    return ChemPipeline(
        tile=False,
        conf=0.30,
        matcher=HungarianMatcher(),
        ocr=NullOCR(),
        smiles_extractor=NullSmilesExtractor(),
    )


def test_pipeline_version_structure():
    info = _hermetic_pipeline().version
    assert isinstance(info, PipelineInfo)

    # Mapping behaviour
    assert info["package"]["name"] == "structflo-cser"
    assert info["package"]["version"]
    assert set(dict(info)) == {
        "package",
        "detector",
        "matcher",
        "smiles_extractor",
        "ocr",
        "config",
        "dependencies",
    }

    # Detector weights described without forcing a download
    assert info["detector"]["source"] == "registry"
    assert info["detector"]["loaded"] is False
    assert info["detector"]["path"] is None

    # Config reflects the constructor args
    assert info["config"]["conf"] == 0.30
    assert info["config"]["tile"] is False
    assert info["config"]["imgsz"] == 1280

    # Matcher identity
    assert info["matcher"]["class"] == "HungarianMatcher"
    assert info["matcher"]["weights"] is None


def test_pipeline_version_renders():
    info = _hermetic_pipeline().version
    text = repr(info)
    assert "ChemPipeline" in text
    assert "structflo-cser" in text
    md = info._repr_markdown_()
    assert md.startswith("### ChemPipeline")
    assert "| field | value |" in md
