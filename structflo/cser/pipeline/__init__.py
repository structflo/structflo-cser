"""structflo.cser.pipeline — extraction pipeline with adapter pattern.

Quick start
-----------
>>> from structflo.cser.pipeline import ChemPipeline
>>> pipeline = ChemPipeline()
>>> pairs = pipeline.process("page.png")          # full pipeline
>>> df = ChemPipeline.to_dataframe(pairs)

Step-by-step
------------
>>> detections = pipeline.detect("page.png")
>>> pairs      = pipeline.match(detections)
>>> pairs      = pipeline.enrich(pairs, "page.png")

Custom adapters
---------------
>>> from structflo.cser.pipeline import ChemPipeline, HungarianMatcher
>>> pipeline = ChemPipeline(matcher=HungarianMatcher(max_distance=500))
"""

from structflo.cser.pipeline.matcher import BaseMatcher, HungarianMatcher
from structflo.cser.pipeline.models import BBox, CompoundPair, Detection, PageResult
from structflo.cser.pipeline.ocr import BaseOCR, EasyOCRExtractor, NullOCR
from structflo.cser.pipeline.pipeline import DEFAULT_DPI, ChemPipeline, render_page
from structflo.cser.pipeline.smiles_extractor import (
    BaseSmilesExtractor,
    DecimerExtractor,
    NullSmilesExtractor,
)

__all__ = [
    # Pipeline
    "ChemPipeline",
    "DEFAULT_DPI",
    "render_page",
    # Data models
    "BBox",
    "Detection",
    "CompoundPair",
    "PageResult",
    # Matching adapters
    "BaseMatcher",
    "HungarianMatcher",
    # SMILES adapters
    "BaseSmilesExtractor",
    "DecimerExtractor",
    "NullSmilesExtractor",
    # OCR adapters
    "BaseOCR",
    "EasyOCRExtractor",
    "NullOCR",
]
