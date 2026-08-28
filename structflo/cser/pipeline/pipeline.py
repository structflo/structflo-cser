"""ChemPipeline: detect → match → extract SMILES + OCR text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from structflo.cser.inference.detector import detect_full, detect_tiled
from structflo.cser.inference.dfine import DFineDetector
from structflo.cser.pdf import iter_pages
from structflo.cser.pdf import render_page as _render_pdf_page
from structflo.cser.weights import resolve_weights, weight_info

from structflo.cser.pipeline.matcher import BaseMatcher
from structflo.cser.pipeline.models import BBox, CompoundPair, Detection, PageResult
from structflo.cser.pipeline.ocr import BaseOCR, EasyOCRExtractor
from structflo.cser.pipeline.smiles_extractor import (
    BaseSmilesExtractor,
    DecimerExtractor,
)
from structflo.cser.pipeline.version import (
    PipelineInfo,
    dependency_versions,
    package_version,
    python_version,
)

# Anything the pipeline accepts as an image input
ImageLike = Union[Path, str, np.ndarray, Image.Image]

DEFAULT_DPI = 150
"""Rendering resolution the pipeline is tuned for.

The detector letterboxes to imgsz=1280, so scale matters: 144 dpi drops pairs
that 150 finds, and 200+ dpi finds none. Downstream consumers that store
coordinates must render at this dpi and must import this constant rather than
repeat the number.
"""


def render_page(
    pdf_path: Path | str, page_index: int, *, dpi: int = DEFAULT_DPI
) -> Image.Image:
    """Render a single PDF page to a PIL image at ``dpi``.

    Module-level and model-free on purpose: callers that only need the image
    (to re-display or to export coordinates against) must not have to construct
    a ChemPipeline and pull down detector weights.

    Rendering is done by pypdfium2 (BSD-3) with pixel dimensions identical to
    the earlier PyMuPDF implementation — see :mod:`structflo.cser.pdf`.
    """
    return _render_pdf_page(pdf_path, page_index, dpi=dpi)


def _to_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


class ChemPipeline:
    """End-to-end pipeline from an image to enriched (SMILES, label-text) pairs.

    Designed after the HuggingFace transformers pattern: every step is exposed
    individually for fine-grained control, and a single ``process()`` call runs
    the whole thing for convenience.

    Low-level access
    ----------------
    >>> detections = pipeline.detect(image)
    >>> pairs      = pipeline.match(detections)
    >>> smiles     = pipeline.extract_smiles(image, pair)
    >>> text       = pipeline.extract_text(image, pair)
    >>> pairs      = pipeline.enrich(pairs, image)

    High-level access
    -----------------
    >>> pairs = pipeline.process("page.png")
    >>> df    = ChemPipeline.to_dataframe(pairs)
    >>> data  = ChemPipeline.to_records(pairs)

    Adapter pattern
    ---------------
    Pass custom implementations of ``BaseMatcher``, ``BaseSmilesExtractor``, or
    ``BaseOCR`` to swap out any step without modifying this class.
    """

    def __init__(
        self,
        *,
        weights: Path | str | None = None,
        matcher: BaseMatcher | None = None,
        smiles_extractor: BaseSmilesExtractor | None = None,
        ocr: BaseOCR | None = None,
        tile: bool = False,
        tile_size: int = 1536,
        imgsz: int = 1280,
        conf: float = 0.4,
        grayscale: bool = True,
    ) -> None:
        """
        Args:
            weights:          Weights version tag (e.g. ``"v1.0"``) or path to a
                              local ``.safetensors`` file.  ``None`` auto-downloads
                              the latest published weights.
            matcher:          Pairing strategy.  Defaults to RelationalMatcher
                              (geometry-only optimal-transport matcher; the best
                              learned matcher in our benchmark and the strongest
                              at rejecting unlabelled structures).  Weights
                              auto-download from HuggingFace Hub.  Pass
                              ``HungarianMatcher()`` for a zero-weight
                              distance baseline, or ``LearnedMatcher()`` for the
                              visual LPS.
            smiles_extractor: SMILES model.  Defaults to DecimerExtractor.
            ocr:              OCR engine.  Defaults to PaddleOCRExtractor.
            tile:             Use sliding-window tiling during detection.
            tile_size:        Tile side length in pixels.
            imgsz:            Inference resolution for full-image (non-tiled)
                              detection.  Defaults to 1280, the training
                              resolution; full-image detection at 1280 strictly
                              outperforms tiling on large landscape pages.
            conf:             Detection confidence threshold (0.4 = the D-FINE
                              detector's operating point, tuned on real_val;
                              the retired YOLO detector used 0.3).
            grayscale:        Convert input images to grayscale before detection.
                              Defaults to True to match training data distribution.
        """
        self._weights = weights  # version tag, local path str/Path, or None
        if matcher is None:
            # Lazy import breaks a relmatch <-> pipeline circular import.
            from structflo.cser.relmatch import RelationalMatcher

            matcher = RelationalMatcher()
        self._matcher = matcher
        self._smiles = smiles_extractor or DecimerExtractor()
        self._ocr = ocr or EasyOCRExtractor()
        self.tile = tile
        self.tile_size = tile_size
        self.imgsz = imgsz
        self.conf = conf
        self.grayscale = grayscale
        self._model: DFineDetector | None = None  # lazy-loaded on first detect() call
        self._weights_path: str | None = None  # set once detector weights resolve

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is None:
            weights_path = resolve_weights("cser-detector", version=self._weights)
            self._weights_path = str(weights_path)
            self._model = DFineDetector.from_file(weights_path, imgsz=self.imgsz)

    @staticmethod
    def _crop(image: Image.Image, bbox: BBox) -> Image.Image:
        return image.crop((int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def version(self) -> PipelineInfo:
        """A display-friendly snapshot of every version this pipeline uses.

        Captures the package version, detector + matcher weight provenance
        (version, repo, revision, sha256, local path), the runtime config, and
        key dependency versions.  Reading it never downloads weights — paths are
        only filled in for components that are already loaded.

        >>> pipeline = ChemPipeline(tile=False, conf=0.30, matcher=RelationalMatcher())
        >>> pipeline.version           # renders as a table in Jupyter / terminal
        >>> pipeline.version["package"]["version"]
        '0.4.1'
        """
        det = weight_info("cser-detector", self._weights)
        det["loaded"] = self._model is not None
        det["path"] = self._weights_path

        matcher_weights = None
        descriptor = getattr(self._matcher, "weight_descriptor", None)
        if callable(descriptor):
            matcher_weights = descriptor()

        # Surface whichever common knobs the configured matcher exposes.
        param_names = (
            "min_score",
            "dustbin_margin",
            "max_distance",
            "max_dist_px",
        )
        params = {
            name: getattr(self._matcher, name)
            for name in param_names
            if getattr(self._matcher, name, None) is not None
        }
        device = getattr(self._matcher, "_device", None)
        if device is not None:
            params["device"] = str(device)

        data = {
            "package": {
                "name": "structflo-cser",
                "version": package_version(),
                "python": python_version(),
            },
            "detector": det,
            "matcher": {
                "class": type(self._matcher).__name__,
                "weights": matcher_weights,
                "params": params,
            },
            "smiles_extractor": {"class": type(self._smiles).__name__},
            "ocr": {"class": type(self._ocr).__name__},
            "config": {
                "tile": self.tile,
                "tile_size": self.tile_size,
                "imgsz": self.imgsz,
                "conf": self.conf,
                "grayscale": self.grayscale,
            },
            "dependencies": dependency_versions(),
        }
        return PipelineInfo(data)

    # ------------------------------------------------------------------
    # Low-level step methods
    # ------------------------------------------------------------------

    def detect(self, image: ImageLike) -> list[Detection]:
        """Run the detector on *image* and return a flat list of Detection objects.

        Both ``structure`` (class 0) and ``label`` (class 1) detections are
        returned together; call ``match()`` next to pair them.
        """
        self._load_model()
        img_pil = _to_pil(image)
        if self.grayscale:
            img_pil = img_pil.convert("L").convert("RGB")
        img_np = np.array(img_pil)
        if self.tile:
            raw = detect_tiled(
                self._model, img_np, tile_size=self.tile_size, conf=self.conf
            )
        else:
            raw = detect_full(self._model, img_np, conf=self.conf, imgsz=self.imgsz)
        return [Detection.from_dict(d) for d in raw]

    def match(
        self,
        detections: list[Detection],
        image: ImageLike | None = None,
    ) -> list[CompoundPair]:
        """Pair structure detections with label detections using the configured matcher.

        Args:
            detections: Flat list of all detections from ``detect()``.
            image:      Page image forwarded to the matcher.  Required when
                        using ``LearnedMatcher`` with a visual scorer; ignored
                        by ``HungarianMatcher``.
        """
        img_np: np.ndarray | None = None
        if image is not None:
            img_np = np.array(_to_pil(image))
        return self._matcher.match(detections, image=img_np)

    def extract_smiles(self, image: ImageLike, pair: CompoundPair) -> str | None:
        """Crop the structure region from *image* and extract a SMILES string."""
        img = _to_pil(image)
        crop = self._crop(img, pair.structure.bbox)
        return self._smiles.extract(crop)

    def extract_text(self, image: ImageLike, pair: CompoundPair) -> str | None:
        """Crop the label region from *image* and extract text via OCR."""
        img = _to_pil(image)
        crop = self._crop(img, pair.label.bbox)
        return self._ocr.extract(crop)

    def enrich(self, pairs: list[CompoundPair], image: ImageLike) -> list[CompoundPair]:
        """Populate ``smiles`` and ``label_text`` on every pair in-place.

        The image is decoded once and reused for all crops.  Returns the same
        list for convenience.
        """
        img = _to_pil(image)
        for pair in pairs:
            pair.smiles = self._smiles.extract(self._crop(img, pair.structure.bbox))
            pair.label_text = self._ocr.extract(self._crop(img, pair.label.bbox))
        return pairs

    # ------------------------------------------------------------------
    # High-level entry point
    # ------------------------------------------------------------------

    def process(self, image: ImageLike) -> list[CompoundPair]:
        """Full pipeline in one call: detect → match → enrich.

        Returns a list of CompoundPair objects with ``smiles`` and
        ``label_text`` populated.
        """
        img = _to_pil(image)
        detections = self.detect(img)
        pairs = self.match(detections, image=img)
        return self.enrich(pairs, img)

    def process_pdf_page(
        self,
        pdf_path: Path | str,
        page_index: int,
        *,
        dpi: int = DEFAULT_DPI,
    ) -> PageResult:
        """Run the full pipeline on ONE page and return the render with it.

        Unlike :meth:`process_pdf`, the image and its dimensions come back to
        the caller, so bounding boxes can be persisted against a render the
        caller also keeps.
        """
        image = render_page(pdf_path, page_index, dpi=dpi)
        return PageResult(
            image=image,
            width=image.width,
            height=image.height,
            pairs=self.process(image),
        )

    def process_pdf(
        self,
        pdf_path: Path | str,
        *,
        dpi: int = DEFAULT_DPI,
        output_pdf: Path | str | None = None,
    ) -> list[list[CompoundPair]]:
        """Run the full pipeline on every page of a PDF.

        Pages are processed one at a time so memory usage stays bounded
        regardless of document length.

        Args:
            pdf_path:   Path to the input PDF.
            dpi:        Rendering resolution.  150 dpi works well for typical
                        journal pages; use 200-300 for small or dense text.
            output_pdf: Optional path for an annotated output PDF.  When given,
                        each page is rendered with bounding boxes, pairing
                        lines, and extracted SMILES / label text, then saved
                        as a multi-page PDF.

        Returns:
            A list with one entry per page; each entry is a list of
            ``CompoundPair`` objects with ``smiles`` and ``label_text``
            populated.
        """
        all_results: list[list[CompoundPair]] = []

        if output_pdf is not None:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            from structflo.cser.viz import plot_results

            pdf_out: PdfPages | None = PdfPages(str(output_pdf))
        else:
            pdf_out = None

        try:
            for img in iter_pages(pdf_path, dpi=dpi):
                pairs = self.process(img)
                all_results.append(pairs)
                if pdf_out is not None:
                    fig = plot_results(img, pairs)
                    pdf_out.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
        finally:
            if pdf_out is not None:
                pdf_out.close()

        return all_results

    # ------------------------------------------------------------------
    # Output helpers  (static — can also be called on the class directly)
    # ------------------------------------------------------------------

    @staticmethod
    def to_records(pairs: list[CompoundPair]) -> list[dict]:
        """Serialise pairs to a list of plain dicts (JSON-serialisable)."""
        return [p.to_dict() for p in pairs]

    @staticmethod
    def to_json(pairs: list[CompoundPair], indent: int = 2) -> str:
        """Serialise pairs to a formatted JSON string."""
        return json.dumps(ChemPipeline.to_records(pairs), indent=indent)

    @staticmethod
    def to_dataframe(pairs: list[CompoundPair]):
        """Convert pairs to a pandas DataFrame.

        Requires pandas to be installed (``pip install pandas``).
        """
        import pandas as pd  # type: ignore[import]

        return pd.DataFrame(ChemPipeline.to_records(pairs))
