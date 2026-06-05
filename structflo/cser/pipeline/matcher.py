"""Adapter interface and implementations for structure–label pairing."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from structflo.cser.pipeline.models import CompoundPair, Detection


class BaseMatcher(ABC):
    """Abstract interface for pairing structure detections with label detections.

    Swap implementations to change the matching strategy without touching the pipeline.
    """

    @abstractmethod
    def match(
        self,
        detections: list[Detection],
        image: np.ndarray | None = None,
    ) -> list[CompoundPair]:
        """Accept a flat list of detections (all classes) and return matched pairs.

        Args:
            detections: All detections from YOLO (structures + labels mixed).
            image:      Page image as a uint8 numpy array.  Required by
                        ``LearnedMatcher`` with a visual scorer; ignored by
                        ``HungarianMatcher``.
        """
        ...

    def weight_descriptor(self) -> dict | None:
        """Describe the matcher's learned weights for ``ChemPipeline.version``.

        Returns ``None`` for parameter-free matchers (e.g. ``HungarianMatcher``).
        Learned matchers override this to return a :func:`weight_info` dict,
        optionally with the resolved local ``"path"`` once loaded.
        """
        return None


class HungarianMatcher(BaseMatcher):
    """Optimal 1-to-1 matching via the Hungarian algorithm on centroid Euclidean distance.

    Delegates to inference.pairing.pair_detections — the single implementation
    of the matching algorithm.  Type conversion happens at the boundary here.
    """

    def __init__(self, max_distance: float | None = None) -> None:
        """
        Args:
            max_distance: Drop pairs whose centroid distance exceeds this value (pixels).
                          None means no limit.
        """
        self.max_distance = max_distance

    def match(
        self,
        detections: list[Detection],
        image: np.ndarray | None = None,  # accepted but not used
    ) -> list[CompoundPair]:
        from structflo.cser.inference.pairing import pair_detections

        # Detection → raw dict (what pair_detections expects)
        raw = [
            {"bbox": d.bbox.as_list(), "conf": d.conf, "class_id": d.class_id}
            for d in detections
        ]

        raw_pairs = pair_detections(raw, max_distance=self.max_distance)

        # Raw dict pair → CompoundPair
        return [
            CompoundPair(
                structure=Detection.from_dict(p["structure"]),
                label=Detection.from_dict(p["label"]),
                match_distance=float(p["distance"]),
            )
            for p in raw_pairs
        ]
