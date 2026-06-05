"""LearnedMatcher — BaseMatcher implementation backed by a trained PairScorer.

Replaces the Euclidean cost matrix in Hungarian matching with a learned
association probability.  The assignment algorithm itself (Hungarian /
``linear_sum_assignment``) is preserved; only the cost metric changes.

Usage::

    from structflo.cser.lps import LearnedMatcher
    from structflo.cser.pipeline import ChemPipeline

    pipeline = ChemPipeline(
        matcher=LearnedMatcher(weights="runs/lps/scorer_best.pt")
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from structflo.cser.lps.features import (
    LABEL_CROP_SIZE,
    STRUCT_CROP_SIZE,
    crop_region,
    geom_features,
)
from structflo.cser.lps.scorer import load_checkpoint
from structflo.cser.pipeline.matcher import BaseMatcher
from structflo.cser.pipeline.models import CompoundPair, Detection


class LearnedMatcher(BaseMatcher):
    """Pair structures with labels using a trained association scorer.

    The scorer produces a probability in [0, 1] for every candidate
    (structure, label) pair.  Hungarian matching is then run on
    ``(1 - score)`` as the cost matrix, preserving global optimality.

    ``match_distance`` on the returned ``CompoundPair`` objects is set to
    ``1 - score`` so that lower values mean higher confidence — consistent
    with the convention used by ``HungarianMatcher``.

    Args:
        weights:     Path to a scorer ``.pt`` checkpoint, a version tag
                     (``"v1.0"``), or ``None`` to auto-download the latest
                     published scorer from HuggingFace Hub.
        min_score:   Pairs whose association score is below this threshold
                     are dropped (structure considered unlabelled).
        device:      Torch device string (``"cuda"`` or ``"cpu"``).
        max_dist_px: Optional pre-filter: candidate pairs whose centroid
                     distance exceeds this value are excluded before
                     scoring, saving compute on dense pages.
    """

    def __init__(
        self,
        weights: Path | str | None = None,
        min_score: float = 0.5,
        device: str = "cuda",
        max_dist_px: float | None = None,
    ) -> None:
        self.min_score = min_score
        self.max_dist_px = max_dist_px
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._weights = weights  # version tag, local path, or None (→ latest)
        self._weights_path: str | None = None
        self._model = self._load(weights)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, weights: Path | str | None) -> torch.nn.Module:
        from structflo.cser.weights import resolve_weights

        path = resolve_weights("cser-lps", version=weights)
        self._weights_path = str(path)
        model, _ = load_checkpoint(path, device=str(self._device))
        model.eval()
        return model

    def weight_descriptor(self) -> dict:
        from structflo.cser.weights import weight_info

        info = weight_info("cser-lps", self._weights)
        info["path"] = self._weights_path
        return info

    def _page_size(
        self,
        detections: list[Detection],
        image: np.ndarray | None,
    ) -> tuple[float, float]:
        if image is not None:
            h, w = image.shape[:2]
            return float(w), float(h)
        if detections:
            pw = max(d.bbox.x2 for d in detections)
            ph = max(d.bbox.y2 for d in detections)
            return float(pw), float(ph)
        return 2480.0, 3508.0  # A4 @ 300 DPI fallback

    def _score_matrix(
        self,
        structures: list[Detection],
        labels: list[Detection],
        image: np.ndarray,
        page_w: float,
        page_h: float,
    ) -> np.ndarray:
        n_s, n_l = len(structures), len(labels)

        geom_rows, struct_crops, label_crops = [], [], []
        for s in structures:
            for lbl in labels:
                geom_rows.append(
                    geom_features(
                        s.bbox.as_list(),
                        lbl.bbox.as_list(),
                        page_w,
                        page_h,
                        s.conf,
                        lbl.conf,
                    )
                )
                struct_crops.append(
                    crop_region(image, s.bbox.as_list(), STRUCT_CROP_SIZE)
                )
                label_crops.append(
                    crop_region(image, lbl.bbox.as_list(), LABEL_CROP_SIZE)
                )

        geom_t = torch.from_numpy(np.stack(geom_rows)).to(self._device)
        sc_t = torch.from_numpy(np.stack(struct_crops)).to(self._device)
        lc_t = torch.from_numpy(np.stack(label_crops)).to(self._device)

        with torch.no_grad():
            scores = self._model(sc_t, lc_t, geom_t).sigmoid().squeeze(1).cpu().numpy()

        return scores.reshape(n_s, n_l)

    # ------------------------------------------------------------------
    # BaseMatcher interface
    # ------------------------------------------------------------------

    def match(
        self,
        detections: list[Detection],
        image: np.ndarray | None = None,
    ) -> list[CompoundPair]:
        """Pair structures with labels using learned association scores.

        Args:
            detections: Flat list of all detections (structures + labels mixed).
            image:      Page image as a uint8 numpy array (H, W) or (H, W, 3).
                        Required — the scorer uses visual crops.

        Returns:
            List of ``CompoundPair`` objects.  ``match_distance`` is set to
            ``1 - score`` (lower = more confident match).
        """
        if image is None:
            raise ValueError(
                "LearnedMatcher requires an image for visual scoring. "
                "Pass image= to match()."
            )

        structures = [d for d in detections if d.class_id == 0]
        labels = [d for d in detections if d.class_id == 1]

        if not structures or not labels:
            return []

        page_w, page_h = self._page_size(detections, image)

        if self.max_dist_px is not None:
            valid_s, valid_l = set(), set()
            for i, s in enumerate(structures):
                scx, scy = s.bbox.centroid
                for j, lbl in enumerate(labels):
                    lcx, lcy = lbl.bbox.centroid
                    if ((scx - lcx) ** 2 + (scy - lcy) ** 2) ** 0.5 <= self.max_dist_px:
                        valid_s.add(i)
                        valid_l.add(j)
            structures = [structures[i] for i in sorted(valid_s)]
            labels = [labels[j] for j in sorted(valid_l)]
            if not structures or not labels:
                return []

        score_matrix = self._score_matrix(structures, labels, image, page_w, page_h)
        cost_matrix = 1.0 - score_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        pairs = []
        for r, c in zip(row_ind, col_ind):
            score = float(score_matrix[r, c])
            if score >= self.min_score:
                pairs.append(
                    CompoundPair(
                        structure=structures[r],
                        label=labels[c],
                        match_distance=float(1.0 - score),
                        match_confidence=score,
                    )
                )
        return pairs
