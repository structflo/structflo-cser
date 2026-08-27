"""Core data structures for the extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_list(cls, lst: list[float]) -> BBox:
        return cls(lst[0], lst[1], lst[2], lst[3])

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Detection:
    bbox: BBox
    conf: float
    class_id: int  # 0 = structure, 1 = label

    @classmethod
    def from_dict(cls, d: dict) -> Detection:
        return cls(
            bbox=BBox.from_list(d["bbox"]),
            conf=d["conf"],
            class_id=d["class_id"],
        )


@dataclass
class CompoundPair:
    """A matched (chemical_structure, compound_label) pair.

    Populated by the matcher; optionally enriched with SMILES and OCR text
    by the pipeline extraction steps.
    """

    structure: Detection
    label: Detection
    match_distance: (
        float  # matcher-specific cost (pixels for Hungarian, 1-score for LPS)
    )
    smiles: str | None = None
    label_text: str | None = None
    match_confidence: float | None = (
        None  # LPS probability in [0,1]; None for Hungarian
    )

    def to_dict(self) -> dict:
        d = {
            "structure_bbox": self.structure.bbox.as_list(),
            "structure_conf": round(self.structure.conf, 4),
            "label_bbox": self.label.bbox.as_list(),
            "label_conf": round(self.label.conf, 4),
            "match_distance": round(self.match_distance, 2),
            "match_confidence": (
                round(self.match_confidence, 4)
                if self.match_confidence is not None
                else None
            ),
            "smiles": self.smiles,
            "label_text": self.label_text,
        }
        return d


@dataclass
class PageResult:
    """One PDF page: the render the model saw, its size, and what it found.

    ``width``/``height`` are the pixel dimensions of ``image``. Consumers that
    persist bounding boxes must persist this render alongside them — the boxes
    are in its pixel coordinates and mean nothing without it.
    """

    image: Image
    width: int
    height: int
    pairs: list[CompoundPair]
