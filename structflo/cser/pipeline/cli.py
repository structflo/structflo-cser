"""CLI entry point: sl-extract

Run the full extraction pipeline on one image and emit results as JSON or CSV.

Examples
--------
# Full pipeline to stdout
sl-extract --image page.png

# Save to files, skip SMILES extraction
sl-extract --image page.png --no_smiles --out results.json --csv results.csv

# Custom weights and confidence, sliding-window tiling for a very dense page
sl-extract --image page.png --weights my_model.safetensors --conf 0.4 --tile

# Only detect + match (skip both extraction steps)
sl-extract --image page.png --no_smiles --no_ocr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Absolute imports so this file works both as `sl-extract` (installed entry point)
# and as `python -m structflo.cser.pipeline.cli` (direct module invocation).
# Running `python structflo/cser/pipeline/cli.py` directly will NOT work — use
# `python -m structflo.cser.pipeline.cli` from the project root instead.
from structflo.cser.pipeline.matcher import HungarianMatcher
from structflo.cser.pipeline.ocr import EasyOCRExtractor, NullOCR
from structflo.cser.pipeline.pipeline import ChemPipeline
from structflo.cser.pipeline.smiles_extractor import (
    DecimerExtractor,
    NullSmilesExtractor,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract (SMILES, label-text) pairs from a chemical document image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--image", required=True, help="Input image path (PNG / JPG)")
    p.add_argument(
        "--weights",
        default=None,
        help="Weights version tag (e.g. v1.0) or path to a local .safetensors file. "
        "Defaults to the latest published weights (auto-downloaded).",
    )
    p.add_argument(
        "--conf", type=float, default=0.5, help="Detection confidence threshold"
    )
    p.add_argument(
        "--tile",
        action="store_true",
        help="Sliding-window tiling instead of full-image detection (dense pages only)",
    )
    p.add_argument("--tile_size", type=int, default=1536, help="Tile size in pixels")
    p.add_argument(
        "--max_dist",
        type=float,
        default=None,
        help="Max centroid distance (px) for a valid structure–label pair",
    )
    p.add_argument(
        "--no_smiles", action="store_true", help="Skip DECIMER SMILES extraction"
    )
    p.add_argument("--no_ocr", action="store_true", help="Skip OCR text extraction")
    p.add_argument("--out", default=None, help="Output JSON file (default: stdout)")
    p.add_argument("--csv", default=None, help="Output CSV file (requires pandas)")
    args = p.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    matcher = HungarianMatcher(max_distance=args.max_dist)
    smiles_ext = NullSmilesExtractor() if args.no_smiles else DecimerExtractor()
    ocr_ext = NullOCR() if args.no_ocr else EasyOCRExtractor()

    pipeline = ChemPipeline(
        weights=args.weights,
        matcher=matcher,
        smiles_extractor=smiles_ext,
        ocr=ocr_ext,
        tile=args.tile,
        tile_size=args.tile_size,
        conf=args.conf,
    )

    print(f"Processing: {image_path}", file=sys.stderr)
    pairs = pipeline.process(image_path)
    print(f"Found {len(pairs)} pair(s)", file=sys.stderr)

    json_str = ChemPipeline.to_json(pairs)

    if args.out:
        Path(args.out).write_text(json_str)
        print(f"JSON → {args.out}", file=sys.stderr)
    else:
        print(json_str)

    if args.csv:
        df = ChemPipeline.to_dataframe(pairs)
        df.to_csv(args.csv, index=False)
        print(f"CSV  → {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
