"""Build the fixed synthetic-TEST manifest + a YOLO data.yaml for it.

The synthetic test set (``data/generated_test``, 1000 pages, generation seed 1000000,
seed-disjoint from the train/val generation seeds 7-11006) is the fixed §A held-out
set. This manifest assigns every stem to the "test" split so ``eval_compare_all.py``
scores all 1000 pages as held-out. Both outputs are seed-INDEPENDENT — run once; the
training seed never touches data splits.

Usage:
    uv run python scripts/repro/make_synth_manifest.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/generated_test/val"))
    ap.add_argument(
        "--out-manifest", type=Path, default=Path("runs/repro/synth_test_manifest.json")
    )
    ap.add_argument("--out-yaml", type=Path, default=Path("runs/repro/synth_test.yaml"))
    args = ap.parse_args()

    stems = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(str(args.src / "ground_truth" / "*.json"))
    )
    if not stems:
        raise SystemExit(f"no ground_truth json under {args.src}/ground_truth")

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"test": stems, "val": [], "train": []}, args.out_manifest.open("w"), indent=0
    )

    # YOLO data.yaml: the 1000 test pages live under val/ in data/generated_test.
    root = args.src.parent.resolve()  # .../data/generated_test
    args.out_yaml.write_text(
        f"path: {root}\n"
        "val: val/images\n"
        "train: val/images\n"  # placeholder; never trained with this yaml
        "\nnc: 2\n"
        "names:\n  0: chemical_structure\n  1: compound_label\n"
    )
    print(f"manifest : {args.out_manifest}  pages={len(stems)}")
    print(f"data yaml: {args.out_yaml}  path={root}")


if __name__ == "__main__":
    main()
