"""Build a real-data-scaling fine-tune subset for (N real docs, seed).

For the learning-curve study: reads the FROZEN ``real_split.json`` (never re-splits —
the 100-page test and 75-page val stay fixed), shuffles the 830 real TRAIN stems with
``--seed``, takes the first ``--n``, and writes self-contained YOLO + LPS fine-tune dirs:

    <out-root>/yolo_n{N}_s{seed}/   train/{images,labels} + real_val + real_test + *.yaml
    <out-root>/lps_n{N}_s{seed}/    train/{images,ground_truth} + val + real_test

Train = N real pages (oversampled ×REAL_OVERSAMPLE) + N_SYNTH synthetic pages. The frozen
real val/test are symlinked in from data/finetune/{yolo,lps}/ so every curve point reports
on the identical held-out pages. The synthetic subsample is fixed per seed (independent of
N), so only the real-doc count varies along the curve.

Usage:
    uv run python scripts/repro/prepare_scale_subset.py --n 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
REAL = Path("/net-fs-ins/shared-docker-vols/structflo-cser-annotate/data")
SYNTH = PROJECT_ROOT / "data" / "generated" / "train"
SPLIT = PROJECT_ROOT / "data" / "finetune" / "real_split.json"
# Frozen val/test dirs (built by the original prepare_data.py runs):
YOLO_FROZEN = PROJECT_ROOT / "data" / "finetune" / "yolo"
LPS_FROZEN = PROJECT_ROOT / "data" / "finetune" / "lps"

N_SYNTH = 2000
REAL_OVERSAMPLE = 2


def _symlink(src: Path, dst: Path) -> None:
    dst.unlink(missing_ok=True)
    dst.symlink_to(Path(src).resolve())


def _find(img_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".png", ".jpeg"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _link_dir(src: Path, dst: Path) -> None:
    """Symlink an entire frozen split dir (val/test) under the scale dir."""
    dst.unlink(missing_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    dst.symlink_to(src.resolve())


def _write_yaml(path: Path, root: Path, val_subdir: str) -> None:
    path.write_text(
        f"path: {root.resolve()}\n"
        f"train: train/images\n"
        f"val: {val_subdir}\n\n"
        f"nc: 2\nnames:\n  0: chemical_structure\n  1: compound_label\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of real train docs")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--out-root", type=Path, default=PROJECT_ROOT / "data" / "finetune" / "scale"
    )
    args = ap.parse_args()

    manifest = json.loads(SPLIT.read_text())
    train_stems = list(manifest["train"])
    if args.n > len(train_stems):
        raise SystemExit(
            f"--n {args.n} exceeds available {len(train_stems)} train pages"
        )

    # Shuffle the FROZEN train pool with the seed; take the first N (nested across N).
    random.Random(args.seed).shuffle(train_stems)
    chosen = train_stems[: args.n]

    # Synthetic subsample: fixed per seed, independent of N (separate RNG stream).
    synth_stems = sorted(p.stem for p in (SYNTH / "ground_truth").glob("*.json"))
    synth_sample = random.Random(args.seed).sample(
        synth_stems, min(N_SYNTH, len(synth_stems))
    )

    # ---- YOLO dir ----
    ydir = args.out_root / f"yolo_n{args.n}_s{args.seed}"
    if ydir.exists():
        shutil.rmtree(ydir, ignore_errors=True)
    (ydir / "train" / "images").mkdir(parents=True)
    (ydir / "train" / "labels").mkdir(parents=True)
    for stem in synth_sample:
        img = _find(SYNTH / "images", stem)
        lbl = SYNTH / "labels" / f"{stem}.txt"
        if img and lbl.exists():
            _symlink(img, ydir / "train" / "images" / img.name)
            _symlink(lbl, ydir / "train" / "labels" / lbl.name)
    for c in range(REAL_OVERSAMPLE):
        for stem in chosen:
            img = _find(REAL / "images", stem)
            lbl = REAL / "labels" / f"{stem}.txt"
            if img and lbl.exists():
                _symlink(
                    img, ydir / "train" / "images" / f"{stem}_real{c:02d}{img.suffix}"
                )
                _symlink(lbl, ydir / "train" / "labels" / f"{stem}_real{c:02d}.txt")
    _link_dir(YOLO_FROZEN / "real_val", ydir / "real_val")
    _link_dir(YOLO_FROZEN / "real_test", ydir / "real_test")
    _write_yaml(ydir / "data.yaml", ydir, "real_val/images")
    _write_yaml(ydir / "data_real_test.yaml", ydir, "real_test/images")

    # ---- LPS dir ----
    ldir = args.out_root / f"lps_n{args.n}_s{args.seed}"
    if ldir.exists():
        shutil.rmtree(ldir, ignore_errors=True)
    (ldir / "train" / "images").mkdir(parents=True)
    (ldir / "train" / "ground_truth").mkdir(parents=True)
    for stem in synth_sample:
        img = _find(SYNTH / "images", stem)
        gt = SYNTH / "ground_truth" / f"{stem}.json"
        if img and gt.exists():
            _symlink(img, ldir / "train" / "images" / img.name)
            _symlink(gt, ldir / "train" / "ground_truth" / gt.name)
    for c in range(REAL_OVERSAMPLE):
        for stem in chosen:
            img = _find(REAL / "images", stem)
            gt = REAL / "ground_truth" / f"{stem}.json"
            if img and gt.exists():
                _symlink(
                    img, ldir / "train" / "images" / f"{stem}_real{c:02d}{img.suffix}"
                )
                _symlink(
                    gt, ldir / "train" / "ground_truth" / f"{stem}_real{c:02d}.json"
                )
    _link_dir(LPS_FROZEN / "val", ldir / "val")
    _link_dir(LPS_FROZEN / "real_test", ldir / "real_test")

    n_synth_used = (
        len(list((ydir / "train" / "images").glob("*"))) - args.n * REAL_OVERSAMPLE
    )
    print(
        f"N={args.n} seed={args.seed}: {len(chosen)} real ×{REAL_OVERSAMPLE} + ~{n_synth_used} synth"
    )
    print(f"  yolo: {ydir}")
    print(f"  lps : {ldir}")


if __name__ == "__main__":
    main()
