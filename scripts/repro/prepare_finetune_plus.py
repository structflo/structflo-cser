"""Build the fine-tune corpus for the PUBLISHED model with newly annotated docs.

Distinct from the paper experiments: this folds the freshly annotated real documents
into training to refresh the released detector + LPS. The held-out val/test stay FROZEN
(same pages as the paper split) so we can confirm no regression.

Train = (frozen 830 real train ×OLD_OVERSAMPLE) + (new clean real docs ×NEW_OVERSAMPLE)
        + N_SYNTH synthetic.  "New clean" = real docs that have BOTH a ground_truth JSON
and a YOLO label and are NOT already in real_split.json. GT-only docs (no YOLO label) are
EXCLUDED as likely partial saves. Frozen real_val / real_test are symlinked in.

Writes data/finetune/plus/{yolo,lps}/ (mirrors the normal fine-tune dirs).

Usage:
    uv run python scripts/repro/prepare_finetune_plus.py --new-oversample 4
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
YOLO_FROZEN = PROJECT_ROOT / "data" / "finetune" / "yolo"
LPS_FROZEN = PROJECT_ROOT / "data" / "finetune" / "lps"
N_SYNTH = 2000


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
    dst.unlink(missing_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    dst.symlink_to(src.resolve())


def _write_yaml(path: Path, root: Path, val_subdir: str) -> None:
    path.write_text(
        f"path: {root.resolve()}\ntrain: train/images\nval: {val_subdir}\n\n"
        f"nc: 2\nnames:\n  0: chemical_structure\n  1: compound_label\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-oversample", type=int, default=2)
    ap.add_argument("--new-oversample", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-root", type=Path, default=PROJECT_ROOT / "data" / "finetune" / "plus"
    )
    args = ap.parse_args()

    manifest = json.loads(SPLIT.read_text())
    frozen_train = list(manifest["train"])
    split_all = set(manifest["train"]) | set(manifest["val"]) | set(manifest["test"])

    gt = {p.stem for p in (REAL / "ground_truth").glob("*.json")}
    lbl = {p.stem for p in (REAL / "labels").glob("*.txt")}
    new_clean = sorted((gt & lbl) - split_all)  # complete + not in frozen split
    dropped = sorted((gt - lbl) - split_all)  # GT-only new docs (excluded)
    print(
        f"frozen train: {len(frozen_train)} | new clean: {len(new_clean)} | "
        f"GT-only new EXCLUDED: {len(dropped)}"
    )

    synth_stems = sorted(p.stem for p in (SYNTH / "ground_truth").glob("*.json"))
    synth_sample = random.Random(args.seed).sample(
        synth_stems, min(N_SYNTH, len(synth_stems))
    )

    def build(
        kind: str, real_root: Path, ann_sub: str, ann_ext: str, frozen: Path
    ) -> Path:
        out = args.out_root / kind
        if out.exists():
            shutil.rmtree(out)
        (out / "train" / "images").mkdir(parents=True)
        (out / "train" / ann_sub).mkdir(parents=True)

        def add(stem: str, src_img_dir: Path, src_ann_dir: Path, tag: str) -> None:
            img = _find(src_img_dir, stem)
            ann = src_ann_dir / f"{stem}{ann_ext}"
            if img and ann.exists():
                _symlink(img, out / "train" / "images" / f"{stem}{tag}{img.suffix}")
                _symlink(ann, out / "train" / ann_sub / f"{stem}{tag}{ann_ext}")

        for stem in synth_sample:
            add(stem, SYNTH / "images", SYNTH / ann_sub, "")
        for c in range(args.old_oversample):
            for stem in frozen_train:
                add(stem, REAL / "images", real_root, f"_real{c:02d}")
        for c in range(args.new_oversample):
            for stem in new_clean:
                add(stem, REAL / "images", real_root, f"_new{c:02d}")
        _link_dir(
            frozen / ("real_val" if kind == "yolo" else "val"),
            out / ("real_val" if kind == "yolo" else "val"),
        )
        _link_dir(frozen / "real_test", out / "real_test")
        n = len(list((out / "train" / "images").glob("*")))
        print(
            f"  {kind}: {n} train imgs "
            f"({len(synth_sample)} synth + {len(frozen_train)}×{args.old_oversample} + "
            f"{len(new_clean)}×{args.new_oversample} new)"
        )
        return out

    ydir = build("yolo", REAL / "labels", "labels", ".txt", YOLO_FROZEN)
    _write_yaml(ydir / "data.yaml", ydir, "real_val/images")
    _write_yaml(ydir / "data_real_test.yaml", ydir, "real_test/images")
    build("lps", REAL / "ground_truth", "ground_truth", ".json", LPS_FROZEN)
    print(f"output: {args.out_root}")


if __name__ == "__main__":
    main()
