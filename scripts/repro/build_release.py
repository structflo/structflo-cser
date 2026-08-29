"""Assemble the Zenodo-uploadable reproducibility bundle.

Collects every seeded checkpoint, all train/eval logs, the aggregated SUMMARY, the
held-out SYNTHETIC test set, an environment snapshot, and the repro scripts into a
single self-contained folder with a sha256 MANIFEST. The CONFIDENTIAL real dataset is
NEVER copied (images, ground_truth, AND real_split.json are excluded — only split
sizes + seed are documented).

Usage:
    uv run python scripts/repro/build_release.py --out repro_release
    uv run python scripts/repro/build_release.py --out repro_release --no-synth-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPRO = Path("runs/repro")
DET = REPRO / "detector"

# weight bundle path -> (source checkpoint, provenance string)
# Detector checkpoints are D-FINE .safetensors (provenance also embedded in their
# metadata: data/init/seed/epoch); LPS / relmatch checkpoints are our own torch .pt files.
SEEDS_DEFAULT = [42, 43, 44]


def weight_map(seeds: list[int]) -> dict[str, tuple[Path, str]]:
    m: dict[str, tuple[Path, str]] = {}
    for s in seeds:
        m[f"detector/base_synth_s{s}/best.safetensors"] = (
            DET / f"base_synth_s{s}" / "best.safetensors",
            f"sf-train --data config/data.yaml --init ustc-community/dfine-large-coco --imgsz 1280 --batch 8 --epochs 10 --seed {s}  (data: config/data.yaml = data/generated, synthetic-only)"
            + (
                "  [seed 42 == original dfine_l_synth; provenance via safetensors metadata]"
                if s == 42
                else ""
            ),
        )
        m[f"detector/finetuned_s{s}/best.safetensors"] = (
            DET / f"finetuned_s{s}" / "best.safetensors",
            f"sf-train --data data/finetune/yolo/data.yaml --init detector/base_synth_s{s} --imgsz 1280 --batch 8 --epochs 30 --lr 5e-5 --backbone-lr 5e-6 --seed {s}  (data: data/finetune/yolo = real+synth)",
        )
        m[f"lps/synth_s{s}/best.pt"] = (
            REPRO / f"lps_synth_s{s}" / "best.pt",
            f"sf-train-lps --data-dir data/generated --seed {s}  (§A synthetic-only)",
        )
        m[f"lps/finetuned_s{s}/best.pt"] = (
            REPRO / f"lps_ft_s{s}" / "best.pt",
            f"sf-train-lps --finetune lps/synth_s{s} --data-dir data/finetune/lps --seed {s}  (§B real+synth)",
        )
        m[f"relmatch/synth_s{s}/best.pt"] = (
            REPRO / f"relmatch_synth_s{s}" / "best.pt",
            f"sf-train-relmatch --data-dir data/generated --seed {s}  (§A GT-box, synthetic-only)",
        )
        m[f"relmatch/gt_realsynth_s{s}/best.pt"] = (
            REPRO / f"relmatch_gt_s{s}" / "best.pt",
            f"sf-train-relmatch --data-dir data/finetune/lps --seed {s}  (§B GT-box, real+synth — ablation)",
        )
        m[f"relmatch/det_s{s}/best.pt"] = (
            REPRO / f"relmatch_det_s{s}" / "best.pt",
            f"sf-train-relmatch --det-data-dir data/finetune/relmatch_det --seed {s}  (§B detection-box — PUBLISHED kind)",
        )
    return m


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(unavailable: {e})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("repro_release"))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    ap.add_argument(
        "--no-synth-data",
        action="store_true",
        help="skip copying the 239MB synth TEST set",
    )
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    git_commit = run(["git", "rev-parse", "HEAD"])
    git_dirty = bool(run(["git", "status", "--porcelain"]))

    # ---- weights ----
    manifest_files: list[dict] = []
    provenance: dict[str, str] = {}
    missing: list[str] = []
    for rel, (src, prov) in weight_map(args.seeds).items():
        dst = out / "weights" / rel
        provenance[rel] = prov
        if not src.exists():
            missing.append(rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        # carry detector provenance sidecars if present (args.json = sf-train; args.yaml = legacy runs)
        for side in ("results.csv", "args.json", "args.yaml"):
            if (src.parent / side).exists():
                shutil.copy2(src.parent / side, dst.parent / side)
        manifest_files.append(
            {
                "path": f"weights/{rel}",
                "sha256": sha256(dst),
                "bytes": dst.stat().st_size,
            }
        )

    # ---- logs (paper-relevant subdirs; all text, no real filenames) ----
    for sub in ("train", "eval", "scale", "scale_train", "matched"):
        s = REPRO / "logs" / sub
        if s.exists():
            shutil.copytree(s, out / "logs" / sub, dirs_exist_ok=True)

    # ---- results ----
    (out / "results").mkdir(parents=True, exist_ok=True)
    for f in (
        "SUMMARY.md",
        "per_seed.json",
        "SCALE_SUMMARY.md",  # real-data scaling curve (table)
        "scale_per_seed.json",
        "scale_curve.png",  # 3-panel learning-curve figure
    ):
        if (REPRO / f).exists():
            shutil.copy2(REPRO / f, out / "results" / f)

    # ---- scripts (copies of what was actually run) ----
    shutil.copytree("scripts/repro", out / "scripts" / "repro", dirs_exist_ok=True)
    # Legacy detector fine-tune wrapper (the repro drivers now call `sf-train` directly);
    # bundled only if it still exists.
    if Path("scripts/finetune/yolo/train.sh").exists():
        (out / "scripts" / "finetune_yolo").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            "scripts/finetune/yolo/train.sh",
            out / "scripts" / "finetune_yolo" / "train.sh",
        )
    shutil.copy2(
        "scripts/finetune/relmatch/eval_compare_all.py",
        out / "scripts" / "eval_compare_all.py",
    )
    # Sanitize the internal NFS mount path from EVERY bundled script (several default to
    # the confidential real-corpus mount; a licensed user passes their own path). Keeps the
    # released bundle free of internal infrastructure paths.
    mount = "/net-fs-ins/shared-docker-vols/structflo-cser-annotate/data"
    for p in (out / "scripts").rglob("*"):
        if p.suffix in (".py", ".sh"):
            txt = p.read_text()
            if mount in txt:
                p.write_text(txt.replace(mount, "data/real"))

    # ---- environment snapshot ----
    env = out / "environment"
    env.mkdir(parents=True, exist_ok=True)
    if Path("uv.lock").exists():
        shutil.copy2("uv.lock", env / "uv.lock")
    (env / "pip_freeze.txt").write_text(run(["uv", "run", "pip", "freeze"]))
    (env / "python_version.txt").write_text(run(["python", "--version"]))
    (env / "platform.txt").write_text(run(["uname", "-a"]))
    (env / "nvidia_smi.txt").write_text(run(["nvidia-smi"]))
    (env / "torch_cuda.txt").write_text(
        run(
            [
                "uv",
                "run",
                "python",
                "-c",
                "import torch;print('torch',torch.__version__);print('cuda',torch.version.cuda);"
                "print('device',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')",
            ]
        )
    )

    # ---- synthetic TEST set (NOT confidential) ----
    if not args.no_synth_data:
        sd = out / "synthetic_data" / "generated_test"
        if Path("data/generated_test").exists():
            shutil.copytree("data/generated_test", sd, dirs_exist_ok=True)
        if (REPRO / "synth_test.yaml").exists():
            # rewrite path: to a bundle-relative value
            txt = (REPRO / "synth_test.yaml").read_text()
            txt = "\n".join(
                "path: ./generated_test" if ln.startswith("path:") else ln
                for ln in txt.splitlines()
            )
            (out / "synthetic_data" / "synth_test.yaml").write_text(txt + "\n")
        (out / "synthetic_data" / "GENERATE.md").write_text(GENERATE_MD)

    # ---- docs ----
    (out / "README.md").write_text(
        readme_md(args.seeds, git_commit, git_dirty, missing)
    )
    (out / "CONFIDENTIAL_DATA_NOTE.md").write_text(CONFIDENTIAL_MD)
    manifest = {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "seeds": args.seeds,
        "eval_imgsz": 1280,
        "provenance": provenance,
        "missing_weights": missing,
        "files": manifest_files,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    total_mb = sum(f["bytes"] for f in manifest_files) / 1e6
    print(
        f"bundle: {out}  weights={len(manifest_files)} ({total_mb:.0f} MB)  missing={len(missing)}"
    )
    if missing:
        print("  MISSING (not yet trained):")
        for r in missing:
            print(f"    - {r}")


GENERATE_MD = """\
# Regenerating the synthetic data

The held-out synthetic TEST set is included here (`generated_test/`, 1000 pages, JPEG +
YOLO labels + per-compound ground-truth JSON with struct/label bboxes + SMILES). It was
generated seed-disjoint from train/val.

```bash
# TEST set (this folder) — generation seed 1000000, WITH distractors:
uv run sf-generate --out data/generated_test --num-train 0 --num-val 1000 \\
    --seed 1000000 --smiles data/smiles/chembl_smiles.csv --fonts-dir data/fonts \\
    --distractors-dir data/distractors --workers 0

# TRAIN/VAL (10k/1k, generation seeds 7-11006) are NOT shipped (118 GB). Regenerate:
uv run sf-generate --out data/generated --num-train 10000 --num-val 1000 \\
    --smiles data/smiles/chembl_smiles.csv --fonts-dir data/fonts \\
    --distractors-dir data/distractors
```

`synth_test.yaml` is a YOLO data config pointing at `./generated_test` (val/ holds the
1000 test pages). Exact RDKit/font versions are pinned in `../environment/`.
"""

CONFIDENTIAL_MD = """\
# Confidential real dataset — withheld

The real internal-document corpus is confidential and is **not** included in this bundle:
no images, no ground-truth, and not even the split manifest (`real_split.json`), whose
filenames could leak internal document identifiers.

What IS documented for reproducibility:

- **Split** (made with split seed 42, fixed across all training seeds):
  test = 100 pages, val = 75 pages, train = 830 pages.
- **Fine-tune corpora layout** a licensed holder of the real data would recreate with the
  project's `scripts/finetune/{yolo,lps}/prepare_data.py` + the relational det-box prep
  (`scripts/finetune/relmatch/prepare_det_data.py`):
  - `data/finetune/lps/{train,val,real_test}`  (GT boxes; LPS + relmatch GT-box training)
  - `data/finetune/relmatch_det/{train,val}`   (detection boxes; published relmatch)
  - `data/finetune/yolo/` + `data_real_test.yaml`  (detector fine-tune + real-test eval)

To reproduce §B numbers with your own licensed corpus, point the eval/train commands at
your data dirs (see `../README.md`). The §A synthetic results are fully reproducible from
this bundle alone.
"""


def readme_md(seeds, commit, dirty, missing) -> str:
    return f"""\
# structflo-cser — reproducibility bundle

Multi-seed weights + logs + results for the structure-label matcher paper. Every number
is reported as **mean ± std over seeds {seeds}**. Two sections, separated by design:

- **§A Synthetic** — synthetic-only weights, evaluated on the held-out synthetic TEST set
  (`synthetic_data/generated_test`, 1000 pages). Fully reproducible from this bundle.
- **§B Internal docs** — real-fine-tuned weights, evaluated on a confidential real TEST
  set (withheld; see `CONFIDENTIAL_DATA_NOTE.md`).

Source commit: `{commit}`{" (dirty working tree)" if dirty else ""}. All eval @ imgsz 1280.

## Layout
- `weights/` — seeded checkpoints. `detector/{{base_synth,finetuned}}`, `lps/{{synth,finetuned}}`,
  `relmatch/{{synth,gt_realsynth,det}}`, each `_s{{seed}}`. Per-weight training command is in
  `MANIFEST.json:provenance`; every file has a sha256 there.
- `results/SUMMARY.md` — the mean ± std tables (1-5 + LPS rows). `per_seed.json` — raw values.
- `logs/{{train,eval}}/` — full stdout of every run.
- `synthetic_data/` — the synth TEST set + `GENERATE.md` recipe + `synth_test.yaml`.
- `environment/` — `uv.lock`, `pip_freeze.txt`, python/cuda/platform/nvidia-smi.
- `scripts/` — the exact repro scripts used.

## Reproduce
```bash
uv sync --dev
# §A synthetic detection + matching (per seed, from this bundle's synth test set):
uv run python scripts/eval_compare_all.py --src synthetic_data/generated_test/val \\
    --manifest <all-stems-as-test.json> --detector weights/detector/base_synth_s42/best.safetensors \\
    --lps weights/lps/synth_s42/best.pt --relmatch weights/relmatch/synth_s42/best.pt \\
    --imgsz 1280 --conf 0.3
# Retrain from scratch (matchers minutes; base detector hours/seed):  scripts/repro/run_train.sh
```
{("## NOTE: " + str(len(missing)) + " weight(s) not yet present — partial bundle.") if missing else ""}
"""


if __name__ == "__main__":
    main()
