"""Compare baseline (synthetic-only) vs fine-tuned D-FINE detector.

Evaluates both models on three val sets:
  1. Real test (held out) — the paper number; never trained on or selected against
  2. Real val (selection set) — drove early-stopping/checkpointing (reference only)
  3. Original synthetic val (1000 pages) — regression check

Prints a summary table with deltas at the end.

This script lives in scripts/finetune/yolo/ for git-history continuity; the
detector it evaluates is now D-FINE (``structflo.cser.inference.detector``,
``.safetensors`` weights), scored with the backend-neutral
``structflo.cser.inference.evaluate`` (COCO-style AP over each yaml's ``val``
split; P/R at the deployment operating point, default conf 0.3).

Usage:
    uv run python scripts/finetune/yolo/eval_compare.py
    uv run python scripts/finetune/yolo/eval_compare.py \
        --finetuned runs/labels_detect/finetune_trial/weights/best.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]

REAL_TEST_YAML = PROJECT_ROOT / "data" / "finetune" / "yolo" / "data_real_test.yaml"
REAL_VAL_YAML = PROJECT_ROOT / "data" / "finetune" / "yolo" / "data.yaml"
SYNTH_YAML = PROJECT_ROOT / "config" / "data.yaml"

# Run names under runs/labels_detect/: dfine_l_synth = synthetic-only base,
# dfine_l_plus = real fine-tune. train.sh's default RUN_NAME is finetune_trial;
# point --finetuned at that run to compare a trial.
BASELINE_RUN = "dfine_l_synth"
RUN_NAME = "dfine_l_plus"
BASELINE = (
    PROJECT_ROOT
    / "runs"
    / "labels_detect"
    / BASELINE_RUN
    / "weights"
    / "best.safetensors"
)
FINETUNED = (
    PROJECT_ROOT / "runs" / "labels_detect" / RUN_NAME / "weights" / "best.safetensors"
)

# Pinned inference resolution so the baseline and the fine-tuned model are
# compared at the same deployment resolution regardless of what each was
# trained at (otherwise the fine-tuning effect is confounded with resolution).
IMGSZ = 1280
# Operating point at which Precision / Recall are reported.
OP_CONF = 0.3

METRICS = ["mAP50", "mAP50-95", "Precision", "Recall"]


def _val(
    weights: Path, data_yaml: Path, imgsz: int = IMGSZ, op_conf: float = OP_CONF
) -> dict[str, float] | None:
    if not weights.exists():
        print(f"  weights not found: {weights}")
        return None
    if not data_yaml.exists():
        print(f"  data config not found: {data_yaml}")
        return None

    from structflo.cser.inference.detector import load_detector
    from structflo.cser.inference.evaluate import evaluate_detector_on_yaml

    model = load_detector(weights, imgsz=imgsz)
    r = evaluate_detector_on_yaml(
        model, data_yaml, key="val", imgsz=imgsz, op_conf=op_conf
    )

    return {
        "mAP50": r["all"]["mAP50"],
        "mAP50-95": r["all"]["mAP50-95"],
        "Precision": r["all"]["P"],
        "Recall": r["all"]["R"],
    }


def _delta_str(baseline: float, finetuned: float) -> str:
    d = finetuned - baseline
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def _print_comparison(label: str, baseline: dict, finetuned: dict) -> None:
    print(f"\n  {label}")
    print(f"  {'Metric':<12} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print(f"  {'-' * 46}")
    for m in METRICS:
        b, f = baseline[m], finetuned[m]
        print(f"  {m:<12} {b:>10.4f} {f:>12.4f} {_delta_str(b, f):>10}")


def _run_pair(
    label: str,
    data_yaml: Path,
    results: dict[str, dict[str, dict[str, float]]],
    key: str,
    baseline: Path,
    finetuned: Path,
    imgsz: int,
    op_conf: float,
) -> None:
    print(f"Running baseline on {label} ...")
    r = _val(baseline, data_yaml, imgsz, op_conf)
    if r:
        results.setdefault(key, {})["baseline"] = r

    print(f"Running fine-tuned on {label} ...")
    r = _val(finetuned, data_yaml, imgsz, op_conf)
    if r:
        results.setdefault(key, {})["finetuned"] = r


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare baseline vs fine-tuned D-FINE detector on real test / "
        "real val / synthetic val."
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE,
        help="synthetic-only base checkpoint (.safetensors)",
    )
    ap.add_argument(
        "--finetuned",
        type=Path,
        default=FINETUNED,
        help="fine-tuned checkpoint (.safetensors)",
    )
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--op-conf", type=float, default=OP_CONF)
    args = ap.parse_args()

    results: dict[str, dict[str, dict[str, float]]] = {}
    common = (args.baseline, args.finetuned, args.imgsz, args.op_conf)

    _run_pair("real test (held out)", REAL_TEST_YAML, results, "real_test", *common)
    _run_pair("real val (selection)", REAL_VAL_YAML, results, "real_val", *common)
    _run_pair("synthetic val", SYNTH_YAML, results, "synth_val", *common)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DETECTOR EVAL SUMMARY")
    print("=" * 60)

    rt = results.get("real_test", {})
    if "baseline" in rt and "finetuned" in rt:
        _print_comparison(
            "Real test — held out (paper number)", rt["baseline"], rt["finetuned"]
        )

    rv = results.get("real_val", {})
    if "baseline" in rv and "finetuned" in rv:
        _print_comparison(
            "Real val — selection set (reference)", rv["baseline"], rv["finetuned"]
        )

    sv = results.get("synth_val", {})
    if "baseline" in sv and "finetuned" in sv:
        _print_comparison(
            "Synthetic val — regression check", sv["baseline"], sv["finetuned"]
        )

    # --- Verdict ---
    if "finetuned" in rt and "finetuned" in sv:
        real_delta = rt["finetuned"]["mAP50"] - rt["baseline"]["mAP50"]
        synth_delta = sv["finetuned"]["mAP50"] - sv["baseline"]["mAP50"]
        print("\n  Verdict:")
        print(f"    Real-data mAP50 delta      : {real_delta:+.4f}")
        print(f"    Synthetic mAP50 delta      : {synth_delta:+.4f}")
        if real_delta > 0.005 and synth_delta > -0.01:
            print(
                "    --> Fine-tuning helped on real data with no synthetic regression"
            )
        elif real_delta > 0.005 and synth_delta <= -0.01:
            print("    --> Fine-tuning helped on real data BUT regressed on synthetic")
            print(
                "        Consider reducing REAL_OVERSAMPLE or increasing N_SYNTH_TRAIN"
            )
        elif real_delta <= 0.005:
            print(
                "    --> Minimal effect on real data — may need more annotations or epochs"
            )

    print()


if __name__ == "__main__":
    main()
