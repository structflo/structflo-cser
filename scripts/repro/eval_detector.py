"""Parameterized detector eval for the reproducibility tables (per seed).

Emits machine-readable per-class metrics (P / R / mAP50 / mAP50-95) at a FIXED
inference resolution (default 1280, the deployment res) so the base and the
fine-tuned detector are compared at the same resolution, regardless of the
resolution each checkpoint was trained at.

Backend-neutral: weights are loaded through ``structflo.cser.inference.detector``
(D-FINE, ``.safetensors``) and scored with ``structflo.cser.inference.evaluate``
— COCO-style AP over the yaml's ``val`` split, with P/R reported at the deployment
operating point ``--op-conf`` (default 0.3). The same evaluator is used for every
checkpoint, so base and fine-tuned numbers share one protocol.

Produces (written as JSON):
  * table1_synth          — base (synthetic-only) detector on the synthetic TEST set.
  * table3_real           — base vs fine-tuned on the real TEST set.
  * table3_synth_regress  — base vs fine-tuned on the synthetic TEST set (regression).

Each block is ``{"all": {P,R,mAP50,mAP50-95}, "chemical_structure": {...},
"compound_label": {...}}`` (consumed by scripts/repro/aggregate.py).

Usage:
    uv run python scripts/repro/eval_detector.py --seed 42 \
        --base runs/repro/detector/base_synth_s42/best.safetensors \
        --ft   runs/repro/detector/finetuned_s42/best.safetensors \
        --synth-test-yaml runs/repro/synth_test.yaml \
        --real-test-yaml  data/finetune/yolo/data_real_test.yaml \
        --out runs/repro/logs/eval/detector_s42.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRIC_KEYS = ("P", "R", "mAP50", "mAP50-95")


def _val(weights: Path, data_yaml: Path, imgsz: int, op_conf: float) -> dict | None:
    if not weights.exists():
        print(f"  [skip] weights not found: {weights}")
        return None
    if not data_yaml.exists():
        print(f"  [skip] data config not found: {data_yaml}")
        return None

    from structflo.cser.inference.detector import load_detector
    from structflo.cser.inference.evaluate import (
        CLASS_NAMES,
        evaluate_detector_on_yaml,
    )

    model = load_detector(weights, imgsz=imgsz)
    r = evaluate_detector_on_yaml(
        model, data_yaml, key="val", imgsz=imgsz, op_conf=op_conf
    )
    # Keep the block shape the retired ultralytics ``model.val()`` version emitted:
    # "all" + one entry per class ({0: chemical_structure, 1: compound_label}),
    # each with P / R / mAP50 / mAP50-95.
    out = {"all": {k: float(r["all"][k]) for k in METRIC_KEYS}}
    for name in CLASS_NAMES.values():
        out[name] = {k: float(r[name][k]) for k in METRIC_KEYS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--base", type=Path, required=True, help="synthetic-only base detector"
    )
    ap.add_argument("--ft", type=Path, required=True, help="real-fine-tuned detector")
    ap.add_argument("--synth-test-yaml", type=Path, required=True)
    ap.add_argument("--real-test-yaml", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument(
        "--op-conf",
        type=float,
        default=0.3,
        help="operating point (conf threshold) at which P/R are reported",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = {"seed": args.seed, "imgsz": args.imgsz, "op_conf": args.op_conf}

    print(f"[seed {args.seed}] Table 1 — base on synthetic TEST @ {args.imgsz} ...")
    result["table1_synth"] = _val(
        args.base, args.synth_test_yaml, args.imgsz, args.op_conf
    )

    print(f"[seed {args.seed}] Table 3 — base vs FT on real TEST @ {args.imgsz} ...")
    result["table3_real"] = {
        "base": _val(args.base, args.real_test_yaml, args.imgsz, args.op_conf),
        "ft": _val(args.ft, args.real_test_yaml, args.imgsz, args.op_conf),
    }

    print(
        f"[seed {args.seed}] Table 3 — synthetic regression (base vs FT) @ {args.imgsz} ..."
    )
    result["table3_synth_regress"] = {
        "base": _val(args.base, args.synth_test_yaml, args.imgsz, args.op_conf),
        "ft": _val(args.ft, args.synth_test_yaml, args.imgsz, args.op_conf),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
