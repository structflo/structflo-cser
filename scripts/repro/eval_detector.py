"""Parameterized detector eval for the reproducibility tables (per seed).

Emits machine-readable per-class metrics (P / R / mAP50 / mAP50-95) at a FIXED
inference resolution (default 1280, the deployment res) so the base (@2048) and the
fine-tuned (@1280) detector are compared at the same resolution. ``model.val()``
otherwise silently uses each checkpoint's own training imgsz.

Produces (written as JSON):
  * table1_synth          — base (synthetic-only) detector on the synthetic TEST set.
  * table3_real           — base vs fine-tuned on the real TEST set.
  * table3_synth_regress  — base vs fine-tuned on the synthetic TEST set (regression).

Usage:
    uv run python scripts/repro/eval_detector.py --seed 42 \
        --base runs/repro/detector/base_synth_s42/best.pt \
        --ft   runs/repro/detector/finetuned_s42/best.pt \
        --synth-test-yaml runs/repro/synth_test.yaml \
        --real-test-yaml  data/finetune/yolo/data_real_test.yaml \
        --out runs/repro/logs/eval/detector_s42.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _val(weights: Path, data_yaml: Path, imgsz: int) -> dict | None:
    if not weights.exists():
        print(f"  [skip] weights not found: {weights}")
        return None
    if not data_yaml.exists():
        print(f"  [skip] data config not found: {data_yaml}")
        return None

    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), imgsz=imgsz, verbose=False)
    box = m.box
    names = m.names  # {0: chemical_structure, 1: compound_label}
    out = {
        "all": {
            "P": float(box.mp),
            "R": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
        }
    }
    for i, ci in enumerate(box.ap_class_index):
        out[names[int(ci)]] = {
            "P": float(box.p[i]),
            "R": float(box.r[i]),
            "mAP50": float(box.ap50[i]),
            "mAP50-95": float(box.ap[i]),
        }
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
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = {"seed": args.seed, "imgsz": args.imgsz}

    print(f"[seed {args.seed}] Table 1 — base on synthetic TEST @ {args.imgsz} ...")
    result["table1_synth"] = _val(args.base, args.synth_test_yaml, args.imgsz)

    print(f"[seed {args.seed}] Table 3 — base vs FT on real TEST @ {args.imgsz} ...")
    result["table3_real"] = {
        "base": _val(args.base, args.real_test_yaml, args.imgsz),
        "ft": _val(args.ft, args.real_test_yaml, args.imgsz),
    }

    print(
        f"[seed {args.seed}] Table 3 — synthetic regression (base vs FT) @ {args.imgsz} ..."
    )
    result["table3_synth_regress"] = {
        "base": _val(args.base, args.synth_test_yaml, args.imgsz),
        "ft": _val(args.ft, args.synth_test_yaml, args.imgsz),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
