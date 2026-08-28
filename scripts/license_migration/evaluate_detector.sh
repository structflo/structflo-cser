#!/usr/bin/env bash
# Full real-data evaluation of a D-FINE detector checkpoint against the frozen YOLO v0.4 baseline.
#   bash scripts/license_migration/evaluate_detector.sh <weights.safetensors> <tag> [relmatch_weights]
# Writes runs/license_migration/preds/<tag>/*.json and runs/license_migration/eval/<tag>_*.json
set -uo pipefail
cd "$(dirname "$0")/../.."
W="$1"; TAG="$2"; RELMATCH="${3:-}"
P=runs/license_migration/preds/$TAG; E=runs/license_migration/eval; mkdir -p "$P" "$E"
lbl() { case $1 in synth_test) echo data/generated_test/val/labels;; *) echo data/finetune/yolo/$1/labels;; esac; }
REL=""; [ -n "$RELMATCH" ] && REL="--relmatch $RELMATCH"

echo "== dump (full res) =="
[ -f "$P/synth_test.json" ] || uv run python scripts/license_migration/dump_preds.py --weights "$W" --out-dir "$P"
for sp in real_test real_val synth_test; do
  uv run python scripts/license_migration/eval_preds.py --preds "$P/$sp.json" --labels "$(lbl $sp)" --conf 0.3 --out "$E/${TAG}_$sp.json" | tail -4
done
echo "== dump (deployment scale 0.48 / 0.5) =="
for sc in 0.48 0.5; do
  [ -f "$P/scale$sc/real_val.json" ] || uv run python scripts/license_migration/dump_preds.py --weights "$W" --out-dir "$P/scale$sc" --scale $sc --splits real_test real_val
  for sp in real_test real_val; do
    uv run python scripts/license_migration/eval_preds.py --preds "$P/scale$sc/$sp.json" --labels "$(lbl $sp)" --conf 0.3 --out "$E/${TAG}_scale${sc}_$sp.json" | tail -2
  done
done
echo "== end-to-end pairing (real test / val) =="
for sp in test val; do
  uv run python scripts/license_migration/e2e_from_preds.py --preds "$P/real_$sp.json" --split $sp $REL --out "$E/${TAG}_e2e_$sp.json" | tail -5
done
echo "== conf operating-point sweep on real VAL (Hungarian; never test) =="
for c in 0.1 0.2 0.25 0.3 0.35 0.4 0.5; do
  uv run python scripts/license_migration/e2e_from_preds.py --preds "$P/real_val.json" --split val --conf $c $REL --out "$E/${TAG}_e2e_val_conf$c.json" | grep -E "Hungarian|Relational" | tr '\n' ' '; echo "  <- conf $c"
done
echo "EVAL_DONE $TAG"
