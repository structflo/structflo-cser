#!/usr/bin/env bash
# Re-cache detection boxes with the NEW detector and retrain the det-trained relational matcher
# (its node feature 8 is the detector confidence and its dustbin prior was fitted to YOLO's FP/miss rates).
#   bash scripts/license_migration/recalibrate_relmatch.sh <detector.safetensors> [conf=0.3] [seed=42]
set -uo pipefail
cd "$(dirname "$0")/../.."
W="$1"; CONF="${2:-0.3}"; SEED="${3:-42}"
DETDATA=data/finetune/relmatch_det_dfine_c${CONF}_s${SEED}
OUT=runs/relmatch_det_dfine_c${CONF}_s${SEED}
if [ -f "$DETDATA/.done" ]; then echo "[skip] det-box cache $DETDATA"; else
  uv run python scripts/finetune/relmatch/prepare_det_data.py --src data/finetune/plus/lps --detector "$W" \
      --out "$DETDATA" --conf "$CONF" --imgsz 1280 && touch "$DETDATA/.done"
fi
if [ -f "$OUT/.done" ]; then echo "[skip] relmatch train $OUT"; else
  uv run sf-train-relmatch --det-data-dir "$DETDATA" --seed "$SEED" --output-dir "$OUT" && touch "$OUT/.done"
fi
echo "RELMATCH_DONE $OUT/best.pt"
