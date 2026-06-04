#!/usr/bin/env bash
# Refresh the PUBLISHED det-trained relational matcher on the new corpus, matched to the
# new detector. Caches the finetune_plus detector's boxes over the data/finetune/plus
# corpus, retrains relmatch_det on them, then re-evaluates the full refreshed trio
# (detector + LPS + relational) on the FROZEN real test set.
#
#   setsid nohup bash scripts/repro/run_relmatch_plus.sh >> runs/repro/logs/relmatch_plus_driver.log 2>&1 </dev/null &
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

S=42
RLOG="runs/repro/logs/relmatch_plus"
mkdir -p "$RLOG"
DET="runs/labels_detect/finetune_plus/weights/best.pt"
LPS="runs/lps_finetune_plus/best.pt"
DETDATA="data/finetune/relmatch_det_plus"
OUT="runs/relmatch_det_plus"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
banner() { echo; echo "[$(ts)] === $* ==="; }

# 1. cache detection boxes (new detector over the new corpus train+val)
if [ -f "$DETDATA/.done" ]; then echo "[skip] det-box caching"; else
    banner "cache det boxes (finetune_plus detector over data/finetune/plus/lps)"
    uv run python scripts/finetune/relmatch/prepare_det_data.py \
        --src data/finetune/plus/lps --detector "$DET" --out "$DETDATA" \
        --conf 0.3 --imgsz 1280 2>&1 | tee "$RLOG/prep.log"
    touch "$DETDATA/.done"
fi

# 2. train the relational matcher on the new det-box data
if [ -f "$OUT/.done" ]; then echo "[skip] relmatch_det_plus train"; else
    banner "train relmatch_det_plus (detection-box, new corpus)"
    uv run sf-train-relmatch --det-data-dir "$DETDATA" --seed "$S" \
        --output-dir "$OUT" 2>&1 | tee "$RLOG/train.log"
    touch "$OUT/.done"
fi

# 3. re-evaluate the full refreshed trio on the frozen real test
banner "e2e — refreshed trio (detector_plus + lps_plus + relmatch_det_plus)"
uv run python scripts/finetune/relmatch/eval_compare_all.py \
    --detector "$DET" --lps "$LPS" --relmatch "$OUT/best.pt" \
    --imgsz 1280 --conf 0.3 --margin 2.0 2>&1 | tee "$RLOG/e2e.txt"

banner "RELMATCH-PLUS COMPLETE"
