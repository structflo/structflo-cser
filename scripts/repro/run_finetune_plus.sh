#!/usr/bin/env bash
# Refresh the PUBLISHED detector + LPS by fine-tuning on the corpus that includes the
# newly annotated docs (data/finetune/plus, built by prepare_finetune_plus.py). Single
# canonical seed (42), warm-started from the synthetic base. Evaluates on the FROZEN
# 100-page real test set so we can confirm no regression vs the current release.
#
#   setsid nohup bash scripts/repro/run_finetune_plus.sh >> runs/repro/logs/plus_driver.log 2>&1 </dev/null &
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

S=42
PLOG="runs/repro/logs/plus"
mkdir -p "$PLOG"
DET_OUT="runs/labels_detect/finetune_plus"
LPS_OUT="runs/lps_finetune_plus"
REAL_TEST_YAML="data/finetune/plus/yolo/data_real_test.yaml"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
banner() { echo; echo "[$(ts)] === $* ==="; }

banner "prepare corpus (idempotent)"
uv run python scripts/repro/prepare_finetune_plus.py --new-oversample 4

# ---- detector fine-tune ----
if [ -f "$DET_OUT/weights/best.pt" ]; then echo "[skip] detector_plus"; else
    banner "FT detector_plus (imgsz 1280, 25ep, warm-start base_synth_s42)"
    rm -rf "$DET_OUT"
    SEED="$S" RUN_NAME="finetune_plus" EPOCHS=25 \
        DATA_YAML="$PROJECT_ROOT/data/finetune/plus/yolo/data.yaml" \
        CHECKPOINT="$PROJECT_ROOT/runs/repro/detector/base_synth_s42/best.pt" \
        bash scripts/finetune/yolo/train.sh 2>&1 | tee "$PLOG/detector_train.log"
fi

# ---- LPS fine-tune ----
if [ -f "$LPS_OUT/.done" ]; then echo "[skip] lps_plus"; else
    banner "FT lps_plus (warm-start lps_synth_s42)"
    uv run sf-train-lps --finetune runs/repro/lps_synth_s42/best.pt \
        --data-dir data/finetune/plus/lps --seed "$S" \
        --output-dir "$LPS_OUT" 2>&1 | tee "$PLOG/lps_train.log"
    touch "$LPS_OUT/.done"
fi

# ---- evaluate on the FROZEN real test set ----
banner "detector mAP50 on frozen real test @1280"
uv run python - "$DET_OUT/weights/best.pt" "$REAL_TEST_YAML" <<'PY' 2>&1 | tee "$PLOG/detval.log"
import sys
from ultralytics import YOLO
m = YOLO(sys.argv[1]).val(data=sys.argv[2], imgsz=1280, verbose=False)
print(f"DETVAL mAP50={m.box.map50:.4f} mAP50_95={m.box.map:.4f} P={m.box.mp:.4f} R={m.box.mr:.4f}", flush=True)
PY

banner "LPS pair-class accuracy on frozen real test"
uv run python scripts/repro/eval_lps_acc.py --weights "$LPS_OUT/best.pt" \
    --data data/finetune/plus/lps/real_test 2>&1 | tee "$PLOG/lps_acc.log"

banner "end-to-end (detector_plus + matchers) on real test"
uv run python scripts/finetune/relmatch/eval_compare_all.py \
    --detector "$DET_OUT/weights/best.pt" --lps "$LPS_OUT/best.pt" \
    --relmatch runs/repro/relmatch_det_s42/best.pt \
    --imgsz 1280 --conf 0.3 --margin 2.0 2>&1 | tee "$PLOG/e2e.txt"

banner "FINETUNE-PLUS COMPLETE"
