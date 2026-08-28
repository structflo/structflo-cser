#!/usr/bin/env bash
# Refresh the PUBLISHED detector + LPS by fine-tuning on the corpus that includes the
# newly annotated docs (data/finetune/plus, built by prepare_finetune_plus.py). Single
# canonical seed (42), warm-started from the synthetic base. Evaluates on the FROZEN
# 100-page real test set so we can confirm no regression vs the current release.
#
# Detector = D-FINE (sf-train); the run is runs/labels_detect/dfine_l_plus (the shipped
# real fine-tune). Env-overridable: DET_RUN, BASE_CKPT, FT_EPOCHS (30), FT_LR (5e-5),
# FT_BACKBONE_LR (5e-6), FT_BATCH (8), TRAIN_EXTRA.
#
#   setsid nohup bash scripts/repro/run_finetune_plus.sh >> runs/repro/logs/plus_driver.log 2>&1 </dev/null &
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

S=42
PLOG="runs/repro/logs/plus"
mkdir -p "$PLOG"
DET_RUN="${DET_RUN:-dfine_l_plus}"
DET_OUT="runs/labels_detect/$DET_RUN"
LPS_OUT="runs/lps_finetune_plus"
REAL_TEST_YAML="data/finetune/plus/yolo/data_real_test.yaml"
# Warm-start: the repro-staged seed-42 base (== dfine_l_synth) if run_train.sh has staged it,
# else the original dfine_l_synth run directly.
if [ -z "${BASE_CKPT:-}" ]; then
    if [ -f runs/repro/detector/base_synth_s42/best.safetensors ]; then
        BASE_CKPT=runs/repro/detector/base_synth_s42/best.safetensors
    else
        BASE_CKPT=runs/labels_detect/dfine_l_synth/weights/best.safetensors
    fi
fi
FT_EPOCHS="${FT_EPOCHS:-30}"
FT_LR="${FT_LR:-5e-5}"
FT_BACKBONE_LR="${FT_BACKBONE_LR:-5e-6}"
FT_BATCH="${FT_BATCH:-8}"
TRAIN_EXTRA="${TRAIN_EXTRA:-}"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
banner() { echo; echo "[$(ts)] === $* ==="; }

banner "prepare corpus (idempotent)"
uv run python scripts/repro/prepare_finetune_plus.py --new-oversample 4

# ---- detector fine-tune ----
if [ -f "$DET_OUT/weights/best.safetensors" ]; then echo "[skip] detector_plus ($DET_RUN)"; else
    banner "FT detector_plus ($DET_RUN: D-FINE, imgsz 1280, ${FT_EPOCHS}ep, lr $FT_LR/$FT_BACKBONE_LR, warm-start $BASE_CKPT)"
    rm -rf "$DET_OUT"
    # shellcheck disable=SC2086
    uv run sf-train --data "$PROJECT_ROOT/data/finetune/plus/yolo/data.yaml" \
        --init "$BASE_CKPT" --imgsz 1280 --batch "$FT_BATCH" --epochs "$FT_EPOCHS" \
        --lr "$FT_LR" --backbone-lr "$FT_BACKBONE_LR" --seed "$S" \
        --project runs/labels_detect --name "$DET_RUN" --exist-ok $TRAIN_EXTRA \
        2>&1 | tee "$PLOG/detector_train.log"
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
banner "detector mAP50 on frozen real test @1280 (operating-point P/R at conf 0.3)"
uv run python - "$DET_OUT/weights/best.safetensors" "$REAL_TEST_YAML" <<'PY' 2>&1 | tee "$PLOG/detval.log"
import sys
from structflo.cser.inference.detector import load_detector
from structflo.cser.inference.evaluate import evaluate_detector_on_yaml
r = evaluate_detector_on_yaml(load_detector(sys.argv[1], imgsz=1280), sys.argv[2], key="val", imgsz=1280, op_conf=0.3)
a = r["all"]
print(f"DETVAL mAP50={a['mAP50']:.4f} mAP50_95={a['mAP50-95']:.4f} P={a['P']:.4f} R={a['R']:.4f}", flush=True)
PY

banner "LPS pair-class accuracy on frozen real test"
uv run python scripts/repro/eval_lps_acc.py --weights "$LPS_OUT/best.pt" \
    --data data/finetune/plus/lps/real_test 2>&1 | tee "$PLOG/lps_acc.log"

banner "end-to-end (detector_plus + matchers) on real test"
uv run python scripts/finetune/relmatch/eval_compare_all.py \
    --detector "$DET_OUT/weights/best.safetensors" --lps "$LPS_OUT/best.pt" \
    --relmatch runs/repro/relmatch_det_s42/best.pt \
    --imgsz 1280 --conf 0.3 --margin 2.0 2>&1 | tee "$PLOG/e2e.txt"

banner "FINETUNE-PLUS COMPLETE"
