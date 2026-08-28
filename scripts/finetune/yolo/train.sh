#!/usr/bin/env bash
# Fine-tune the D-FINE detector on combined synthetic + real data.
#
# Drives `uv run sf-train` (structflo.cser.training.trainer). This directory is
# still scripts/finetune/yolo/ for git-history continuity: the on-disk data
# layout (YOLO-txt labels + data yaml, written by prepare_data.py) is unchanged,
# but the detector being trained is D-FINE (weights are .safetensors).
#
# Prerequisites:
#   uv run python scripts/finetune/yolo/prepare_data.py
#
# Usage:
#   bash scripts/finetune/yolo/train.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_DIR="$PROJECT_ROOT/runs/labels_detect"

# Overridable via env vars (defaults reproduce the original trial run).
# CHECKPOINT = model to warm-start from: a .safetensors checkpoint (per-seed base
#              for seed isolation) or an HF Hub model id (passed to --init as-is).
# SEED varies init/shuffle/augmentation only — data splits stay fixed.
# LR / BACKBONE_LR = fine-tune-stage learning rates (below the from-scratch
#              1e-4 / 1e-5 that sf-train defaults to).
# MOSAIC is a legacy knob from the retired trainer: D-FINE training has no mosaic
#              augmentation, so it is accepted but ignored (a note is printed).
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/runs/labels_detect/dfine_l_synth/weights/best.safetensors}"
DATA_YAML="${DATA_YAML:-$PROJECT_ROOT/data/finetune/yolo/data.yaml}"
RUN_NAME="${RUN_NAME:-finetune_trial}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-5}"
SEED="${SEED:-42}"
LR="${LR:-5e-5}"
BACKBONE_LR="${BACKBONE_LR:-5e-6}"
MOSAIC="${MOSAIC:-}"

if [ ! -f "$DATA_YAML" ]; then
    echo "ERROR: $DATA_YAML not found. Run prepare_data.py first."
    exit 1
fi

case "$CHECKPOINT" in
    *.pt)
        echo "ERROR: Checkpoint $CHECKPOINT is a legacy ultralytics .pt file;"
        echo "       the D-FINE trainer needs a .safetensors checkpoint or an HF Hub model id."
        exit 1
        ;;
    *.safetensors)
        if [ ! -f "$CHECKPOINT" ]; then
            echo "ERROR: Checkpoint $CHECKPOINT not found."
            exit 1
        fi
        ;;
    *)
        # Anything else is treated as an HF Hub model id and resolved by sf-train.
        ;;
esac

echo "=== D-FINE detector fine-tune (sf-train) ==="
echo "  Data:        $DATA_YAML"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Run name:    $RUN_NAME"
echo "  Epochs:      $EPOCHS  (patience $PATIENCE)"
echo "  LR:          $LR  (backbone $BACKBONE_LR)"
echo "  Seed:        $SEED"
if [ -n "$MOSAIC" ]; then
    echo "  NOTE: MOSAIC=$MOSAIC is ignored — the D-FINE trainer has no mosaic augmentation."
else
    echo "  NOTE: mosaic augmentation no longer applies (D-FINE trainer); MOSAIC env var is ignored."
fi
echo ""

# Optimiser/schedule mirror the retired recipe (AdamW, cosine LR, 1-epoch warmup,
# batch 8 @ 1280). Augmentation: scale jitter 0.3 + brightness 0.1 (the old
# scale/hsv_v knobs); degrees/translate/shear/mosaic have no D-FINE equivalent.
# Checkpoint selection is on the yaml's val split (fitness = 0.1*mAP50 + 0.9*mAP50-95).
uv run sf-train \
    --data "$DATA_YAML" \
    --init "$CHECKPOINT" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch 8 \
    --imgsz 1280 \
    --lr "$LR" \
    --backbone-lr "$BACKBONE_LR" \
    --lrf 0.01 \
    --weight-decay 1e-4 \
    --warmup-epochs 1 \
    --grad-clip 0.1 \
    --ema-decay 0.9999 \
    --conf 0.3 \
    --scale-jitter 0.3 \
    --brightness 0.1 \
    --workers 8 \
    --seed "$SEED" \
    --project "$OUT_DIR" \
    --name "$RUN_NAME" \
    --exist-ok \
    --save-period 5

# Validate the selected checkpoint on the yaml's val split at the deployment
# resolution with the backend-neutral evaluator (P/R at conf 0.3).
BEST="$OUT_DIR/$RUN_NAME/weights/best.safetensors"
if [ -f "$BEST" ]; then
    uv run python - "$BEST" "$DATA_YAML" <<'PY'
import sys

from structflo.cser.inference.detector import load_detector
from structflo.cser.inference.evaluate import evaluate_detector_on_yaml

best, data_yaml = sys.argv[1], sys.argv[2]
model = load_detector(best, imgsz=1280)
r = evaluate_detector_on_yaml(model, data_yaml, key="val", imgsz=1280, op_conf=0.3)
print()
print("--- Fine-tuned model metrics (val split @1280, P/R at conf 0.3) ---")
print(f"mAP50:     {r['all']['mAP50']:.4f}")
print(f"mAP50-95:  {r['all']['mAP50-95']:.4f}")
print(f"Precision: {r['all']['P']:.4f}")
print(f"Recall:    {r['all']['R']:.4f}")
print(f"Weights:   {best}")
PY
else
    echo "WARNING: $BEST not found after training; skipping validation."
fi
