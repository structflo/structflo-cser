#!/usr/bin/env bash
# Reproducibility TRAINING driver — multi-seed detector + matchers.
#
# Trains every model the paper depends on, with EXPLICIT --data-dir + --seed so
# provenance is unambiguous. Seeds vary init/shuffle/augmentation only; data splits
# are fixed on disk (see scripts/repro/README.md).
#
#   §A (synthetic): base detector, lps_synth, relmatch_synth   (--data-dir data/generated)
#   §B (real+synth): fine-tuned detector, lps_ft, relmatch_gt, relmatch_det
#
# Detector = D-FINE (sf-train). Seed 42 base == the originally-trained dfine_l_synth run
# (provenance confirmed via the data/init/seed metadata embedded in best.safetensors);
# every other detector is trained fresh with the same recipe. There is no pre-existing
# D-FINE run on data/finetune/yolo, so the seed-42 fine-tune is trained fresh too unless
# ORIG_FT points at a matching run.
# Matchers are ALL trained fresh for every seed (their provenance is what we're fixing).
#
# Idempotent: every detector step is skipped if its best.safetensors is already staged,
# so the job is safe to interrupt and resume. Override seeds with:
#   SEEDS="42 43" bash run_train.sh
# Detector recipes are env-overridable (defaults below):
#   BASE_INIT / BASE_DATA / BASE_IMGSZ / BASE_EPOCHS / BASE_BATCH   (synthetic base)
#   FT_DATA / FT_IMGSZ / FT_EPOCHS / FT_LR / FT_BACKBONE_LR / FT_BATCH (real fine-tune)
#   TRAIN_EXTRA="..."   extra args appended to every sf-train call
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

REPRO="runs/repro"
LOGS="$REPRO/logs/train"
DET="$REPRO/detector"
mkdir -p "$LOGS" "$DET"
SEEDS="${SEEDS:-42 43 44}"

# Original seed-42 runs to reuse (set to "" to force a fresh train).
ORIG_BASE="${ORIG_BASE:-runs/labels_detect/dfine_l_synth}"   # seed-42 base (imgsz 1280, 10ep, dfine-large-coco init)
ORIG_FT="${ORIG_FT:-}"                                        # no matching seed-42 FT run on data/finetune/yolo

# ---- detector recipes (D-FINE via sf-train) ----
BASE_INIT="${BASE_INIT:-ustc-community/dfine-large-coco}"
BASE_DATA="${BASE_DATA:-config/data.yaml}"       # = data/generated, synthetic-only
BASE_IMGSZ="${BASE_IMGSZ:-1280}"
BASE_EPOCHS="${BASE_EPOCHS:-10}"
BASE_BATCH="${BASE_BATCH:-8}"
FT_DATA="${FT_DATA:-data/finetune/yolo/data.yaml}"   # real+synth (830-doc corpus)
FT_IMGSZ="${FT_IMGSZ:-1280}"
FT_EPOCHS="${FT_EPOCHS:-30}"
FT_LR="${FT_LR:-5e-5}"
FT_BACKBONE_LR="${FT_BACKBONE_LR:-5e-6}"
FT_BATCH="${FT_BATCH:-8}"
TRAIN_EXTRA="${TRAIN_EXTRA:-}"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
banner() { echo; echo "============================================================"; echo "[$(ts)] $*"; echo "============================================================"; }

# stage_run <src_run_dir> <dst_dir>  — copy best.safetensors + provenance (results.csv, args.json)
stage_run() {
    local src="$1" dst="$2"
    mkdir -p "$dst"
    cp -f "$src/weights/best.safetensors" "$dst/best.safetensors"
    [ -f "$src/results.csv" ] && cp -f "$src/results.csv" "$dst/results.csv" || true
    [ -f "$src/args.json" ] && cp -f "$src/args.json" "$dst/args.json" || true
}

for S in $SEEDS; do
    banner "SEED $S"

    # ---------- §A/§B detector: base (synthetic-only) ----------
    BASE_DST="$DET/base_synth_s$S"
    if [ -f "$BASE_DST/best.safetensors" ]; then
        echo "[skip] base detector s$S already staged"
    elif [ "$S" = "42" ] && [ -n "$ORIG_BASE" ] && [ -f "$ORIG_BASE/weights/best.safetensors" ]; then
        echo "[reuse] base detector s42 <- $ORIG_BASE (provenance: safetensors metadata data=$BASE_DATA,imgsz=$BASE_IMGSZ,seed=42)"
        stage_run "$ORIG_BASE" "$BASE_DST"
    else
        banner "TRAIN base detector s$S (D-FINE, init $BASE_INIT, imgsz $BASE_IMGSZ, batch $BASE_BATCH, ${BASE_EPOCHS}ep)"
        rm -rf "runs/labels_detect/dfine_l_synth_s$S"  # clear partial from any prior killed run (fresh retrain)
        # shellcheck disable=SC2086
        uv run sf-train --data "$BASE_DATA" --init "$BASE_INIT" \
            --imgsz "$BASE_IMGSZ" --batch "$BASE_BATCH" --epochs "$BASE_EPOCHS" --seed "$S" \
            --project runs/labels_detect --name "dfine_l_synth_s$S" $TRAIN_EXTRA \
            2>&1 | tee "$LOGS/base_detector_s$S.log"
        stage_run "runs/labels_detect/dfine_l_synth_s$S" "$BASE_DST"
    fi

    # ---------- §B detector: fine-tuned (real+synth) ----------
    FT_DST="$DET/finetuned_s$S"
    if [ -f "$FT_DST/best.safetensors" ]; then
        echo "[skip] fine-tuned detector s$S already staged"
    elif [ "$S" = "42" ] && [ -n "$ORIG_FT" ] && [ -f "$ORIG_FT/weights/best.safetensors" ]; then
        echo "[reuse] fine-tuned detector s42 <- $ORIG_FT (provenance: safetensors metadata)"
        stage_run "$ORIG_FT" "$FT_DST"
    else
        banner "TRAIN fine-tuned detector s$S (D-FINE, imgsz $FT_IMGSZ, ${FT_EPOCHS}ep, lr $FT_LR/$FT_BACKBONE_LR, warm-start base_synth_s$S)"
        # shellcheck disable=SC2086
        uv run sf-train --data "$FT_DATA" --init "$BASE_DST/best.safetensors" \
            --imgsz "$FT_IMGSZ" --batch "$FT_BATCH" --epochs "$FT_EPOCHS" \
            --lr "$FT_LR" --backbone-lr "$FT_BACKBONE_LR" --seed "$S" \
            --project runs/labels_detect --name "dfine_l_ft_s$S" --exist-ok $TRAIN_EXTRA \
            2>&1 | tee "$LOGS/ft_detector_s$S.log"
        stage_run "runs/labels_detect/dfine_l_ft_s$S" "$FT_DST"
    fi

    # ---------- §A matchers (synthetic-only) ----------
    # Matchers gate on a .done marker (written only after a clean exit) — NOT best.pt,
    # which the trainers write mid-run, so a killed step is correctly retrained on resume.
    if [ -f "$REPRO/lps_synth_s$S/.done" ]; then echo "[skip] lps_synth s$S"; else
        banner "TRAIN lps_synth s$S (--data-dir data/generated)"
        uv run sf-train-lps --data-dir data/generated --seed "$S" \
            --output-dir "$REPRO/lps_synth_s$S" 2>&1 | tee "$LOGS/lps_synth_s$S.log"
        touch "$REPRO/lps_synth_s$S/.done"
    fi
    if [ -f "$REPRO/relmatch_synth_s$S/.done" ]; then echo "[skip] relmatch_synth s$S"; else
        banner "TRAIN relmatch_synth s$S (GT-box, --data-dir data/generated)"
        uv run sf-train-relmatch --data-dir data/generated --seed "$S" \
            --output-dir "$REPRO/relmatch_synth_s$S" 2>&1 | tee "$LOGS/relmatch_synth_s$S.log"
        touch "$REPRO/relmatch_synth_s$S/.done"
    fi

    # ---------- §B matchers (real-fine-tuned / published-kind) ----------
    if [ -f "$REPRO/lps_ft_s$S/.done" ]; then echo "[skip] lps_ft s$S"; else
        banner "TRAIN lps_ft s$S (warm-start lps_synth_s$S, --data-dir data/finetune/lps)"
        uv run sf-train-lps --finetune "$REPRO/lps_synth_s$S/best.pt" \
            --data-dir data/finetune/lps --seed "$S" \
            --output-dir "$REPRO/lps_ft_s$S" 2>&1 | tee "$LOGS/lps_ft_s$S.log"
        touch "$REPRO/lps_ft_s$S/.done"
    fi
    if [ -f "$REPRO/relmatch_gt_s$S/.done" ]; then echo "[skip] relmatch_gt s$S"; else
        banner "TRAIN relmatch_gt s$S (GT-box, real+synth — clean-variant ablation)"
        uv run sf-train-relmatch --data-dir data/finetune/lps --seed "$S" \
            --output-dir "$REPRO/relmatch_gt_s$S" 2>&1 | tee "$LOGS/relmatch_gt_s$S.log"
        touch "$REPRO/relmatch_gt_s$S/.done"
    fi
    if [ -f "$REPRO/relmatch_det_s$S/.done" ]; then echo "[skip] relmatch_det s$S"; else
        banner "TRAIN relmatch_det s$S (detection-box — the PUBLISHED kind)"
        uv run sf-train-relmatch --det-data-dir data/finetune/relmatch_det --seed "$S" \
            --output-dir "$REPRO/relmatch_det_s$S" 2>&1 | tee "$LOGS/relmatch_det_s$S.log"
        touch "$REPRO/relmatch_det_s$S/.done"
    fi
done

banner "TRAINING MATRIX COMPLETE  (seeds: $SEEDS)"
