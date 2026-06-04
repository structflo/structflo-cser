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
# Detector seed 42 == the originally-trained checkpoints (provenance confirmed via the
# train_args embedded in each .pt); only s43/s44 are trained fresh with the same recipe.
# Matchers are ALL trained fresh for every seed (their provenance is what we're fixing).
#
# Idempotent: every step is skipped if its best.pt already exists, so the job is safe
# to interrupt and resume. Override seeds with:  SEEDS="42 43" bash run_train.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

REPRO="runs/repro"
LOGS="$REPRO/logs/train"
DET="$REPRO/detector"
mkdir -p "$LOGS" "$DET"
SEEDS="${SEEDS:-42 43 44}"

ORIG_BASE="runs/labels_detect/yolo11l_panels"   # seed-42 base (imgsz 2048, 30ep)
ORIG_FT="runs/labels_detect/finetune_3way"      # seed-42 FT   (imgsz 1280, 25ep)

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
banner() { echo; echo "============================================================"; echo "[$(ts)] $*"; echo "============================================================"; }

# stage_run <src_run_dir> <dst_dir>  — copy best.pt + provenance (results.csv, args.yaml)
stage_run() {
    local src="$1" dst="$2"
    mkdir -p "$dst"
    cp -f "$src/weights/best.pt" "$dst/best.pt"
    [ -f "$src/results.csv" ] && cp -f "$src/results.csv" "$dst/results.csv" || true
    [ -f "$src/args.yaml" ] && cp -f "$src/args.yaml" "$dst/args.yaml" || true
}

for S in $SEEDS; do
    banner "SEED $S"

    # ---------- §A/§B detector: base (synthetic-only) ----------
    BASE_DST="$DET/base_synth_s$S"
    if [ -f "$BASE_DST/best.pt" ]; then
        echo "[skip] base detector s$S already staged"
    elif [ "$S" = "42" ]; then
        echo "[reuse] base detector s42 <- $ORIG_BASE (provenance: train_args imgsz=2048,seed=42)"
        stage_run "$ORIG_BASE" "$BASE_DST"
    else
        banner "TRAIN base detector s$S (imgsz 2048, batch 4, 30ep)"
        rm -rf "runs/labels_detect/yolo11l_panels_s$S"  # clear partial from any prior killed run (exist_ok=False)
        uv run sf-train --imgsz 2048 --batch 4 --epochs 30 --seed "$S" \
            --name "yolo11l_panels_s$S" 2>&1 | tee "$LOGS/base_detector_s$S.log"
        stage_run "runs/labels_detect/yolo11l_panels_s$S" "$BASE_DST"
    fi

    # ---------- §B detector: fine-tuned (real+synth) ----------
    FT_DST="$DET/finetuned_s$S"
    if [ -f "$FT_DST/best.pt" ]; then
        echo "[skip] fine-tuned detector s$S already staged"
    elif [ "$S" = "42" ]; then
        echo "[reuse] fine-tuned detector s42 <- $ORIG_FT (provenance: train_args imgsz=1280,seed=42)"
        stage_run "$ORIG_FT" "$FT_DST"
    else
        banner "TRAIN fine-tuned detector s$S (imgsz 1280, 25ep, warm-start base_synth_s$S)"
        SEED="$S" RUN_NAME="finetune_s$S" EPOCHS=25 \
            CHECKPOINT="$PROJECT_ROOT/$BASE_DST/best.pt" \
            bash scripts/finetune/yolo/train.sh 2>&1 | tee "$LOGS/ft_detector_s$S.log"
        stage_run "runs/labels_detect/finetune_s$S" "$FT_DST"
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
