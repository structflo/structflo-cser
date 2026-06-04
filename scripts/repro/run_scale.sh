#!/usr/bin/env bash
# Real-data SCALING curve driver: fine-tune on increasing numbers of real docs and
# evaluate on the FROZEN real test set. Three metrics per point: detector mAP50,
# LPS pair-classification accuracy, and end-to-end pairing F1 (detector + parameter-free
# Hungarian matcher, so the e2e curve reflects detector scaling without a matcher-training
# confound). Mean ± s.d. over seeds.
#
# Curve points N (real docs): 0 (synthetic-only base, reused) .. 830 (full FT, reused);
# only the middle points are trained here. Idempotent (.done markers + completion-grep).
#
#   setsid nohup bash scripts/repro/run_scale.sh >> runs/repro/logs/scale_driver.log 2>&1 </dev/null &
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

REPRO="runs/repro"
SCALE="$REPRO/scale"            # staged checkpoints
SLOG="$REPRO/logs/scale"        # per-point eval logs
TLOG="$REPRO/logs/scale_train"  # training stdout
mkdir -p "$SCALE" "$SLOG" "$TLOG"

SEEDS="${SEEDS:-42 43 44}"
NS_MID="${NS_MID:-50 100 200 400}"   # points trained here
REAL_TEST_YAML="data/finetune/yolo/data_real_test.yaml"
LPS_REAL_TEST="data/finetune/lps/real_test"
REAL_SRC="/net-fs-ins/shared-docker-vols/structflo-cser-annotate/data"
EC_ALL="scripts/finetune/relmatch/eval_compare_all.py"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
banner() { echo; echo "[$(ts)] === $* ==="; }
done_marker() { [ -f "$1" ] && grep -q "$2" "$1" 2>/dev/null; }
have() { [ -f "$1" ]; }

# eval_point <N> <S> <detector.pt> <lps.pt> — run the 3 metrics for one curve point.
eval_point() {
    local N="$1" S="$2" det="$3" lps="$4" relm="$REPRO/relmatch_det_s$2/best.pt"
    # (a) detector mAP50 on frozen real test @1280
    local out="$SLOG/point_n${N}_s${S}_detval.log"
    if done_marker "$out" "DETVAL"; then echo "[skip] detval n$N s$S"; else
        banner "detector mAP50  n=$N s=$S"
        uv run python - "$det" "$REAL_TEST_YAML" <<'PY' 2>&1 | tee "$out"
import sys
from ultralytics import YOLO
m = YOLO(sys.argv[1]).val(data=sys.argv[2], imgsz=1280, verbose=False)
print(f"DETVAL mAP50={m.box.map50:.4f} mAP50_95={m.box.map:.4f} P={m.box.mp:.4f} R={m.box.mr:.4f}", flush=True)
PY
    fi
    # (b) LPS pair-classification accuracy on frozen real test
    out="$SLOG/point_n${N}_s${S}_lpsacc.log"
    if done_marker "$out" "RESULT"; then echo "[skip] lpsacc n$N s$S"; else
        banner "LPS acc  n=$N s=$S"
        uv run python scripts/repro/eval_lps_acc.py --weights "$lps" --data "$LPS_REAL_TEST" 2>&1 | tee "$out"
    fi
    # (c) end-to-end pairing F1 (read the Hungarian Part-B [TEST] row downstream)
    out="$SLOG/point_n${N}_s${S}_e2e.txt"
    if done_marker "$out" "PART B"; then echo "[skip] e2e n$N s$S"; else
        banner "e2e (detector + matchers)  n=$N s=$S"
        uv run python "$EC_ALL" --detector "$det" --lps "$lps" --relmatch "$relm" \
            --imgsz 1280 --conf 0.3 --margin 2.0 2>&1 | tee "$out"
    fi
}

for S in $SEEDS; do
    banner "SCALE SEED $S"
    # ---- endpoint N=0: synthetic-only base ----
    eval_point 0 "$S" "$REPRO/detector/base_synth_s$S/best.pt" "$REPRO/lps_synth_s$S/best.pt"

    # ---- middle points: train then eval ----
    for N in $NS_MID; do
        uv run python scripts/repro/prepare_scale_subset.py --n "$N" --seed "$S" >/dev/null

        det_dst="$SCALE/det_n${N}_s${S}"
        if [ -f "$det_dst/.done" ]; then echo "[skip] FT detector n$N s$S"; else
            banner "FT detector  n=$N s=$S (warm-start base_synth_s$S)"
            rm -rf "runs/labels_detect/scale_det_n${N}_s${S}"
            SEED="$S" RUN_NAME="scale_det_n${N}_s${S}" EPOCHS=25 \
                DATA_YAML="$PROJECT_ROOT/data/finetune/scale/yolo_n${N}_s${S}/data.yaml" \
                CHECKPOINT="$PROJECT_ROOT/$REPRO/detector/base_synth_s$S/best.pt" \
                bash scripts/finetune/yolo/train.sh 2>&1 | tee "$TLOG/det_n${N}_s${S}.log"
            mkdir -p "$det_dst"
            cp -f "runs/labels_detect/scale_det_n${N}_s${S}/weights/best.pt" "$det_dst/best.pt"
            cp -f "runs/labels_detect/scale_det_n${N}_s${S}/results.csv" "$det_dst/results.csv" 2>/dev/null || true
            touch "$det_dst/.done"
        fi

        lps_dst="$SCALE/lps_n${N}_s${S}"
        if [ -f "$lps_dst/.done" ]; then echo "[skip] FT lps n$N s$S"; else
            banner "FT LPS  n=$N s=$S (warm-start lps_synth_s$S)"
            uv run sf-train-lps --finetune "$REPRO/lps_synth_s$S/best.pt" \
                --data-dir "data/finetune/scale/lps_n${N}_s${S}" --seed "$S" \
                --output-dir "$lps_dst" 2>&1 | tee "$TLOG/lps_n${N}_s${S}.log"
            touch "$lps_dst/.done"
        fi

        eval_point "$N" "$S" "$det_dst/best.pt" "$lps_dst/best.pt"
    done

    # ---- endpoint N=830: full fine-tune (reuse main repro checkpoints) ----
    eval_point 830 "$S" "$REPRO/detector/finetuned_s$S/best.pt" "$REPRO/lps_ft_s$S/best.pt"
done

banner "SCALE SWEEP COMPLETE (seeds: $SEEDS; N: 0 $NS_MID 830)"
uv run python scripts/repro/aggregate_scale.py
