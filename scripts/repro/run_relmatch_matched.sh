#!/usr/bin/env bash
# FAIR relational benchmark: fix the train/eval detector MISMATCH in the paper's relmatch_det.
# The paper trained all 3 seeds on ONE shared det-box cache (matched only to seed 42's detector),
# so seeds 43/44 were tested on a different detector's box distribution than they trained on.
# Here we recache EACH seed's OWN detector boxes, retrain relmatch_det, and re-eval e2e — all on
# the 830-corpus (paper-consistent, no confidential docs). Seed 42 is already matched
# (relmatch_det_s42, e2e 0.804) and is reused at aggregation.
#
#   setsid nohup bash scripts/repro/run_relmatch_matched.sh >> runs/repro/logs/matched_driver.log 2>&1 </dev/null &
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

REPRO="runs/repro"
MLOG="$REPRO/logs/matched"
mkdir -p "$MLOG"
SEEDS="${SEEDS:-43 44}"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
banner() { echo; echo "[$(ts)] === $* ==="; }
done_marker() { [ -f "$1" ] && grep -q "$2" "$1" 2>/dev/null; }

for S in $SEEDS; do
    banner "MATCHED SEED $S"
    DET="$REPRO/detector/finetuned_s$S/best.safetensors"
    CACHE="data/finetune/relmatch_det_matched_s$S"
    OUT="$REPRO/relmatch_det_matched_s$S"

    # 1. recache detection boxes with THIS seed's own detector (the fix)
    if [ -f "$CACHE/.done" ]; then echo "[skip] recache s$S"; else
        banner "recache det boxes — finetuned_s$S over data/finetune/lps"
        uv run python scripts/finetune/relmatch/prepare_det_data.py \
            --src data/finetune/lps --detector "$DET" --out "$CACHE" \
            --conf 0.3 --imgsz 1280 2>&1 | tee "$MLOG/recache_s$S.log"
        touch "$CACHE/.done"
    fi

    # 2. retrain the relational matcher on the matched cache
    if [ -f "$OUT/.done" ]; then echo "[skip] train s$S"; else
        banner "train relmatch_det_matched_s$S"
        uv run sf-train-relmatch --det-data-dir "$CACHE" --seed "$S" \
            --output-dir "$OUT" 2>&1 | tee "$MLOG/train_s$S.log"
        touch "$OUT/.done"
    fi

    # 3. re-eval e2e (matched detector + matched relational; Hungarian is the parameter-free ref)
    out="$MLOG/eval_s$S.txt"
    if done_marker "$out" "PART B"; then echo "[skip] eval s$S"; else
        banner "e2e eval s$S (matched detector + matched relational)"
        uv run python scripts/finetune/relmatch/eval_compare_all.py \
            --detector "$DET" --lps "$REPRO/lps_ft_s$S/best.pt" --relmatch "$OUT/best.pt" \
            --imgsz 1280 --conf 0.3 --margin 2.0 2>&1 | tee "$out"
    fi
done

banner "MATCHED-RELMATCH COMPLETE (seeds: $SEEDS)"
