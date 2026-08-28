#!/usr/bin/env bash
# STAGE-1 conf sweep (eval-only, present weights, NO training): re-run end-to-end at
# label-conf 0.10 (structures stay 0.30) to test how matchers handle more label candidates.
# Part A (clean, GT boxes) is conf-independent — only Part B (e2e) changes.
# Relational uses the present matched weights (conf-0.3-trained -> conf-mismatched at 0.10; flagged).
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$PROJECT_ROOT"
REPRO=runs/repro; CLOG=$REPRO/logs/conf010; mkdir -p "$CLOG"
SEEDS="${SEEDS:-42 43 44}"
EC=scripts/finetune/relmatch/eval_compare_all.py
ts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }; banner(){ echo; echo "[$(ts)] === $* ==="; }
is_done(){ [ -f "$1" ] && grep -q "PART B" "$1" 2>/dev/null; }
relmatch_for(){ [ "$1" = 42 ] && echo "$REPRO/relmatch_det_s42/best.pt" || echo "$REPRO/relmatch_det_matched_s$1/best.pt"; }

# Phase 1: §B real (fast, the decision signal)
for S in $SEEDS; do
  out=$CLOG/eval_real_s$S.txt
  if is_done "$out"; then echo "[skip] real s$S"; continue; fi
  banner "real e2e @ label-conf 0.10 -- s$S"
  uv run python $EC --detector $REPRO/detector/finetuned_s$S/best.safetensors \
    --lps $REPRO/lps_ft_s$S/best.pt --relmatch "$(relmatch_for $S)" \
    --imgsz 1280 --conf 0.3 --label-conf 0.10 --margin 2.0 2>&1 | tee "$out"
done
banner "CONF010 REAL PHASE COMPLETE"

# Phase 2: §A synthetic (slower, low expected signal)
for S in $SEEDS; do
  out=$CLOG/eval_synth_s$S.txt
  if is_done "$out"; then echo "[skip] synth s$S"; continue; fi
  banner "synth e2e @ label-conf 0.10 -- s$S"
  uv run python $EC --src data/generated_test/val --manifest $REPRO/synth_test_manifest.json \
    --detector $REPRO/detector/base_synth_s$S/best.safetensors \
    --lps $REPRO/lps_synth_s$S/best.pt --relmatch $REPRO/relmatch_synth_s$S/best.pt \
    --imgsz 1280 --conf 0.3 --label-conf 0.10 2>&1 | tee "$out"
done
banner "CONF010 COMPLETE"
