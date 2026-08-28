#!/usr/bin/env bash
# e2e F1 vs LABEL-conf sweep on the real VAL set (eval-only, present weights). Picks the
# operating point and yields the matcher-robustness curve. Structures fixed at 0.30.
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$PROJECT_ROOT"
REPRO=runs/repro; SLOG=$REPRO/logs/confsweep; mkdir -p "$SLOG"
SEEDS="${SEEDS:-42 43 44}"; CONFS="${CONFS:-0.05 0.10 0.20 0.30 0.40 0.50 0.70}"
MAN=$REPRO/real_val_manifest.json
EC=scripts/finetune/relmatch/eval_compare_all.py
relmatch_for(){ [ "$1" = 42 ] && echo "$REPRO/relmatch_det_s42/best.pt" || echo "$REPRO/relmatch_det_matched_s$1/best.pt"; }
is_done(){ [ -f "$1" ] && grep -q "PART B" "$1" 2>/dev/null; }
for S in $SEEDS; do
  for C in $CONFS; do
    out=$SLOG/eval_val_s${S}_c${C}.txt
    is_done "$out" && { echo "[skip] s$S c$C"; continue; }
    echo "=== val sweep s$S label-conf $C ==="
    uv run python $EC --manifest "$MAN" \
      --detector $REPRO/detector/finetuned_s$S/best.safetensors --lps $REPRO/lps_ft_s$S/best.pt \
      --relmatch "$(relmatch_for $S)" --imgsz 1280 --conf 0.3 --label-conf "$C" --margin 2.0 \
      2>&1 | tee "$out"
  done
done
echo "CONFSWEEP-VAL COMPLETE"
