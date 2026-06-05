#!/usr/bin/env bash
# Train the relational at LOW conf (label 0.1 = max false-positive exposure), then sweep its
# inference conf on val. Tests "train-on-negatives -> better low-conf robustness" vs the
# existing 0.3-trained relational sweep. Detectors/LPS unchanged.
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$PROJECT_ROOT"
REPRO=runs/repro; SLOG=$REPRO/logs/c01sweep; mkdir -p "$SLOG"
SEEDS="${SEEDS:-42 43 44}"; CONFS="${CONFS:-0.05 0.10 0.20 0.30 0.40 0.50 0.70}"
MAN=$REPRO/real_val_manifest.json
EC=scripts/finetune/relmatch/eval_compare_all.py
PREP=scripts/finetune/relmatch/prepare_det_data.py
ts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }; banner(){ echo; echo "[$(ts)] === $* ==="; }
is_done(){ [ -f "$1" ] && grep -q "PART B" "$1" 2>/dev/null; }
for S in $SEEDS; do
  DET=$REPRO/detector/finetuned_s$S/best.pt
  CACHE=data/finetune/relmatch_det_c01_s$S
  OUT=$REPRO/relmatch_det_c01_s$S
  if [ ! -f "$CACHE/.done" ]; then
    banner "recache c01 s$S (struct 0.3 / label 0.1)"
    uv run python $PREP --src data/finetune/lps --detector "$DET" --out "$CACHE" \
      --conf 0.3 --label-conf 0.1 2>&1 | tee "$SLOG/recache_s$S.log"; touch "$CACHE/.done"
  fi
  if [ ! -f "$OUT/.done" ]; then
    banner "train relmatch_det_c01 s$S"
    uv run sf-train-relmatch --det-data-dir "$CACHE" --seed "$S" --output-dir "$OUT" \
      2>&1 | tee "$SLOG/train_s$S.log"; touch "$OUT/.done"
  fi
  for C in $CONFS; do
    out=$SLOG/eval_val_s${S}_c${C}.txt
    is_done "$out" && { echo "[skip] eval s$S c$C"; continue; }
    banner "c01 sweep-eval s$S label-conf $C"
    uv run python $EC --manifest "$MAN" --detector "$DET" --lps $REPRO/lps_ft_s$S/best.pt \
      --relmatch "$OUT/best.pt" --imgsz 1280 --conf 0.3 --label-conf "$C" --margin 2.0 \
      2>&1 | tee "$out"
  done
done
banner "C01-SWEEP COMPLETE"
