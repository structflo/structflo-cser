#!/usr/bin/env bash
# Reproducibility EVAL driver — runs every paper eval per seed, all @ imgsz 1280.
#
# Pairs each seed's matchers with the SAME-seed detector (clean seed isolation):
#   §A synthetic : base_synth_sS detector + lps_synth_sS + relmatch_synth_sS, on synth TEST
#   §B real      : finetuned_sS detector + lps_ft_sS    + relmatch_det_sS,    on real TEST
#                  (+ a GT-trained relmatch ablation for the §B Table-4 decision)
#   LPS pair-acc : lps_synth/lps_ft on {real_test, synth test}  (Table 3 rows)
#   detector     : base vs FT, per-class, on real + synth TEST  (Tables 1 & 3)
#
# Idempotent: each eval is skipped if its output already carries a completion marker.
# A seed is skipped (with a message) until its required checkpoints exist, so this is
# safe to run repeatedly while training is still in flight.
set -uo pipefail   # NOT -e: one missing-seed skip must not abort the whole sweep

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

REPRO="runs/repro"
DET="$REPRO/detector"
ELOG="$REPRO/logs/eval"
mkdir -p "$ELOG"
SEEDS="${SEEDS:-42 43 44}"

SYNTH_SRC="data/generated_test/val"
SYNTH_MANIFEST="$REPRO/synth_test_manifest.json"
SYNTH_YAML="$REPRO/synth_test.yaml"
REAL_TEST_YAML="data/finetune/yolo/data_real_test.yaml"
EC_ALL="scripts/finetune/relmatch/eval_compare_all.py"
LPS_ACC="scripts/repro/eval_lps_acc.py"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
banner() { echo; echo "[$(ts)] === $* ==="; }
# done_marker <file> <grep-pattern>  -> 0 if file exists AND contains pattern (i.e. completed)
done_marker() { [ -f "$1" ] && grep -q "$2" "$1" 2>/dev/null; }
have() { [ -f "$1" ]; }   # checkpoint present?

[ -f "$SYNTH_MANIFEST" ] || uv run python scripts/repro/make_synth_manifest.py

for S in $SEEDS; do
    banner "SEED $S"
    BASE="$DET/base_synth_s$S/best.safetensors"
    FT="$DET/finetuned_s$S/best.safetensors"

    # ---------- §A synthetic matching (Table 2) ----------
    out="$ELOG/eval_synth_s$S.txt"
    if done_marker "$out" "PART B"; then echo "[skip] §A synth matching s$S"
    elif have "$BASE" && have "$REPRO/lps_synth_s$S/best.pt" && have "$REPRO/relmatch_synth_s$S/best.pt"; then
        banner "§A synthetic matching s$S"
        uv run python "$EC_ALL" --src "$SYNTH_SRC" --manifest "$SYNTH_MANIFEST" \
            --detector "$BASE" --lps "$REPRO/lps_synth_s$S/best.pt" \
            --relmatch "$REPRO/relmatch_synth_s$S/best.pt" \
            --imgsz 1280 --conf 0.3 2>&1 | tee "$out"
    else echo "[wait] §A synth s$S — missing base/lps_synth/relmatch_synth checkpoint"; fi

    # ---------- §B real matching (Tables 4 & 5; published-kind relmatch_det) ----------
    out="$ELOG/eval_real_s$S.txt"
    if done_marker "$out" "PART B"; then echo "[skip] §B real matching s$S"
    elif have "$FT" && have "$REPRO/lps_ft_s$S/best.pt" && have "$REPRO/relmatch_det_s$S/best.pt"; then
        banner "§B real matching s$S (relmatch_det, margin 2.0)"
        uv run python "$EC_ALL" --detector "$FT" --lps "$REPRO/lps_ft_s$S/best.pt" \
            --relmatch "$REPRO/relmatch_det_s$S/best.pt" \
            --imgsz 1280 --conf 0.3 --margin 2.0 2>&1 | tee "$out"
    else echo "[wait] §B real s$S — missing ft/lps_ft/relmatch_det checkpoint"; fi

    # ---------- §B real matching, GT-trained relmatch ablation (Table 4 decision) ----------
    out="$ELOG/eval_real_gtabl_s$S.txt"
    if done_marker "$out" "PART B"; then echo "[skip] §B GT-ablation s$S"
    elif have "$FT" && have "$REPRO/lps_ft_s$S/best.pt" && have "$REPRO/relmatch_gt_s$S/best.pt"; then
        banner "§B real matching s$S — GT-trained relmatch ablation"
        uv run python "$EC_ALL" --detector "$FT" --lps "$REPRO/lps_ft_s$S/best.pt" \
            --relmatch "$REPRO/relmatch_gt_s$S/best.pt" \
            --imgsz 1280 --conf 0.3 --margin 2.0 2>&1 | tee "$out"
    else echo "[wait] §B GT-ablation s$S — missing relmatch_gt checkpoint"; fi

    # ---------- LPS pair-classification accuracy (Table 3 rows) ----------
    # variant=baseline(lps_synth) / finetuned(lps_ft)  x  dataset=real_test / synth test
    declare -A LPSW=( [baseline]="$REPRO/lps_synth_s$S/best.pt" [finetuned]="$REPRO/lps_ft_s$S/best.pt" )
    declare -A LPSD=( [real]="data/finetune/lps/real_test" [synth]="$SYNTH_SRC" )
    for v in baseline finetuned; do
        for d in real synth; do
            out="$ELOG/lps_acc_${v}_${d}_s$S.log"
            if done_marker "$out" "RESULT"; then echo "[skip] lps_acc $v/$d s$S"
            elif have "${LPSW[$v]}"; then
                banner "LPS acc $v on $d s$S"
                uv run python "$LPS_ACC" --weights "${LPSW[$v]}" --data "${LPSD[$d]}" 2>&1 | tee "$out"
            else echo "[wait] lps_acc $v/$d s$S — missing ${LPSW[$v]}"; fi
        done
    done

    # ---------- Detector Tables 1 & 3 (per-class, base vs FT) ----------
    out="$ELOG/detector_s$S.json"
    if [ -f "$out" ]; then echo "[skip] detector eval s$S"
    elif have "$BASE" && have "$FT"; then
        banner "detector eval s$S (Tables 1 & 3)"
        uv run python scripts/repro/eval_detector.py --seed "$S" \
            --base "$BASE" --ft "$FT" \
            --synth-test-yaml "$SYNTH_YAML" --real-test-yaml "$REAL_TEST_YAML" \
            --imgsz 1280 --out "$out" 2>&1 | tee "$ELOG/detector_s$S.log"
    else echo "[wait] detector s$S — missing base/ft checkpoint"; fi
done

banner "EVAL SWEEP COMPLETE (seeds: $SEEDS)"
