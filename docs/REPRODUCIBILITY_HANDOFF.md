# Reproducibility Handoff — Retrain matchers (≥3 seeds) & re-verify all paper numbers

**Created:** 2026-06-01. **For:** a fresh session.
**Goal:** Remove all weight-provenance ambiguity and establish **reproducible, multi-seed
(≥3)** results for the structure–label matcher benchmark (and the detection/LPS numbers it
depends on), cleanly separated into the paper's two sections. Report **mean ± std** across
training seeds.

---

## 0. Why this handoff exists (the problem to fix)

During paper review we could **not determine from artifacts** whether the existing relational
checkpoints were trained on synthetic-only vs real+synthetic data:

- `runs/relmatch` — **GT-box-trained** variant (clean geometry). `sf-train-relmatch`'s
  **default `--data-dir` is `data/finetune/lps` (real+synthetic)** and its default output is
  `runs/relmatch`, so this checkpoint was *most likely trained on real+synthetic*, **not**
  synthetic-only. Unconfirmed.
- `runs/relmatch_det` — **detection-box-trained** variant. Trained on
  `data/finetune/relmatch_det` (1,242 real + 1,834 synthetic json). This is the **PUBLISHED**
  matcher (`cser-relmatcher` v0.1 = `runs/relmatch_det/best.pt`; the `ChemPipeline` default).
- Checkpoints store no `data_dir`; no run logs survive. Hence the ambiguity.

This matters because the paper is organized as:
- **§A "Performance on Synthetic Data"** → must use **synthetic-only** weights, evaluated on a
  held-out **synthetic test set @1280 px**.
- **§B "Generalization to Internal Documents"** → must use the **real-fine-tuned / published**
  weights (baseline = synthetic → fine-tuned improvement, on the real test set).

If `runs/relmatch` is actually real-trained, then §A's relational row currently uses the wrong
(real) weights, and there is **no synthetic-only relational matcher at all**. The retrain
resolves this by training every matcher **from scratch with an explicit `--data-dir` and an
explicit `--seed`**, logging everything.

---

## 1. The principle (don't deviate)

| Section | Weights to use | Eval data |
|---|---|---|
| §A synthetic | **synthetic-only** (`--data-dir data/generated`) | synthetic **test** set, @1280 px |
| §B internal | **real-fine-tuned / published** (mix `data/finetune/*`) | real **test** set (100 pages), @1280 px |

- All detection/e2e evaluation at **imgsz 1280** (the deployment resolution). Do **not** rely on
  `model.val()`'s default imgsz — it silently uses each checkpoint's *training* imgsz (base
  detector = 2048, fine-tuned = 1280), which previously confounded a baseline-vs-finetuned
  comparison. Always pass `imgsz=1280` explicitly. (`scripts/finetune/yolo/eval_compare.py`
  already pins it.)
- §A draws **no matcher conclusions** (matchers saturate in-distribution); all matcher
  conclusions are on real data in §B. (User decision.)

---

## 2. Current single-seed numbers (the baseline to reproduce)

These are what's in the paper draft now (`docs/paper_draft.md`, gitignored). The retrain should
reproduce them within seed variance; **mean ± std replaces the point values**.

**§A synthetic TEST set (1,000 pages, seed 1000000, WITH distractors), @1280:**
- *Table 1 detection (base `yolo11l_panels`):* structure 0.998/1.000/0.995/0.990 ·
  label 0.993/0.992/0.995/0.890 · all 0.995/0.996/0.995/0.940 (P/R/mAP50/mAP50-95)
- *Table 2 matching (synthetic-only weights — relational here used `runs/relmatch`, TO BE
  RE-ESTABLISHED as truly synthetic):* clean assign/reject/prec + e2e P/R/F1 —
  Distance 99.0/89.8/99.0 · 0.971/0.974/0.973 ;
  Relational 98.9/86.6/98.9 · 0.977/0.977/0.977 ;
  LPS 97.0/94.1/99.5 · 0.991/0.944/0.967

**§B real TEST set (100 pages; 247 labelled, 25 unlabelled), @1280:**
- *Table 3 detector:* base→FT mAP50 0.691→0.871, mAP50-95 0.404→0.552, P 0.668→0.850,
  R 0.755→0.897. Synthetic regression (synth test): mAP50 0.995→0.995, mAP50-95 0.940→0.953.
- *Table 3 LPS pair-classification:* real test 85.2%→90.7% ; synth test 96.7%→96.2%.
- *Table 4 clean matching:* Distance 99.6/96.0/99.6 ; LPS 91.1/96.0/98.3 ;
  Relational **with published `relmatch_det`** 98.0/96.0/99.6 **(current paper value)** ;
  Relational with GT-trained `runs/relmatch` 99.2/100.0/100.0 (alternative — unpublished variant).
- *Table 5 e2e:* Distance 0.815/0.818/0.816 ; LPS 0.842/0.713/0.772 ;
  Relational (`relmatch_det`) 0.838/0.773/0.804.

> Open decision the user was mid-answering: whether §B Table 4 reports the **published
> `relmatch_det`** (96% rejection, drops the "relational rejects best/100%" claim — already
> edited into the draft + abstract + conclusions) or also shows the GT-trained variant's
> 99.2/100/100 as an ablation. After the retrain, re-surface this with clean numbers.

---

## 3. Prerequisites / data inventory

```bash
uv sync --dev
```

- **Synthetic train/val:** `data/generated/{train,val}/{images,ground_truth,labels}` (10k/1k).
- **Synthetic TEST set:** `data/generated_test/val/...` (1,000 pages). If missing, regenerate
  EXACTLY (seed-disjoint from train/val, with distractors):
  ```bash
  uv run sf-generate --out data/generated_test --num-train 0 --num-val 1000 \
    --seed 1000000 --smiles data/smiles/chembl_smiles.csv --fonts-dir data/fonts \
    --distractors-dir data/distractors --workers 0
  ```
- **Real (CONFIDENTIAL):** mounted at `/net-fs-ins/shared-docker-vols/structflo-cser-annotate/data`
  (`ground_truth/`, `images/`); split manifest `data/finetune/real_split.json`
  (test=100, val=75, train=830, seed 42). Fine-tune corpora:
  `data/finetune/lps/{train,val,real_test}` (GT boxes, 1,660 real + 2,000 synth in train),
  `data/finetune/relmatch_det/{train,val}` (detection boxes), `data/finetune/yolo/` (+ `data_real_test.yaml`).
- **Detector checkpoints (reuse, do NOT retrain unless you accept the GPU cost — see §6):**
  base synthetic `runs/labels_detect/yolo11l_panels/weights/best.pt` (trained imgsz **2048**);
  fine-tuned `runs/labels_detect/finetune_3way/weights/best.pt` (imgsz 1280).

---

## 4. Retrain plan — matchers, ≥3 seeds, explicit provenance

Use **SEEDS = {42, 43, 44}**. Put everything under `runs/repro/`. Both trainers honor `--seed`.

```bash
for S in 42 43 44; do
  # ---- §A: SYNTHETIC-ONLY matchers (explicit --data-dir data/generated) ----
  uv run sf-train-lps        --data-dir data/generated --seed $S \
        --output-dir runs/repro/lps_synth_s$S
  uv run sf-train-relmatch   --data-dir data/generated --seed $S \
        --output-dir runs/repro/relmatch_synth_s$S          # GT-box, synthetic

  # ---- §B: REAL-FINE-TUNED matchers ----
  uv run sf-train-lps        --finetune runs/repro/lps_synth_s$S/best.pt \
        --data-dir data/finetune/lps --seed $S \
        --output-dir runs/repro/lps_ft_s$S
  uv run sf-train-relmatch   --data-dir data/finetune/lps --seed $S \
        --output-dir runs/repro/relmatch_gt_s$S             # GT-box, real+synth (clean variant)
  uv run sf-train-relmatch   --det-data-dir data/finetune/relmatch_det --seed $S \
        --output-dir runs/repro/relmatch_det_s$S            # detection-box, real+synth (PUBLISHED kind)
done
```

Notes:
- `sf-train-relmatch` has **no `--finetune`** (always trains from scratch); `--det-data-dir`
  switches it to detection-box training (overrides `--data-dir`).
- LPS fine-tune warm-starts from the **same seed's** synthetic base for clean seed isolation.
- Training is cheap: LPS ~minutes, relmatch ~minutes/seed. The whole matrix is well under an hour.

---

## 5. Eval plan — per seed, on the FIXED test sets

Create the synthetic-test manifest once:
```bash
uv run python - <<'PY'
import json, glob, os
stems = sorted(os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob('data/generated_test/val/ground_truth/*.json'))
json.dump({'test': stems, 'val': [], 'train': []}, open('runs/repro/synth_test_manifest.json','w'))
print('pages', len(stems))
PY
```

**(a) Synthetic matching — Table 2 (per seed):** synthetic-only weights on the synthetic test set.
```bash
for S in 42 43 44; do
  uv run python scripts/finetune/relmatch/eval_compare_all.py \
    --src data/generated_test/val --manifest runs/repro/synth_test_manifest.json \
    --detector runs/labels_detect/yolo11l_panels/weights/best.pt \
    --lps runs/repro/lps_synth_s$S/best.pt \
    --relmatch runs/repro/relmatch_synth_s$S/best.pt \
    --imgsz 1280 --conf 0.3 | tee runs/repro/eval_synth_s$S.txt
done
# Read the [TEST] block: PART A = clean (assign/reject/prec), PART B = e2e (P/R/F1).
```

**(b) Real matching — Tables 4 & 5 (per seed):** fine-tuned/published weights on the real test set.
```bash
for S in 42 43 44; do
  uv run python scripts/finetune/relmatch/eval_compare_all.py \
    --detector runs/labels_detect/finetune_3way/weights/best.pt \
    --lps runs/repro/lps_ft_s$S/best.pt \
    --relmatch runs/repro/relmatch_det_s$S/best.pt \
    --imgsz 1280 --conf 0.3 --margin 2.0 | tee runs/repro/eval_real_s$S.txt
done
# (default --src is the network mount; default --manifest is data/finetune/real_split.json)
# For the §B Table 4 "GT-trained clean variant" ablation, also run with
#   --relmatch runs/repro/relmatch_gt_s$S/best.pt   and record PART A only.
```

**(c) LPS pair-classification accuracy — Table 3 LPS row + synth regression (per seed):**
use the helper `scripts/repro/eval_lps_acc.py` (created in this handoff).
```bash
for S in 42 43 44; do
  uv run python scripts/repro/eval_lps_acc.py \
     --weights runs/repro/lps_synth_s$S/best.pt --data data/finetune/lps/real_test   # baseline, real
  uv run python scripts/repro/eval_lps_acc.py \
     --weights runs/repro/lps_ft_s$S/best.pt    --data data/finetune/lps/real_test   # fine-tuned, real
  uv run python scripts/repro/eval_lps_acc.py \
     --weights runs/repro/lps_synth_s$S/best.pt --data data/generated_test/val       # baseline, synth
  uv run python scripts/repro/eval_lps_acc.py \
     --weights runs/repro/lps_ft_s$S/best.pt    --data data/generated_test/val       # fine-tuned, synth
done
```

**(d) Detection — Tables 1 & 3 (detector is single-model; see §6):**
```bash
# Synthetic Table 1 (base @1280 on synth test):
uv run yolo detect val model=runs/labels_detect/yolo11l_panels/weights/best.pt \
  data=<synth_test.yaml> split=val imgsz=1280 verbose=True
# Real Table 3 detector (base vs FT @1280) + synth regression:
uv run python scripts/finetune/yolo/eval_compare.py
```
`<synth_test.yaml>`: a data.yaml with `path: .../data/generated_test`, `val: val/images`,
`nc: 2`, names {0: chemical_structure, 1: compound_label}.

---

## 6. Detector seeds — cost note / decision

The detector is standard YOLO11l; the paper's **contribution is the matcher**. A 3-seed detector
retrain is **expensive** (base ~hours/seed at imgsz 2048; fine-tune ~45 min/seed). Recommended:
- **Keep the existing detector single-seed** for Tables 1 & 3 (detection is not the novelty), and
  report matcher reproducibility via **Part A (clean), which is detector-independent**, plus e2e
  with the fixed detector.
- Only do multi-seed detector if the reviewer/user explicitly wants e2e variance to include
  detector seeds. If so, retrain base (`config/data.yaml`, imgsz 2048, 30 ep) and fine-tune
  (`scripts/finetune/yolo/train.sh`, imgsz 1280) per seed and re-run (a)/(b)/(d) per detector seed.

---

## 7. Aggregate & update the paper

1. Parse each `runs/repro/eval_*_s*.txt` [TEST] block → collect per-seed values.
2. Compute **mean ± std** across the 3 seeds for every cell of Tables 2, 4, 5 and the LPS rows.
3. Update `docs/paper_draft.md` Tables 1–5 to report mean ± std (keep §A "no conclusions";
   keep §B conclusions). Note in each caption: "mean ± std over 3 training seeds (42–44);
   synthetic-only weights for §A, fine-tuned for §B; eval @1280."
4. Re-confirm the §B Table 4 relational decision (published `relmatch_det` vs GT-trained
   ablation) with the user using the fresh multi-seed numbers.
5. Sanity-check: fresh numbers should land within ~±1% of §2. Large drift ⇒ investigate.

---

## 8. Publish (after numbers are locked)

Pick a canonical seed (or the median-seed checkpoint) for the released weights and bump versions:
```bash
python scripts/publish_weights.py --model cser-detector  --version vX.Y --weights-file <...>
python scripts/publish_weights.py --model cser-lps        --version vX.Y --weights-file runs/repro/lps_ft_s42/best.pt
python scripts/publish_weights.py --model cser-relmatcher --version vX.Y --weights-file runs/repro/relmatch_det_s42/best.pt
```
Document the seed + training `--data-dir` in the HF model card so provenance is never ambiguous again.

---

## 9. Gotchas / lessons (don't relearn these)

- **`model.val()` imgsz**: silently uses the checkpoint's training imgsz. Always pass `imgsz=1280`.
- **Two relational variants**: GT-box (`relmatch`) vs detection-box (`relmatch_det`). Published =
  `relmatch_det`. Keep them straight; record `--data-dir`/`--det-data-dir` per run.
- **Synthetic test set** must be **seed-disjoint** from train/val (train/val used seeds 7–11006;
  test uses 1000000) and generated **with distractors** to match the realistic distribution.
- **Real data is confidential** — never copy it into git, notebooks, or the released package.
- **Paper draft is gitignored** (`.gitignore:48` `docs/*draft*.md`); edits live on disk only.
- The synthetic LPS-classification eval over 1,000 pages is slow (visual-crop extraction) — run
  in the background; real_test (100 pages) is fast.
- `runs/`, `data/` are gitignored; `runs/repro/` will not be committed (that's fine — logs + the
  aggregated table are the durable artifacts; copy the aggregated table into the paper).

---

## 10. Definition of done

- [ ] `runs/repro/` holds all seeded checkpoints + `eval_*_s*.txt` logs (3 seeds).
- [ ] Aggregated mean ± std table (e.g. `runs/repro/SUMMARY.md`) for Tables 2, 4, 5 + LPS rows.
- [ ] `runs/relmatch` provenance resolved: a **confirmed synthetic-only** relational matcher
      exists (`runs/repro/relmatch_synth_s*`) and is used in §A; published `relmatch_det`-kind in §B.
- [ ] Paper Tables 1–5 updated to mean ± std with provenance noted in captions.
- [ ] §B Table 4 relational decision re-confirmed with the user on fresh numbers.
- [ ] (Optional) Re-published weights with documented seed + data provenance.
