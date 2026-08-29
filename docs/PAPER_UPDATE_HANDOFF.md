# Handoff: update the paper for the 1.0 detector (text, references, multi-seed re-runs)

Written 2026-08-29 at the end of the migration session. Everything below is verified state, not
plan-speculation. Start by reading this file, `docs/LICENSE_MIGRATION.md` (numbers + method) and
`docs/publishing-weights.md` (release checklist). Memory notes for the assistant live in
`~/.claude/projects/-home-sidx-workspace-struct-labels/memory/`.

## 1. What changed vs. the paper's system (main @ v1.0.1)

| Paper (docs/paper_draft.md) | Now |
|---|---|
| YOLO11l via Ultralytics, ref [28]; 2048-px base + 1280-px fine-tune; ultralytics `.val()` metrics | **D-FINE-L** (`transformers.DFineForObjectDetection`, HGNet-V2, 31.1 M params, NMS-free, 300 queries), init `ustc-community/dfine-large-coco`, letterbox 1280; own COCO-style evaluator (`structflo.cser.inference.metrics`, pycocotools-identical) |
| conf 0.3 operating point; tiling discussed | **conf 0.5** (val flat 0.4–0.55; 0.5 best on 144-dpi input); full-image default, `--tile` opt-in |
| PyMuPDF rendering | pypdfium2 (`structflo.cser.pdf`), pixel sizes identical |
| 2-stage training (synthetic → real) | 4 stages: synthetic (10 ep) → real fine-tune (`data/finetune/plus`, 30 ep, es@26 best 16) → rasteriser/DPI robustness (`--downscale-aug 0.5`, 10 ep) → photometric/polarity robustness (`--photometric-aug 0.3 --photometric-mix v1 --val-variants invert`, 12 ep) |
| LPS features 12/13 = live detector conf | pinned to 1.0 at inference (`LearnedMatcher(use_detector_conf=False)`, +0.045 F1 on identical YOLO detections) |
| relmatch det-trained per detector seed | recalibration on D-FINE boxes does NOT beat published v0.2 (test 0.809/0.808 vs 0.814/0.825); v0.2 kept; margin 0.0 (shipped) ≈ 2.0 (paper) |
| denoising groups on (HF default) | **off** (`--num-denoising 0`): with DN on, eval-mode scores collapse (HF transformers 5.16 D-FINE; documented in trainer.py + LICENSE_MIGRATION.md) |

**Do not frame the change as licence-driven in the paper or README** (user instruction); describe it as an
architecture/quality change. `docs/LICENSE_MIGRATION.md` is the internal engineering record.

## 2. Current numbers (single seed 42; same frozen 100-page real_test, same evaluator for every row)

Sources: `runs/license_migration/eval/*.json`; regenerate tables with
`uv run python scripts/migration/summarize.py --tags yolo_v0.4 dfine_l_plus_ms dfine_l_plus_photo`.

| real_test | YOLO v0.4 @0.3 (re-scored) | YOLO 3 seeds re-scored | **D-FINE v1.0 @0.5** |
|---|---|---|---|
| mAP50 / mAP50-95 | 0.853 / 0.523 | 0.846±0.007 / 0.522±0.008 | 0.921 / 0.612 |
| label mAP50; label P/R | 0.748; 0.736/0.826 | — | 0.874; 0.805/0.850 |
| struct P/R | 0.880/0.995 | — | 0.927/0.979 |
| e2e Hungarian F1 | 0.816 | 0.807±0.007 | 0.821 |
| e2e Relational F1 (paper margin 2.0 / shipped 0.0) | 0.834 / 0.820 | 0.805±0.015 | 0.830 / ~0.825 |
| e2e LPS F1 (pinned) | 0.823 | 0.774±0.017 (unpinned) | 0.804 |
| 0.48× input (144 dpi) Hungarian | 0.823 | — | 0.820 |
| dark-page proxies (polarity / held-out) Hungarian | — | — | 0.81 / 0.80 (before stage 4: 0.54 / 0.58); control run w/o photometric 0.57 / 0.61 |
| synth_test mAP50 / mAP50-95 | 0.995 / 0.931 | 0.995 | 0.995 / 0.977 |
| real_val mAP50 / mAP50-95 | 0.783 / 0.473 | — | 0.827 / 0.535 |
| GPU latency | 12.5 ms/page | — | ~17 ms/page (bf16); CPU 0.9 s/page |

Paired bootstraps (`scripts/migration/paired_bootstrap.py`, 100 pages): every e2e difference vs. YOLO
or between D-FINE stages has a 95 % CI spanning zero (±0.02–0.03) — e2e is a statistical tie;
detection mAP gains are far outside seed noise (YOLO band ±0.007).

## 3. Multi-seed: what the paper requires and what exists

The paper reports mean ± s.d. over seeds 42/43/44 for every trained component on a FIXED split
(`data/finetune/real_split.json`, test 100 / val 75 / train 830 — verified restored and identical to the
YOLO dumps; **never** run `scripts/finetune/*/prepare_data.py` without `--yes`, it rebuilds the split).

Existing D-FINE runs are seed 42 only, and the shipped recipe fine-tunes on `data/finetune/plus`
(830+60 new real pages), whereas the paper protocol (§Real-data evaluation, Table 3) fine-tunes on
`data/finetune/yolo` (830 real). Decide explicitly which the paper reports; the ported drivers default to
the paper corpus. Suggested: paper tables = paper corpus, 3 seeds, full 4-stage recipe; the
"release/shipped" row (plus corpus, seed 42) noted separately.

Drivers (ported off ultralytics by agents, `--help`/syntax verified, **not yet exercised end-to-end**):
- `scripts/repro/run_train.sh` — per seed: synthetic base (`BASE_*` env, 10 ep) → real fine-tune
  (`FT_*`, default `data/finetune/yolo`, 30 ep) via `sf-train`; add the two robustness stages yourself
  (see the exact commands in `CLAUDE.md` § Detector fine-tune) or extend the script.
- `scripts/repro/run_eval.sh` + `scripts/repro/eval_detector.py` (uses `evaluate_detector_on_yaml`,
  JSON keys table1_synth / table3_real / table3_synth_regress) + `aggregate.py`.
- Per-detector e2e: `scripts/migration/dump_preds.py` → `eval_preds.py` (+`--cocoeval`) →
  `e2e_from_preds.py --split test --conf 0.5` (writes `per_page`) → `paired_bootstrap.py`.
- Matched-detector relmatch per seed (paper Table 5 protocol): `scripts/migration/recalibrate_relmatch.sh
  <det.safetensors> 0.5 <seed>` (prepare_det_data → sf-train-relmatch); ~1.5 h each. Expect ties with v0.2.
- LPS is GT-trained (detector-independent); the paper's 3 LPS seeds (`runs/repro/lps_ft_s4{2,3,4}`) can be
  re-used; re-report Part B with the pinned conf features (`e2e_from_preds.py` default).
- Dark-page proxy suite (new in the paper): `scripts/migration/proxy_dark_eval.py --weights <ckpt> --split
  real_test --conf 0.5 [--scale 0.48]` — report in-distribution groups and the held-out block separately;
  state that no real dark test set exists (1 dark page in the corpus).

GPU budget (RTX 6000 Ada, ~26–31 GB per run, so runs are sequential; docu-store containers hold ~5 GB):
stage 1 ≈ 1.6 h, stage 2 ≈ 2 h (paper corpus 3 660 imgs ≈ 1.9 h), stage 3 ≈ 0.75 h, stage 4 ≈ 0.9 h →
≈ 5.3 h per seed; seeds 43+44 ≈ 11 h; relmatch matched protocol +1.5 h per seed; evals ≈ 15 min per
checkpoint. Launch detached (`setsid nohup … < /dev/null &`), monitor logs; a queue script pattern is in
`runs/license_migration/logs/{stage2_chain.sh,photo_queue.sh}`. Training env: project `.venv` (no
ultralytics) or `runs/license_migration/venv-train`.

## 4. Paper edits to make (docs/paper_draft.md; mirror the short version docs/paper_dd_draft.md)

- Abstract/§Pipeline/§Detection: replace "fine-tuned YOLO11l [28] (Ultralytics)" with D-FINE-L; update
  Figure 1 placeholder text; §Training recipe (4 stages, AdamW 1e-4/1e-5 → 5e-5/5e-6 → 2e-5/2e-6, warmup+cosine,
  bf16, EMA 0.9999, fitness 0.1·mAP50+0.9·mAP50-95, no flips, scale/translate/brightness, downscale and
  photometric augmentation; denoising groups disabled — cite the observed train/eval score leak).
- §Evaluation: metrics now from the package's own COCO-style evaluator (101-pt AP, IoU .5:.95), verified
  against pycocotools; note that legacy ultralytics numbers are not comparable and were re-scored.
- Tables 1, 3, 3b, 5, 5b: replace with D-FINE mean ± s.d. (after the seed runs); add a row/table for the
  robustness results (144-dpi input, dark-page proxies, control ablation) and the operating-point sweep at
  0.5 (Table 5b analogue: conf 0.35–0.6).
- Keep Tables 2/4/6 (GT-box matching, OpenChemIE) — detector-independent — but re-check Table 5's LPS row
  with pinned conf and state the pinning.
- §Discussion: e2e is bounded by label detection *and* by GT incompleteness (many "FP" labels are
  unannotated real labels — `runs/repro/ocr_gate`, memory `miss-autopsy`); dark decks now handled
  (proxy-only evidence); LPS pinning; relational recalibration wash.
- Dependencies line (paper_draft.md:517): PyTorch, transformers, DECIMER, EasyOCR, RDKit, SciPy, OpenCV,
  pypdfium2, safetensors (drop Ultralytics, PyMuPDF).
- References: replace [28] Ultralytics YOLO with **D-FINE** — Peng Y, Li H, Wu P, Zhang Y, Sun X, Wu F.
  "D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement", arXiv:2410.13842
  (ICLR 2025). Add **RT-DETR** — Zhao Y et al. "DETRs Beat YOLOs on Real-time Object Detection", CVPR 2024
  (D-FINE's base design; HGNet-V2 backbone). Add **transformers** — Wolf T et al., EMNLP 2020 demo. Add
  pypdfium2 / PDFium (software citation, https://github.com/pypdfium2-team/pypdfium2). COCO (Lin et al.
  2014) for pretraining. Remove the PyMuPDF citation if present; keep DECIMER [13], ChEMBL [31], etc.
  Renumber consistently (refs are numeric, in order of first appearance).

## 5. Pitfalls learned this session (read before running anything)

- `pkill -f <pattern>` kills your own shell if the pattern appears in the command line; use `[.]` tricks.
- `prepare_data.py` scripts rebuild splits; they now require `--yes`. Verify the split before any eval:
  `uv run python scripts/migration/e2e_from_preds.py --preds runs/license_migration/preds/yolo_v0.4/real_test.json --split test`
  must print Hungarian 0.798/0.834/0.816.
- Patching files with exact-string replaces after `ruff format` silently misses; assert every replacement.
- Registry: every `LATEST` entry must accept the package version (test exists; publish script refuses).
- Build + smoke the Docker image (`--preload`) **before** pushing a release tag (1.0.0 shipped broken).
- HF D-FINE: keep `num_denoising=0`; checkpoints drop the DN embedding (`dfine.py`).
- Do not annotate/evaluate on portrait "paper" PDFs as the deployment proxy: docu-store input = landscape
  slide decks (the annotated corpus IS the deployment domain).
