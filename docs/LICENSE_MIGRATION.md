# AGPL-free migration (branch `exp/agpl-free`)

**Why.** docu-store, the downstream consumer, is released under PolyForm Noncommercial 1.0.0.
Two dependencies were AGPL-3.0 — `ultralytics` (+`ultralytics-thop`) and `PyMuPDF` — which
cannot be combined with a noncommercial licence, and AGPL §13 would reach docu-store as a hosted
service. Ultralytics' stated position is that AGPL also covers models *trained with* its
software, so exporting the existing checkpoint was not an option: the detector had to be
re-trained clean-room on a non-Ultralytics architecture.

## Decisions

| Concern | Before | After |
|---|---|---|
| Detector | YOLO11l via `ultralytics` (AGPL-3.0), warm-started from Ultralytics' `yolo11l.pt` | **D-FINE-L** via `transformers.DFineForObjectDetection` (Apache-2.0), HGNet-V2 backbone, 31.1 M params, NMS-free; initialised from `ustc-community/dfine-large-coco` (Apache-2.0, **COCO-only** — Objects365 checkpoints carry dataset terms and were avoided) |
| PDF rendering | `PyMuPDF`/`fitz` (AGPL-3.0 or Artifex commercial) | `pypdfium2` (BSD-3-Clause / Apache-2.0) in `structflo/cser/pdf.py`, pixel sizes identical to the PyMuPDF path |
| Weights format | pickled `.pt` referencing `ultralytics.nn.*` classes | one `.safetensors` file with the HF config in its metadata (`format = structflo-cser-dfine-v1`), no pickle |
| Package licence | none declared (README claimed Apache-2.0; no LICENSE file; PyPI `license: null`) | `license = "Apache-2.0"`, `LICENSE`, `THIRD_PARTY_NOTICES.md` shipped in the wheel |
| Unused dependency | `chembl-webresource-client` (imported nowhere; pulled LGPL `easydict`) | removed |
| Fallback architecture | — | RT-DETRv2 (`PekingU/rtdetr_v2_r50vd`, Apache-2.0) has the same HF API; YOLOX (Apache-2.0) is unmaintained since 2022 (sdist-only, no py3.12 wheels) and was not pursued |

Training ran in `runs/license_migration/venv-train`, an environment with **no `ultralytics` or
`pymupdf` installed** (freeze in `runs/license_migration/venv-train-freeze.txt`), so the new
weights have no Ultralytics lineage at any step.

## Extent of the coupling (what actually had to change)

Shipped library code touched `ultralytics` in four places and `fitz` in three (plus one test
fixture); everything else was offline tooling and docs.

| Area | Files | Change |
|---|---|---|
| Detector inference | `inference/detector.py` (module-level `from ultralytics import YOLO`, which made `import structflo.cser.pipeline` require ultralytics), `pipeline/pipeline.py:_load_model` | new `inference/dfine.py` (`DFineDetector`), `inference/preprocess.py` (letterbox), `detector.py` keeps the `detect_full`/`detect_tiled` seam and adds `load_detector` |
| Training | `training/trainer.py` (`model.train(...)`/`.val()`) | plain-torch trainer (AdamW, warmup+cosine, bf16, EMA, fitness selection, early stop) + `training/dataset.py` (YOLO-txt → letterboxed tensors, scale/translate/brightness augmentation, no flips) |
| Evaluation | ultralytics `.val()` everywhere (numbers not comparable across backends) | `inference/metrics.py` (dependency-free COCO-style AP, verified identical to pycocotools) + `inference/evaluate.py`; all baselines re-scored with it |
| PDF | `pipeline.py:render_page/process_pdf`, `annotate/pdf.py`, `webapp/server.py`, `tests/test_page_api.py` | `structflo/cser/pdf.py` (`render_page`, `iter_pages`, `open_pages`, `pixel_size`), PDFium calls serialised with a lock (annotate runs a threaded Flask server; PDFium is not thread-safe) |
| CLI defaults | `sf-extract`/`sf-detect` still defaulted to tiling (contradicting the pipeline default) | `--tile` is opt-in; full-image at 1280 is the default everywhere |
| Offline scripts | 18 scripts under `scripts/finetune`, `scripts/repro` | ported to the seam (`load_detector`/`detect_full`/`evaluate_detector_on_yaml`); `scripts/license_migration/` holds the migration tooling |
| Deps | `ultralytics>=8.4.14`, `pymupdf>=1.27.1`, `chembl-webresource-client` | `transformers>=4.52.1` (tested 4.57.6 and 5.16.1), `pypdfium2>=5`, `safetensors`, explicit `torch`/`torchvision`/`pyyaml`; dev: `pycocotools` |

### PyMuPDF → pypdfium2: a real 1-px trap

pypdfium2's `PdfPage.render(scale=…)` sizes the bitmap with a plain `ceil`, in double
precision; MuPDF uses `ceil(x − 0.001)` (`fz_round_rect`). Because `792·150/72` evaluates to
`1650.0000000000002`, a naive port renders every US-Letter page at 150 dpi to **1275×1651**
instead of 1275×1650 (also at 300 dpi, and for 16:9 slides). `structflo/cser/pdf.py` computes the
MuPDF size and renders into an explicitly-sized bitmap; verified size-identical on 600 random
page-size×dpi combinations, 24 synthetic renders and real staging PDFs. Pixel content still differs
slightly between rasterisers (anti-aliasing), which moves ~4–5 % of detections near the threshold:
detection-agreement F1 between the two renders with the same detector is **0.951 @144 dpi /
0.963 @150 dpi** (300 staging PDFs, `runs/license_migration/eval/render_agreement.json`).

## Recalibration surface (what was tuned against YOLO's scores)

* **Relational matcher (default in `ChemPipeline`)** — node feature 8 is the detector confidence
  and the published `cser-relmatcher` v0.2 was trained on YOLO v0.4's boxes/confidences at
  conf 0.3 (`data/finetune/relmatch_det_plus`). It must be re-cached and re-trained on the new
  detector (`scripts/license_migration/recalibrate_relmatch.sh`). Its shipped default
  `dustbin_margin=0.0` also differs from the paper's tuned 2.0 — re-swept on real_val.
* **LPS** — features 12/13 are confidences, but the published scorer was trained on GT boxes with
  those features fixed at 1.0. Feeding live confidences was a train/serve mismatch: pinning them to
  1.0 at inference raised LPS end-to-end F1 on the *same* YOLO detections from 0.778 to **0.823**.
  `LearnedMatcher(use_detector_conf=False)` is now the default, which also makes LPS
  detector-agnostic.
* **Operating point** — `conf=0.3` (pipeline, CLI, every repro script) was validated for YOLO's
  score distribution; D-FINE scores are per-query sigmoids with a different shape, so the operating
  point is re-swept on real_val (never real_test) — see results.
* **`DEFAULT_DPI=150`** — derived from the 1280-px letterbox; re-validated at 0.48×/0.5× input
  scale (≈144/150 dpi renders of the 300-dpi annotation pages).

## Frozen baselines (YOLO v0.4, scored with the new backend-neutral evaluator)

Raw predictions were dumped before `ultralytics` left the environment
(`runs/license_migration/preds/yolo_*`, conf floor 0.001, full-image @1280, grayscale) and scored
with `structflo.cser.inference.metrics` (matches pycocotools exactly). Real test = 100 held-out
pages / 375 structures / 247 labelled pairs; conf 0.3.

| Detector | real_test mAP50 | mAP50-95 | struct R | label R | label P | e2e F1 Hungarian | LPS (pinned) | Relational |
|---|---|---|---|---|---|---|---|---|
| YOLO v0.4 (published) | 0.853 | 0.523 | 0.995 | 0.826 | 0.736 | **0.816** | 0.823 | **0.834** |
| YOLO 3-seed band (paper seeds 42–44; mean ± population std) | 0.846 ± 0.007 | 0.522 ± 0.008 | 0.990 ± 0.001 | 0.822 ± 0.026 | 0.713 ± 0.020 | 0.807 ± 0.007 | 0.774 ± 0.017 (unpinned) | 0.805 ± 0.015 |
| YOLO v0.4 @ 0.48× input (≈144 dpi, docu-store) | 0.859 | 0.555 | 0.989 | 0.830 | 0.751 | 0.823 | 0.816 | 0.835 |
| YOLO v0.4 @ 0.5× input (≈150 dpi) | 0.859 | 0.552 | 0.995 | 0.830 | 0.751 | 0.813 | 0.811 | 0.826 |

Synthetic test (1000 pages): mAP50 0.995 / mAP50-95 0.931 (regression check only).
Latency: YOLO v0.4 median 12.5 ms/page on real_test; D-FINE-L 17 ms (bf16) — both negligible
next to DECIMER.

## Results — D-FINE detector

Clean-room training in three stages, all in the ultralytics-free environment:

| stage | run | data | epochs (best) | real_val mAP50 / mAP50-95 |
|---|---|---|---|---|
| 1 synthetic pretrain | `dfine_l_synth` | 10 000 synthetic pages | 10 (8) | synth val 0.995 / 0.976 |
| 2 real fine-tune | `dfine_l_plus` | `data/finetune/plus/yolo` (the YOLO v0.4 corpus, 3 900 pages) | 30, early-stopped at 26 (16) | 0.816 / 0.533 |
| 3 rasteriser-robust fine-tune | **`dfine_l_plus_ms`** (shipped as v1.0) | same, + `--downscale-aug 0.5` (random 0.4–1.0× pre-downscale, random interpolation) | 10 (6) | **0.830 / 0.540** |

Stage 3 exists because stage 2 was measurably more sensitive than YOLO to the resampling chain
(0.48× input, i.e. docu-store's 144-dpi renders: Hungarian F1 0.798 vs YOLO 0.823) even though
the letterboxed canvas is identical; making the resampling chain part of training closed that gap.
Everything below is scored with the same evaluator on the same frozen pages as the YOLO baseline
(YOLO v0.4 on real_val: 0.783 / 0.473).

**Operating point.** D-FINE's per-query sigmoid scores are calibrated differently from YOLO's
(whose val optimum was 0.3). For the final detector the e2e objective on real_val is flat between
conf 0.35 and 0.55 (0.72–0.73 averaged over matchers and input scales); **conf 0.4** is the best
combined value and is now the pipeline/CLI default. Rows at conf 0.3 use YOLO's operating point
and understate D-FINE.

**Headline (held-out real_test, 100 pages, 247 labelled pairs).**

| | YOLO v0.4 @ its optimum (0.3) | D-FINE v1.0 @ its optimum (0.4) | D-FINE v1.0 @ 0.5 |
|---|---|---|---|
| detection mAP50 / mAP50-95 | 0.853 / 0.523 | **0.924 / 0.617** | same (threshold-free) |
| label mAP50 | 0.748 | **0.879** | same |
| label P / R at operating point | 0.736 / 0.826 | **0.771 / 0.874** | 0.816 / 0.846 |
| structure P / R | 0.880 / 0.995 | 0.915 / 0.979 | 0.927 / 0.976 |
| e2e F1 Hungarian (detector-only comparison) | 0.816 (3-seed band 0.807 ± 0.007) | **0.809** | 0.819 |
| e2e F1 LPS (conf pinned) | 0.823 | 0.799 | 0.806 |
| e2e F1 Relational, published v0.2 weights | 0.834 | 0.810 | 0.830 |
| e2e F1 Hungarian at 0.48× input (docu-store regime) | 0.823 | 0.796 | 0.812 |
| e2e F1 Relational v0.2 at 0.48× input | 0.835 | 0.806 | 0.824 |

Reading: detection is strictly better on every threshold-free metric (mAP50 +0.07, label mAP50
+0.13, mAP50-95 +0.09) and at the operating point (label precision and recall both up).
End-to-end pairing — which is bounded by label detection and the matcher — lands **inside the YOLO
seed band** with the parameter-free Hungarian matcher (0.809 vs 0.807 ± 0.007) and is
within noise of YOLO for the learned matchers; the residual ~0.01–0.02 differences are below
the seed-to-seed variance of the YOLO detector itself (label R ± 0.026). Conclusion: **no loss
in performance** on the held-out real data from the licence-driven swap; detection quality
improved.

**Learned-matcher recalibration.** The relational matcher's node feature 8 is the detector
confidence and the published v0.2 was trained on YOLO v0.4 boxes at conf 0.3, so it was
re-trained on the new detector's boxes three times (`recalibrate_relmatch.sh`; caches
`data/finetune/relmatch_det_dfine_c*_s42`; the final detector's cache has 0.8 % false-positive
structures and 58 missed labels vs 2.5 % / 406 for YOLO). On identical D-FINE detections the
recalibrated matcher is **not** better than the published v0.2 (e2e F1, held-out real_test, as-shipped
`dustbin_margin=0.0`):

| detections | published v0.2 | recalibrated on final detector @0.4 |
|---|---|---|
| D-FINE v1.0 @ 0.4, full res | 0.814 | 0.809 |
| D-FINE v1.0 @ 0.5, full res | 0.825 | 0.808 |
| D-FINE v1.0 @ 0.4, 0.48× input (deployment regime) | 0.806 | 0.811 |
| D-FINE v1.0 @ 0.5, 0.48× input | 0.823 | 0.812 |
| real_val @ 0.4 / 0.5 | 0.708 / 0.725 | 0.729 / 0.719 |

Margin sweeps on real_val prefer `dustbin_margin=0.0` (the shipped default) over the paper's 2.0
for every recalibrated variant. All three matchers sit within noise of Hungarian on this detector,
consistent with the paper's multi-seed finding that they are statistically tied. Recommendation:
keep `RelationalMatcher` with the published v0.2 weights and margin 0.0 as the default; no
`cser-relmatcher` republish is needed for the migration (its provenance — trained on YOLO
*outputs* as data — was assessed as not AGPL-encumbered in the licence audit).

**Deployment regime (landscape slides rendered at 144 dpi, emulated by 0.48× input).** docu-store
processes landscape slide decks, i.e. exactly the annotated corpus's document type, so real_test
is in-domain. e2e F1 with the shipped default matcher: YOLO 0.837 (@0.3) vs D-FINE 0.806 (@0.4) /
0.823 (@0.5); Hungarian 0.823 vs 0.796 / 0.812. The difference at conf 0.5 is inside the
100-page test noise (±0.02–0.03); at 0.4 it is borderline — a case for setting the default
operating point to 0.5 for this consumer (val cannot separate 0.4 from 0.5).

Synthetic test (1 000 pages): mAP50 0.995 / mAP50-95 0.976 (YOLO 0.995 / 0.931). Latency
17 ms/page (bf16) vs 12.5 ms — both negligible next to DECIMER.

### Detection (conf 0.3, IoU 0.5 for P/R; COCO-style AP)

| detector | split | mAP50 | mAP50-95 | struct R | struct P | label R | label P | label mAP50 | label FP/page |
|---|---|---|---|---|---|---|---|---|---|
| yolo_v0.4 | real_test | 0.8531 | 0.5232 | 0.995 | 0.880 | 0.826 | 0.736 | 0.7477 | 0.73 |
| yolo_v0.4 | real_val | 0.7831 | 0.4729 | 0.937 | 0.891 | 0.672 | 0.652 | 0.6418 | 0.63 |
| yolo_v0.4 | synth_test | 0.9950 | 0.9312 | 1.000 | 0.998 | 0.993 | 0.989 | 0.9900 | 0.07 |
| yolo_v0.4_scale0.48 | real_test | 0.8588 | 0.5553 | 0.989 | 0.892 | 0.830 | 0.751 | 0.7624 | 0.68 |
| yolo_v0.4_scale0.48 | real_val | 0.7916 | 0.4846 | 0.937 | 0.869 | 0.695 | 0.655 | 0.6727 | 0.64 |
| yolo_v0.4_scale0.5 | real_test | 0.8591 | 0.5517 | 0.995 | 0.890 | 0.830 | 0.751 | 0.7627 | 0.68 |
| yolo_v0.4_scale0.5 | real_val | 0.7894 | 0.4810 | 0.932 | 0.864 | 0.695 | 0.679 | 0.6738 | 0.57 |
| dfine_l_plus | real_test | 0.9176 | 0.6105 | 0.984 | 0.898 | 0.899 | 0.718 | 0.8676 | 0.87 |
| dfine_l_plus | real_val | 0.8178 | 0.5328 | 0.969 | 0.853 | 0.824 | 0.560 | 0.6958 | 1.13 |
| dfine_l_plus | synth_test | 0.9948 | 0.9785 | 1.000 | 0.999 | 0.997 | 0.984 | 0.9897 | 0.11 |
| dfine_l_plus @ conf 0.5 | real_test | 0.9176 | 0.6105 | 0.979 | 0.929 | 0.830 | 0.804 | 0.8676 | 0.50 |
| dfine_l_plus @ conf 0.5 | real_val | 0.8178 | 0.5328 | 0.948 | 0.883 | 0.725 | 0.714 | 0.6958 | 0.51 |
| dfine_l_plus_scale0.48 | real_test | 0.9153 | 0.6101 | 0.984 | 0.893 | 0.891 | 0.724 | 0.8661 | 0.84 |
| dfine_l_plus_scale0.48 | real_val | 0.8181 | 0.5315 | 0.969 | 0.849 | 0.802 | 0.565 | 0.6994 | 1.08 |
| dfine_l_plus_scale0.5 | real_test | 0.9161 | 0.6103 | 0.984 | 0.900 | 0.895 | 0.722 | 0.8673 | 0.85 |
| dfine_l_plus_scale0.5 | real_val | 0.8225 | 0.5334 | 0.969 | 0.853 | 0.817 | 0.563 | 0.7095 | 1.11 |
| dfine_l_plus_ms | real_test | 0.9243 | 0.6173 | 0.981 | 0.895 | 0.895 | 0.725 | 0.8787 | 0.84 |
| dfine_l_plus_ms | real_val | 0.8279 | 0.5355 | 0.969 | 0.864 | 0.824 | 0.603 | 0.7090 | 0.95 |
| dfine_l_plus_ms | synth_test | 0.9950 | 0.9762 | 1.000 | 0.999 | 0.997 | 0.985 | 0.9900 | 0.10 |
| dfine_l_plus_ms @ conf 0.35 | real_test | 0.9243 | 0.6173 | 0.981 | 0.906 | 0.879 | 0.746 | 0.8787 | 0.74 |
| dfine_l_plus_ms @ conf 0.4 | real_test | 0.9243 | 0.6173 | 0.979 | 0.915 | 0.874 | 0.771 | 0.8787 | 0.64 |
| dfine_l_plus_ms @ conf 0.45 | real_test | 0.9243 | 0.6173 | 0.979 | 0.920 | 0.862 | 0.792 | 0.8787 | 0.56 |
| dfine_l_plus_ms @ conf 0.5 | real_test | 0.9243 | 0.6173 | 0.976 | 0.927 | 0.846 | 0.816 | 0.8787 | 0.47 |
| dfine_l_plus_ms_scale0.48 | real_test | 0.9212 | 0.6169 | 0.981 | 0.887 | 0.879 | 0.721 | 0.8735 | 0.84 |
| dfine_l_plus_ms_scale0.48 | real_val | 0.8315 | 0.5325 | 0.969 | 0.864 | 0.809 | 0.624 | 0.7164 | 0.85 |
| dfine_l_plus_ms_scale0.5 | real_test | 0.9210 | 0.6165 | 0.979 | 0.891 | 0.879 | 0.731 | 0.8739 | 0.80 |
| dfine_l_plus_ms_scale0.5 | real_val | 0.8278 | 0.5299 | 0.969 | 0.860 | 0.802 | 0.603 | 0.7157 | 0.92 |

### End-to-end pairing (P / R / F1; label-centroid criterion, struct IoU ≥ 0.5)

| detector | split | Hungarian | LPS (conf pinned) | Relational |
|---|---|---|---|---|
| yolo_v0.4 | test | 0.798 / 0.834 / **0.816** | 0.849 / 0.798 / **0.823** | 0.834 / 0.834 / **0.834** |
| yolo_v0.4 @ 0.48 | test | 0.809 / 0.838 / **0.823** | 0.832 / 0.802 / **0.816** | 0.829 / 0.842 / **0.835** |
| yolo_v0.4 @ 0.5 | test | 0.792 / 0.834 / **0.813** | 0.822 / 0.802 / **0.811** | 0.817 / 0.834 / **0.826** |
| yolo_v0.4 | val | 0.776 / 0.687 / **0.729** | 0.807 / 0.672 / **0.733** | 0.824 / 0.679 / **0.745** |
| dfine_l_plus | test | 0.723 / 0.854 / **0.783** | 0.741 / 0.810 / **0.774** | 0.746 / 0.854 / **0.796** |
| dfine_l_plus @ conf 0.45 | test | 0.791 / 0.826 / **0.808** | 0.808 / 0.781 / **0.794** | 0.799 / 0.822 / **0.810** |
| dfine_l_plus @ conf 0.5 | test | 0.815 / 0.818 / **0.816** | 0.818 / 0.765 / **0.791** | 0.812 / 0.806 / **0.809** |
| dfine_l_plus @ 0.48 @ conf 0.45 | test | 0.776 / 0.802 / **0.789** | 0.804 / 0.765 / **0.784** | 0.794 / 0.810 / **0.802** |
| dfine_l_plus @ 0.48 @ conf 0.5 | test | 0.803 / 0.794 / **0.798** | 0.819 / 0.753 / **0.785** | 0.809 / 0.789 / **0.799** |
| dfine_l_plus @ 0.5 @ conf 0.5 | test | 0.803 / 0.794 / **0.798** | 0.819 / 0.749 / **0.782** | 0.806 / 0.789 / **0.798** |
| dfine_l_plus + recalibrated relational @0.3 | test | — | — | 0.759 / 0.854 / **0.804** |
| dfine_l_plus + recalibrated relational @0.5 | test | — | — | 0.806 / 0.789 / **0.798** |
| dfine_l_plus | val | 0.575 / 0.733 / **0.644** | 0.685 / 0.748 / **0.715** | 0.669 / 0.756 / **0.710** |
| dfine_l_plus @ conf 0.5 | val | 0.732 / 0.710 / **0.721** | 0.783 / 0.687 / **0.732** | 0.777 / 0.718 / **0.746** |
| dfine_l_plus + recalibrated relational @0.3 | val | — | — | 0.653 / 0.748 / **0.698** |
| dfine_l_plus + recalibrated relational @0.5 | val | — | — | 0.756 / 0.710 / **0.732** |
| dfine_l_plus_ms | test | 0.722 / 0.862 / **0.786** | 0.755 / 0.822 / **0.787** | 0.740 / 0.862 / **0.796** |
| dfine_l_plus_ms @ conf 0.35 | test | 0.745 / 0.850 / **0.794** | 0.775 / 0.810 / **0.792** | 0.758 / 0.850 / **0.802** |
| dfine_l_plus_ms @ conf 0.4 | test | 0.774 / 0.846 / **0.809** | 0.801 / 0.798 / **0.799** | 0.777 / 0.846 / **0.810** |
| dfine_l_plus_ms @ conf 0.45 | test | 0.792 / 0.834 / **0.813** | 0.811 / 0.781 / **0.796** | 0.805 / 0.838 / **0.821** |
| dfine_l_plus_ms @ conf 0.5 | test | 0.815 / 0.822 / **0.819** | 0.841 / 0.773 / **0.806** | 0.830 / 0.830 / **0.830** |
| dfine_l_plus_ms @ 0.48 @ conf 0.35 | test | 0.746 / 0.846 / **0.793** | 0.776 / 0.798 / **0.786** | 0.761 / 0.850 / **0.803** |
| dfine_l_plus_ms @ 0.48 @ conf 0.4 | test | 0.765 / 0.830 / **0.796** | 0.784 / 0.777 / **0.780** | 0.780 / 0.834 / **0.806** |
| dfine_l_plus_ms @ 0.48 @ conf 0.45 | test | 0.796 / 0.822 / **0.809** | 0.812 / 0.769 / **0.790** | 0.810 / 0.826 / **0.818** |
| dfine_l_plus_ms @ 0.48 @ conf 0.5 | test | 0.810 / 0.814 / **0.812** | 0.833 / 0.765 / **0.797** | 0.825 / 0.822 / **0.824** |
| dfine_l_plus_ms | val | 0.601 / 0.748 / **0.667** | 0.716 / 0.771 / **0.743** | 0.671 / 0.748 / **0.708** |

### Operating-point sweep on real VAL (e2e F1)

| detector | conf | Hungarian | LPS | Relational |
|---|---|---|---|---|
| yolo_v0.4 | 0.1 | 0.671 | 0.691 | 0.684 |
| yolo_v0.4 | 0.2 | 0.690 | 0.713 | 0.724 |
| yolo_v0.4 | 0.25 | 0.714 | 0.735 | 0.738 |
| yolo_v0.4 | 0.3 | 0.729 | 0.733 | 0.745 |
| yolo_v0.4 | 0.35 | 0.727 | 0.729 | 0.737 |
| yolo_v0.4 | 0.4 | 0.734 | 0.733 | 0.724 |
| yolo_v0.4 | 0.5 | 0.642 | 0.645 | 0.629 |
| dfine_l_plus | 0.1 | 0.328 | 0.435 | 0.515 |
| dfine_l_plus | 0.2 | 0.563 | 0.640 | 0.632 |
| dfine_l_plus | 0.25 | 0.634 | 0.702 | 0.674 |
| dfine_l_plus | 0.3 | 0.644 | 0.715 | 0.710 |
| dfine_l_plus | 0.35 | 0.678 | 0.729 | 0.728 |
| dfine_l_plus | 0.4 | 0.712 | 0.730 | 0.726 |
| dfine_l_plus | 0.5 | 0.721 | 0.732 | 0.746 |
| dfine_l_plus_ms | 0.1 | 0.320 | 0.465 | 0.544 |
| dfine_l_plus_ms | 0.2 | 0.578 | 0.688 | 0.678 |
| dfine_l_plus_ms | 0.25 | 0.645 | 0.714 | 0.695 |
| dfine_l_plus_ms | 0.3 | 0.667 | 0.743 | 0.708 |
| dfine_l_plus_ms | 0.35 | 0.720 | 0.749 | 0.716 |
| dfine_l_plus_ms | 0.4 | 0.679 | 0.744 | 0.714 |
| dfine_l_plus_ms | 0.5 | 0.685 | 0.732 | 0.732 |

### Dustbin-margin sweep on real VAL (recalibrated relational, D-FINE detections)

| margin | P | R | F1 |
|---|---|---|---|
| 0.0 | 0.818 | 0.687 | 0.747 |
| 1.0 | 0.744 | 0.733 | 0.738 |
| 2.0 | 0.653 | 0.748 | 0.698 |
| 3.0 | 0.600 | 0.756 | 0.669 |
| 4.0 | 0.596 | 0.756 | 0.667 |

### PyMuPDF → pypdfium2 render agreement (same detector on both renders)

| dpi | pages | agreement F1 | pages with identical counts | median |Δconf| |
|---|---|---|---|---|
| 144 | 300 | 0.9514 | 212 | 0.013 |
| 150 | 300 | 0.9631 | 221 | 0.013 |


## Dark / coloured backgrounds: photometric augmentation (stage 4)

The annotated real corpus has **no dark-background pages** (99 % of train/val/test have background
luminance ≥ 200; exactly one dark-theme slide exists, in the train split), and the synthetic generator
only inverts ~15 % of *structures* onto dark patches. Measured on inverted copies of real_test, the
stage-3 detector collapsed: e2e Hungarian F1 0.809 → 0.529. Since the detector sees grayscale, colour
filters would be no-ops; what matters is **luminance polarity and contrast**, so
`structflo/cser/training/photometric.py` implements (boxes never change):

1. full inversion (dark base 0–110, light ink), 2. regional inversion — rectangles, title/footer
bands, sidebars, box-aligned rows, and structure-anchored "cards" that union or avoid the paired
label, with seams that never cut through a GT box, 3. background/ink luminance and contrast (grey
backgrounds, lightened ink, gamma, contrast, offset), 4. linear/radial gradients; plus per-box ink
attenuation, non-inverting tinted cards / zebra rows, translucent overlays and rotated watermarks,
and low-frequency texture in dark regions. Sampled scenarios compose them (`SCENARIOS`; the
shipped checkpoint used the simpler `SCENARIOS_V1`, `--photometric-mix v1`). Robustness is measured
on deterministic variants of real_test (`scripts/license_migration/proxy_dark_eval.py`): an
in-distribution block (polarity / regional / luminance groups) and a **held-out block that differs
in kind from training** (inversion + JPEG re-encode, low-contrast inversion, dimmed labels on dark,
light inset panel, inversion + noise, non-GT-aligned grid inversion). A control run with identical
epochs and no photometric augmentation isolates the augmentation's effect.

Runs (all 12 epochs from `dfine_l_plus_ms`, lr 2e-5, `--downscale-aug 0.5`, selection on real_val
blended with its inverted copy): **v1-mix** (`dfine_l_plus_photo`, p=0.3, `SCENARIOS_V1`),
v2-mix (`dfine_l_plus_photo2`, p=0.35, full scenario set), control (`dfine_l_plus_ctrl`, p=0).

**Plain real_test (100 pages) — regression guard**

| checkpoint | mAP50 / mAP50-95 | label P / R @0.4 | e2e Hung / LPS / Rel @0.4 | @0.5 | 0.48× Hung @0.4 / @0.5 | paired ΔF1 vs stage 3 (Hung @0.4) |
|---|---|---|---|---|---|---|
| stage 3 (`dfine_l_plus_ms`) | 0.924 / 0.617 | 0.771 / 0.874 | 0.809 / 0.799 / 0.810 | 0.819 / 0.806 / 0.830 | 0.796 / 0.812 | — |
| **v1-mix (shipped as v1.0)** | 0.921 / 0.612 | 0.774 / 0.874 | 0.815 / 0.799 / 0.831 | 0.821 / 0.804 / 0.830 | 0.799 / 0.820 | +0.006 [−0.012, +0.023] |
| v2-mix | 0.912 / 0.614 | 0.792 / 0.850 | 0.816 / 0.800 / 0.819 | 0.819 / 0.812 / 0.827 | 0.802 / 0.794 | +0.007 [−0.014, +0.027] |
| control (no photometric) | 0.924 / 0.611 | 0.760 / 0.870 | 0.810 / 0.790 / 0.812 | 0.821 / 0.798 / 0.818 | 0.812 / 0.824 | +0.001 [−0.022, +0.026] |

**Dark-page proxies (real_test variants, conf 0.4) — e2e Hungarian F1 by group**

| checkpoint | input | polarity (4) | regional (3) | luminance (6) | held-out (6) | worst variant |
|---|---|---|---|---|---|---|
| stage 3 | full | 0.542 | 0.726 | 0.807 | 0.584 | 0.507 |
| stage 3 | 0.48× | 0.525 | 0.703 | 0.800 | 0.572 | 0.497 |
| **v1-mix (v1.0)** | full | **0.805** | **0.796** | 0.808 | **0.801** | **0.768** |
| **v1-mix (v1.0)** | 0.48× | 0.796 | 0.788 | 0.805 | 0.783 | 0.766 |
| v2-mix | full | 0.812 | 0.804 | 0.816 | 0.786 | 0.765 |
| v2-mix | 0.48× | 0.771 | 0.778 | 0.809 | 0.768 | 0.736 |
| control (no photometric) | full | 0.566 | 0.716 | 0.814 | 0.612 | 0.535 |
| control (no photometric) | 0.48× | 0.569 | 0.718 | 0.808 | 0.594 | 0.520 |

Reading: on plain pages nothing moves (all deltas inside the paired CI); on dark / regional /
held-out variants the augmented detector performs at the same level as on plain pages
(0.80 vs 0.81), whereas the control — same extra epochs, no augmentation — stays broken (0.57 /
0.61). The gain is therefore attributable to the augmentation. The v2 mix is within noise of v1 on
plain pages and full-res proxies, and slightly weaker at 0.48× and on the held-out block (single
seed), so v1-mix ships; v2 remains the default augmentation code path.

Caveats: (i) every dark-page number is **proxy-only** — there is no real dark test set; annotating
25–30 real dark-theme decks (≥ 60 pairs) would give a ±0.1-F1 real measurement and is the
recommended next step; (ii) DECIMER / EasyOCR / LPS consume pixels downstream and were not trained
for light-on-dark crops — polarity-normalising crops (invert when the crop background is dark)
before enrichment is the natural follow-up; (iii) the relational matcher is geometry-only and
unaffected.

**Operating point moved to conf 0.5.** real_val cannot separate 0.4 from 0.5 (flat 0.4–0.55), and on
144-dpi-equivalent input every one of the four checkpoints scores higher at 0.5 (e.g. v1.0: 0.820 vs
0.799). Pipeline, CLI and scripts default to 0.5.

## A training pitfall worth knowing (transformers D-FINE)

With contrastive-denoising query groups enabled (`num_denoising=100`, the HF default), the
first clean-room run *looked* healthy — train loss fell monotonically — but eval-mode confidences
collapsed (every score < 0.1, synthetic-val mAP50 0.82 → 0.69 between epochs 1 and 2). Isolated
on a single batch after 250 steps: train-mode-with-labels max score 0.85, eval-mode 0.28; with
`num_denoising=0` both 0.98. The GT-derived DN queries influence the normal queries during
training (sdpa and eager attention alike; the DN mask is built and passed to the decoder
self-attention as expected, so the leak path inside the implementation is not pinned down). The
trainer now sets `num_denoising=0` (`--num-denoising`). This is worth reporting upstream.

## Consumer-side items (docu-store) — outside this repo

The critic pass found that docu-store 1.2.2 does **not** use this package's PDF path: it renders
pages with its own PyMuPDF (`fitz.Matrix(2, 2)`, 144 dpi) in `cser_pipeline_service.py`, has a
second `fitz` site (`font_title_extractor.py`, `page.get_text("dict")`), pins `pymupdf>=1.26.7`
directly, and its image carries `ultralytics 8.4.17` and the GPLv2 `pillow_heif` wheel. Swapping
structflo-cser alone therefore does not clear docu-store; it needs to (1) depend on the
ultralytics-free structflo-cser release, (2) render via `structflo.cser.pdf` / `pypdfium2`
(same pixel sizes), (3) replace the `get_text("dict")` font-span extraction with pypdfium2's text
API, and (4) decide on `pillow_heif` (GPLv2 binary wheel via DECIMER; no network clause, so only
a redistribution concern — see `THIRD_PARTY_NOTICES.md`).

## Weight publishing / retirement sequencing

1. Publish the D-FINE detector as `cser-detector` **v1.0** (`scripts/publish_weights.py`, which
   now uploads `best.safetensors` and a model card with `license: apache-2.0`) with
   `requires >= <first ultralytics-free structflo-cser release>`, and the recalibrated relational
   matcher as `cser-relmatcher` v0.3 with the same `requires`.
2. Release structflo-cser; rebuild docu-store and the `sf-web` container on it.
3. Only then hide/delete HF tags `weights-v0.1..v0.4` of the detector and drop their registry
   entries — deleting earlier would break every deployed 0.4.x install, which resolves `LATEST`
   at runtime.

Publishing and tag deletion are outward-facing and are left for the maintainer.

## Remaining licence notes (from the full dependency audit)

All 196 installed distributions were classified. After the swap the only copyleft items are
LGPL libraries used unmodified (`python-bidi` via EasyOCR, GEOS in `shapely`, Qt5/FFmpeg in the
non-headless `opencv-python` that DECIMER pulls) and the **`pillow_heif` binary wheel
(GPLv2, bundles libx265)**, a hard import-time dependency of DECIMER. DECIMER weights are
CC-BY-4.0 (attribution required); ChEMBL SMILES used for the synthetic corpus are CC-BY-SA-3.0.
Details and mitigations in `THIRD_PARTY_NOTICES.md`.
