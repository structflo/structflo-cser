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

Two-stage clean-room training: `dfine_l_synth` = 10 epochs on the 10 000 synthetic pages
(best epoch 8; synthetic val mAP50 0.995 / mAP50-95 0.976), then `dfine_l_plus` = fine-tune on the
same corpus as YOLO v0.4 (`data/finetune/plus/yolo`, 3 900 pages; selection on the frozen 75-page
real_val; early-stopped at epoch 26, best epoch 16). Everything below is scored with the same
evaluator on the same frozen pages as the YOLO baseline.

**Headline (held-out real_test, 100 pages).** At its own val-tuned operating point (conf 0.5 for
both classes — D-FINE's sigmoid scores are calibrated differently from YOLO's, whose val optimum is
0.3), the D-FINE detector gives **Hungarian end-to-end F1 0.816, identical to YOLO v0.4 (0.816)
and inside the 3-seed band (0.807 ± 0.007)**, with better label precision at equal recall
(0.804 vs 0.736). Detection quality is strictly better: mAP50 **0.918 vs 0.853**, mAP50-95
**0.611 vs 0.523**, label mAP50 0.868 vs 0.748, label recall 0.899 vs 0.826 at conf 0.3.
The deployment-regime renders (0.48×/0.5× input ≈ 144/150 dpi) give the same numbers
(mAP50 0.915–0.916), so `DEFAULT_DPI = 150` stands. Synthetic test: mAP50 0.995 / mAP50-95 0.979
(YOLO 0.995 / 0.931). Latency 17 ms/page (bf16) vs 12.5 ms.

**Learned matchers need the recalibration that was predicted.** With the *old* YOLO-calibrated
weights, LPS (0.791) and the relational matcher (0.809 at conf 0.5) sit below their YOLO numbers
(0.823 / 0.834); the relational matcher is being re-trained on the new detector's boxes
(`recalibrate_relmatch.sh`, conf 0.3 and 0.5 variants) — see the "recalibrated relational" rows.

Operating point: e2e F1 on real_val for the fine-tuned D-FINE peaks at struct conf 0.5 / label
conf 0.5 (Hungarian 0.721, LPS 0.732, Relational 0.746 — a tie with YOLO's val optimum
0.729 / 0.733 / 0.745). Rows tagged "@ conf 0.3" use YOLO's operating point and understate D-FINE.

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
| dfine_l_synth | real_test | 0.6734 | 0.3837 | 0.963 | 0.732 | 0.692 | 0.325 | 0.4409 | 3.55 |
| dfine_l_synth | real_val | 0.6132 | 0.3547 | 0.942 | 0.583 | 0.550 | 0.224 | 0.3802 | 3.32 |
| dfine_l_synth | synth_test | 0.9950 | 0.9777 | 1.000 | 0.999 | 0.998 | 0.976 | 0.9900 | 0.16 |
| dfine_l_synth_scale0.48 | real_test | 0.6802 | 0.3810 | 0.968 | 0.733 | 0.721 | 0.328 | 0.4474 | 3.64 |
| dfine_l_synth_scale0.48 | real_val | 0.6045 | 0.3509 | 0.937 | 0.576 | 0.542 | 0.213 | 0.3627 | 3.51 |
| dfine_l_synth_scale0.5 | real_test | 0.6723 | 0.3793 | 0.965 | 0.734 | 0.700 | 0.321 | 0.4360 | 3.66 |
| dfine_l_synth_scale0.5 | real_val | 0.6019 | 0.3491 | 0.942 | 0.581 | 0.550 | 0.212 | 0.3598 | 3.57 |
| dfine_l_plus | real_test | 0.9176 | 0.6105 | 0.984 | 0.898 | 0.899 | 0.718 | 0.8676 | 0.87 |
| dfine_l_plus | real_val | 0.8178 | 0.5328 | 0.969 | 0.853 | 0.824 | 0.560 | 0.6958 | 1.13 |
| dfine_l_plus | synth_test | 0.9948 | 0.9785 | 1.000 | 0.999 | 0.997 | 0.984 | 0.9897 | 0.11 |
| dfine_l_plus @ conf 0.5 (val-tuned) | real_test | 0.9176 | 0.6105 | 0.979 | 0.929 | 0.830 | 0.804 | 0.8676 | 0.50 |
| dfine_l_plus @ conf 0.5 (val-tuned) | real_val | 0.8178 | 0.5328 | 0.948 | 0.883 | 0.725 | 0.714 | 0.6958 | 0.51 |
| dfine_l_plus_scale0.48 | real_test | 0.9153 | 0.6101 | 0.984 | 0.893 | 0.891 | 0.724 | 0.8661 | 0.84 |
| dfine_l_plus_scale0.48 | real_val | 0.8181 | 0.5315 | 0.969 | 0.849 | 0.802 | 0.565 | 0.6994 | 1.08 |
| dfine_l_plus_scale0.5 | real_test | 0.9161 | 0.6103 | 0.984 | 0.900 | 0.895 | 0.722 | 0.8673 | 0.85 |
| dfine_l_plus_scale0.5 | real_val | 0.8225 | 0.5334 | 0.969 | 0.853 | 0.817 | 0.563 | 0.7095 | 1.11 |

### End-to-end pairing (P / R / F1; label-centroid criterion, struct IoU ≥ 0.5)

| detector | split | Hungarian | LPS (conf pinned) | Relational |
|---|---|---|---|---|
| yolo_v0.4 | test | 0.798 / 0.834 / **0.816** | 0.849 / 0.798 / **0.823** | 0.834 / 0.834 / **0.834** |
| yolo_v0.4_scale0.48 | test | 0.809 / 0.838 / **0.823** | 0.832 / 0.802 / **0.816** | 0.829 / 0.842 / **0.835** |
| yolo_v0.4_scale0.5 | test | 0.792 / 0.834 / **0.813** | 0.822 / 0.802 / **0.811** | 0.817 / 0.834 / **0.826** |
| yolo_v0.4 | val | 0.776 / 0.687 / **0.729** | 0.807 / 0.672 / **0.733** | 0.824 / 0.679 / **0.745** |
| dfine_l_synth | test | 0.340 / 0.587 / **0.430** | 0.432 / 0.603 / **0.503** | 0.360 / 0.567 / **0.440** |
| dfine_l_synth | val | 0.285 / 0.534 / **0.371** | 0.376 / 0.565 / **0.451** | 0.352 / 0.588 / **0.440** |
| dfine_l_plus | test | 0.723 / 0.854 / **0.783** | 0.741 / 0.810 / **0.774** | 0.746 / 0.854 / **0.796** |
| dfine_l_plus @ conf 0.5 (val-tuned) | test | 0.815 / 0.818 / **0.816** | 0.818 / 0.765 / **0.791** | 0.812 / 0.806 / **0.809** |
| dfine_l_plus | val | 0.575 / 0.733 / **0.644** | 0.685 / 0.748 / **0.715** | 0.669 / 0.756 / **0.710** |
| dfine_l_plus @ conf 0.5 (val-tuned) | val | 0.732 / 0.710 / **0.721** | 0.783 / 0.687 / **0.732** | 0.777 / 0.718 / **0.746** |

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
| dfine_l_synth | 0.1 | 0.326 | 0.376 | 0.398 |
| dfine_l_synth | 0.2 | 0.378 | 0.451 | 0.444 |
| dfine_l_synth | 0.25 | 0.382 | 0.447 | 0.442 |
| dfine_l_synth | 0.3 | 0.371 | 0.451 | 0.440 |
| dfine_l_synth | 0.35 | 0.404 | 0.455 | 0.428 |
| dfine_l_synth | 0.4 | 0.429 | 0.476 | 0.444 |
| dfine_l_synth | 0.5 | 0.445 | 0.495 | 0.446 |
| dfine_l_plus | 0.1 | 0.328 | 0.435 | 0.515 |
| dfine_l_plus | 0.2 | 0.563 | 0.640 | 0.632 |
| dfine_l_plus | 0.25 | 0.634 | 0.702 | 0.674 |
| dfine_l_plus | 0.3 | 0.644 | 0.715 | 0.710 |
| dfine_l_plus | 0.35 | 0.678 | 0.729 | 0.728 |
| dfine_l_plus | 0.4 | 0.712 | 0.730 | 0.726 |
| dfine_l_plus | 0.5 | 0.721 | 0.732 | 0.746 |

### PyMuPDF → pypdfium2 render agreement (same detector on both renders)

| dpi | pages | agreement F1 | pages with identical counts | median |Δconf| |
|---|---|---|---|---|
| 144 | 300 | 0.9514 | 212 | 0.013 |
| 150 | 300 | 0.9631 | 221 | 0.013 |


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
