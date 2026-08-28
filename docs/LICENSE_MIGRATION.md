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

_(filled in from `runs/license_migration/eval/dfine_l_*.json` when the training chain finishes)_

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
