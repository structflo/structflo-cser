# Third-party notices

structflo-cser is licensed under the Apache License 2.0 (see `LICENSE`). It depends on and
downloads at runtime the following third-party software and model weights. This file lists the
components whose licences carry obligations beyond attribution, or which are worth knowing about
when redistributing an environment or container image built from this package.

## Components replaced in 1.0

| Component | Licence | Replaced by |
|---|---|---|
| `ultralytics` (YOLO11), `ultralytics-thop` | AGPL-3.0 | D-FINE via `transformers` (Apache-2.0); detector re-trained clean-room from `ustc-community/dfine-large-coco` (Apache-2.0, COCO-only pretraining) |
| `PyMuPDF` (`fitz`) | AGPL-3.0 / Artifex commercial | `pypdfium2` (BSD-3-Clause / Apache-2.0; bundles PDFium, BSD-3-Clause) |
| `chembl-webresource-client` (unused import; pulled LGPL `easydict`) | Apache-2.0 (+ LGPL-3.0 dep) | removed |

The detector weights `cser-detector` v0.1–v0.4 were produced with the previous (Ultralytics) stack and
are not loadable by this version; `cser-detector` v1.0 was trained from scratch with the new stack.

## Runtime Python dependencies with notable terms

| Component | Licence | Notes |
|---|---|---|
| `transformers`, `huggingface_hub`, `safetensors`, `tokenizers` | Apache-2.0 | detector backend + weight download |
| `pypdfium2` / PDFium | BSD-3-Clause, Apache-2.0 (PDFium third-party notices in the wheel) | ship the wheel's `LICENSES` notices with any binary redistribution |
| `torch` | BSD-3-Clause (bundles NVIDIA CUDA runtime wheels: NVIDIA proprietary EULA; libgomp: GPL with GCC Runtime Library Exception) | not a copyleft issue; CUDA EULA applies to redistribution of the wheels |
| `easyocr` (+ CRAFT) | Apache-2.0 (CRAFT code MIT) | pulls `python-bidi` (LGPL) and `shapely` (BSD; bundles GEOS, LGPL-2.1) — LGPL libraries used unmodified via dynamic import/linking |
| `decimer` | MIT | pulls `pillow-heif` and non-headless `opencv-python` (see below) |
| `pillow-heif` (via `decimer`) | BSD-3-Clause source; **binary wheel GPLv2** because it bundles `libx265` (+ libheif/libde265, LGPLv3) | GPLv2 has no network clause; obligations arise only when redistributing an environment/container containing the wheel. Only needed by DECIMER for `.heic` input. Mitigation options: install `pillow-heif` from source against a libheif built without x265, or ask upstream to make it optional. |
| `opencv-python` (non-headless, via `decimer`) | Apache-2.0 (bundles Qt5 LGPL-3.0 and FFmpeg LGPL-2.1+) | coexists with `opencv-python-headless`; LGPL notices must accompany image distribution |
| `tqdm`, `certifi` | MPL-2.0 (+MIT) | file-level weak copyleft, no obligations for use |
| `rdkit` | BSD-3-Clause | |

## Model weights downloaded at runtime

| Weights | Licence | Attribution |
|---|---|---|
| `ustc-community/dfine-large-coco` (D-FINE pretraining, used as the initialisation of `cser-detector` ≥ v1.0) | Apache-2.0 | Peng et al., *D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement*, 2024 |
| DECIMER Image Transformer weights (Zenodo 10.5281/zenodo.8300489) | CC-BY-4.0 | Rajan, Zielesny, Steinbeck — attribution required |
| EasyOCR detection/recognition weights | no separate statement (code Apache-2.0 / CRAFT MIT) | |
| `cser-detector`, `cser-lps`, `cser-relmatcher` (this project, HF Hub `sidxz/*`) | Apache-2.0 (to be declared in the model cards) | |

## Training data

| Data | Licence | Notes |
|---|---|---|
| ChEMBL (SMILES used to render synthetic structures) | CC-BY-SA-3.0 | attribution; ShareAlike applies if the synthetic corpus itself is published |
| COCO (D-FINE pretraining) | annotations CC-BY-4.0 | via the upstream checkpoint |
