# Publishing New Model Weights

This guide is for developers who have retrained a model and need to make the new
weights available to users of the `structflo-cser` package.

---

## Concepts

Two things are versioned independently:

| Thing | Where | Example |
|---|---|---|
| Python package | PyPI | `structflo-cser==0.2.0` |
| Model weights | HuggingFace Hub | `weights-v1.1` (git tag) |

A weights release does **not** require a code/PyPI release unless the model
architecture changed (different classes, input shape, head structure).

---

## One-time setup: create the HF Hub repo

Do this once per model, not per release.

```bash
pip install huggingface_hub
huggingface-cli login          # paste your HF write token

# Create the repo (public)
huggingface-cli repo create sidxz/structflo-cser-detector --type model
```

---

## Case 1: Retrain on more data, same architecture

This is the common case — better accuracy, same D-FINE-L 2-class model.

### Steps 1–3 in one command

```bash
# Dry-run first to confirm everything looks right
python scripts/publish_weights.py \
    --model cser-detector \
    --version v1.1 \
    --dry-run

# Then publish for real
python scripts/publish_weights.py \
    --model cser-detector \
    --version v1.1
```

The script:
1. Computes the `sha256` of the weights file
2. Uploads `best.safetensors` to `sidxz/structflo-cser-detector` on HF Hub (LPS / relmatcher: `best.pt`)
3. Creates the git tag `weights-v1.1` on the HF repo
4. Patches [structflo/cser/weights.py](../structflo/cser/weights.py) — adds the
   registry entry and bumps `LATEST`

Default weights file path: `runs/labels_detect/dfine_l_plus/weights/best.safetensors`.
The script also (re)uploads a model card (`README.md` with `license: apache-2.0`).
Override with `--weights-file` if yours is elsewhere.

### Step 4 — commit and push

```bash
git add structflo/cser/weights.py
git commit -m "weights: publish v1.1"
git push
```

No PyPI release required. Users get the new weights automatically on their next
run once they have the updated package.

---

## Case 2: Architecture change (new classes, different model)

This requires both a new weights major version **and** a new package version,
because the old code cannot load the new weights.

### Step 1 — publish weights with a custom `--requires`

```bash
python scripts/publish_weights.py \
    --model cser-detector \
    --version v2.0 \
    --requires ">=1.0.0,<2.0.0"
```

The `--requires` flag sets which package versions are compatible with these
weights.  Use a new major range whenever the architecture changes.

### Step 2 — bump the package version

In `pyproject.toml`:
```toml
version = "1.0.0"
```

### Step 3 — commit, tag, and publish to PyPI

```bash
git add structflo/cser/weights.py pyproject.toml
git commit -m "v1.0.0: 3-class detector"
git tag v1.0.0
git push && git push --tags
uv build && uv publish
```

Users on the old package who try to use `v2.0` weights will see a clear error:

```
WeightsCompatibilityError: Weights 'cser-detector/v2.0' require
structflo-cser>=1.0.0 (you have 0.2.0).
Upgrade with:  uv add 'structflo-cser>=1.0.0'
```

---

## Adding a new model entirely

For a second model (e.g. `ner-tagger`):

1. Create a new HF Hub repo: `sidxz/structflo-ner-tagger`

2. Add the repo to `MODEL_REPOS` in [scripts/publish_weights.py](../scripts/publish_weights.py):
   ```python
   MODEL_REPOS = {
       "cser-detector": { ... },   # existing
       "ner-tagger": {
           "repo_id":  "sidxz/structflo-ner-tagger",
           "filename": "model.pt",
       },
   }
   ```

3. Add its default weights path to `DEFAULT_WEIGHTS_PATHS` in the same file.

4. Add the model key to `REGISTRY` and `LATEST` in `weights.py`:
   ```python
   REGISTRY = {
       "cser-detector": { ... },
       "ner-tagger": {},           # entries added by publish_weights.py
   }
   LATEST = {
       "cser-detector": "v1.1",
       "ner-tagger":    None,      # set after first publish
   }
   ```

5. Run `publish_weights.py --model ner-tagger --version v1.0` as normal.

---

## Decision tree

```
Did anything in the model architecture change?
(classes, imgsz, backbone, head)
│
├── No  → Case 1: run publish_weights.py, no PyPI release
│         same --requires range, bump LATEST automatically
│
└── Yes → Case 2: run publish_weights.py --requires ">=X.0.0,<Y.0.0"
          bump pyproject.toml version, publish to PyPI
```

---

## Checklist

- [ ] `huggingface-cli login` done on this machine
- [ ] Training finished, `best.pt` exists at the expected path
- [ ] `--dry-run` reviewed and looks correct
- [ ] Published with `publish_weights.py` (uploads + tags + patches `weights.py`)
- [ ] `git add structflo/cser/weights.py && git commit` done
- [ ] If arch changed: `pyproject.toml` version bumped and published to PyPI


---

## Release checklist (learned from 1.0.0)

1. **Major version bump ⇒ re-open every `LATEST` entry's `requires`** in `structflo/cser/weights.py`,
   not just the model you changed. `scripts/publish_weights.py` now refuses to publish when another
   model's `LATEST` entry rejects the package version, and
   `tests/test_weights_registry_consistency.py` fails on dev checkouts heading for an incompatible
   release.
2. Resolve **all three** models under the exact release version (no dev suffix):
   `uv sync --reinstall-package structflo-cser` after tagging locally, then
   `python -c "from structflo.cser.weights import resolve_weights; [resolve_weights(m) for m in ('cser-detector','cser-lps','cser-relmatcher')]"`.
3. **Build and smoke the container before pushing the tag**:
   `docker build --build-arg VERSION=X.Y.Z -t structflo-cser:X.Y.Z .` then run it with `--preload`
   and POST a page to `/extract`; `--preload` constructs `ChemPipeline()` and resolves every default
   weight, which is exactly where 1.0.0 failed.
4. Only then `git push origin main && git push origin vX.Y.Z` (PyPI publish is tag-triggered and
   cannot be undone — a broken release must be yanked manually on PyPI).
