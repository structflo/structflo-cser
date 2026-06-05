"""Structured, display-friendly version/configuration snapshot for a pipeline.

``ChemPipeline.version`` returns a :class:`PipelineInfo`: a read-only mapping
that also renders as a readable table in a terminal (``repr``) and in Jupyter
(``_repr_markdown_``).  It gathers the package version, the detector + matcher
weight provenance (version, repo, revision, sha256, local path), the runtime
config, and key dependency versions — without triggering any weight download.
"""

from __future__ import annotations

import importlib.metadata
import platform
from collections.abc import Mapping

# Dependencies worth surfacing for a reproducible bug report.
_KEY_DEPS = (
    "ultralytics",
    "torch",
    "huggingface-hub",
    "numpy",
    "scipy",
    "pillow",
    "easyocr",
    "decimer",
)


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_version() -> str:
    """Installed ``structflo-cser`` version, or the source ``__version__``."""
    try:
        return importlib.metadata.version("structflo-cser")
    except importlib.metadata.PackageNotFoundError:
        from structflo.cser import __version__

        return __version__


def dependency_versions() -> dict[str, str | None]:
    return {name: _dist_version(name) for name in _KEY_DEPS}


def _fmt_weight(w: dict | None) -> str:
    """One-line summary of a ``weight_info``/``weight_descriptor`` dict."""
    if not w:
        return "—  (parameter-free)"
    source = w.get("source")
    if source == "local":
        return f"local file  {w.get('version') or w.get('path')}"
    if source == "unpublished":
        return "unpublished (no weights for this model yet)"
    if source == "unknown":
        return f"{w.get('version')}  (not in registry)"
    # registry
    parts = [str(w.get("version"))]
    if w.get("repo_id"):
        parts.append(f"{w['repo_id']}@{w.get('revision')}")
    if w.get("sha256"):
        parts.append(f"sha256:{w['sha256'][:12]}…")
    return "  ".join(parts)


class PipelineInfo(Mapping):
    """Read-only mapping snapshot of a pipeline's versions and configuration.

    Supports ``info["package"]``, ``dict(info)`` and ``info.to_dict()`` for
    programmatic access, and renders nicely via ``print(info)`` / Jupyter.
    """

    def __init__(self, data: dict) -> None:
        self._data = data

    # -- Mapping interface --------------------------------------------------
    def __getitem__(self, key: str):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict:
        return dict(self._data)

    # -- Rendering ----------------------------------------------------------
    def __repr__(self) -> str:
        d = self._data
        pkg = d["package"]
        cfg = d["config"]
        lines = [
            "ChemPipeline",
            f"  package        : {pkg['name']} {pkg['version']}  (Python {pkg['python']})",
            f"  detector       : {_fmt_weight(d['detector'])}",
            f"                   path: {d['detector'].get('path') or '(not downloaded yet)'}",
            f"  matcher        : {d['matcher']['class']}  →  {_fmt_weight(d['matcher']['weights'])}",
        ]
        if d["matcher"]["weights"] and d["matcher"]["weights"].get("path"):
            lines.append(f"                   path: {d['matcher']['weights']['path']}")
        if d["matcher"].get("params"):
            params = "  ".join(f"{k}={v}" for k, v in d["matcher"]["params"].items())
            lines.append(f"                   params: {params}")
        lines += [
            f"  smiles         : {d['smiles_extractor']['class']}",
            f"  ocr            : {d['ocr']['class']}",
            "  config         : " + "  ".join(f"{k}={v}" for k, v in cfg.items()),
            "  dependencies   : "
            + "  ".join(f"{k}={v or '—'}" for k, v in d["dependencies"].items()),
        ]
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        d = self._data
        pkg = d["package"]
        rows = [
            "### ChemPipeline",
            "",
            "| field | value |",
            "|---|---|",
            f"| **package** | `{pkg['name']}` **{pkg['version']}** (Python {pkg['python']}) |",
            f"| **detector** | {_fmt_weight(d['detector'])} |",
            f"| detector path | `{d['detector'].get('path') or '(not downloaded yet)'}` |",
            f"| **matcher** | `{d['matcher']['class']}` → {_fmt_weight(d['matcher']['weights'])} |",
        ]
        mw = d["matcher"]["weights"]
        if mw and mw.get("path"):
            rows.append(f"| matcher path | `{mw['path']}` |")
        if d["matcher"].get("params"):
            params = ", ".join(f"{k}={v}" for k, v in d["matcher"]["params"].items())
            rows.append(f"| matcher params | {params} |")
        rows += [
            f"| **smiles** | `{d['smiles_extractor']['class']}` |",
            f"| **ocr** | `{d['ocr']['class']}` |",
            "| **config** | "
            + ", ".join(f"{k}={v}" for k, v in d["config"].items())
            + " |",
            "| **dependencies** | "
            + ", ".join(f"{k} {v or '—'}" for k, v in d["dependencies"].items())
            + " |",
        ]
        return "\n".join(rows)


def python_version() -> str:
    return platform.python_version()
