"""Every LATEST weights entry must be loadable by the installed package version and by the
current major release — a registry where the detector requires >=1.0.0 while the default
matcher's weights require <1.0.0 breaks ChemPipeline() on a fresh install (shipped in 1.0.0)."""

from __future__ import annotations

import importlib.metadata

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from structflo.cser.weights import LATEST, REGISTRY


def _installed() -> Version:
    return Version(importlib.metadata.version("structflo-cser"))


def test_latest_entries_share_a_compatible_release():
    """The LATEST entries of all models must accept one common release: the installed one
    (or, on dev checkouts, the release the dev version is heading for)."""
    v = _installed()
    target = Version(v.base_version)  # 1.0.1.dev3 -> 1.0.1
    for model, tag in LATEST.items():
        if tag is None:
            continue
        spec = SpecifierSet(REGISTRY[model][tag]["requires"])
        assert target in spec, (
            f"{model}/{tag} requires {spec} but the package is {target}"
        )


def test_latest_entries_accept_the_current_major():
    majors = {Version(v.base_version).major for v in [_installed()]}
    for model, tag in LATEST.items():
        if tag is None:
            continue
        spec = SpecifierSet(REGISTRY[model][tag]["requires"])
        for major in majors:
            assert Version(f"{major}.0.0") in spec or any(
                Version(f"{major}.{minor}.0") in spec for minor in range(0, 20)
            ), f"{model}/{tag} ({spec}) accepts no {major}.x release"
