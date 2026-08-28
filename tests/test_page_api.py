"""Per-page render/inference API (0.4.2)."""

from __future__ import annotations

import pytest
from PIL import Image

from structflo.cser.pipeline import DEFAULT_DPI, PageResult, render_page


@pytest.fixture
def two_page_pdf(tmp_path):
    """A 2-page US-Letter PDF (612x792 pt), written with matplotlib — no PDF library needed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = tmp_path / "letter.pdf"
    with PdfPages(path) as pp:
        for text in ("page one", "page two"):
            fig = plt.figure(figsize=(612 / 72, 792 / 72), dpi=72)
            fig.text(0.1, 0.9, text, fontsize=20)
            pp.savefig(fig)
            plt.close(fig)
    return path


def test_default_dpi_is_the_operating_point():
    # 144 loses pairs, >=200 finds none. If this changes, DocuStore's stored
    # coordinates silently stop matching its stored renders.
    assert DEFAULT_DPI == 150


def test_render_page_size_follows_dpi(two_page_pdf):
    img = render_page(two_page_pdf, 0)
    # 612pt * 150/72 = 1275, 792pt * 150/72 = 1650
    assert (img.width, img.height) == (1275, 1650)
    assert img.mode == "RGB"


def test_render_page_selects_the_requested_page(two_page_pdf):
    first = render_page(two_page_pdf, 0)
    second = render_page(two_page_pdf, 1)
    assert first.tobytes() != second.tobytes()


def test_render_page_is_deterministic(two_page_pdf):
    assert (
        render_page(two_page_pdf, 0).tobytes() == render_page(two_page_pdf, 0).tobytes()
    )


def test_process_pdf_page_reports_the_size_of_the_image_it_returns(
    two_page_pdf, monkeypatch
):
    from structflo.cser.pipeline import ChemPipeline

    sentinel_pairs = ["pair-a", "pair-b"]
    monkeypatch.setattr(ChemPipeline, "__init__", lambda self: None)
    monkeypatch.setattr(ChemPipeline, "process", lambda self, image: sentinel_pairs)

    result = ChemPipeline().process_pdf_page(two_page_pdf, 1)

    assert isinstance(result, PageResult)
    assert result.pairs == sentinel_pairs
    assert isinstance(result.image, Image.Image)
    assert (result.width, result.height) == result.image.size
    assert (result.width, result.height) == (1275, 1650)
