"""pypdfium2-backed PDF rendering (replaces PyMuPDF)."""

from __future__ import annotations

import pytest
from PIL import Image

from structflo.cser.pdf import iter_pages, page_count, pixel_size, render_page


@pytest.fixture
def two_page_pdf(tmp_path):
    """A 2-page US-Letter PDF (612x792 pt) written with matplotlib — no PDF library needed."""
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


def test_pixel_size_matches_pymupdf_rounding():
    # 792*150/72 = 1650.0000000000002 in floating point: a plain ceil gives 1651,
    # PyMuPDF (fz_round_rect, ceil(x - 0.001)) gives 1650 — the size DocuStore stored.
    assert pixel_size(612, 792, 150) == (1275, 1650)
    assert pixel_size(960, 540, 150) == (2000, 1125)
    assert pixel_size(595.276, 841.89, 150) == (1241, 1754)
    assert pixel_size(612, 792, 300) == (2550, 3300)


def test_render_page_size_and_mode(two_page_pdf):
    img = render_page(two_page_pdf, 0, dpi=150)
    assert (img.width, img.height) == (1275, 1650)
    assert img.mode == "RGB"


def test_render_page_accepts_bytes(two_page_pdf):
    img = render_page(two_page_pdf.read_bytes(), 1, dpi=150)
    assert (img.width, img.height) == (1275, 1650)


def test_render_page_selects_page_and_is_deterministic(two_page_pdf):
    first = render_page(two_page_pdf, 0, dpi=150)
    second = render_page(two_page_pdf, 1, dpi=150)
    assert first.tobytes() != second.tobytes()
    assert render_page(two_page_pdf, 0, dpi=150).tobytes() == first.tobytes()


def test_page_count_and_iter_pages(two_page_pdf):
    assert page_count(two_page_pdf) == 2
    pages = list(iter_pages(two_page_pdf, dpi=72))
    assert len(pages) == 2
    assert all(isinstance(p, Image.Image) and p.size == (612, 792) for p in pages)
