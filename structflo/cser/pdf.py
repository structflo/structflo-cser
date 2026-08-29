"""PDF page rendering via pypdfium2 (BSD-3-Clause) — replaces PyMuPDF (AGPL-3.0).

Rendering geometry matches the retired PyMuPDF path exactly. MuPDF sizes a
pixmap with ``fz_round_rect`` — ``ceil(extent − 0.001)`` — whereas pypdfium2's
``PdfPage.render(scale=…)`` uses a plain ``ceil``, which overshoots by one
pixel whenever ``points·dpi/72`` lands on an integer with floating-point error
(US-Letter at 150 dpi: 792·150/72 → 1650.0000000000002 → 1651). Downstream
consumers persist bounding boxes against these renders, so we compute the
PyMuPDF size ourselves and render into an explicitly-sized bitmap. Verified
identical on 600 random page-size × dpi combinations.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

PdfSource = Path | str | bytes

_ROUND_EPS = 0.001  # MuPDF fz_round_rect tolerance

# PDFium is not thread-safe (not even across documents); the annotate tool runs a
# threaded Flask server, so every PDFium call is serialised through this lock.
_PDFIUM_LOCK = threading.RLock()


def pixel_size(width_pt: float, height_pt: float, dpi: int) -> tuple[int, int]:
    """Pixel size PyMuPDF would produce for a page of the given point size."""
    scale = dpi / 72
    return (
        max(1, math.ceil(width_pt * scale - _ROUND_EPS)),
        max(1, math.ceil(height_pt * scale - _ROUND_EPS)),
    )


def _open(src: PdfSource):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(src) if isinstance(src, (str, Path)) else src)
    pdf.init_forms()  # draw form-field widgets, as PyMuPDF did by default
    return pdf


def page_count(src: PdfSource) -> int:
    with _PDFIUM_LOCK:
        pdf = _open(src)
        try:
            return len(pdf)
        finally:
            pdf.close()


def _render(page, dpi: int) -> Image.Image:
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    width, height = pixel_size(page.get_width(), page.get_height(), dpi)
    bitmap = pdfium.PdfBitmap.new_native(
        width, height, format=pdfium_c.FPDFBitmap_BGR, rev_byteorder=True
    )
    try:
        bitmap.fill_rect((255, 255, 255, 255), 0, 0, width, height)
        flags = pdfium_c.FPDF_ANNOT | pdfium_c.FPDF_REVERSE_BYTE_ORDER
        render_args = (bitmap, page, 0, 0, width, height, 0, flags)
        pdfium_c.FPDF_RenderPageBitmap(*render_args)
        if page.formenv:
            pdfium_c.FPDF_FFLDraw(page.formenv, *render_args)
        return bitmap.to_pil().convert("RGB")
    finally:
        bitmap.close()


def render_page(src: PdfSource, page_index: int, *, dpi: int) -> Image.Image:
    """Render one page (0-based) of a PDF path or bytes to an RGB PIL image."""
    with _PDFIUM_LOCK:
        pdf = _open(src)
        try:
            page = pdf[page_index]
            try:
                return _render(page, dpi)
            finally:
                page.close()
        finally:
            pdf.close()


def iter_pages(src: PdfSource, *, dpi: int) -> Iterator[Image.Image]:
    """Yield every page as an RGB PIL image, one at a time (bounded memory)."""
    with _PDFIUM_LOCK:
        pdf = _open(src)
        n = len(pdf)
    try:
        for i in range(n):
            with _PDFIUM_LOCK:
                page = pdf[i]
                try:
                    img = _render(page, dpi)
                finally:
                    page.close()
            yield img
    finally:
        with _PDFIUM_LOCK:
            pdf.close()


def open_pages(src: PdfSource, *, dpi: int) -> tuple[int, Iterator[Image.Image]]:
    """Return ``(page_count, page_iterator)`` for a PDF path or bytes.

    The document is opened once; the iterator closes it when exhausted or
    garbage-collected. Callers that consume lazily (e.g. streaming one page at
    a time to a client) get bounded memory.
    """
    with _PDFIUM_LOCK:
        pdf = _open(src)
        n = len(pdf)

    def pages() -> Iterator[Image.Image]:
        try:
            for i in range(n):
                with _PDFIUM_LOCK:
                    page = pdf[i]
                    try:
                        img = _render(page, dpi)
                    finally:
                        page.close()
                yield img
        finally:
            with _PDFIUM_LOCK:
                pdf.close()

    return n, pages()
