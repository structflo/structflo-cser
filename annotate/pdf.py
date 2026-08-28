"""PDF → PNG page rendering (pypdfium2 via structflo.cser.pdf)."""

import uuid
from pathlib import Path

from structflo.cser.pdf import iter_pages


def render_pdf(pdf_path: Path, output_dir: Path, dpi: int) -> list[dict]:
    """Render every page of *pdf_path* to a PNG at *dpi*, return page metadata.

    A 6-char unique suffix is appended to the stem so that re-uploading the
    same PDF never collides with a previous session's ground-truth files.

    Returns:
        list of {"id": str, "path": str (absolute), "w": int, "h": int}
    """
    img_dir = output_dir / "tmp"    # staging — moved to images/ on export
    img_dir.mkdir(parents=True, exist_ok=True)

    uid  = uuid.uuid4().hex[:6]             # unique per upload
    stem = f"{pdf_path.stem}_{uid}"
    pages = []

    for i, img in enumerate(iter_pages(pdf_path, dpi=dpi)):
        pid = f"{stem}_p{i:03d}"
        out_path = img_dir / f"{pid}.png"
        if not out_path.exists():
            img.save(str(out_path))
        pages.append({
            "id":   pid,
            "path": str(out_path),
            "w":    img.width,
            "h":    img.height,
        })

    return pages
