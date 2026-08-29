"""Flask UI: upload an image or PDF, stream back label IDs + SMILES per page.

Run:  sf-web            (or: python -m webapp)

No auth — this is meant to sit behind Traefik / LDAP forward-auth.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from collections.abc import Iterator
from pathlib import Path

from flask import Flask, Response, render_template, request, send_file
from PIL import Image

from structflo.cser.pipeline import BBox, CompoundPair

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # reject oversized uploads

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

DPI = 150           # PDF render resolution, overwritten by main()
MAX_PAGE_PX = 1600  # page preview is downscaled; boxes stay in original coords

_pipeline = None


def get_pipeline():
    """Lazy singleton — loading the detector + DECIMER + EasyOCR takes ~30 s."""
    global _pipeline
    if _pipeline is None:
        from structflo.cser.pipeline import ChemPipeline

        _pipeline = ChemPipeline()
    return _pipeline


# ── Image helpers ─────────────────────────────────────────────────────────────

def _data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, fmt)
    mime = "jpeg" if fmt == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _crop_uri(img: Image.Image, bbox: BBox, max_px: int) -> str:
    crop = img.crop((int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)))
    crop.thumbnail((max_px, max_px))
    return _data_uri(crop)


def _render_smiles(smiles: str | None) -> str | None:
    """Re-draw the extracted SMILES so it can be eyeballed against the crop."""
    if not smiles:
        return None
    from rdkit import Chem
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # unparseable — the UI flags it
    return _data_uri(Draw.MolToImage(mol, size=(240, 240)))


def _open_pages(data: bytes, filename: str) -> tuple[int, Iterator[Image.Image]]:
    """Return (page count, page iterator) for an uploaded PDF or image."""
    if not filename.lower().endswith(".pdf"):
        return 1, iter([Image.open(io.BytesIO(data)).convert("RGB")])

    from structflo.cser.pdf import open_pages

    return open_pages(data, dpi=DPI)


def _page_payload(index: int, img: Image.Image, pairs: list[CompoundPair]) -> dict:
    view = img.copy()
    view.thumbnail((MAX_PAGE_PX, MAX_PAGE_PX))
    return {
        "index": index,
        "w": img.width,
        "h": img.height,
        "image": _data_uri(view, "JPEG"),
        "pairs": [
            {
                "i": k,
                "id": p.label_text or "",
                "smiles": p.smiles or "",
                "structure_bbox": p.structure.bbox.as_list(),
                "label_bbox": p.label.bbox.as_list(),
                "match_confidence": p.match_confidence,
                "structure_crop": _crop_uri(img, p.structure.bbox, 260),
                "label_crop": _crop_uri(img, p.label.bbox, 220),
                "smiles_img": _render_smiles(p.smiles),
            }
            for k, p in enumerate(pairs)
        ],
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract():
    """Stream one NDJSON line per page so long PDFs show results as they finish."""
    f = request.files.get("file")
    if not f or not f.filename:
        return {"error": "no file"}, 400
    name = f.filename
    if not name.lower().endswith((".pdf", *IMAGE_EXTS)):
        return {"error": f"unsupported file type: {name}"}, 400
    data = f.read()  # read inside the request context; the generator runs after it

    def stream() -> Iterator[str]:
        try:
            n_pages, pages = _open_pages(data, name)
            yield json.dumps({"n_pages": n_pages, "name": name}) + "\n"
            pipe = get_pipeline()
            for i, img in enumerate(pages):
                yield json.dumps(_page_payload(i, img, pipe.process(img))) + "\n"
        except Exception as e:  # surface it in the UI instead of a dead stream
            yield json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n"

    return Response(stream(), mimetype="application/x-ndjson")


XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@app.route("/export.xlsx", methods=["POST"])
def export_xlsx():
    """Build a workbook from the rows the browser collected while streaming.

    Written as .xlsx rather than CSV because Excel re-parses CSV text on open and
    mangles compound IDs — "7178-39-6" becomes a date.  Cell values written as
    Python str stay text.
    """
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {"error": "expected {'rows': [...]}"}, 400

    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "compounds"
    ws.append(["ID", "SMILES", "Page", "Match confidence"])
    for r in rows:
        ws.append([
            str(r.get("id") or ""),
            str(r.get("smiles") or ""),
            r.get("page"),
            r.get("confidence"),
        ])
    for col, width in zip("ABCD", (28, 70, 8, 18)):
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A2"

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    stem = Path(str(payload.get("name") or "extract")).stem
    return send_file(buf, as_attachment=True, download_name=f"{stem}.xlsx",
                     mimetype=XLSX_MIME)


def main() -> None:
    p = argparse.ArgumentParser(description="structflo.cser extraction UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--dpi", type=int, default=150, help="PDF render DPI")
    p.add_argument("--preload", action="store_true",
                   help="load models at startup instead of on first upload")
    args = p.parse_args()

    global DPI
    DPI = args.dpi
    if args.preload:
        get_pipeline()

    print(f"structflo.cser UI: http://{args.host}:{args.port}")
    # ponytail: threaded=False serialises requests so two uploads can't fight over
    # the GPU. Put a job queue in front if more than one person uses it at once.
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
