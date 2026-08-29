"""Bound the effect of swapping the PDF rasteriser (PyMuPDF → pypdfium2) on detection.

LEGACY TOOL — intentionally imports PyMuPDF (fitz) and ultralytics, which are no longer
project dependencies. It was run once, before the migration, in an environment that still had
both (`uv run --with pymupdf --with ultralytics`); its result lives in
runs/license_migration/eval/render_agreement.json (agreement F1 0.951 @144 dpi / 0.963 @150 dpi).

GT-free: for N staging PDFs, render the page with BOTH engines at the same dpi, run the
detector on both renders, and measure how well the two detection sets agree (same-class
IoU>=0.5 greedy matching at the conf operating point).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from structflo.cser.inference.metrics import box_iou
from structflo.cser.pdf import render_page


def _fitz_render(path: Path, dpi: int) -> Image.Image:
    import fitz

    doc = fitz.open(str(path))
    try:
        pix = doc[0].get_pixmap(
            matrix=fitz.Matrix(dpi / 72, dpi / 72), colorspace=fitz.csRGB
        )
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()


def _dets(model, im: Image.Image, conf: float, imgsz: int) -> list[tuple]:
    arr = np.array(im.convert("L").convert("RGB"))
    res = model(arr, conf=conf, imgsz=imgsz, verbose=False)[0]
    return [
        (box.xyxy[0].cpu().numpy(), float(box.conf[0]), int(box.cls[0]))
        for box in res.boxes
    ]


def _agree(a: list, b: list, thr: float = 0.5) -> tuple[int, int, int, list]:
    matched, dconf = 0, []
    for c in (0, 1):
        A = [x for x in a if x[2] == c]
        B = [x for x in b if x[2] == c]
        if not A or not B:
            continue
        ious = box_iou(np.array([x[0] for x in A]), np.array([x[0] for x in B]))
        used = set()
        for i in np.argsort([-x[1] for x in A]):
            cand = [
                j for j in np.argsort(-ious[i]) if j not in used and ious[i, j] >= thr
            ]
            if cand:
                used.add(cand[0])
                matched += 1
                dconf.append(abs(A[i][1] - B[cand[0]][1]))
    return matched, len(a), len(b), dconf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pdf-dir", type=Path, default=Path("data/cser_staging_all/staging_pdfs")
    )
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--dpis", nargs="*", type=int, default=[144, 150])
    ap.add_argument(
        "--weights", default="runs/labels_detect/finetune_plus/weights/best.pt"
    )
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("runs/license_migration/eval/render_agreement.json"),
    )
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    random.seed(0)
    pdfs = random.sample(pdfs, min(args.n, len(pdfs)))
    summary = {}
    for dpi in args.dpis:
        size_mismatch = n_pages = tot_m = tot_a = tot_b = same_count = 0
        dconfs, pix_diff = [], []
        for p in pdfs:
            try:
                fz = _fitz_render(p, dpi)
                pf = render_page(p, 0, dpi=dpi)
            except Exception as e:  # corrupt PDFs in staging
                print(f"skip {p.name}: {type(e).__name__}")
                continue
            n_pages += 1
            if fz.size != pf.size:
                size_mismatch += 1
                continue
            a = np.asarray(fz.convert("L"), dtype=np.int16)
            b = np.asarray(pf.convert("L"), dtype=np.int16)
            pix_diff.append(float((np.abs(a - b) > 32).mean()))
            da, db = (
                _dets(model, fz, args.conf, args.imgsz),
                _dets(model, pf, args.conf, args.imgsz),
            )
            m, na, nb, dc = _agree(da, db)
            tot_m += m
            tot_a += na
            tot_b += nb
            dconfs += dc
            same_count += int(na == nb)
            if n_pages % 50 == 0:
                print(f"[{dpi} dpi] {n_pages}/{len(pdfs)}", flush=True)
        f1 = 2 * tot_m / max(tot_a + tot_b, 1)
        summary[dpi] = {
            "pages": n_pages,
            "size_mismatch_pages": size_mismatch,
            "dets_fitz": tot_a,
            "dets_pdfium": tot_b,
            "matched": tot_m,
            "agreement_f1": f1,
            "pages_same_count": same_count,
            "median_abs_conf_diff": float(np.median(dconfs)) if dconfs else None,
            "mean_pixel_frac_diff_gt32": float(np.mean(pix_diff)) if pix_diff else None,
        }
        print(
            f"[{dpi} dpi] pages {n_pages} (size mismatch {size_mismatch}); dets fitz {tot_a} / pdfium {tot_b}; "
            f"matched {tot_m} → agreement F1 {f1:.4f}; same-count pages {same_count}/{n_pages}; "
            f"median |Δconf| {summary[dpi]['median_abs_conf_diff']}; pixel-diff frac {summary[dpi]['mean_pixel_frac_diff_gt32']}",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "weights": args.weights,
                "conf": args.conf,
                "imgsz": args.imgsz,
                "n_pdfs": len(pdfs),
                "by_dpi": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
