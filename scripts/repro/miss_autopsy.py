"""Residual label-miss autopsy on the real val/test splits.

Runs the deployed detector (full-image @1280) ONCE per page at a low conf floor,
then classifies every GT label at the operating point (conf 0.3, IoU 0.5):

  HIT          candidate with conf>=0.3 and IoU>=0.5
  CONF_FLOOR   best candidate has IoU>=0.5 but conf in [floor, 0.3)
  LOOSE        candidate with conf>=0.3 but only 0.1<=IoU<0.5  (localization)
  WEAK_LOOSE   candidate with conf<0.3 and 0.1<=IoU<0.5
  UNDETECTED   no candidate with IoU>=0.1 even at the floor

Per-miss record: page, bbox, size, aspect, position, best IoU/conf. Saves an
annotated context crop per miss (GT label green, GT struct blue, best candidate
orange) plus contact sheets for visual review. Also collects FP label boxes at
conf 0.3 (no GT label overlap at IoU>=0.1) with their own crops/sheets.

Usage:
    uv run python scripts/repro/miss_autopsy.py --split test
    uv run python scripts/repro/miss_autopsy.py --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REAL = Path("/net-fs-ins/shared-docker-vols/structflo-cser-annotate/data")
SPLIT = Path("data/finetune/real_split.json")
FLOOR = 0.01
CONF_OP = 0.30
IOU_OP = 0.5
IOU_LOOSE = 0.1

CELL_W, CELL_H, CAP_H = 440, 330, 30
COLS, ROWS = 4, 4


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def classify(gt_box, cands):
    """Return (category, best) where best = candidate dict maximizing IoU."""
    best, best_iou = None, 0.0
    for d in cands:
        v = _iou(gt_box, d["bbox"])
        if v > best_iou:
            best, best_iou = d, v
    if best_iou >= IOU_OP:
        cat = "HIT" if best["conf"] >= CONF_OP else "CONF_FLOOR"
    elif best_iou >= IOU_LOOSE:
        cat = "LOOSE" if best["conf"] >= CONF_OP else "WEAK_LOOSE"
    else:
        cat = "UNDETECTED"
        best = None
    return cat, best, best_iou


def context_crop(img, gt_label, gt_struct, best, pad=220):
    """Annotated context crop around a GT label box."""
    w, h = img.size
    x1, y1, x2, y2 = gt_label
    cx1 = max(0, int(x1) - pad)
    cy1 = max(0, int(y1) - pad)
    cx2 = min(w, int(x2) + pad)
    cy2 = min(h, int(y2) + pad)
    crop = img.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    dr = ImageDraw.Draw(crop)

    def draw(box, color, width=3):
        if box is None:
            return
        bx1, by1, bx2, by2 = (box[0] - cx1, box[1] - cy1, box[2] - cx1, box[3] - cy1)
        dr.rectangle([bx1, by1, bx2, by2], outline=color, width=width)

    if gt_struct is not None:
        draw(gt_struct, (60, 120, 255), 2)
    draw(gt_label, (0, 200, 0), 3)
    if best is not None:
        draw(best["bbox"], (255, 140, 0), 2)
    return crop


def fp_crop(img, det_box, pad=220):
    w, h = img.size
    x1, y1, x2, y2 = det_box
    cx1 = max(0, int(x1) - pad)
    cy1 = max(0, int(y1) - pad)
    cx2 = min(w, int(x2) + pad)
    cy2 = min(h, int(y2) + pad)
    crop = img.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    dr = ImageDraw.Draw(crop)
    dr.rectangle([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], outline=(255, 0, 0), width=3)
    return crop


def contact_sheets(records, out_dir, prefix):
    """records: list of (crop_path, caption). Builds COLSxROWS sheets."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    per = COLS * ROWS
    paths = []
    for s in range((len(records) + per - 1) // per):
        batch = records[s * per : (s + 1) * per]
        sheet = Image.new("RGB", (COLS * CELL_W, ROWS * (CELL_H + CAP_H)), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        for i, (cp, cap) in enumerate(batch):
            r, c = divmod(i, COLS)
            cell = Image.open(cp)
            cell.thumbnail((CELL_W - 8, CELL_H - 8))
            ox = c * CELL_W + (CELL_W - cell.width) // 2
            oy = r * (CELL_H + CAP_H) + (CELL_H - cell.height) // 2
            sheet.paste(cell, (ox, oy))
            dr.text((c * CELL_W + 6, r * (CELL_H + CAP_H) + CELL_H + 6), cap, fill=(0, 0, 0), font=font)
        p = out_dir / f"{prefix}_sheet{s:02d}.png"
        sheet.save(p)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--detector", type=Path, default=Path("runs/labels_detect/finetune_plus/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", type=Path, default=Path("runs/repro/miss_autopsy"))
    args = ap.parse_args()

    from ultralytics import YOLO

    stems = json.loads(SPLIT.read_text())[args.split]
    model = YOLO(str(args.detector))

    out = args.out / args.split
    crops_dir = out / "crops"
    fp_dir = out / "fp_crops"
    sheets_dir = out / "sheets"
    for d in (crops_dir, fp_dir, sheets_dir):
        d.mkdir(parents=True, exist_ok=True)

    cats = ["HIT", "CONF_FLOOR", "LOOSE", "WEAK_LOOSE", "UNDETECTED"]
    counts = dict.fromkeys(cats, 0)
    miss_records = []  # full per-miss dicts
    fp_records = []
    hit_sizes, miss_sizes = [], []
    rec_thr = {0.5: [0, 0], 0.3: [0, 0]}  # IoU thr -> [hit, total] @ conf 0.3
    per_page_miss = {}
    n_pages = 0

    for k, stem in enumerate(stems):
        gtf = REAL / "ground_truth" / f"{stem}.json"
        ip = REAL / "images" / f"{stem}.jpg"
        if not ip.exists():
            ip = REAL / "images" / f"{stem}.png"
        if not gtf.exists() or not ip.exists():
            continue
        n_pages += 1
        entries = json.loads(gtf.read_text())
        pil = Image.open(ip).convert("L").convert("RGB")
        img_np = np.array(pil)
        res = model(img_np, conf=FLOOR, imgsz=args.imgsz, verbose=False)[0]
        cands = []
        for b in res.boxes:
            if int(b.cls[0]) == 1:
                cands.append({"bbox": [float(v) for v in b.xyxy[0].cpu().numpy()], "conf": float(b.conf[0])})
        cands_op = [d for d in cands if d["conf"] >= CONF_OP]

        gt_labeled = [e for e in entries if e.get("label_bbox")]
        for e in gt_labeled:
            g = e["label_bbox"]
            for thr in rec_thr:
                rec_thr[thr][1] += 1
                if any(_iou(g, d["bbox"]) >= thr for d in cands_op):
                    rec_thr[thr][0] += 1

            cat, best, best_iou = classify(g, cands)
            counts[cat] += 1
            bw, bh = g[2] - g[0], g[3] - g[1]
            if cat == "HIT":
                hit_sizes.append((bw * bh) ** 0.5)
                continue
            miss_sizes.append((bw * bh) ** 0.5)
            per_page_miss[stem] = per_page_miss.get(stem, 0) + 1
            idx = len(miss_records)
            crop = context_crop(pil, g, e.get("struct_bbox"), best)
            cp = crops_dir / f"{idx:03d}_{cat}_{stem[:40]}.png"
            crop.save(cp)
            miss_records.append({
                "idx": idx,
                "page": stem,
                "category": cat,
                "bbox": g,
                "w": bw,
                "h": bh,
                "sqrt_area": (bw * bh) ** 0.5,
                "aspect": bw / bh if bh > 0 else 0,
                "pos_norm": [(g[0] + g[2]) / 2 / pil.width, (g[1] + g[3]) / 2 / pil.height],
                "best_iou": best_iou,
                "best_conf": best["conf"] if best else None,
                "crop": str(cp),
            })

        # FP labels at the operating point
        gt_l = [e["label_bbox"] for e in gt_labeled]
        for d in cands_op:
            if not any(_iou(g, d["bbox"]) >= IOU_LOOSE for g in gt_l):
                idx = len(fp_records)
                cp = fp_dir / f"{idx:03d}_{stem[:40]}.png"
                fp_crop(pil, d["bbox"]).save(cp)
                fp_records.append({
                    "idx": idx, "page": stem, "bbox": d["bbox"], "conf": d["conf"], "crop": str(cp),
                })

        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(stems)} pages")

    # contact sheets, misses grouped by category then FPs by conf desc
    miss_sorted = sorted(miss_records, key=lambda r: (cats.index(r["category"]), -(r["best_conf"] or 0)))
    sheet_recs = [
        (r["crop"], f"#{r['idx']} {r['category']}  {r['w']:.0f}x{r['h']:.0f}px  "
                    f"IoU {r['best_iou']:.2f}  conf {r['best_conf'] if r['best_conf'] is None else round(r['best_conf'], 2)}")
        for r in miss_sorted
    ]
    sheets = contact_sheets(sheet_recs, sheets_dir, "miss")
    fp_sorted = sorted(fp_records, key=lambda r: -r["conf"])
    fp_sheets = contact_sheets([(r["crop"], f"#{r['idx']} FP conf {r['conf']:.2f}") for r in fp_sorted], sheets_dir, "fp")

    total = sum(counts.values())
    n_miss = total - counts["HIT"]
    lines = [
        f"# Label miss autopsy — real {args.split} ({n_pages} pages)",
        "",
        f"Detector: `{args.detector}` full@{args.imgsz}, floor {FLOOR}, operating point conf {CONF_OP} IoU {IOU_OP}.",
        "",
        f"GT labels: {total}.  Recall@conf0.3: IoU.5 {rec_thr[0.5][0]/max(rec_thr[0.5][1],1):.1%}, "
        f"IoU.3 {rec_thr[0.3][0]/max(rec_thr[0.3][1],1):.1%}",
        "",
        "| category | n | % of GT | % of misses |",
        "|---|---|---|---|",
    ]
    for c in cats:
        pm = f"{counts[c]/n_miss:.0%}" if c != "HIT" and n_miss else "-"
        lines.append(f"| {c} | {counts[c]} | {counts[c]/max(total,1):.1%} | {pm} |")
    if hit_sizes and miss_sizes:
        hs, ms = np.array(hit_sizes), np.array(miss_sizes)
        lines += [
            "",
            f"Label size sqrt-area px — HIT median {np.median(hs):.0f} (p25 {np.percentile(hs,25):.0f}, p75 {np.percentile(hs,75):.0f})"
            f" vs MISS median {np.median(ms):.0f} (p25 {np.percentile(ms,25):.0f}, p75 {np.percentile(ms,75):.0f})",
        ]
    lines += ["", f"FP labels @0.3: {len(fp_records)} ({len(fp_records)/max(n_pages,1):.2f}/page)", "", "Pages by miss count:"]
    for stem, n in sorted(per_page_miss.items(), key=lambda kv: -kv[1])[:15]:
        lines.append(f"- {n}  {stem}")
    lines += ["", f"Miss sheets: {len(sheets)}, FP sheets: {len(fp_sheets)} in `{sheets_dir}`"]

    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "misses.json").write_text(json.dumps(miss_records, indent=1))
    (out / "fps.json").write_text(json.dumps(fp_records, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
