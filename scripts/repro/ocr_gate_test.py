"""Quantify the OCR-grammar gate for recovering low-confidence label detections.

The gate idea: among label candidates in the low-conf recovery band [0.05, 0.30),
keep only those whose OCR text parses as a compound ID. This should convert
CONF_FLOOR misses -> hits (real labels parse) without the precision flood that a
plain threshold drop causes (numeric/percentage data cells don't parse).

For each page we re-detect at a 0.05 floor and OCR the ACTUAL candidate box pixels
(tight crop from the grayscale original), bucketing each label candidate:

  HIT          conf>=0.30, matches a GT label (IoU>=0.5)   — kept anyway; sanity upper bound
  RECOVER_TP   conf in [0.05,0.30), matches GT (IoU>=0.5)   — gate SHOULD keep (recall)
  RECOVER_FP   conf in [0.05,0.30), no GT match (IoU<0.1)   — gate SHOULD drop (precision)
  HICONF_FP    conf>=0.30, no GT match                      — current FP flood

Reports parse-rate per bucket (the gate's operating characteristic) and dumps every
(bucket, conf, text, parses) record + sample texts.

Usage:
    uv run python scripts/repro/ocr_gate_test.py --split test
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

REAL = Path("/net-fs-ins/shared-docker-vols/structflo-cser-annotate/data")
SPLIT = Path("data/finetune/real_split.json")
FLOOR = 0.05
CONF_OP = 0.30
IOU_HIT = 0.5
IOU_FP = 0.1

# assay / measurement / unit tokens that contain letters+digits but are NOT compound ids
STOP = {
    "ic50", "ic90", "ec50", "ec90", "mic", "mic50", "mic90", "mic99", "ki", "kd",
    "logp", "clogp", "logd", "mw", "pic50", "gi50", "cc50", "ld50", "tm", "cmax",
    "auc", "ph", "ki", "kd", "hg2", "hepg2", "mlm", "hlm", "t1", "t2", "h2o", "co2",
    "p450", "cyp3a4", "n1", "n2",
}


def looks_like_compound_id(text: str | None) -> bool:
    """Heuristic compound-ID grammar: an alphanumeric code token (letters+digits,
    len>=3), not an assay/unit token, not a pure number/percent/comparator."""
    if not text:
        return False
    for tok in re.split(r"\s+", text.strip()):
        core = tok.strip(".,;:()[]{}").lstrip("=<>~±")
        low = core.lower()
        if not core or low in STOP:
            continue
        if re.fullmatch(r"[\d.,%±<>=~+/x*-]+", core):  # pure numeric/percent/comparator
            continue
        has_alpha = any(c.isalpha() for c in core)
        has_digit = any(c.isdigit() for c in core)
        if has_alpha and has_digit and len(core) >= 3:
            return True
    return False


def is_pure_numeric(text: str | None) -> bool:
    if not text:
        return False
    toks = [t for t in re.split(r"\s+", text.strip()) if t]
    return bool(toks) and all(re.fullmatch(r"[\d.,%±<>=~+/x*µuMnNmkg-]+", t) for t in toks)


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def crop_for_ocr(pil, box, pad=6):
    w, h = pil.size
    x1 = max(0, int(box[0]) - pad)
    y1 = max(0, int(box[1]) - pad)
    x2 = min(w, int(box[2]) + pad)
    y2 = min(h, int(box[3]) + pad)
    c = pil.crop((x1, y1, x2, y2))
    if c.width < 80 and c.width > 0:  # upscale tiny crops for OCR
        s = max(1, int(120 / c.width))
        c = c.resize((c.width * s, c.height * s), Image.LANCZOS)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--detector", type=Path, default=Path("runs/labels_detect/finetune_plus/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", type=Path, default=Path("runs/repro/ocr_gate"))
    args = ap.parse_args()

    from ultralytics import YOLO

    from structflo.cser.pipeline.ocr import EasyOCRExtractor

    stems = json.loads(SPLIT.read_text())[args.split]
    model = YOLO(str(args.detector))
    ocr = EasyOCRExtractor()

    buckets = ["HIT", "RECOVER_TP", "RECOVER_FP", "HICONF_FP"]
    records = {b: [] for b in buckets}

    for k, stem in enumerate(stems):
        gtf = REAL / "ground_truth" / f"{stem}.json"
        ip = REAL / "images" / f"{stem}.jpg"
        if not ip.exists():
            ip = REAL / "images" / f"{stem}.png"
        if not gtf.exists() or not ip.exists():
            continue
        entries = json.loads(gtf.read_text())
        gt_l = [e["label_bbox"] for e in entries if e.get("label_bbox")]
        pil = Image.open(ip).convert("L")
        img_np = np.array(pil.convert("RGB"))
        res = model(img_np, conf=FLOOR, imgsz=args.imgsz, verbose=False)[0]

        cands = []
        for b in res.boxes:
            if int(b.cls[0]) == 1:
                cands.append((float(b.conf[0]), [float(v) for v in b.xyxy[0].cpu().numpy()]))

        # which GT labels are matched by a candidate, split by conf band
        for conf, box in cands:
            best = max((_iou(g, box) for g in gt_l), default=0.0)
            matched = best >= IOU_HIT
            nomatch = best < IOU_FP
            if matched and conf >= CONF_OP:
                bucket = "HIT"
            elif matched and conf >= FLOOR:
                bucket = "RECOVER_TP"
            elif nomatch and conf >= CONF_OP:
                bucket = "HICONF_FP"
            elif nomatch:
                bucket = "RECOVER_FP"
            else:
                continue  # partial-overlap (loose) — not a clean gate case
            text = ocr.extract(crop_for_ocr(pil, box))
            records[bucket].append({
                "page": stem, "conf": round(conf, 3), "iou": round(best, 3),
                "text": text, "parses": looks_like_compound_id(text),
                "pure_numeric": is_pure_numeric(text),
            })
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(stems)} pages")

    # report
    lines = [
        f"# OCR-grammar gate test — real {args.split}",
        "",
        f"Detector full@{args.imgsz}, candidates floor {FLOOR}. OCR = pipeline EasyOCRExtractor on the "
        f"candidate box. Gate = `looks_like_compound_id(text)`.",
        "",
        "| bucket | n | parse-rate | pure-numeric | gate wants |",
        "|---|---|---|---|---|",
    ]
    want = {"HIT": "keep", "RECOVER_TP": "KEEP", "RECOVER_FP": "DROP", "HICONF_FP": "keep*"}
    for b in buckets:
        rs = records[b]
        n = len(rs)
        pr = sum(r["parses"] for r in rs) / n if n else 0
        pn = sum(r["pure_numeric"] for r in rs) / n if n else 0
        lines.append(f"| {b} | {n} | {pr:.0%} | {pn:.0%} | {want[b]} |")

    rt = records["RECOVER_TP"]
    rf = records["RECOVER_FP"]
    gate_recall = sum(r["parses"] for r in rt) / len(rt) if rt else 0
    gate_keepfp = sum(r["parses"] for r in rf) / len(rf) if rf else 0
    lines += [
        "",
        f"**Gate in the recovery band [{FLOOR},{CONF_OP}):** keeps {gate_recall:.0%} of real labels "
        f"(RECOVER_TP), keeps {gate_keepfp:.0%} of FP candidates (RECOVER_FP). "
        f"Separation = {gate_recall - gate_keepfp:+.0%}.",
        f"Of the kept FPs, many are unannotated real labels; pure-numeric FP share = "
        f"{sum(r['pure_numeric'] for r in rf) / len(rf) if rf else 0:.0%} (these the gate correctly drops).",
        "",
        "## Sample OCR text per bucket",
    ]
    for b in buckets:
        lines.append(f"\n### {b}")
        for r in sorted(records[b], key=lambda r: -r["conf"])[:18]:
            flag = "OK " if r["parses"] else ("NUM" if r["pure_numeric"] else "no ")
            lines.append(f"- [{flag}] conf {r['conf']:.2f} iou {r['iou']:.2f}  «{r['text']}»")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.split}_summary.md").write_text("\n".join(lines) + "\n")
    (out / f"{args.split}_records.json").write_text(json.dumps(records, indent=1))
    print("\n".join(lines[:30]))
    print(f"\n... full report: {out / f'{args.split}_summary.md'}")


if __name__ == "__main__":
    main()
