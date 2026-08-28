"""Summarise runs/license_migration/eval/*.json into comparison tables (markdown).

Usage:
    uv run python scripts/license_migration/summarize.py [--tags yolo_v0.4 dfine_l_synth dfine_l_plus] [--out docs/x.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

E = Path("runs/license_migration/eval")


def _load(name: str) -> dict | None:
    p = E / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def det_row(tag: str, split: str, suffix: str = "") -> str | None:
    d = _load(f"{tag}{suffix}_{split}")
    if not d:
        return None
    a, pc = d["all"], d["per_class"]
    return (
        f"| {tag}{suffix} | {split} | {a['mAP50']:.4f} | {a['mAP50-95']:.4f} | "
        f"{pc['structure']['R']:.3f} | {pc['structure']['P']:.3f} | {pc['label']['R']:.3f} | {pc['label']['P']:.3f} | "
        f"{pc['label']['mAP50']:.4f} | {pc['label']['FP'] / max(d['n_images'], 1):.2f} |"
    )


def e2e_row(tag: str, split: str, suffix: str = "") -> str | None:
    d = _load(f"{tag}{suffix}_e2e_{split}")
    if not d:
        return None
    m = d["matchers"]
    cells = " | ".join(f"{m[k]['P']:.3f} / {m[k]['R']:.3f} / **{m[k]['F1']:.3f}**" for k in ("Hungarian", "LPS", "Relational"))
    return f"| {tag}{suffix} | {split} | {cells} |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=["yolo_v0.4", "dfine_l_synth", "dfine_l_plus"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    lines = []
    lines += ["### Detection (conf 0.3, IoU 0.5 for P/R; COCO-style AP)", "",
              "| detector | split | mAP50 | mAP50-95 | struct R | struct P | label R | label P | label mAP50 | label FP/page |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for tag in args.tags:
        for split in ("real_test", "real_val", "synth_test"):
            r = det_row(tag, split)
            if r:
                lines.append(r)
        for split in ("real_test", "real_val"):
            r = det_row(tag, split, "_conf0.5")
            if r:
                lines.append(r.replace(f"| {tag}_conf0.5 |", f"| {tag} @ conf 0.5 (val-tuned) |"))
        for sc in ("0.48", "0.5"):
            for split in ("real_test", "real_val"):
                r = det_row(tag, split, f"_scale{sc}")
                if r:
                    lines.append(r)
    lines += ["", "### End-to-end pairing (P / R / F1; label-centroid criterion, struct IoU ≥ 0.5)", "",
              "| detector | split | Hungarian | LPS (conf pinned) | Relational |", "|---|---|---|---|---|"]
    for tag in args.tags:
        for split in ("test", "val"):
            for suffix in ("", "_conf0.5", "_scale0.48", "_scale0.5"):
                r = e2e_row(tag, split, suffix)
                if r:
                    lines.append(r.replace(f"| {tag}_conf0.5 |", f"| {tag} @ conf 0.5 (val-tuned) |"))
            for suffix, label in (("_relnew", "recalibrated relational @0.3"), ("_relnew_c0.5", "recalibrated relational @0.5")):
                r = _load(f"{tag}_e2e_{split}{suffix}")
                if r:
                    m = r["matchers"]["Relational"]
                    lines.append(f"| {tag} + {label} | {split} | — | — | {m['P']:.3f} / {m['R']:.3f} / **{m['F1']:.3f}** |")
    # conf sweep on val
    sweep = []
    for tag in args.tags:
        for c in ("0.1", "0.2", "0.25", "0.3", "0.35", "0.4", "0.5"):
            r = _load(f"{tag}_e2e_val_conf{c}")
            if r:
                m = r["matchers"]
                sweep.append(f"| {tag} | {c} | {m['Hungarian']['F1']:.3f} | {m['LPS']['F1']:.3f} | {m['Relational']['F1']:.3f} |")
    if sweep:
        lines += ["", "### Operating-point sweep on real VAL (e2e F1)", "", "| detector | conf | Hungarian | LPS | Relational |", "|---|---|---|---|---|", *sweep]
    ms = []
    for m_ in ("0.0", "1.0", "2.0", "3.0", "4.0"):
        r = _load(f"dfine_l_plus_e2e_val_relnew_m{m_}")
        if r:
            ms.append(f"| {m_} | {r['matchers']['Relational']['P']:.3f} | {r['matchers']['Relational']['R']:.3f} | {r['matchers']['Relational']['F1']:.3f} |")
    if ms:
        lines += ["", "### Dustbin-margin sweep on real VAL (recalibrated relational, D-FINE detections)", "", "| margin | P | R | F1 |", "|---|---|---|---|", *ms]
    ra = _load("render_agreement")
    if ra:
        lines += ["", "### PyMuPDF → pypdfium2 render agreement (same detector on both renders)", "", "| dpi | pages | agreement F1 | pages with identical counts | median |Δconf| |", "|---|---|---|---|---|"]
        for dpi, v in ra["by_dpi"].items():
            lines.append(f"| {dpi} | {v['pages']} | {v['agreement_f1']:.4f} | {v['pages_same_count']} | {v['median_abs_conf_diff']:.3f} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
