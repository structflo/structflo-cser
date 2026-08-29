"""Re-grade the saved OCR-gate records with improved compound-ID grammars.

No re-OCR — operates on the texts dumped by ocr_gate_test.py. Compares grammar
variants to show how much of the apparent low gate-recall is a grammar gap vs a
real limit, and inspects what each variant newly keeps/drops.

  v1   single token with letters+digits, len>=3 (the original)
  v2   v1 + multi-token join ("JNJ 4229") + multi-hyphen numeric codes ("542-1410-250-215")
  v3   v2 + long single alpha word as a drug/protein name (len>=8)

Usage:
    uv run python scripts/repro/ocr_gate_regrade.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path("runs/repro/ocr_gate")
STOP = {
    "ic50", "ic90", "ec50", "ec90", "mic", "mic50", "mic90", "mic99", "ki", "kd",
    "logp", "clogp", "logd", "mw", "pic50", "gi50", "cc50", "ld50", "tm", "cmax",
    "auc", "ph", "hg2", "hepg2", "mlm", "hlm", "h2o", "co2", "p450", "cyp3a4",
    "nadh", "nadph", "atp", "adp", "gtp", "coa", "dsf", "mps", "ppb", "ppant",
}


def _tok(text):
    return [t.strip(".,;:()[]{}").lstrip("=<>~±") for t in re.split(r"\s+", text.strip())]


def _alnum_code(core):
    has_a = any(c.isalpha() for c in core)
    has_d = any(c.isdigit() for c in core)
    return has_a and has_d and len(core) >= 3


def v1(text):
    if not text:
        return False
    for core in _tok(text):
        low = core.lower()
        if not core or low in STOP or re.fullmatch(r"[\d.,%±<>=~+/x*-]+", core):
            continue
        if _alnum_code(core):
            return True
    return False


def v2(text):
    if v1(text):
        return True
    if not text:
        return False
    toks = [t for t in _tok(text) if t]
    # multi-token: an alpha prefix followed by a numeric token  ("JNJ 4229", "CK 263")
    for a, b in zip(toks, toks[1:]):
        if a.lower() in STOP:
            continue
        if re.fullmatch(r"[A-Za-z]{1,5}", a) and re.fullmatch(r"\d{2,}[A-Za-z]?", b):
            return True
    # multi-hyphen numeric catalog code  ("542-1410-250-215")
    for core in toks:
        if core.count("-") >= 2 and re.fullmatch(r"[\d-]+", core) and len(core) >= 7:
            return True
    return False


def v3(text):
    if v2(text):
        return True
    if not text:
        return False
    for core in _tok(text):
        if core.lower() not in STOP and re.fullmatch(r"[A-Za-z]{8,}", core):
            return True
    return False


GRAMMARS = {"v1": v1, "v2": v2, "v3": v3}


def main():
    for split in ["test", "val"]:
        p = OUT / f"{split}_records.json"
        if not p.exists():
            print(f"(skip {split}: no records)")
            continue
        rec = json.loads(p.read_text())
        print(f"\n================  {split}  ================")
        tp, fp = rec["RECOVER_TP"], rec["RECOVER_FP"]
        hifp = rec["HICONF_FP"]
        print(f"{'grammar':>7} | {'TP-keep (recall)':>18} | {'FP-keep':>10} | {'sep':>6} | {'HICONF_FP-keep':>14}")
        for name, g in GRAMMARS.items():
            tpr = sum(g(r["text"]) for r in tp) / len(tp) if tp else 0
            fpr = sum(g(r["text"]) for r in fp) / len(fp) if fp else 0
            hfr = sum(g(r["text"]) for r in hifp) / len(hifp) if hifp else 0
            print(f"{name:>7} | {tpr:>17.0%} | {fpr:>9.0%} | {tpr - fpr:>+5.0%} | {hfr:>13.0%}")

        # what v2 newly recovers in TP (over v1)
        new_tp = [r["text"] for r in tp if v2(r["text"]) and not v1(r["text"])]
        print(f"\nv2 newly-kept RECOVER_TP (real labels recovered): {new_tp}")
        # TP still missed by v3 (the residual)
        miss = [r["text"] for r in tp if not v3(r["text"])]
        print(f"v3 STILL-missed RECOVER_TP (true residual): {miss}")
        # FP that v2 keeps — are they real labels or junk?
        fp_keep = [r["text"] for r in fp if v2(r["text"])]
        fp_drop = [r["text"] for r in fp if not v2(r["text"])]
        print(f"\nv2 RECOVER_FP KEPT ({len(fp_keep)}): {fp_keep}")
        print(f"v2 RECOVER_FP DROPPED ({len(fp_drop)}): {fp_drop}")


if __name__ == "__main__":
    main()
