"""Aggregate per-seed reproducibility evals into mean ± std tables.

Parses the outputs written by ``run_eval.sh`` for every seed found under
``runs/repro/logs/eval/`` and writes:
  * ``runs/repro/SUMMARY.md``    — human-readable Tables 1-5 + LPS rows (mean ± std)
  * ``runs/repro/per_seed.json`` — machine-readable raw values per seed

Runs on partial data (reports only seeds that have completed evals), so it is safe to
call while the sweep is still in flight. std is sample std (ddof=1) over the available
seeds; with a single seed it is shown as "—".
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

REPRO = Path("runs/repro")
ELOG = REPRO / "logs" / "eval"
MATCHERS = ("Hungarian", "LPS", "Relational")
ROW_RE = re.compile(r"^\s*(Hungarian|LPS|Relational)\s*\|\s*(.+?)\s*$")


# ---------------------------------------------------------------- parsers
def parse_eval_compare(path: Path) -> dict | None:
    """{'A': {split: {matcher: [assign,reject,prec]}}, 'B': {split: {matcher:[P,R,F1]}}}.

    Part A values are percentages (e.g. 99.6); Part B values are floats (e.g. 0.816).
    """
    if not path.exists():
        return None
    part = None
    split = None
    out: dict[str, dict] = {"A": {}, "B": {}}
    for line in path.read_text().splitlines():
        if "PART A" in line:
            part = "A"
            continue
        if "PART B" in line:
            part = "B"
            continue
        m = re.search(r"\[(TEST|VAL|TRAIN|ALL)\]", line)
        if m:
            split = m.group(1)
            continue
        rm = ROW_RE.match(line)
        if rm and part and split:
            name = rm.group(1)
            nums = [float(x.replace("%", "")) for x in rm.group(2).split()]
            if len(nums) == 3:
                out[part].setdefault(split, {})[name] = nums
    return out


def parse_lps_acc(path: Path) -> float | None:
    if not path.exists():
        return None
    m = re.search(r"acc=([0-9.]+)", path.read_text())
    return float(m.group(1)) if m else None


def load_detector(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------- aggregation helpers
def cell(values: list[float], dec: int = 3) -> str:
    """Format a list of per-seed values as 'mean ± std' (std '—' for n<2)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "—"
    mean = statistics.mean(vals)
    if len(vals) < 2:
        return f"{mean:.{dec}f} (n=1)"
    return f"{mean:.{dec}f} ± {statistics.stdev(vals):.{dec}f}"


def collect(seeds: list[int]) -> dict:
    per = {}
    for s in seeds:
        per[s] = {
            "synth_match": parse_eval_compare(ELOG / f"eval_synth_s{s}.txt"),
            "real_match": parse_eval_compare(ELOG / f"eval_real_s{s}.txt"),
            "real_gtabl": parse_eval_compare(ELOG / f"eval_real_gtabl_s{s}.txt"),
            "lps_acc": {
                "baseline_real": parse_lps_acc(
                    ELOG / f"lps_acc_baseline_real_s{s}.log"
                ),
                "finetuned_real": parse_lps_acc(
                    ELOG / f"lps_acc_finetuned_real_s{s}.log"
                ),
                "baseline_synth": parse_lps_acc(
                    ELOG / f"lps_acc_baseline_synth_s{s}.log"
                ),
                "finetuned_synth": parse_lps_acc(
                    ELOG / f"lps_acc_finetuned_synth_s{s}.log"
                ),
            },
            "detector": load_detector(ELOG / f"detector_s{s}.json"),
        }
    return per


def _series(per, seeds, getter):
    """Collect getter(per[s]) across seeds, dropping None / KeyErrors."""
    out = []
    for s in seeds:
        try:
            v = getter(per[s])
        except (KeyError, TypeError):
            v = None
        if v is not None:
            out.append(v)
    return out


# ---------------------------------------------------------------- table builders
def det_block(per, seeds, table_key, sub, cls):
    """Per-class detector metric series across seeds -> dict metric -> formatted cell."""
    metrics = ["P", "R", "mAP50", "mAP50-95"]
    row = {}
    for met in metrics:

        def get(p, met=met):
            d = p["detector"][table_key]
            d = d[sub] if sub else d
            return d[cls][met]

        row[met] = cell(_series(per, seeds, get))
    return row


def match_block(per, seeds, key, part, split, idx):
    """matcher -> formatted cell for metric `idx` of part A/B at `split`."""
    pct = part == "A"
    return {
        mtr: cell(
            _series(per, seeds, lambda p, mtr=mtr: p[key][part][split][mtr][idx]),
            dec=1 if pct else 3,
        )
        for mtr in MATCHERS
    }


def main() -> None:
    seeds = sorted(
        int(m.group(1))
        for f in ELOG.glob("eval_*_s*.txt")
        if (m := re.search(r"_s(\d+)\.txt$", f.name))
    )
    seeds = sorted(set(seeds)) or [42, 43, 44]
    per = collect(seeds)
    md = []
    A = md.append
    A(f"# Reproducibility SUMMARY — mean ± std over seeds {seeds}\n")
    A(
        "All eval @ imgsz 1280 (deployment resolution). §A = synthetic-only weights on the "
        "held-out synthetic TEST set (1000 pages); §B = real-fine-tuned weights on the real "
        "TEST set (100 pages). Splits fixed across seeds. std = sample std (ddof=1).\n"
    )

    # ---- Table 1: synthetic detection (base detector, synth TEST) ----
    A("\n## Table 1 — Synthetic detection (base detector, synthetic TEST) [§A]\n")
    A("| class | P | R | mAP50 | mAP50-95 |")
    A("|---|---|---|---|---|")
    for cls in ("chemical_structure", "compound_label", "all"):
        r = det_block(per, seeds, "table1_synth", None, cls)
        A(f"| {cls} | {r['P']} | {r['R']} | {r['mAP50']} | {r['mAP50-95']} |")

    # ---- Table 2: synthetic matching (§A — descriptive, no conclusions) ----
    A("\n## Table 2 — Synthetic matching (synthetic-only weights, synth TEST) [§A]\n")
    A(
        "Relational here = relmatch_synth (GT-box, synthetic-only). §A draws no matcher "
        "conclusions (matchers saturate in-distribution).\n"
    )
    A("| matcher | assign | reject | prec | e2e P | e2e R | e2e F1 |")
    A("|---|---|---|---|---|---|---|")
    pa = {i: match_block(per, seeds, "synth_match", "A", "TEST", i) for i in range(3)}
    pb = {i: match_block(per, seeds, "synth_match", "B", "TEST", i) for i in range(3)}
    for mtr in MATCHERS:
        A(
            f"| {mtr} | {pa[0][mtr]} | {pa[1][mtr]} | {pa[2][mtr]} | "
            f"{pb[0][mtr]} | {pb[1][mtr]} | {pb[2][mtr]} |"
        )

    # ---- Table 3: real detector + LPS pair-classification ----
    A("\n## Table 3 — Generalization: detector & LPS, base → fine-tuned [§B]\n")
    A("**Detector** (real TEST):\n")
    A("| metric | base | fine-tuned |")
    A("|---|---|---|")
    for met in ("mAP50", "mAP50-95", "P", "R"):
        b = cell(
            _series(
                per,
                seeds,
                lambda p, met=met: p["detector"]["table3_real"]["base"]["all"][met],
            )
        )
        f = cell(
            _series(
                per,
                seeds,
                lambda p, met=met: p["detector"]["table3_real"]["ft"]["all"][met],
            )
        )
        A(f"| {met} | {b} | {f} |")
    A("\n**Detector synthetic regression** (synth TEST):\n")
    A("| metric | base | fine-tuned |")
    A("|---|---|---|")
    for met in ("mAP50", "mAP50-95"):
        b = cell(
            _series(
                per,
                seeds,
                lambda p, met=met: p["detector"]["table3_synth_regress"]["base"]["all"][
                    met
                ],
            )
        )
        f = cell(
            _series(
                per,
                seeds,
                lambda p, met=met: p["detector"]["table3_synth_regress"]["ft"]["all"][
                    met
                ],
            )
        )
        A(f"| {met} | {b} | {f} |")
    A("\n**LPS pair-classification accuracy** (acc @ 0.5):\n")
    A("| set | base | fine-tuned |")
    A("|---|---|---|")
    for label, bk, fk in (
        ("real TEST", "baseline_real", "finetuned_real"),
        ("synth TEST", "baseline_synth", "finetuned_synth"),
    ):
        b = cell(_series(per, seeds, lambda p, k=bk: p["lps_acc"][k]))
        f = cell(_series(per, seeds, lambda p, k=fk: p["lps_acc"][k]))
        A(f"| {label} | {b} | {f} |")

    # ---- Table 4: real clean matching (Part A) ----
    A("\n## Table 4 — Real clean (GT-box) matching [§B]\n")
    A("| matcher | assign | reject | prec |")
    A("|---|---|---|---|")
    ra = {i: match_block(per, seeds, "real_match", "A", "TEST", i) for i in range(3)}
    for mtr in MATCHERS:
        suffix = " (det-trained, published)" if mtr == "Relational" else ""
        A(f"| {mtr}{suffix} | {ra[0][mtr]} | {ra[1][mtr]} | {ra[2][mtr]} |")
    # GT-trained relmatch ablation (Part A Relational row from the ablation eval)
    ga = {
        i: match_block(per, seeds, "real_gtabl", "A", "TEST", i)["Relational"]
        for i in range(3)
    }
    A(f"| Relational (GT-trained, ablation) | {ga[0]} | {ga[1]} | {ga[2]} |")

    # ---- Table 5: real end-to-end (Part B) ----
    A("\n## Table 5 — Real end-to-end pairing F1 [§B]\n")
    A("| matcher | P | R | F1 |")
    A("|---|---|---|---|")
    rb = {i: match_block(per, seeds, "real_match", "B", "TEST", i) for i in range(3)}
    for mtr in MATCHERS:
        A(f"| {mtr} | {rb[0][mtr]} | {rb[1][mtr]} | {rb[2][mtr]} |")

    A(f"\n---\nSeeds aggregated: {seeds}. Per-seed raw values in `per_seed.json`.\n")

    (REPRO / "SUMMARY.md").write_text("\n".join(md))
    (REPRO / "per_seed.json").write_text(json.dumps(per, indent=2))
    print(f"wrote {REPRO / 'SUMMARY.md'} and {REPRO / 'per_seed.json'} (seeds={seeds})")


if __name__ == "__main__":
    main()
