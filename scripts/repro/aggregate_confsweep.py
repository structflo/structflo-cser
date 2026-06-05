"""Aggregate the VAL label-conf sweep -> e2e pairing F1 (and P/R) vs label-conf, per matcher.

Reads runs/repro/logs/confsweep/eval_val_s{seed}_c{conf}.txt (the [VAL] Part-B block),
prints a mean +/- std table + each matcher's peak-F1 conf, and plots F1-vs-label-conf per
matcher (the robustness curve). The operating point = the conf maximising mean e2e F1.

Usage:
    uv run python scripts/repro/aggregate_confsweep.py
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

REPRO = Path("runs/repro")
SLOG = REPRO / "logs" / "confsweep"
CONFS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70]
SEEDS = [42, 43, 44]
MATCHERS = ("Hungarian", "LPS", "Relational")


def partB_val(path: Path) -> dict:
    out: dict = {}
    if not path.exists():
        return out
    partB = False
    split = None
    for ln in path.read_text().splitlines():
        if "PART B" in ln:
            partB = True
        m = re.search(r"\[(TEST|VAL|TRAIN|ALL)\]", ln)
        if m:
            split = m.group(1)
        rm = re.match(r"\s*(Hungarian|LPS|Relational)\s*\|\s*(.+)", ln)
        if rm and partB and split == "VAL":
            nums = [float(x) for x in rm.group(2).split()]
            if len(nums) == 3:
                out[rm.group(1)] = nums
    return out


def main() -> None:
    # data[conf][matcher] = list of (P,R,F1) across seeds
    data = {c: {m: [] for m in MATCHERS} for c in CONFS}
    for s in SEEDS:
        for c in CONFS:
            b = partB_val(SLOG / f"eval_val_s{s}_c{c:.2f}.txt")
            for m in MATCHERS:
                if m in b:
                    data[c][m].append(b[m])

    def stat(c, m, i):
        vals = [v[i] for v in data[c][m]]
        if not vals:
            return None, 0.0
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    # ---- table ----
    md = [
        "# Label-conf sweep — real VAL — e2e pairing F1 (mean ± std over seeds 42-44)\n",
        "Structures fixed at conf 0.30; label conf varied. Relational = present matched weights "
        "(conf-0.3-trained → lower bound at off-0.3 confs).\n",
        "| label conf | Hungarian F1 | LPS F1 | Relational F1 |",
        "|---|---|---|---|",
    ]
    for c in CONFS:
        cells = []
        for m in MATCHERS:
            mean, sd = stat(c, m, 2)
            cells.append(f"{mean:.3f} ± {sd:.3f}" if mean is not None else "—")
        md.append(f"| {c:.2f} | {cells[0]} | {cells[1]} | {cells[2]} |")
    # best conf per matcher (max mean F1)
    md.append("")
    for m in MATCHERS:
        best = max(CONFS, key=lambda c: stat(c, m, 2)[0] or -1)
        bm, _ = stat(best, m, 2)
        md.append(f"- **{m}** peaks at label-conf **{best:.2f}** (F1 {bm:.3f})")
    out_md = REPRO / "CONFSWEEP_VAL.md"
    out_md.write_text("\n".join(md))

    print("\n".join(md))

    # ---- plot ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.6))
        for m in MATCHERS:
            xs, ys, es = [], [], []
            for c in CONFS:
                mean, sd = stat(c, m, 2)
                if mean is not None:
                    xs.append(c)
                    ys.append(mean)
                    es.append(sd)
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=2, label=m)
        ax.set_xlabel("label confidence threshold")
        ax.set_ylabel("end-to-end pairing F1")
        ax.set_title("Matcher robustness vs label confidence (real val, mean ± s.d.)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(REPRO / "confsweep_val.png", dpi=150)
        print(f"\nplot: {REPRO / 'confsweep_val.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"\nplot skipped ({e})")


if __name__ == "__main__":
    main()
