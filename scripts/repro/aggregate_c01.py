"""Compare the relational matcher trained @ label-conf 0.1 vs @ 0.3, across inference conf (val).

Tests the "train on more false positives -> better low-conf robustness" hypothesis. Reads the
0.1-trained sweep (logs/c01sweep) and the 0.3-trained sweep (logs/confsweep), plus Hungarian as
a parameter-free reference, and writes a table + overlay plot of e2e F1 vs label-conf.

Usage:
    uv run python scripts/repro/aggregate_c01.py
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

REPRO = Path("runs/repro")
C01 = REPRO / "logs" / "c01sweep"
C03 = REPRO / "logs" / "confsweep"
CONFS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70]
SEEDS = [42, 43, 44]


def partB_val(path: Path, matcher: str):
    if not path.exists():
        return None
    pB = False
    sp = None
    for ln in path.read_text().splitlines():
        if "PART B" in ln:
            pB = True
        m = re.search(r"\[(TEST|VAL|TRAIN|ALL)\]", ln)
        if m:
            sp = m.group(1)
        rm = re.match(rf"\s*{matcher}\s*\|\s*(.+)", ln)
        if rm and pB and sp == "VAL":
            nums = [float(x) for x in rm.group(1).split()]
            if len(nums) == 3:
                return nums[2]  # F1
    return None


def series(logdir: Path, matcher: str):
    out = {}
    for c in CONFS:
        vals = [
            partB_val(logdir / f"eval_val_s{s}_c{c:.2f}.txt", matcher) for s in SEEDS
        ]
        vals = [v for v in vals if v is not None]
        out[c] = (
            (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
            if vals
            else (None, 0.0)
        )
    return out


def main() -> None:
    rel01 = series(C01, "Relational")  # trained @0.1
    rel03 = series(C03, "Relational")  # trained @0.3
    hung = series(C03, "Hungarian")  # reference (same in both)

    md = [
        "# Relational trained @0.1 vs @0.3 — e2e F1 vs inference label-conf (val, mean ± std)\n",
        "| label conf | Relational @0.1-trained | Relational @0.3-trained | Hungarian (ref) |",
        "|---|---|---|---|",
    ]

    def cell(d, c):
        m, sd = d[c]
        return f"{m:.3f} ± {sd:.3f}" if m is not None else "—"

    for c in CONFS:
        md.append(
            f"| {c:.2f} | {cell(rel01, c)} | {cell(rel03, c)} | {cell(hung, c)} |"
        )
    # low-conf verdict
    md.append("\nΔ (0.1-trained − 0.3-trained), low conf:")
    for c in (0.05, 0.10, 0.20):
        a, b = rel01[c][0], rel03[c][0]
        if a is not None and b is not None:
            md.append(f"- conf {c:.2f}: {a - b:+.3f}")
    (REPRO / "C01_COMPARE.md").write_text("\n".join(md))
    print("\n".join(md))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.6))
        for d, lbl, style in [
            (rel01, "Relational @0.1-trained", "-o"),
            (rel03, "Relational @0.3-trained", "-s"),
            (hung, "Hungarian (ref)", "--^"),
        ]:
            xs = [c for c in CONFS if d[c][0] is not None]
            ys = [d[c][0] for c in xs]
            es = [d[c][1] for c in xs]
            ax.errorbar(xs, ys, yerr=es, fmt=style, capsize=3, lw=2, label=lbl)
        ax.set_xlabel("inference label confidence")
        ax.set_ylabel("end-to-end pairing F1")
        ax.set_title("Relational train-conf: 0.1 vs 0.3 across inference conf (val)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(REPRO / "c01_compare.png", dpi=150)
        print(f"\nplot: {REPRO / 'c01_compare.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"\nplot skipped ({e})")


if __name__ == "__main__":
    main()
