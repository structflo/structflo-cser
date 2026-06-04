"""Aggregate the real-data scaling curve into mean ± std vs. number of real docs.

Parses runs/repro/logs/scale/point_n{N}_s{S}_{detval,lpsacc,e2e} for every (N, seed)
and writes:
  * runs/repro/SCALE_SUMMARY.md   — table: N docs → detector mAP50 / LPS acc / e2e F1
  * runs/repro/scale_per_seed.json
  * runs/repro/scale_curve.png    — 3-panel learning curve with mean ± std error bars

Safe to run on partial data. e2e F1 is the parameter-free Hungarian matcher on each
point's fine-tuned detector (isolates detector scaling).
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

REPRO = Path("runs/repro")
SLOG = REPRO / "logs" / "scale"


def _detval(path: Path) -> float | None:
    if not path.exists():
        return None
    m = re.search(r"DETVAL mAP50=([0-9.]+)", path.read_text())
    return float(m.group(1)) if m else None


def _lpsacc(path: Path) -> float | None:
    if not path.exists():
        return None
    m = re.search(r"acc=([0-9.]+)", path.read_text())
    return float(m.group(1)) if m else None


def _e2e_hungarian_f1(path: Path) -> float | None:
    """F1 of the Hungarian row in the PART B [TEST] block."""
    if not path.exists():
        return None
    part = split = None
    for line in path.read_text().splitlines():
        if "PART B" in line:
            part = "B"
        elif "PART A" in line:
            part = "A"
        m = re.search(r"\[(TEST|VAL|TRAIN|ALL)\]", line)
        if m:
            split = m.group(1)
        rm = re.match(r"\s*Hungarian\s*\|\s*(.+)", line)
        if rm and part == "B" and split == "TEST":
            nums = [float(x) for x in rm.group(1).split()]
            if len(nums) == 3:
                return nums[2]
    return None


def _cell(vals: list[float], dec: int = 3) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    mean = statistics.mean(vals)
    if len(vals) < 2:
        return f"{mean:.{dec}f} (n=1)"
    return f"{mean:.{dec}f} ± {statistics.stdev(vals):.{dec}f}"


def _stats(vals: list[float]):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0.0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main() -> None:
    # discover N values and seeds from filenames
    ns, seeds = set(), set()
    for f in SLOG.glob("point_n*_s*_detval.log"):
        m = re.search(r"point_n(\d+)_s(\d+)_detval", f.name)
        if m:
            ns.add(int(m.group(1)))
            seeds.add(int(m.group(2)))
    ns = sorted(ns) or [0, 50, 100, 200, 400, 830]
    seeds = sorted(seeds) or [42, 43, 44]

    per: dict = {}
    for N in ns:
        per[N] = {}
        for S in seeds:
            base = SLOG / f"point_n{N}_s{S}"
            per[N][S] = {
                "map50": _detval(Path(f"{base}_detval.log")),
                "lps_acc": _lpsacc(Path(f"{base}_lpsacc.log")),
                "e2e_f1": _e2e_hungarian_f1(Path(f"{base}_e2e.txt")),
            }

    # ---- markdown table ----
    md = [
        f"# Real-data scaling curve — mean ± std over seeds {seeds}\n",
        "Fine-tune on N real documents (+2000 synthetic, real ×2), evaluate on the frozen "
        "100-page real test set @1280. N=0 is the synthetic-only baseline; N=830 the full "
        "fine-tune. e2e F1 uses the parameter-free Hungarian matcher on each point's detector.\n",
        "| Real docs (N) | Detector mAP@50 | LPS pair-class acc | End-to-end F1 (Hungarian) |",
        "|---|---|---|---|",
    ]
    for N in ns:
        rows = [per[N][S] for S in seeds]
        md.append(
            f"| {N} | {_cell([r['map50'] for r in rows])} | "
            f"{_cell([r['lps_acc'] for r in rows])} | {_cell([r['e2e_f1'] for r in rows])} |"
        )
    md.append(f"\nSeeds: {seeds}. Raw per-seed values in `scale_per_seed.json`.\n")
    (REPRO / "SCALE_SUMMARY.md").write_text("\n".join(md))
    (REPRO / "scale_per_seed.json").write_text(json.dumps(per, indent=2))

    # ---- plot ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = [
            ("map50", "Detector mAP@50"),
            ("lps_acc", "LPS pair-class acc"),
            ("e2e_f1", "End-to-end F1 (Hungarian)"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, (key, title) in zip(axes, metrics):
            xs, ys, es = [], [], []
            for N in ns:
                mean, sd = _stats([per[N][S][key] for S in seeds])
                if mean is not None:
                    xs.append(N)
                    ys.append(mean)
                    es.append(sd)
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, lw=2)
            ax.set_title(title)
            ax.set_xlabel("# real training documents")
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            f"Real-data scaling (mean ± s.d. over seeds {seeds}; frozen 100-page real test)"
        )
        fig.tight_layout()
        fig.savefig(REPRO / "scale_curve.png", dpi=150)
        plot_msg = f"plot: {REPRO / 'scale_curve.png'}"
    except Exception as e:  # noqa: BLE001
        plot_msg = f"plot skipped ({e})"

    print(f"wrote {REPRO / 'SCALE_SUMMARY.md'} (N={ns}, seeds={seeds}); {plot_msg}")


if __name__ == "__main__":
    main()
