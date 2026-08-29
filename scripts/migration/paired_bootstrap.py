"""Paired page-level bootstrap of the end-to-end pairing metric between two e2e runs.

Inputs are two JSONs written by ``e2e_from_preds.py --out`` (they carry ``per_page``) over the
same split, e.g. two detectors' predictions scored by the same matchers. For each matcher the
pages are resampled with replacement — the SAME page draw is applied to A and B, so the
difference is paired — and the metric is recomputed from the resampled tp/npred/GT-pair sums
(the micro-averaged quantity the aggregate F1 is built from, not a mean of per-page scores).
Reported per matcher: point estimates, delta = B - A, percentile 95% CI of the delta and
P(delta < 0), the bootstrap fraction in which B is worse than A.

Usage:
    uv run python scripts/migration/paired_bootstrap.py \\
        --a runs/license_migration/eval/dfine_l_plus_ms_conf0.4_e2e_test.json \\
        --b runs/license_migration/eval/yolo_v0.4_e2e_test.json \\
        [--metric f1|precision|recall] [--n-boot 10000] [--seed 0] [--out delta.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRIC_LABEL = {"f1": "F1", "precision": "P", "recall": "R"}


def _metric(
    tp: np.ndarray, npred: np.ndarray, gt: np.ndarray, metric: str
) -> np.ndarray:
    """Same definitions as e2e_from_preds.py: P = tp/max(npred,1), R = tp/max(gt,1)."""
    tp = np.asarray(tp, dtype=float)
    P = tp / np.maximum(npred, 1)
    R = tp / np.maximum(gt, 1)
    if metric == "precision":
        return P
    if metric == "recall":
        return R
    denom = P + R
    return np.divide(2 * P * R, denom, out=np.zeros_like(denom), where=denom > 0)


def _load(path: Path) -> dict:
    d = json.loads(path.read_text())
    if "per_page" not in d:
        raise SystemExit(
            f"{path}: no 'per_page' key — regenerate with e2e_from_preds.py --out"
        )
    return d


def paired_bootstrap(
    a: dict, b: dict, metric: str = "f1", n_boot: int = 10_000, seed: int = 0
) -> dict:
    """Return {matcher: {A, B, delta, ci95, p_delta_lt_0, boot_mean, boot_std}} plus meta."""
    if a["split"] != b["split"]:
        raise SystemExit(f"split mismatch: {a['split']} vs {b['split']}")
    stems = sorted(a["per_page"])
    if stems != sorted(b["per_page"]):
        only_a = set(a["per_page"]) - set(b["per_page"])
        only_b = set(b["per_page"]) - set(a["per_page"])
        raise SystemExit(
            f"page sets differ ({len(only_a)} only in A, {len(only_b)} only in B); "
            "both runs must score the same split"
        )
    matchers = [m for m in a["matchers"] if m in b["matchers"]]
    if not matchers:
        raise SystemExit("no matcher common to both files")

    gt_a = np.array([a["per_page"][s]["gt_pairs"] for s in stems], dtype=np.int64)
    gt_b = np.array([b["per_page"][s]["gt_pairs"] for s in stems], dtype=np.int64)
    if not np.array_equal(gt_a, gt_b):
        raise SystemExit(
            "per-page GT pair counts differ between A and B (different GT?)"
        )

    n = len(stems)
    rng = np.random.default_rng(seed)
    # counts[i, j] = how many times page j was drawn in resample i; shared by A and B.
    idx = rng.integers(0, n, size=(n_boot, n))
    counts = np.zeros((n_boot, n), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1)
    gt_boot = counts @ gt_a

    out: dict = {}
    for m in matchers:
        tp_a = np.array([a["per_page"][s][m]["tp"] for s in stems], dtype=np.int64)
        np_a = np.array([a["per_page"][s][m]["npred"] for s in stems], dtype=np.int64)
        tp_b = np.array([b["per_page"][s][m]["tp"] for s in stems], dtype=np.int64)
        np_b = np.array([b["per_page"][s][m]["npred"] for s in stems], dtype=np.int64)

        m_a = float(_metric(tp_a.sum(), np_a.sum(), gt_a.sum(), metric))
        m_b = float(_metric(tp_b.sum(), np_b.sum(), gt_b.sum(), metric))
        boot_a = _metric(counts @ tp_a, counts @ np_a, gt_boot, metric)
        boot_b = _metric(counts @ tp_b, counts @ np_b, gt_boot, metric)
        delta = boot_b - boot_a
        lo, hi = np.percentile(delta, [2.5, 97.5])
        out[m] = {
            "A": m_a,
            "B": m_b,
            "delta": m_b - m_a,
            "ci95": [float(lo), float(hi)],
            "p_delta_lt_0": float(np.mean(delta < 0)),
            "boot_mean": float(delta.mean()),
            "boot_std": float(delta.std(ddof=1)),
        }
    return {
        "split": a["split"],
        "metric": metric,
        "n_pages": n,
        "n_boot": n_boot,
        "seed": seed,
        "matchers": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--a", type=Path, required=True, help="baseline e2e JSON (with per_page)"
    )
    ap.add_argument(
        "--b", type=Path, required=True, help="comparison e2e JSON (with per_page)"
    )
    ap.add_argument("--metric", default="f1", choices=sorted(METRIC_LABEL))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="write the result as JSON")
    args = ap.parse_args()

    a, b = _load(args.a), _load(args.b)
    res = paired_bootstrap(a, b, metric=args.metric, n_boot=args.n_boot, seed=args.seed)
    res = {"a": str(args.a), "b": str(args.b), **res}

    lab = METRIC_LABEL[args.metric]
    print(f"A = {args.a}\nB = {args.b}")
    print(
        f"[{res['split']}: {res['n_pages']} pages; paired page bootstrap, "
        f"{res['n_boot']} resamples, seed {res['seed']}; delta = B - A]"
    )
    print(
        f"  {'matcher':>11} | {lab + '_A':>7} {lab + '_B':>7} {'Δ' + lab:>8} | "
        f"{'95% CI [lo, hi]':>18} | {'P(Δ<0)':>7}"
    )
    for m, r in res["matchers"].items():
        lo, hi = r["ci95"]
        print(
            f"  {m:>11} | {r['A']:7.3f} {r['B']:7.3f} {r['delta']:+8.3f} | "
            f"[{lo:+.3f}, {hi:+.3f}] | {r['p_delta_lt_0']:7.3f}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
