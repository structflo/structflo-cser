"""Standalone LPS pair-classification accuracy on any LPS data dir.

Reports per-candidate binary-classification accuracy at a 0.5 threshold — the metric
used for the paper's "LPS pair-classification accuracy" row (Table 3). Works on any
directory containing ``ground_truth/`` + ``images/`` (real_test, synthetic test, etc.).

Usage:
    uv run python scripts/repro/eval_lps_acc.py --weights runs/repro/lps_ft_s42/best.pt \
        --data data/finetune/lps/real_test

num_workers=0 avoids spawn/import issues when launched in the background.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from structflo.cser.lps.dataset import LPSDataset, PageGroupSampler
from structflo.cser.lps.scorer import PairScorer


@torch.no_grad()
def evaluate(weights: Path, data_dir: Path) -> tuple[float, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LPSDataset(data_dir, neg_per_pos=3, bbox_jitter=0.0, augment=False, seed=42)
    model = PairScorer()
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    loader = DataLoader(
        ds,
        batch_size=512,
        sampler=PageGroupSampler(ds._path_idx, shuffle=False, seed=42),
        num_workers=0,
    )
    correct = n = 0
    for b in loader:
        logits = model(
            b["struct_crop"].to(device),
            b["label_crop"].to(device),
            b["geom"].to(device),
        )
        target = b["target"].to(device).unsqueeze(1)
        correct += ((logits.sigmoid() >= 0.5).float() == target).sum().item()
        n += target.size(0)
    return correct / max(n, 1), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    acc, n = evaluate(args.weights, args.data)
    print(
        f"RESULT weights={args.weights} data={args.data} acc={acc:.4f} n={n}",
        flush=True,
    )


if __name__ == "__main__":
    main()
