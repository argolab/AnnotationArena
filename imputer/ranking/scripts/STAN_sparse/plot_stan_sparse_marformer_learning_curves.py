#!/usr/bin/env python3
"""
Ad-hoc learning curves for STAN_sparse Marformer runs (non-hard-mask by default).

Reads training_history.json under RESULTS/MARFORMER/STAN_sparse/<run_name>/ and writes a PNG.

Example:
  cd imputer/ranking
  python scripts/STAN_sparse/plot_stan_sparse_marformer_learning_curves.py
  python scripts/STAN_sparse/plot_stan_sparse_marformer_learning_curves.py \\
    --runs Factor_225_25_9_ItemTest_Size_175 Normal_225_25_9_ItemTest_Size_175 \\
    --output RESULTS/MARFORMER/STAN_sparse/plots/learning_curves_soft_mask.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _ranking_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_history(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "training_history.json"
    with path.open() as f:
        return json.load(f)


def series_total_loss(history: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    e, y = [], []
    for row in history:
        if "epoch" in row and "total_loss" in row:
            e.append(int(row["epoch"]))
            y.append(float(row["total_loss"]))
    return e, y


def series_val_missing(
    history: list[dict[str, Any]], key: str
) -> tuple[list[int], list[float]]:
    e, y = [], []
    for row in history:
        ve = row.get("val_eval") or {}
        m = (ve.get("metrics") or {}).get("missing") or {}
        r = m.get("rating") or {}
        if key not in r:
            continue
        e.append(int(row["epoch"]))
        y.append(float(r[key]))
    return e, y


def main() -> None:
    root = _ranking_root()
    default_runs = [
        "Factor_225_25_9_ItemTest_Size_175",
        "Normal_225_25_9_ItemTest_Size_175",
    ]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--results-root",
        type=Path,
        default=root / "RESULTS" / "MARFORMER" / "STAN_sparse",
        help="Directory containing run subfolders with training_history.json",
    )
    p.add_argument("--runs", nargs="+", default=default_runs, help="Run folder names under results-root")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: <results-root>/plots/learning_curves_<suffix>.png)",
    )
    p.add_argument(
        "--suffix",
        default="soft_mask",
        help="Used in default output filename if --output not set",
    )
    args = p.parse_args()
    out = args.output
    if out is None:
        out = args.results_root / "plots" / f"learning_curves_{args.suffix}.png"

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    for name in args.runs:
        run_dir = args.results_root / name
        if not (run_dir / "training_history.json").exists():
            raise SystemExit(f"Missing {run_dir / 'training_history.json'}")
        h = load_history(run_dir)
        label = "Factor" if name.startswith("Factor") else "Normal"

        ep, loss = series_total_loss(h)
        if ep:
            axes[0].plot(ep, loss, label=label, linewidth=1.5)

        ep, xent = series_val_missing(h, "xent")
        if ep:
            axes[1].plot(ep, xent, label=label, linewidth=1.5)

        ep, acc = series_val_missing(h, "acc")
        if ep:
            axes[2].plot(ep, acc, label=label, linewidth=1.5)

    axes[0].set_title("Train loss (weighted objective)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("total_loss")
    # axes[0].set_ylim(bottom=0.0, top=3.0)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].set_title("Val — missing rating CE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("cross-entropy")
    axes[1].set_ylim(bottom=0.8, top=1.7)
    axes[1].grid(True, linestyle="--", alpha=0.35)

    axes[2].set_title("Val — missing rating accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("accuracy")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, linestyle="--", alpha=0.35)

    for ax in axes:
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("STAN_sparse Marformer (soft mask)", fontsize=11, y=1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
