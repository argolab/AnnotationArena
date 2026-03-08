#!/usr/bin/env python3
"""
Plot learning curves for Imputer ablation runs.

Produces:
- train_loss_curves.png: training rating loss vs epoch
- test_missing_xent_curves.png: test-missing rating loss vs epoch
- test_missing_acc_curves.png: test-missing accuracy vs epoch (optional)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def discover_runs(output_root: Path, run_prefix: str) -> List[Path]:
    """Find run directories matching prefix, sorted by name."""
    output_root = Path(output_root)
    if not output_root.exists():
        return []
    runs: List[Path] = []
    for d in output_root.iterdir():
        if d.is_dir() and d.name.startswith(run_prefix):
            runs.append(d)
    return sorted(runs, key=lambda p: p.name)


def load_training_loss_history(run_dir: Path) -> List[Dict[str, Any]]:
    """Load training_loss_history.json."""
    p = run_dir / "training_loss_history.json"
    if not p.exists():
        return []
    with p.open("r") as f:
        return json.load(f)


def load_test_instance_history(run_dir: Path) -> List[Dict[str, Any]]:
    """Load test_instance_training_history.json."""
    p = run_dir / "test_instance_training_history.json"
    if not p.exists():
        return []
    with p.open("r") as f:
        return json.load(f)


def extract_ablation_id(run_dir: Path, run_prefix: str) -> str:
    """Derive ablation ID from run dir name."""
    name = run_dir.name
    if name.startswith(run_prefix):
        suffix = name[len(run_prefix) :].lstrip("_")
        return suffix if suffix else "BASE"
    return name


# Color cycle for distinct lines
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_curves(
    runs: List[Path],
    run_prefix: str,
    plot_dir: Path,
) -> None:
    """Plot train loss, test-missing xent, and test-missing acc for all runs."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig_train, ax_train = plt.subplots(figsize=(8, 5))
    fig_test_xent, ax_test_xent = plt.subplots(figsize=(8, 5))
    fig_test_acc, ax_test_acc = plt.subplots(figsize=(8, 5))

    for idx, run_dir in enumerate(runs):
        ablation_id = extract_ablation_id(run_dir, run_prefix)
        color = COLORS[idx % len(COLORS)]

        train_hist = load_training_loss_history(run_dir)
        test_hist = load_test_instance_history(run_dir)

        # Training: epoch -> rating_loss
        if train_hist:
            epochs = [e.get("epoch", i) for i, e in enumerate(train_hist)]
            losses = [float(e.get("rating_loss", 0)) for e in train_hist]
            # Sort by epoch
            paired = sorted(zip(epochs, losses), key=lambda x: x[0])
            epochs, losses = [p[0] for p in paired], [p[1] for p in paired]
            ax_train.plot(epochs, losses, label=ablation_id, color=color, alpha=0.9)

        # Test-missing: epoch -> rating_loss, rating_accuracy
        if test_hist:
            test_epochs = []
            test_xents = []
            test_accs = []
            for e in sorted(test_hist, key=lambda x: x.get("epoch", 0)):
                mm = e.get("missing_metrics") or {}
                xent = mm.get("rating_loss")
                acc = mm.get("rating_accuracy")
                test_epochs.append(e.get("epoch", len(test_epochs)))
                if xent is not None:
                    test_xents.append(float(xent))
                else:
                    test_xents.append(np.nan)
                if acc is not None:
                    test_accs.append(float(acc))
                else:
                    test_accs.append(np.nan)
            if test_epochs:
                ax_test_xent.plot(test_epochs, test_xents, label=ablation_id, color=color, alpha=0.9)
                ax_test_acc.plot(test_epochs, test_accs, label=ablation_id, color=color, alpha=0.9)

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Training rating loss (xent)")
    ax_train.set_title("Training loss curves (ablation sweep)")
    ax_train.legend(loc="best", fontsize=8)
    ax_train.grid(True, alpha=0.3)
    fig_train.tight_layout()
    fig_train.savefig(plot_dir / "train_loss_curves.png", dpi=150)
    plt.close(fig_train)

    ax_test_xent.set_xlabel("Epoch")
    ax_test_xent.set_ylabel("Test-missing rating loss (xent)")
    ax_test_xent.set_title("Test-missing loss curves (ablation sweep)")
    ax_test_xent.legend(loc="best", fontsize=8)
    ax_test_xent.grid(True, alpha=0.3)
    # Fix y-axis range for x-ent curves for comparability
    ax_test_xent.set_ylim(0.0, 2.0)
    fig_test_xent.tight_layout()
    fig_test_xent.savefig(plot_dir / "test_missing_xent_curves.png", dpi=150)
    plt.close(fig_test_xent)

    ax_test_acc.set_xlabel("Epoch")
    ax_test_acc.set_ylabel("Test-missing rating accuracy")
    ax_test_acc.set_title("Test-missing accuracy curves (ablation sweep)")
    ax_test_acc.legend(loc="best", fontsize=8)
    ax_test_acc.grid(True, alpha=0.3)
    fig_test_acc.tight_layout()
    fig_test_acc.savefig(plot_dir / "test_missing_acc_curves.png", dpi=150)
    plt.close(fig_test_acc)

    print(f"Saved plots to {plot_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot Imputer ablation learning curves")
    parser.add_argument("--output-root", default="OUTPUT/IMPUTER", help="Root output directory")
    parser.add_argument(
        "--run-prefix",
        default="llm_rubric_marformer_ablation",
        help="Prefix of run directory names",
    )
    parser.add_argument(
        "--plot-dir",
        default="OUTPUT/IMPUTER/plots/IMPUTER_ABLATION",
        help="Directory for output plots",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    runs = discover_runs(output_root, args.run_prefix)

    if not runs:
        print(f"No runs found under {output_root} with prefix {args.run_prefix}")
        return

    plot_curves(runs, args.run_prefix, args.plot_dir)


if __name__ == "__main__":
    main()
