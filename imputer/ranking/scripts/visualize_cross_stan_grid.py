#!/usr/bin/env python3
"""
Plot Stan cross-type performance as an N×N grid: rows = data generator type,
columns = domain model type. Two subplots: log-loss (negative log-likelihood)
and prediction accuracy (argmax).

Usage:
    python scripts/visualize_cross_stan_grid.py --output stan_grid.png \\
        --stan-types "normal-noise-dot-product,factored-dot-product,discrete,tensor" \\
        --metrics-paths path_00.json path_01.json ... path_33.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path) -> Optional[dict]:
    """Load metrics JSON; return None if file missing or invalid (e.g. Stan run failed)."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Plot Stan N×N (and optional Marformer) performance grid")
    parser.add_argument("--output", required=True, help="Output figure path")
    parser.add_argument("--stan-types", required=True, help="Comma-separated list of stan_type labels (e.g. normal-noise-dot-product,factored-dot-product,discrete)")
    parser.add_argument("--metrics-paths", nargs="+", required=True,
                        help="Paths to Stan predictive_metrics.json in row-major order: (data0,model0), (data0,model1), (data1,model0), (data1,model1)")
    parser.add_argument("--imputer-metrics-paths", nargs="*", default=None,
                        help="Optional paths to imputer test_metrics.json, one per data type (row order)")
    args = parser.parse_args()

    types = [s.strip() for s in args.stan_types.split(",")]
    n = len(types)
    if n * n != len(args.metrics_paths):
        raise ValueError(f"Expected {n * n} metrics paths for {n} types, got {len(args.metrics_paths)}")

    # Build short labels for axes (e.g. "normal-noise", "factored", "discrete", "tensor")
    def short_label(s: str) -> str:
        if s == "normal-noise-dot-product": return "normal-noise"
        if s == "factored-dot-product": return "factored"
        if s == "discrete": return "discrete"
        if s == "tensor": return "tensor"
        return s.replace("_", " ")
    labels = [short_label(t) for t in types]

    has_imputer = bool(args.imputer_metrics_paths)
    cols = n + (1 if has_imputer else 0)

    logloss = np.full((n, cols), np.nan)
    accuracy = np.full((n, cols), np.nan)

    # Fill Stan columns 0..n-1
    for idx, path in enumerate(args.metrics_paths):
        i, j = idx // n, idx % n
        m = load_metrics(Path(path))
        if m is None:
            continue  # keep nan for failed/missing run
        # Stan stores mean log-likelihood; we plot as loss = -log_lik
        logloss[i, j] = -float(m.get("rating_missing_log_likelihood", np.nan))
        accuracy[i, j] = float(m.get("rating_missing_accuracy", np.nan))

    # Fill Marformer (imputer) column if provided
    if has_imputer:
        if len(args.imputer_metrics_paths) != n:
            raise ValueError(f"Expected {n} imputer metrics paths (one per data type), got {len(args.imputer_metrics_paths)}")
        for i, path in enumerate(args.imputer_metrics_paths):
            m = load_metrics(Path(path))
            if m is None:
                continue
            missing = m.get("missing_metrics", {})
            # Imputer stores rating_loss and rating_accuracy under missing_metrics
            rating_loss = missing.get("rating_loss", np.nan)
            rating_acc = missing.get("rating_accuracy", np.nan)
            logloss[i, n] = float(rating_loss) if rating_loss is not None else np.nan
            accuracy[i, n] = float(rating_acc) if rating_acc is not None else np.nan

    # Scale figure with grid size (e.g. 4×4 needs more space)
    cell_size = 3.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * cell_size * cols, cell_size * n))

    # Subplot 1: Log-loss (negative log-likelihood); mask nan for colormap
    logloss_plot = np.ma.masked_invalid(logloss)
    cmap1 = plt.cm.viridis_r.copy()
    cmap1.set_bad(color="lightgray", alpha=0.8)
    im1 = ax1.imshow(logloss_plot, cmap=cmap1, aspect="equal")
    ax1.set_xticks(range(cols))
    ax1.set_yticks(range(n))
    xtick_labels = [f"Model: {l}" for l in labels]
    if has_imputer:
        xtick_labels.append("Model: marformer")
    ax1.set_xticklabels(xtick_labels)
    ax1.set_yticklabels([f"Data: {l}" for l in labels])
    ax1.set_title("Rating log-loss (missing)\n− log p(y_true)")
    for i in range(n):
        for j in range(cols):
            val = logloss[i, j]
            text = "N/A" if np.isnan(val) else f"{val:.3f}"
            ax1.text(j, i, text, ha="center", va="center", color="black" if np.isnan(val) else "white", fontsize=11)
    plt.colorbar(im1, ax=ax1, label="− log-likelihood")

    # Subplot 2: Accuracy (argmax); mask nan for colormap
    accuracy_plot = np.ma.masked_invalid(accuracy)
    cmap2 = plt.cm.plasma.copy()
    cmap2.set_bad(color="lightgray", alpha=0.8)
    im2 = ax2.imshow(accuracy_plot, cmap=cmap2, aspect="equal", vmin=0, vmax=1)
    ax2.set_xticks(range(cols))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(xtick_labels)
    ax2.set_yticklabels([f"Data: {l}" for l in labels])
    ax2.set_title("Rating accuracy (missing)\nargmax prediction")
    for i in range(n):
        for j in range(cols):
            val = accuracy[i, j]
            text = "N/A" if np.isnan(val) else f"{val:.3f}"
            ax2.text(j, i, text, ha="center", va="center", color="black" if np.isnan(val) else "white", fontsize=11)
    plt.colorbar(im2, ax=ax2, label="Accuracy")

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
