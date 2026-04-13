#!/usr/bin/env python3
"""
Test log-loss vs training size for Marformer and Stan Tensor (Tensor_500_25_9_ItemTest).

Reads best-*.json from each Marformer TEST_RESULTS/ and predictive_metrics.json
from each Stan eval dir, then plots missing log-loss vs K_train size.

Usage (run from imputer/ranking):
    python scripts/utils/plot_marformer_test_loss_by_size.py

Output: PLOTS/TALK/Tensor_500_25_9_test_loss_by_size.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family":       "serif",
    "font.size":         12,
    "axes.labelsize":    14,
    "axes.titlesize":    14,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   11,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "0.75",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "0.88",
    "grid.linewidth":    0.6,
    "lines.linewidth":   2.0,
    "lines.markersize":  6,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

MARFORMER_ROOT = Path("RESULTS/MARFORMER/STAN/SPARSE/Tensor")
STAN_ROOT      = Path("RESULTS/STAN/SPARSE/Tensor500")
SIZES          = [10, 50, 100, 200, 400]


def load_marformer_log_loss(size: int) -> float | None:
    run_dir = MARFORMER_ROOT / f"Tensor_500_25_9_ItemTest_{size}_CLUSTER_NOITEMDEV_TRANS" / "TEST_RESULTS"
    jsons = sorted(run_dir.glob("best-*.json"))
    if not jsons:
        print(f"  [SKIP Marformer] size={size} — no best-*.json in {run_dir}")
        return None
    return json.loads(jsons[0].read_text())["missing"]["log_loss"]


def load_stan_log_loss(size: int) -> float | None:
    metrics_path = STAN_ROOT / f"Tensor_500_25_9_ItemTest_{size}_TEN_eval" / "predictive_metrics.json"
    if not metrics_path.exists():
        print(f"  [SKIP Stan] size={size} — {metrics_path} not found")
        return None
    data = json.loads(metrics_path.read_text())
    return -data["rating_missing_log_likelihood"]


def main() -> None:
    mf_sizes, mf_losses = [], []
    stan_sizes, stan_losses = [], []

    for size in SIZES:
        ll = load_marformer_log_loss(size)
        if ll is not None:
            mf_sizes.append(size)
            mf_losses.append(ll)

        ll = load_stan_log_loss(size)
        if ll is not None:
            stan_sizes.append(size)
            stan_losses.append(ll)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(mf_sizes,   mf_losses,   color="#1f6fba", marker="o", label="Marformer")
    ax.plot(stan_sizes, stan_losses, color="#e67e22", marker="s", label="Stan Tensor")

    ax.set_xlabel("Training Size (K_train)")
    ax.set_ylabel("Test Missing Log-Loss")
    ax.set_title("Test Loss by Training Size\n(Tensor_500_25_9_ItemTest)")
    ax.set_xticks(SIZES)
    ax.legend()

    out = Path("PLOTS/TALK/Tensor_500_25_9_test_loss_by_size.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
