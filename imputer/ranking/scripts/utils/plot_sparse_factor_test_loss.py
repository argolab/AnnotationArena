#!/usr/bin/env python3
"""
Test log loss vs training item size — Sparse Factor_225_25_9_ItemTest.

Methods:
  - Marformer (no-item-dev, transductive): reads TEST_RESULTS/best.json
      → run first: bash scripts/TESTING_SCRIPTS/run_test_sparse_factor_itemtest.sh
  - STAN Normal:  reads RESULTS/STAN/ItemTestSparse/.../predictive_metrics.json
  - STAN Factor:  reads RESULTS/STAN/ItemTestSparse/.../predictive_metrics.json

Usage (run from imputer/ranking):
    python scripts/utils/plot_sparse_factor_test_loss.py

Output: PLOTS/TALK/sparse_factor_test_log_loss.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT              = Path(__file__).resolve().parents[2]
MF_RESULTS_ROOT   = ROOT / "RESULTS/MARFORMER/STAN/SPARSE"
STAN_RESULTS_ROOT = ROOT / "RESULTS/STAN/ItemTestSparse"
OUT_FILE          = ROOT / "PLOTS/TALK/sparse_factor_test_log_loss.png"

# Marformer: size → run dir name
MF_SIZE_TO_RUN = {
    10:  "Factor_225_25_9_ItemTest_10_LOCAL_NOITEMDEV_TRANS",
    20:  "Factor_225_25_9_ItemTest_20_LOCAL_NOITEMDEV_TRANS",
    50:  "Factor_225_25_9_ItemTest_50_LOCAL_NOITEMDEV_TRANS",
    100: "Factor_225_25_9_ItemTest_100_LOCAL_NOITEMDEV_TRANS",
    175: "Factor_225_25_9_ItemTest_Size_175_LOCAL_NOITEMDEV_TRANS",
}

# STAN: all available sizes on Factor dataset
STAN_SIZES = [10, 20, 30, 40, 50, 75, 100, 125, 150, 175]

# ── rcParams ──────────────────────────────────────────────────────────────────
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
    "lines.markersize":  7,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})


def _mf_log_loss(run_dir: Path) -> float | None:
    p = run_dir / "TEST_RESULTS" / "best.json"
    if not p.exists():
        return None
    v = json.load(open(p)).get("missing", {}).get("log_loss")
    return float(v) if v is not None else None


def _stan_log_loss(size: int, stan_model: str) -> float | None:
    p = STAN_RESULTS_ROOT / f"Factor_225_25_9_ItemTest_{size}_nt_{stan_model}_eval" / "predictive_metrics.json"
    if not p.exists():
        return None
    ll = json.load(open(p)).get("rating_missing_log_likelihood")
    return float(-ll) if ll is not None else None


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_plotted = False

    # ── Marformer ─────────────────────────────────────────────────────────────
    mf_pairs: list[tuple[int, float]] = []
    for size in sorted(MF_SIZE_TO_RUN):
        run_dir = MF_RESULTS_ROOT / MF_SIZE_TO_RUN[size]
        if not run_dir.exists():
            print(f"[Marformer skip] size={size}  (run dir not found)")
            continue
        ll = _mf_log_loss(run_dir)
        if ll is None:
            print(f"[Marformer skip] size={size}  (no TEST_RESULTS — run the test script first)")
            continue
        print(f"[Marformer] size={size}  log_loss={ll:.4f}")
        mf_pairs.append((size, ll))

    if mf_pairs:
        xs, ys = zip(*mf_pairs)
        ax.plot(xs, ys, color="#1f6fba", marker="o", linestyle="-", label="Marformer")
        any_plotted = True

    # ── STAN Normal ───────────────────────────────────────────────────────────
    sn_pairs: list[tuple[int, float]] = []
    for size in STAN_SIZES:
        ll = _stan_log_loss(size, "Normal")
        if ll is None:
            print(f"[STAN Normal skip] size={size}")
            continue
        print(f"[STAN Normal] size={size}  log_loss={ll:.4f}")
        sn_pairs.append((size, ll))

    if sn_pairs:
        xs, ys = zip(*sn_pairs)
        ax.plot(xs, ys, color="#e67e22", marker="D", linestyle="--", label="STAN Normal")
        any_plotted = True

    # ── STAN Factor ───────────────────────────────────────────────────────────
    sf_pairs: list[tuple[int, float]] = []
    for size in STAN_SIZES:
        ll = _stan_log_loss(size, "Factor")
        if ll is None:
            print(f"[STAN Factor skip] size={size}")
            continue
        print(f"[STAN Factor] size={size}  log_loss={ll:.4f}")
        sf_pairs.append((size, ll))

    if sf_pairs:
        xs, ys = zip(*sf_pairs)
        ax.plot(xs, ys, color="#27ae60", marker="^", linestyle="--", label="STAN Factor")
        any_plotted = True

    if not any_plotted:
        print("No data to plot.")
        return

    ax.set_xlabel("Training Items")
    ax.set_ylabel("Test Log Loss")
    ax.set_title("Sparse Factor — Test Log Loss vs Training Size")
    ax.legend(loc="upper right")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE)
    plt.close(fig)
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
