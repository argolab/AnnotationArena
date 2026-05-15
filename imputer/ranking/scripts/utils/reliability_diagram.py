"""
Shared reliability-diagram helpers (relplot).

Used by ``plot_realdata_calibration.py``, ``plot_llm_rubric_cpm_with_structured_baselines.py``,
and domain-3 result scripts. Keeps smECE / all-class calibration logic in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import relplot as rp

rp.config.use_tex_fonts = False


def all_class_calibration(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (n, C) probs into confidence; accuracy is 1 iff label matches that class column."""
    classes = np.arange(probs.shape[1], dtype=np.int64)
    conf = probs.flatten()
    acc = (labels[:, None] == classes[None, :]).astype(np.float32).flatten()
    return conf, acc


def draw_empty(ax: plt.Axes, title: str) -> None:
    ax.plot([0, 1], [0, 1], color="0.75", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.text(0.5, 0.5, "no data", ha="center", va="center", color="0.6", transform=ax.transAxes)
    ax.set_title(title)


def plot_ece(ax: plt.Axes, probs: np.ndarray, labels: np.ndarray, title: str, color: str) -> None:
    """Draw one reliability panel with smECE in the title."""
    conf, acc = all_class_calibration(probs, labels)
    diag = rp.prepare_rel_diagram(
        conf,
        acc,
        num_bootstrap=500,
        report_CE=True,
        report_CE_std=True,
    )
    ce = diag.get("ce", float("nan"))
    rp.plot_rel_diagram(
        diag,
        fig=ax.get_figure(),
        ax=ax,
        color=color,
        use_default_style=True,
        plot_density_ticks=True,
        plot_labels=True,
        legend=False,
    )
    for txt in ax.texts:
        txt.remove()
    ci_w = diag.get("ce_ci_width", 0.0)
    ax.set_title(f"{title}\nsmECE = {ce:.3f} ± {ci_w:.3f}", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)


def plot_reliability_panels(
    panels: Sequence[tuple[str, np.ndarray | None, np.ndarray | None, str]],
    *,
    suptitle: str,
    output_path: Path,
    ncols: int = 3,
    figsize_per_col: float = 5.2,
    row_height: float = 5.0,
) -> None:
    """
    Save a grid of reliability diagrams.

    Each panel is ``(title, probs, labels, color)``. If ``probs`` is None, draws an empty panel.
    """
    n = len(panels)
    if n == 0:
        return
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_col * ncols, row_height * nrows),
        squeeze=False,
    )
    for ax, (title, probs, labels, color) in zip(axes.flat, panels):
        if probs is None or labels is None or len(labels) == 0:
            draw_empty(ax, title)
            continue
        plot_ece(ax, probs, labels, title, color)
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=18, y=0.98)
    fig.subplots_adjust(top=0.88, hspace=0.34, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
