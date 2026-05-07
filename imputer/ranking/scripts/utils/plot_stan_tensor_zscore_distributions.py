#!/usr/bin/env python3
"""
Plot aggregated latent score distributions from a CPM Stan CSV or synthetic bundle.

For the tensor model,
    z_ijk = base_scores[(i-1) * J + j, k]
and the standardized latent score entering the ordinal likelihood is
    z_std_ijk = z_ijk / total_std[(i-1) * J + j]

This script computes posterior means of either
  - standardized latent scores z_ijk / total_std_ij, or
  - raw latent scores z_ijk,
then plots:
  1. Distribution over all (attribute, item) cells for each annotator j
  2. Heatmaps over (attribute, annotator) of:
       - mean z_std_ijk across items k
       - std  z_std_ijk across items k
       - mean posterior SD of z_std_ijk across items k
  3. Five per-annotator violin panels over the nine (attribute, annotator) pairs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 17,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.9",
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

BASE_RE = re.compile(r"base_scores\.(\d+)\.(\d+)$")
TOTAL_RE = re.compile(r"total_std\.(\d+)$")
THRESH_RE = re.compile(r"rating_thresholds\.(\d+)\.(\d+)$")


def _read_header(csv_path: Path) -> tuple[list[str], int]:
    skiprows = 0
    with open(csv_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                skiprows += 1
                continue
            header = line.strip().split(",")
            return header, skiprows + 1
    raise ValueError(f"No header found in {csv_path}")


def _load_dims(run_dir: Path) -> tuple[int, int, int, str]:
    cfg_path = run_dir / "configs.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    bundle_rel = cfg["inference"]["data_bundle"]
    bundle_path = ROOT / bundle_rel
    with open(bundle_path, "r") as f:
        bundle = json.load(f)
    stats = bundle["stats"]
    return int(stats["I"]), int(stats["J"]), int(stats["K"]), bundle_path.stem


def _load_score_and_threshold_stats_from_csv(
    csv_path: Path, I: int, J: int, K: int, score_type: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    header, skiprows = _read_header(csv_path)

    base_meta: list[tuple[int, int, int]] = []
    total_meta: list[tuple[int, int]] = []
    thresh_meta: list[tuple[int, int, int]] = []
    for idx, name in enumerate(header):
        m = BASE_RE.match(name)
        if m:
            ij = int(m.group(1))
            k = int(m.group(2))
            base_meta.append((ij, k, idx))
            continue
        m = TOTAL_RE.match(name)
        if m:
            ij = int(m.group(1))
            total_meta.append((ij, idx))
            continue
        m = THRESH_RE.match(name)
        if m:
            ij = int(m.group(1))
            c = int(m.group(2))
            thresh_meta.append((ij, c, idx))

    base_meta.sort(key=lambda x: (x[0], x[1]))
    total_meta.sort(key=lambda x: x[0])
    thresh_meta.sort(key=lambda x: (x[0], x[1]))

    expected_ij = I * J
    if len(base_meta) != expected_ij * K:
        raise ValueError(f"Expected {expected_ij * K} base_scores columns, found {len(base_meta)}")
    if len(total_meta) != expected_ij:
        raise ValueError(f"Expected {expected_ij} total_std columns, found {len(total_meta)}")
    if len(thresh_meta) % expected_ij != 0:
        raise ValueError("rating_thresholds columns do not align with I*J")

    num_thresholds = len(thresh_meta) // expected_ij

    usecols = [idx for _, _, idx in base_meta] + [idx for _, idx in total_meta] + [idx for _, _, idx in thresh_meta]
    data = np.loadtxt(csv_path, delimiter=",", comments="#", skiprows=skiprows, usecols=usecols, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]

    n_base = len(base_meta)
    n_total = len(total_meta)
    base = data[:, :n_base].reshape(data.shape[0], expected_ij, K)
    total = data[:, n_base : n_base + n_total].reshape(data.shape[0], expected_ij, 1)
    thresh = data[:, n_base + n_total :].reshape(data.shape[0], expected_ij, num_thresholds)
    if score_type == "standardized":
        score = base / total
    elif score_type == "raw":
        score = base
    else:
        raise ValueError(f"Unknown score_type={score_type}")
    return score.mean(axis=0), score.std(axis=0), thresh.mean(axis=0), thresh.std(axis=0)


def _load_score_and_threshold_stats_from_bundle(
    bundle_path: Path,
    score_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    with open(bundle_path, "r") as f:
        bundle = json.load(f)
    stats = bundle["stats"]
    I, J = int(stats["I"]), int(stats["J"])
    K = int(stats.get("K", stats.get("total_items")))
    base_scores = np.asarray(bundle["base_scores"], dtype=np.float64).reshape(I * J, K)
    thresh = np.asarray(bundle["rating_thresholds_z"], dtype=np.float32).reshape(I * J, -1)
    if score_type == "raw":
        score = base_scores.astype(np.float32)
    elif score_type == "standardized":
        cfg_path = bundle_path.parent / "configs.json"
        if not cfg_path.exists():
            raise ValueError(f"Need configs.json next to bundle to recover total_std: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        datagen = cfg["datagen"]
        E = np.asarray(bundle["embeddings"], dtype=np.float64)  # [K, D]
        # Solve E @ x ~= base_scores[ij, :] for all ij jointly.
        X, *_ = np.linalg.lstsq(E, base_scores.T, rcond=None)  # [D, IJ]
        eff_norm = np.linalg.norm(X, axis=0)  # [IJ]
        if datagen.get("use_dawid_skene_noise", False):
            noise = 0.05  # same fixed bin_smoothing used in tensor_model.stan
        else:
            noise = float(datagen["sigma_measurement"])
        total_std = np.sqrt(eff_norm ** 2 + noise ** 2)  # [IJ]
        score = (base_scores / total_std[:, None]).astype(np.float32)
    else:
        raise ValueError(f"Unknown score_type={score_type}")
    score_sd = np.zeros_like(score, dtype=np.float32)
    thresh_sd = np.zeros_like(thresh)
    return score, score_sd, thresh, thresh_sd, I, J, K


def _plot_annotator_violins(
    z_mean: np.ndarray,
    thresh_mean: np.ndarray,
    thresh_sd: np.ndarray,
    I: int,
    J: int,
    out_path: Path,
    title_prefix: str,
    x_label: str,
) -> None:
    annot_values: list[np.ndarray] = []
    annot_labels: list[str] = []
    annot_means: list[float] = []
    annot_thresh: list[np.ndarray] = []
    annot_thresh_sd: list[np.ndarray] = []

    for j in range(J):
        vals = np.concatenate([z_mean[i * J + j, :] for i in range(I)])
        annot_values.append(vals)
        annot_labels.append(f"J{j + 1}")
        annot_means.append(float(vals.mean()))
        # Threshold indices 1..C are the finite interior cutpoints; 0 and C+1 are +/-inf.
        finite_thresh = np.vstack([thresh_mean[i * J + j, 1:-1] for i in range(I)]).mean(axis=0)
        finite_thresh_sd = np.vstack([thresh_sd[i * J + j, 1:-1] for i in range(I)]).mean(axis=0)
        annot_thresh.append(finite_thresh)
        annot_thresh_sd.append(finite_thresh_sd)

    order = np.argsort(annot_means)
    ordered_values = [annot_values[idx] for idx in order]
    ordered_labels = [annot_labels[idx] for idx in order]
    ordered_means = [annot_means[idx] for idx in order]
    ordered_thresh = [annot_thresh[idx] for idx in order]
    ordered_thresh_sd = [annot_thresh_sd[idx] for idx in order]
    all_vals = np.concatenate(ordered_values) if ordered_values else np.array([0.0, 1.0])
    x_span = float(np.nanmax(all_vals) - np.nanmin(all_vals))
    min_whisker = max(1e-6, 0.015 * x_span)

    fig, ax = plt.subplots(figsize=(11.8, 9.2))
    parts = ax.violinplot(ordered_values, vert=False, showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#1b9e77")
        body.set_edgecolor("#0f5f4e")
        body.set_alpha(0.65)

    ypos = np.arange(1, J + 1)
    ax.scatter(ordered_means, ypos, color="#0f5f4e", s=18, zorder=3, label="Mean")
    thresh_colors = ["#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    thresh_labels_done: set[int] = set()
    for y, thresholds, thresh_sds in zip(ypos, ordered_thresh, ordered_thresh_sd):
        for t_idx, (x, x_sd) in enumerate(zip(thresholds, thresh_sds), start=1):
            color = thresh_colors[(t_idx - 1) % len(thresh_colors)]
            label = None
            if t_idx not in thresh_labels_done:
                label = f"Threshold {t_idx}"
                thresh_labels_done.add(t_idx)
            ax.vlines(x, y - 0.34, y + 0.34, color=color, linewidth=1.8, alpha=0.95, zorder=4, label=label)
            whisker = float(x_sd) if np.isfinite(x_sd) and x_sd > 0 else min_whisker
            ax.hlines(y, x - whisker, x + whisker, color=color, linewidth=2.0, alpha=0.9, zorder=5)
            ax.vlines([x - whisker, x + whisker], y - 0.08, y + 0.08, color=color, linewidth=1.4, alpha=0.9, zorder=5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ordered_labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Annotator")
    ax.set_title(f"{title_prefix}: Standardized Latent Score Distribution by Annotator", pad=14)
    ax.grid(True, axis="x", alpha=0.35)
    ax.grid(False, axis="y")
    ax.legend(loc="lower right")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def _plot_attr_annot_heatmaps(
    z_mean: np.ndarray,
    z_sd: np.ndarray,
    I: int,
    J: int,
    out_path: Path,
    title_prefix: str,
    value_name: str,
) -> None:
    mean_map = z_mean.mean(axis=1).reshape(I, J)
    item_sd_map = z_mean.std(axis=1).reshape(I, J)
    post_sd_map = z_sd.mean(axis=1).reshape(I, J)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    panels = [
        (mean_map, f"Mean {value_name} over items", "coolwarm"),
        (item_sd_map, "Std over items", "viridis"),
        (post_sd_map, "Mean posterior SD", "magma"),
    ]

    for ax, (mat, subtitle, cmap) in zip(axes, panels):
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(J))
        ax.set_xticklabels([f"J{j + 1}" for j in range(J)], rotation=45, ha="right")
        ax.set_yticks(np.arange(I))
        ax.set_yticklabels([f"I{i + 1}" for i in range(I)])
        ax.set_xlabel("Annotator")
        ax.set_ylabel("Attribute")
        ax.set_title(subtitle, pad=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{title_prefix}: Standardized Latent Score by Attribute-Annotator Pair", y=1.02, fontsize=18)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def _plot_single_annotator_attribute_violins(
    z_mean: np.ndarray,
    thresh_mean: np.ndarray,
    thresh_sd: np.ndarray,
    I: int,
    J: int,
    annotator_index: int,
    out_path: Path,
    title_prefix: str,
    x_label: str,
) -> None:
    values: list[np.ndarray] = []
    labels: list[str] = []
    means: list[float] = []
    thresholds: list[np.ndarray] = []
    threshold_sds: list[np.ndarray] = []

    for i in range(I):
        ij = i * J + annotator_index
        vals = z_mean[ij, :]
        values.append(vals)
        labels.append(f"I={i + 1}, J={annotator_index + 1}")
        means.append(float(vals.mean()))
        thresholds.append(thresh_mean[ij, 1:-1])
        threshold_sds.append(thresh_sd[ij, 1:-1])

    ypos = np.arange(1, I + 1)
    all_vals = np.concatenate(values) if values else np.array([0.0, 1.0])
    x_span = float(np.nanmax(all_vals) - np.nanmin(all_vals))
    min_whisker = max(1e-6, 0.015 * x_span)
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    parts = ax.violinplot(values, vert=False, showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#1f78b4")
        body.set_edgecolor("#0c4c78")
        body.set_alpha(0.68)

    ax.scatter(means, ypos, color="#0c4c78", s=20, zorder=3, label="Mean")
    thresh_colors = ["#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    for y, thresh_row, thresh_sd_row in zip(ypos, thresholds, threshold_sds):
        for t_idx, (x, x_sd) in enumerate(zip(thresh_row, thresh_sd_row), start=1):
            color = thresh_colors[(t_idx - 1) % len(thresh_colors)]
            ax.vlines(
                x,
                y - 0.34,
                y + 0.34,
                color=color,
                linewidth=1.8,
                alpha=0.95,
                zorder=4,
                label=f"Threshold {t_idx}" if y == 1 else None,
            )
            whisker = float(x_sd) if np.isfinite(x_sd) and x_sd > 0 else min_whisker
            ax.hlines(y, x - whisker, x + whisker, color=color, linewidth=2.0, alpha=0.9, zorder=5)
            ax.vlines([x - whisker, x + whisker], y - 0.08, y + 0.08, color=color, linewidth=1.4, alpha=0.9, zorder=5)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Attribute-Annotator Pair")
    ax.set_title(f"{title_prefix}: Attribute-Level Distributions for J={annotator_index + 1}", pad=14)
    ax.grid(True, axis="x", alpha=0.35)
    ax.grid(False, axis="y")
    ax.legend(loc="lower right")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def _choose_annotators(J: int) -> list[int]:
    rng = np.random.default_rng(42)
    forced = [idx for idx in (8, 24) if idx < J]
    remaining = [idx for idx in range(J) if idx not in forced]
    need = max(0, min(5, J) - len(forced))
    sampled = list(rng.choice(remaining, size=need, replace=False)) if need > 0 else []
    chosen = forced + sampled
    return sorted(chosen[: min(5, J)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CPM latent score distributions.")
    parser.add_argument("--csv", help="Path to Stan CSV with tensor model draws.")
    parser.add_argument("--bundle", help="Path to data_bundle.json for synthetic raw-score analysis.")
    parser.add_argument("--out-dir", default=None, help="Optional output directory.")
    parser.add_argument("--score-type", choices=["standardized", "raw"], default="standardized")
    parser.add_argument("--per-annotator-only", action="store_true")
    args = parser.parse_args()

    if bool(args.csv) == bool(args.bundle):
        raise ValueError("Pass exactly one of --csv or --bundle")

    if args.csv:
        csv_path = Path(args.csv).resolve()
        run_dir = csv_path.parent
        I, J, K, _ = _load_dims(run_dir)
        z_mean, z_sd, thresh_mean, thresh_sd = _load_score_and_threshold_stats_from_csv(csv_path, I, J, K, args.score_type)
        run_name = run_dir.name
        title_prefix = f"CPM Stan {run_name} ({args.score_type})"
    else:
        bundle_path = Path(args.bundle).resolve()
        z_mean, z_sd, thresh_mean, thresh_sd, I, J, K = _load_score_and_threshold_stats_from_bundle(bundle_path, args.score_type)
        run_name = bundle_path.parent.name
        title_prefix = f"Synthetic Tensor {run_name} ({args.score_type})"

    out_dir = Path(args.out_dir).resolve() if args.out_dir else ROOT / "PLOTS/TALK/LLMRubric"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_label = (
        r"Posterior Mean Standardized Score $\bar{z}_{ijk} / \sigma_{ij}$"
        if args.score_type == "standardized"
        else r"Score $\bar{z}_{ijk}$"
    )
    value_name = "standardized score" if args.score_type == "standardized" else "score"

    if not args.per_annotator_only:
        _plot_annotator_violins(
            z_mean,
            thresh_mean,
            thresh_sd,
            I,
            J,
            out_dir / f"{run_name}_{args.score_type}_by_annotator.png",
            title_prefix,
            x_label,
        )
        _plot_attr_annot_heatmaps(
            z_mean,
            z_sd,
            I,
            J,
            out_dir / f"{run_name}_{args.score_type}_by_attr_annotator.png",
            title_prefix,
            value_name,
        )
    for annotator_index in _choose_annotators(J):
        _plot_single_annotator_attribute_violins(
            z_mean,
            thresh_mean,
            thresh_sd,
            I,
            J,
            annotator_index,
            out_dir / f"{run_name}_{args.score_type}_attr_violin_J{annotator_index + 1}.png",
            title_prefix,
            x_label,
        )


if __name__ == "__main__":
    main()
