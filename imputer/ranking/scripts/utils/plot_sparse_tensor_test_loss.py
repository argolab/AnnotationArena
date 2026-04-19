#!/usr/bin/env python3
"""
Plot sparse Tensor_400_25_9 item-test results by training size.

Outputs:
  - PLOTS/TALK/Item/sparse_tensor_test_loss_by_size.png
  - PLOTS/TALK/Item/sparse_tensor_test_loss_by_size_nt.png
  - PLOTS/TALK/Item/sparse_tensor_correlation_by_method.png
  - PLOTS/TALK/Item/sparse_tensor_mbr_l2_size300.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "DATA/STAN/SPARSE/Tensor_400_25_9_ItemTest"
MARFORMER_ROOT = ROOT / "RESULTS/MARFORMER/STAN/SPARSE/Tensor400"
MARFORMER_NT_ROOT = ROOT / "RESULTS/MARFORMER/STAN/SPARSE/Tensor400NT"
STAN_ROOT = ROOT / "RESULTS/STAN/SPARSE/Tensor400"
STAN_NT_ROOT = ROOT / "RESULTS/STAN/SPARSE/Tensor400NT"
OUT_DIR = ROOT / "PLOTS/TALK/Item"

SIZES = [10, 50, 100, 200, 300]
PROB_COLS = ["prob_cat_1", "prob_cat_2", "prob_cat_3", "prob_cat_4", "prob_cat_5"]
DISCRETE_FALLBACK = {300: 200}
STAN_TENSOR_NT_FALLBACK: dict[int, int] = {}

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.75",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.4,
    "lines.markersize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _bundle_path(size: int) -> Path:
    return DATA_ROOT / f"Tensor_400_25_9_ItemTest_{size}" / "data_bundle.json"


def _load_bundle(size: int) -> dict | None:
    path = _bundle_path(size)
    if not path.exists():
        return None
    return _read_json(path)


def _test_missing_labels(size: int) -> np.ndarray | None:
    bundle = _load_bundle(size)
    if bundle is None:
        return None
    idxs = bundle["missing_ratings_indexes_in_test_instance"]
    rows = [bundle["missing_ratings"][idx] for idx in idxs]
    return np.asarray([r["value"] - 1 for r in rows], dtype=np.int64)


def _expected_rmse(probs: np.ndarray, labels: np.ndarray) -> float:
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    expected = probs.astype(np.float64) @ classes
    truth = labels.astype(np.float64) + 1.0
    return float(np.sqrt(np.mean((expected - truth) ** 2)))


def _mean_xent(probs: np.ndarray, labels: np.ndarray) -> float:
    idx = np.arange(labels.shape[0])
    return float(-np.log(probs[idx, labels] + 1e-12).mean())


def _discrete_source_size(size: int) -> int:
    return DISCRETE_FALLBACK.get(size, size)


def _stan_tensor_nt_source_size(size: int) -> int:
    return STAN_TENSOR_NT_FALLBACK.get(size, size)


def _stan_source_size(size: int, discrete: bool, nontrans: bool) -> int:
    if discrete:
        return _discrete_source_size(size)
    if nontrans:
        return _stan_tensor_nt_source_size(size)
    return size


def _stan_eval_dir(size: int, discrete: bool, nontrans: bool) -> Path:
    source_size = _stan_source_size(size, discrete, nontrans)
    root = STAN_NT_ROOT if nontrans and not discrete else STAN_ROOT
    if discrete:
        run_name = f"Tensor_400_25_9_ItemTest_{source_size}_DISCRETE_MISP_eval"
    elif nontrans:
        run_name = f"Tensor_400_25_9_ItemTest_{source_size}_NONTRANS_eval"
    else:
        run_name = f"Tensor_400_25_9_ItemTest_{source_size}_eval"
    return root / run_name


def _stan_metrics_path(size: int, discrete: bool, nontrans: bool = False) -> Path:
    return _stan_eval_dir(size, discrete, nontrans) / "predictive_metrics.json"


def _stan_probs_path(size: int, discrete: bool, nontrans: bool = False) -> Path:
    return _stan_eval_dir(size, discrete, nontrans) / "rating_probabilities.csv"


def _load_stan_missing_log_loss(size: int, discrete: bool, nontrans: bool = False) -> float | None:
    path = _stan_metrics_path(size, discrete, nontrans)
    if not path.exists():
        return None
    data = _read_json(path)
    return float(-data["rating_missing_log_likelihood"])


def _load_stan_probs_and_labels(size: int, discrete: bool, nontrans: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
    probs_path = _stan_probs_path(size, discrete, nontrans)
    source_size = _stan_source_size(size, discrete, nontrans)
    bundle = _load_bundle(source_size)
    if not probs_path.exists() or bundle is None:
        return None

    test_idxs = bundle["missing_ratings_indexes_in_test_instance"]
    labels = np.asarray([bundle["missing_ratings"][idx]["value"] - 1 for idx in test_idxs], dtype=np.int64)
    df = pd.read_csv(probs_path)
    grouped = (
        df[df["missing_rating_idx"].isin(test_idxs)]
        .groupby("missing_rating_idx")[PROB_COLS]
        .mean()
        .loc[test_idxs]
    )
    probs = grouped.to_numpy(dtype=np.float64)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Stan predictions/labels mismatch for size={size} discrete={discrete} nontrans={nontrans}: "
            f"{probs.shape[0]} vs {labels.shape[0]}"
        )
    return probs, labels


def _select_best_marformer_json(size: int, nontrans: bool = False) -> Path | None:
    root = MARFORMER_NT_ROOT if nontrans else MARFORMER_ROOT
    suffix = "NOITEMDEV_NONTRANS_MARFORMER" if nontrans else "NOITEMDEV_TRANS_MARFORMER"
    run_dir = root / f"Tensor_400_25_9_ItemTest_{size}_{suffix}" / "TEST_RESULTS"
    candidates = sorted(run_dir.glob("best-*.json"))
    if not candidates:
        return None
    return min(candidates, key=lambda p: _read_json(p)["missing"]["log_loss"])


def _load_marformer_metric(size: int, key: str, nontrans: bool = False) -> float | None:
    best_json = _select_best_marformer_json(size, nontrans=nontrans)
    if best_json is None:
        return None
    data = _read_json(best_json)
    return float(data["missing"][key])


def _load_unigram_probs_and_labels(size: int) -> tuple[np.ndarray, np.ndarray] | None:
    bundle = _load_bundle(size)
    if bundle is None:
        return None
    observed = [r for r in bundle["observed_ratings"] if r["instance"] in {"train", "val", "test"}]
    test_missing = [r for r in bundle["missing_ratings"] if r["instance"] == "test"]
    if not observed or not test_missing:
        return None

    num_classes = max(r["value"] for r in observed + test_missing)
    counts = np.zeros(num_classes, dtype=np.float64)
    for row in observed:
        counts[row["value"] - 1] += 1.0
    probs = counts / counts.sum()
    labels = np.asarray([r["value"] - 1 for r in test_missing], dtype=np.int64)
    tiled = np.tile(probs[None, :], (labels.shape[0], 1))
    return tiled, labels


def _mbr_l2_from_probs_labels(probs: np.ndarray, labels: np.ndarray) -> float:
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    expected = probs.astype(np.float64) @ classes
    truth = labels.astype(np.float64) + 1.0
    return float(np.mean((expected - truth) ** 2))


def _largest_available_mbr_l2_marformer(nontrans: bool = False) -> float | None:
    for size in reversed(SIZES):
        rmse = _load_marformer_metric(size, "rmse", nontrans=nontrans)
        if rmse is not None:
            return float(rmse) ** 2
    return None


def _largest_available_mbr_l2_stan(discrete: bool, nontrans: bool = False) -> float | None:
    for size in reversed(SIZES):
        loaded = _load_stan_probs_and_labels(size, discrete=discrete, nontrans=nontrans)
        if loaded is not None:
            return _mbr_l2_from_probs_labels(*loaded)
    return None


def _largest_available_mbr_l2_unigram() -> float | None:
    for size in reversed(SIZES):
        loaded = _load_unigram_probs_and_labels(size)
        if loaded is not None:
            return _mbr_l2_from_probs_labels(*loaded)
    return None


def _expected_values_from_probs(probs: np.ndarray) -> np.ndarray:
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    return probs.astype(np.float64) @ classes


def _correlations_from_preds_labels(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    truth = labels.astype(np.float64) + 1.0
    pearson = float(stats.pearsonr(preds, truth).statistic)
    spearman = float(stats.spearmanr(preds, truth).statistic)
    kendall = float(stats.kendalltau(preds, truth).statistic)
    return {"pearson": pearson, "spearman": spearman, "kendall": kendall}


def _load_marformer_correlations(size: int, nontrans: bool = False) -> dict[str, float] | None:
    best_json = _select_best_marformer_json(size, nontrans=nontrans)
    if best_json is None:
        return None
    missing = _read_json(best_json)["missing"]
    return {
        "pearson": float(missing["pearson_r"]),
        "spearman": float(missing["spearman_r"]),
        "kendall": float(missing["kendall_tau"]),
    }


def _load_stan_correlations(size: int, discrete: bool, nontrans: bool = False) -> dict[str, float] | None:
    loaded = _load_stan_probs_and_labels(size, discrete=discrete, nontrans=nontrans)
    if loaded is None:
        return None
    probs, labels = loaded
    return _correlations_from_preds_labels(_expected_values_from_probs(probs), labels)


def _series_from_loader(loader) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for size in SIZES:
        value = loader(size)
        if value is None:
            continue
        xs.append(size)
        ys.append(float(value))
    return xs, ys


def _plot_metric(output_path: Path, ylabel: str, title: str, series: Iterable[tuple[str, list[int], list[float], dict]]) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 8.5))
    for label, xs, ys, style in series:
        if xs:
            ax.plot(xs, ys, label=label, **style)

    ax.set_xlabel("Training Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="center", pad=22)
    ax.set_xticks(SIZES)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(right=0.78, top=0.86)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _plot_metric_broken_y(
    output_path: Path,
    ylabel: str,
    title: str,
    series: Iterable[tuple[str, list[int], list[float], dict]],
    break_at: float,
) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(15.5, 8.5),
        gridspec_kw={"height_ratios": [1.0, 4.2], "hspace": 0.05},
    )

    def draw_broken_series(label: str, xs: list[int], ys: list[float], style: dict) -> None:
        if not xs:
            return

        # Add a proxy artist so the legend reflects the full style once.
        ax_top.plot([], [], label=label, **style)

        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)

        line_style = {k: v for k, v in style.items() if k != "marker"}
        marker_style = {
            "linestyle": "None",
            "marker": style.get("marker", "o"),
            "color": style.get("color", "k"),
            "markersize": mpl.rcParams["lines.markersize"],
        }

        for idx in range(len(xs_arr) - 1):
            x1, x2 = xs_arr[idx], xs_arr[idx + 1]
            y1, y2 = ys_arr[idx], ys_arr[idx + 1]
            y1_top = y1 > break_at
            y2_top = y2 > break_at

            if y1_top == y2_top:
                target_ax = ax_top if y1_top else ax_bottom
                target_ax.plot([x1, x2], [y1, y2], label="_nolegend_", **line_style)
                continue

            x_cross = x1 + (break_at - y1) * (x2 - x1) / (y2 - y1)
            if y1_top:
                ax_top.plot([x1, x_cross], [y1, break_at], label="_nolegend_", **line_style)
                ax_bottom.plot([x_cross, x2], [break_at, y2], label="_nolegend_", **line_style)
            else:
                ax_bottom.plot([x1, x_cross], [y1, break_at], label="_nolegend_", **line_style)
                ax_top.plot([x_cross, x2], [break_at, y2], label="_nolegend_", **line_style)

        top_mask = ys_arr > break_at
        bottom_mask = ~top_mask
        if np.any(top_mask):
            ax_top.plot(xs_arr[top_mask], ys_arr[top_mask], label="_nolegend_", **marker_style)
        if np.any(bottom_mask):
            ax_bottom.plot(xs_arr[bottom_mask], ys_arr[bottom_mask], label="_nolegend_", **marker_style)

    all_y: list[float] = []
    for label, xs, ys, style in series:
        all_y.extend(ys)
        draw_broken_series(label, xs, ys, style)

    below = [y for y in all_y if y <= break_at]
    above = [y for y in all_y if y > break_at]
    lower_min = min(below) if below else 0.0
    lower_pad = max(0.04, 0.08 * (break_at - lower_min))
    ax_bottom.set_ylim(max(0.0, lower_min - lower_pad), break_at + 0.03)

    if above:
        upper_min = min(above)
        upper_max = max(above)
        upper_pad = max(0.08, 0.08 * max(upper_max - upper_min, 0.25))
        ax_top.set_ylim(upper_min - upper_pad, upper_max + upper_pad)
        ax_top.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bottom.xaxis.tick_bottom()

    d = 0.008
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    xs = np.linspace(0.0, 1.0, 33)
    top_y = np.where(np.arange(xs.size) % 2 == 0, -0.006, 0.006)
    bottom_y = np.where(np.arange(xs.size) % 2 == 0, 1.006, 0.994)
    ax_top.plot(xs, top_y, transform=ax_top.transAxes, color="0.35", alpha=0.28, linewidth=0.9, clip_on=False)
    ax_bottom.plot(xs, bottom_y, transform=ax_bottom.transAxes, color="0.35", alpha=0.28, linewidth=0.9, clip_on=False)

    ax_bottom.set_xlabel("Training Size")
    ax_bottom.set_ylabel(ylabel)
    ax_bottom.set_xticks(SIZES)
    fig.suptitle(title, x=0.5, y=0.98, ha="center")
    ax_top.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(right=0.78, top=0.92)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _plot_correlations(output_path: Path) -> None:
    method_rows: list[tuple[str, dict[str, float]]] = []

    marformer_corr = _load_marformer_correlations(300, nontrans=False)
    if marformer_corr is not None:
        method_rows.append(("Marformer", marformer_corr))

    stan_tensor_corr = _load_stan_correlations(300, discrete=False, nontrans=False)
    if stan_tensor_corr is not None:
        method_rows.append(("Oracle Tensor Stan", stan_tensor_corr))

    stan_discrete_corr = _load_stan_correlations(300, discrete=True, nontrans=False)
    if stan_discrete_corr is not None:
        method_rows.append(("Stan Discrete", stan_discrete_corr))

    marformer_nt_corr = _load_marformer_correlations(300, nontrans=True)
    if marformer_nt_corr is not None:
        method_rows.append(("Marformer NT", marformer_nt_corr))

    stan_tensor_nt_corr = _load_stan_correlations(300, discrete=False, nontrans=True)
    if stan_tensor_nt_corr is not None:
        method_rows.append(("Oracle Tensor Stan NT", stan_tensor_nt_corr))

    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    metrics = ["pearson", "spearman", "kendall"]
    metric_labels = ["Pearson", "Spearman", "Kendall"]
    metric_colors = {"pearson": "#1f6fba", "spearman": "#d55e00", "kendall": "#009e73"}
    width = 0.16
    x = np.arange(len(method_rows), dtype=float)

    for offset_idx, (metric_key, metric_label) in enumerate(zip(metrics, metric_labels)):
        offsets = x + (offset_idx - 1) * width
        vals = [row[1][metric_key] for row in method_rows]
        ax.bar(offsets, vals, width=width, label=metric_label, color=metric_colors[metric_key], alpha=0.9)

    ax.set_ylabel("Correlation")
    ax.set_title("Compositional Projection Model: Item Generalization Correlations", loc="center", pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels([row[0] for row in method_rows], rotation=12, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(right=0.8, top=0.87)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _plot_mbr_l2(output_path: Path) -> None:
    method_rows: list[tuple[str, float]] = []

    marformer = _load_marformer_metric(300, "rmse", nontrans=False)
    if marformer is not None:
        method_rows.append(("Marformer", float(marformer) ** 2))

    stan_tensor = _load_stan_probs_and_labels(300, discrete=False, nontrans=False)
    if stan_tensor is not None:
        method_rows.append(("Oracle Tensor Stan", _mbr_l2_from_probs_labels(*stan_tensor)))

    stan_discrete = _load_stan_probs_and_labels(300, discrete=True, nontrans=False)
    if stan_discrete is not None:
        method_rows.append(("Stan Discrete", _mbr_l2_from_probs_labels(*stan_discrete)))

    marformer_nt = _load_marformer_metric(300, "rmse", nontrans=True)
    if marformer_nt is not None:
        method_rows.append(("Marformer NT", float(marformer_nt) ** 2))

    stan_tensor_nt = _load_stan_probs_and_labels(300, discrete=False, nontrans=True)
    if stan_tensor_nt is not None:
        method_rows.append(("Oracle Tensor Stan NT", _mbr_l2_from_probs_labels(*stan_tensor_nt)))

    unigram = _load_unigram_probs_and_labels(300)
    if unigram is not None:
        method_rows.append(("Unigram", _mbr_l2_from_probs_labels(*unigram)))

    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    x = np.arange(len(method_rows), dtype=float)
    colors = ["#1f6fba", "#d55e00", "#009e73", "#5d8fd0", "#f08a43", "#7a7a7a"][: len(method_rows)]
    vals = [row[1] for row in method_rows]
    ax.bar(x, vals, color=colors, alpha=0.9)
    ax.set_ylabel("MBR-L2 (MSE)")
    ax.set_title("Compositional Projection Model: Item Generalization MBR-L2 at Size 300", loc="center", pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels([row[0] for row in method_rows], rotation=12, ha="right")
    fig.subplots_adjust(top=0.87)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def main() -> None:
    marformer_loss = _series_from_loader(lambda size: _load_marformer_metric(size, "log_loss", nontrans=False))
    marformer_nt_loss = _series_from_loader(lambda size: _load_marformer_metric(size, "log_loss", nontrans=True))
    stan_tensor_loss = _series_from_loader(lambda size: _load_stan_missing_log_loss(size, discrete=False, nontrans=False))
    stan_tensor_nt_loss = _series_from_loader(lambda size: _load_stan_missing_log_loss(size, discrete=False, nontrans=True))
    stan_discrete_loss = _series_from_loader(lambda size: _load_stan_missing_log_loss(size, discrete=True, nontrans=False))
    unigram_loss = _series_from_loader(
        lambda size: None if _load_unigram_probs_and_labels(size) is None else _mean_xent(*_load_unigram_probs_and_labels(size))
    )

    print("\n[MBR-L2 @ largest available size]")
    print(f"Marformer:              {(_largest_available_mbr_l2_marformer(False) or float('nan')):.3f}")
    print(f"Marformer NT:           {(_largest_available_mbr_l2_marformer(True) or float('nan')):.3f}")
    print(f"Oracle Tensor Stan:     {(_largest_available_mbr_l2_stan(False, False) or float('nan')):.3f}")
    print(f"Oracle Tensor Stan NT:  {(_largest_available_mbr_l2_stan(False, True) or float('nan')):.3f}")
    print(f"Stan Discrete:          {(_largest_available_mbr_l2_stan(True, False) or float('nan')):.3f}")
    print(f"Unigram:                {(_largest_available_mbr_l2_unigram() or float('nan')):.3f}")

    _plot_metric_broken_y(
        OUT_DIR / "sparse_tensor_test_loss_by_size.png",
        ylabel="Test Missing Log Loss",
        title="Compositional Projection Model: Item Generalization by Training Size",
        series=[
            ("Marformer", *marformer_loss, {"color": "#1f6fba", "marker": "o"}),
            ("Oracle Tensor Stan", *stan_tensor_loss, {"color": "#d55e00", "marker": "s"}),
            ("Stan Discrete", *stan_discrete_loss, {"color": "#009e73", "marker": "^"}),
            ("Unigram", *unigram_loss, {"color": "#7a7a7a", "marker": "D", "linestyle": ":"}),
        ],
        break_at=1.1,
    )

    _plot_metric_broken_y(
        OUT_DIR / "sparse_tensor_test_loss_by_size_nt.png",
        ylabel="Test Missing Log Loss",
        title="Compositional Projection Model: Item Generalization by Training Size (Non-Transductive)",
        series=[
            ("Marformer", *marformer_loss, {"color": "#1f6fba", "marker": "o", "linestyle": "--"}),
            ("Marformer NT", *marformer_nt_loss, {"color": "#1f6fba", "marker": "o"}),
            ("Oracle Tensor Stan NT", *stan_tensor_nt_loss, {"color": "#d55e00", "marker": "s"}),
            ("Unigram", *unigram_loss, {"color": "#7a7a7a", "marker": "D", "linestyle": ":"}),
        ],
        break_at=1.1,
    )

    _plot_correlations(OUT_DIR / "sparse_tensor_correlation_by_method.png")
    _plot_mbr_l2(OUT_DIR / "sparse_tensor_mbr_l2_size300.png")


if __name__ == "__main__":
    main()
