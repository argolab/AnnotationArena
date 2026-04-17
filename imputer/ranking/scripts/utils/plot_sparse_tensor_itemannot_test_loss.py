#!/usr/bin/env python3
"""
Plot sparse Tensor_400_25_9 item+annotator-test results by training size.

Outputs:
  - PLOTS/TALK/ItemAnnot/sparse_tensor_itemannot_test_loss_by_size.png
  - PLOTS/TALK/ItemAnnot/sparse_tensor_itemannot_runtime_by_size.png
  - PLOTS/TALK/ItemAnnot/sparse_tensor_itemannot_correlation_by_method.png
  - PLOTS/TALK/ItemAnnot/sparse_tensor_itemannot_mbr_l2_size300_15.png
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
DATA_ROOT = ROOT / "DATA/STAN/SPARSE/Tensor_400_25_9_ItemAnnotTest"
MARFORMER_ROOT = ROOT / "RESULTS/MARFORMER/STAN/SPARSE/TensorItemAnnot400"
MARFORMER_NT_ROOT = ROOT / "RESULTS/MARFORMER/STAN/SPARSE/TensorItemAnnot400NT"
STAN_ROOT = ROOT / "RESULTS/STAN/SPARSE/Tensor400ItemAnnot"
STAN_NT_ROOT = ROOT / "RESULTS/STAN/SPARSE/Tensor400ItemAnnotNT"
OUT_DIR = ROOT / "PLOTS/TALK/ItemAnnot"

SIZE_PAIRS = [(10, 5), (50, 5), (100, 10), (200, 15), (300, 15)]
X_POS = np.arange(len(SIZE_PAIRS), dtype=float)
X_LABELS = [f"{items}/{anns}" for items, anns in SIZE_PAIRS]
PROB_COLS = ["prob_cat_1", "prob_cat_2", "prob_cat_3", "prob_cat_4", "prob_cat_5"]
DISCRETE_FALLBACK = {(300, 15): (200, 15)}
STAN_TENSOR_NT_FALLBACK: dict[tuple[int, int], tuple[int, int]] = {}

MARFORMER_RUNTIMES_MIN = {
    (10, 5): 27 + 2 / 60,
    (50, 5): 34 + 23 / 60,
    (100, 10): 67 + 8 / 60,
    (200, 15): 122 + 27 / 60,
    (300, 15): 163 + 50 / 60,
}
STAN_TENSOR_RUNTIMES_MIN = {
    (10, 5): 30 + 13 / 60,
    (50, 5): 49 + 32 / 60,
    (100, 10): 69 + 56 / 60,
    (200, 15): 394 + 18 / 60,
    (300, 15): 458 + 9 / 60,
}
STAN_DISCRETE_RUNTIMES_MIN = {
    (10, 5): 162 + 17 / 60,
    (50, 5): 127 + 30 / 60,
    (100, 10): 258 + 45 / 60,
    (200, 15): 515 + 38 / 60,
    (300, 15): 36 * 60,
}

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


def _bundle_path(items: int, anns: int) -> Path:
    return DATA_ROOT / f"Tensor_400_25_9_ItemAnnotTest_{items}_{anns}" / "data_bundle.json"


def _load_bundle(items: int, anns: int) -> dict | None:
    path = _bundle_path(items, anns)
    if not path.exists():
        return None
    return _read_json(path)


def _expected_rmse(probs: np.ndarray, labels: np.ndarray) -> float:
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    expected = probs.astype(np.float64) @ classes
    truth = labels.astype(np.float64) + 1.0
    return float(np.sqrt(np.mean((expected - truth) ** 2)))


def _test_missing_labels(items: int, anns: int) -> np.ndarray | None:
    bundle = _load_bundle(items, anns)
    if bundle is None:
        return None
    test_idxs = bundle["missing_ratings_indexes_in_test_instance"]
    return np.asarray(
        [bundle["missing_ratings"][idx]["value"] - 1 for idx in test_idxs],
        dtype=np.int64,
    )


def _discrete_source_pair(items: int, anns: int) -> tuple[int, int]:
    return DISCRETE_FALLBACK.get((items, anns), (items, anns))


def _stan_tensor_nt_source_pair(items: int, anns: int) -> tuple[int, int]:
    return STAN_TENSOR_NT_FALLBACK.get((items, anns), (items, anns))


def _stan_source_pair(items: int, anns: int, discrete: bool, nontrans: bool) -> tuple[int, int]:
    if discrete:
        return _discrete_source_pair(items, anns)
    if nontrans:
        return _stan_tensor_nt_source_pair(items, anns)
    return items, anns


def _stan_eval_dir(items: int, anns: int, discrete: bool, nontrans: bool) -> Path:
    source_items, source_anns = _stan_source_pair(items, anns, discrete, nontrans)
    root = STAN_NT_ROOT if nontrans and not discrete else STAN_ROOT
    if discrete:
        run_name = f"Tensor_400_25_9_ItemAnnotTest_{source_items}_{source_anns}_DISCRETE_MISP_eval"
    elif nontrans:
        run_name = f"Tensor_400_25_9_ItemAnnotTest_{source_items}_{source_anns}_NONTRANS_eval"
    else:
        run_name = f"Tensor_400_25_9_ItemAnnotTest_{source_items}_{source_anns}_eval"
    return root / run_name


def _stan_metrics_path(items: int, anns: int, discrete: bool, nontrans: bool = False) -> Path:
    return _stan_eval_dir(items, anns, discrete, nontrans) / "predictive_metrics.json"


def _stan_probs_path(items: int, anns: int, discrete: bool, nontrans: bool = False) -> Path:
    return _stan_eval_dir(items, anns, discrete, nontrans) / "rating_probabilities.csv"


def _load_stan_missing_log_loss(items: int, anns: int, discrete: bool, nontrans: bool = False) -> float | None:
    path = _stan_metrics_path(items, anns, discrete, nontrans)
    if not path.exists():
        return None
    data = _read_json(path)
    return float(-data["rating_missing_log_likelihood"])


def _load_stan_probs_and_labels(
    items: int,
    anns: int,
    discrete: bool,
    nontrans: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    source_items, source_anns = _stan_source_pair(items, anns, discrete, nontrans)
    probs_path = _stan_probs_path(items, anns, discrete, nontrans)
    bundle = _load_bundle(source_items, source_anns)
    if not probs_path.exists() or bundle is None:
        return None

    test_idxs = bundle["missing_ratings_indexes_in_test_instance"]
    labels = np.asarray(
        [bundle["missing_ratings"][idx]["value"] - 1 for idx in test_idxs],
        dtype=np.int64,
    )
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
            f"Stan predictions/labels mismatch for size=({items}, {anns}) discrete={discrete}: "
            f"{probs.shape[0]} vs {labels.shape[0]}"
        )
    return probs, labels


def _select_best_marformer_json(items: int, anns: int, nontrans: bool = False) -> Path | None:
    root = MARFORMER_NT_ROOT if nontrans else MARFORMER_ROOT
    suffix = "NOITEMDEV_NONTRANS_MARFORMER" if nontrans else "NOITEMDEV_TRANS_MARFORMER"
    run_dir = root / f"Tensor_400_25_9_ItemAnnotTest_{items}_{anns}_{suffix}" / "TEST_RESULTS"
    candidates = sorted(run_dir.glob("best-*.json"))
    if not candidates:
        return None
    return min(candidates, key=lambda p: _read_json(p)["missing"]["log_loss"])


def _load_marformer_metric(items: int, anns: int, key: str, nontrans: bool = False) -> float | None:
    best_json = _select_best_marformer_json(items, anns, nontrans=nontrans)
    if best_json is None:
        return None
    data = _read_json(best_json)
    return float(data["missing"][key])


def _mbr_l2_from_probs_labels(probs: np.ndarray, labels: np.ndarray) -> float:
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    expected = probs.astype(np.float64) @ classes
    truth = labels.astype(np.float64) + 1.0
    return float(np.mean((expected - truth) ** 2))


def _largest_available_mbr_l2_marformer(nontrans: bool = False) -> float | None:
    for items, anns in reversed(SIZE_PAIRS):
        rmse = _load_marformer_metric(items, anns, "rmse", nontrans=nontrans)
        if rmse is not None:
            return float(rmse) ** 2
    return None


def _largest_available_mbr_l2_stan(discrete: bool, nontrans: bool = False) -> float | None:
    for items, anns in reversed(SIZE_PAIRS):
        loaded = _load_stan_probs_and_labels(items, anns, discrete=discrete, nontrans=nontrans)
        if loaded is not None:
            probs, labels = loaded
            return _mbr_l2_from_probs_labels(probs, labels)
    return None


def _load_unigram_probs_and_labels(items: int, anns: int) -> tuple[np.ndarray, np.ndarray] | None:
    bundle = _load_bundle(items, anns)
    labels = _test_missing_labels(items, anns)
    if bundle is None or labels is None or labels.size == 0:
        return None

    class_mass = np.zeros(len(PROB_COLS), dtype=np.float64)
    for rating in bundle["observed_ratings"]:
        dist = rating.get("rating_dist")
        if dist is not None:
            class_mass += np.asarray(dist, dtype=np.float64)
        else:
            class_mass[int(rating["value"]) - 1] += 1.0

    total = float(class_mass.sum())
    if total <= 0.0:
        return None

    probs = class_mass / total
    return np.tile(probs[None, :], (labels.shape[0], 1)), labels


def _load_unigram_missing_log_loss(items: int, anns: int) -> float | None:
    loaded = _load_unigram_probs_and_labels(items, anns)
    if loaded is None:
        return None
    probs, labels = loaded
    eps = 1e-12
    return float(-np.mean(np.log(np.clip(probs[np.arange(labels.shape[0]), labels], eps, 1.0))))


def _load_unigram_rmse(items: int, anns: int) -> float | None:
    loaded = _load_unigram_probs_and_labels(items, anns)
    if loaded is None:
        return None
    return _expected_rmse(*loaded)


def _largest_available_mbr_l2_unigram() -> float | None:
    for items, anns in reversed(SIZE_PAIRS):
        loaded = _load_unigram_probs_and_labels(items, anns)
        if loaded is not None:
            probs, labels = loaded
            return _mbr_l2_from_probs_labels(probs, labels)
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


def _load_marformer_correlations(items: int, anns: int, nontrans: bool = False) -> dict[str, float] | None:
    best_json = _select_best_marformer_json(items, anns, nontrans=nontrans)
    if best_json is None:
        return None
    missing = _read_json(best_json)["missing"]
    return {
        "pearson": float(missing["pearson_r"]),
        "spearman": float(missing["spearman_r"]),
        "kendall": float(missing["kendall_tau"]),
    }


def _load_stan_correlations(items: int, anns: int, discrete: bool, nontrans: bool = False) -> dict[str, float] | None:
    loaded = _load_stan_probs_and_labels(items, anns, discrete=discrete, nontrans=nontrans)
    if loaded is None:
        return None
    probs, labels = loaded
    preds = _expected_values_from_probs(probs)
    return _correlations_from_preds_labels(preds, labels)


def _series_from_loader(loader) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for xpos, (items, anns) in zip(X_POS, SIZE_PAIRS):
        value = loader(items, anns)
        if value is None:
            continue
        xs.append(float(xpos))
        ys.append(float(value))
    return xs, ys


def _series_from_runtime(runtime_map: dict[tuple[int, int], float]) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for xpos, pair in zip(X_POS, SIZE_PAIRS):
        value = runtime_map.get(pair)
        if value is None:
            continue
        xs.append(float(xpos))
        ys.append(float(value) / 60.0)
    return xs, ys


def _plot_metric(
    output_path: Path,
    ylabel: str,
    title: str,
    series: Iterable[tuple[str, list[float], list[float], dict]],
) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 8.5))
    for label, xs, ys, style in series:
        if xs:
            ax.plot(xs, ys, label=label, **style)

    ax.set_xlabel("Train Size (Items / Annotators)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="center", pad=22)
    ax.set_xticks(X_POS)
    ax.set_xticklabels(X_LABELS)
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
    series: Iterable[tuple[str, list[float], list[float], dict]],
    break_at: float,
) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(15.5, 8.5),
        gridspec_kw={"height_ratios": [1.0, 4.2], "hspace": 0.05},
    )

    all_y: list[float] = []
    for label, xs, ys, style in series:
        all_y.extend(ys)
        if xs:
            ax_top.plot(xs, ys, label=label, **style)
            ax_bottom.plot(xs, ys, label=label, **style)

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

    ax_bottom.set_xlabel("Train Size (Items / Annotators)")
    ax_bottom.set_ylabel(ylabel)
    ax_bottom.set_xticks(X_POS)
    ax_bottom.set_xticklabels(X_LABELS)
    fig.suptitle(title, x=0.5, y=0.98, ha="center")
    ax_top.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(right=0.78, top=0.92)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _plot_runtime(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 7.0))

    runtime_series = [
        ("Marformer", *_series_from_runtime(MARFORMER_RUNTIMES_MIN), {"color": "#1f6fba", "marker": "o"}),
        ("Stan Tensor", *_series_from_runtime(STAN_TENSOR_RUNTIMES_MIN), {"color": "#d55e00", "marker": "s"}),
        ("Stan Discrete", *_series_from_runtime(STAN_DISCRETE_RUNTIMES_MIN), {"color": "#009e73", "marker": "^"}),
    ]

    for label, xs, ys, style in runtime_series:
        if xs:
            ax.plot(xs, ys, label=label, **style)

    x_last = X_POS[-1]
    y_last = STAN_DISCRETE_RUNTIMES_MIN[(300, 15)] / 60.0
    ax.annotate(
        ">36h",
        xy=(x_last, y_last),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=11,
        color="#009e73",
    )

    ax.set_xlabel("Train Size (items/annotators)")
    ax.set_ylabel("Runtime (hours)")
    ax.set_title("Sparse Tensor Item+Annotator Runtime by Training Size")
    ax.set_xticks(X_POS)
    ax.set_xticklabels(X_LABELS)
    ax.set_yscale("log")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.subplots_adjust(right=0.78)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _plot_correlations(output_path: Path) -> None:
    method_rows: list[tuple[str, dict[str, float]]] = []

    marformer_corr = _load_marformer_correlations(300, 15, nontrans=False)
    if marformer_corr is not None:
        method_rows.append(("Marformer", marformer_corr))

    stan_tensor_corr = _load_stan_correlations(300, 15, discrete=False, nontrans=False)
    if stan_tensor_corr is not None:
        method_rows.append(("Oracle Tensor Stan", stan_tensor_corr))

    stan_discrete_corr = _load_stan_correlations(300, 15, discrete=True, nontrans=False)
    if stan_discrete_corr is not None:
        method_rows.append(("Stan Discrete", stan_discrete_corr))

    marformer_nt_corr = _load_marformer_correlations(300, 15, nontrans=True)
    if marformer_nt_corr is not None:
        method_rows.append(("Marformer NT", marformer_nt_corr))

    stan_tensor_nt_corr = _load_stan_correlations(300, 15, discrete=False, nontrans=True)
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
    ax.set_title("Compositional Projection Model: Item-Annotator Generalization Correlations", loc="center", pad=18)
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

    marformer = _load_marformer_metric(300, 15, "rmse", nontrans=False)
    if marformer is not None:
        method_rows.append(("Marformer", float(marformer) ** 2))

    stan_tensor = _load_stan_probs_and_labels(300, 15, discrete=False, nontrans=False)
    if stan_tensor is not None:
        method_rows.append(("Oracle Tensor Stan", _mbr_l2_from_probs_labels(*stan_tensor)))

    stan_discrete = _load_stan_probs_and_labels(300, 15, discrete=True, nontrans=False)
    if stan_discrete is not None:
        method_rows.append(("Stan Discrete", _mbr_l2_from_probs_labels(*stan_discrete)))

    marformer_nt = _load_marformer_metric(300, 15, "rmse", nontrans=True)
    if marformer_nt is not None:
        method_rows.append(("Marformer NT", float(marformer_nt) ** 2))

    stan_tensor_nt = _load_stan_probs_and_labels(300, 15, discrete=False, nontrans=True)
    if stan_tensor_nt is not None:
        method_rows.append(("Oracle Tensor Stan NT", _mbr_l2_from_probs_labels(*stan_tensor_nt)))

    unigram = _load_unigram_probs_and_labels(300, 15)
    if unigram is not None:
        method_rows.append(("Unigram", _mbr_l2_from_probs_labels(*unigram)))

    fig, ax = plt.subplots(figsize=(13.8, 7.0))
    x = np.arange(len(method_rows), dtype=float)
    colors = ["#1f6fba", "#d55e00", "#009e73", "#5d8fd0", "#f08a43", "#7a7a7a"][: len(method_rows)]
    vals = [row[1] for row in method_rows]
    ax.bar(x, vals, color=colors, alpha=0.9)
    ax.set_ylabel("MBR-L2 (MSE)")
    ax.set_title("Compositional Projection Model: Item-Annotator Generalization MBR-L2 at Size 300/15", loc="center", pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels([row[0] for row in method_rows], rotation=12, ha="right")
    fig.subplots_adjust(top=0.87)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def main() -> None:
    marformer_loss = _series_from_loader(
        lambda items, anns: _load_marformer_metric(items, anns, "log_loss", nontrans=False)
    )
    marformer_nt_loss = _series_from_loader(
        lambda items, anns: _load_marformer_metric(items, anns, "log_loss", nontrans=True)
    )
    stan_tensor_loss = _series_from_loader(
        lambda items, anns: _load_stan_missing_log_loss(items, anns, discrete=False, nontrans=False)
    )
    stan_tensor_nt_loss = _series_from_loader(
        lambda items, anns: _load_stan_missing_log_loss(items, anns, discrete=False, nontrans=True)
    )
    stan_discrete_loss = _series_from_loader(lambda items, anns: _load_stan_missing_log_loss(items, anns, discrete=True))
    unigram_loss = _series_from_loader(lambda items, anns: _load_unigram_missing_log_loss(items, anns))

    print("\n[MBR-L2 @ largest available size]")
    print(f"Marformer:              {(_largest_available_mbr_l2_marformer(False) or float('nan')):.3f}")
    print(f"Marformer NT:           {(_largest_available_mbr_l2_marformer(True) or float('nan')):.3f}")
    print(f"Oracle Tensor Stan:     {(_largest_available_mbr_l2_stan(False, False) or float('nan')):.3f}")
    print(f"Oracle Tensor Stan NT:  {(_largest_available_mbr_l2_stan(False, True) or float('nan')):.3f}")
    print(f"Stan Discrete:          {(_largest_available_mbr_l2_stan(True, False) or float('nan')):.3f}")
    print(f"Unigram:                {(_largest_available_mbr_l2_unigram() or float('nan')):.3f}")

    _plot_metric_broken_y(
        OUT_DIR / "sparse_tensor_itemannot_test_loss_by_size.png",
        ylabel="Test Missing Log Loss",
        title="Compositional Projection Model: Item-Annotator Generalization by Training Size",
        series=[
            ("Marformer", *marformer_loss, {"color": "#1f6fba", "marker": "o"}),
            ("Marformer NT", *marformer_nt_loss, {"color": "#1f6fba", "marker": "o", "linestyle": "--"}),
            ("Oracle Tensor Stan", *stan_tensor_loss, {"color": "#d55e00", "marker": "s"}),
            ("Oracle Tensor Stan NT", *stan_tensor_nt_loss, {"color": "#d55e00", "marker": "s", "linestyle": "--"}),
            ("Stan Discrete", *stan_discrete_loss, {"color": "#009e73", "marker": "^"}),
            ("Unigram", *unigram_loss, {"color": "#7a7a7a", "marker": "D", "linestyle": ":"}),
        ],
        break_at=1.5,
    )

    _plot_runtime(OUT_DIR / "sparse_tensor_itemannot_runtime_by_size.png")
    _plot_correlations(OUT_DIR / "sparse_tensor_itemannot_correlation_by_method.png")
    _plot_mbr_l2(OUT_DIR / "sparse_tensor_itemannot_mbr_l2_size300_15.png")


if __name__ == "__main__":
    main()
