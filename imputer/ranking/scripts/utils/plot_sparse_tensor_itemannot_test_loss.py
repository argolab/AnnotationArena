#!/usr/bin/env python3
"""
Plot sparse Tensor_400_25_9 item+annotator-test results by training size.

Outputs:
  - PLOTS/TALK/sparse_tensor_itemannot_test_loss_by_size.png
  - PLOTS/TALK/sparse_tensor_itemannot_test_rmse_by_size.png
  - PLOTS/TALK/sparse_tensor_itemannot_runtime_by_size.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "DATA/STAN/SPARSE/Tensor_400_25_9_ItemAnnotTest"
MARFORMER_ROOT = ROOT / "RESULTS/MARFORMER/STAN/SPARSE/TensorItemAnnot400"
STAN_ROOT = ROOT / "RESULTS/STAN/SPARSE/Tensor400ItemAnnot"
OUT_DIR = ROOT / "PLOTS/TALK"

SIZE_PAIRS = [(10, 5), (50, 5), (100, 10), (200, 15), (300, 15)]
X_POS = np.arange(len(SIZE_PAIRS), dtype=float)
X_LABELS = [f"{items}/{anns}" for items, anns in SIZE_PAIRS]
PROB_COLS = ["prob_cat_1", "prob_cat_2", "prob_cat_3", "prob_cat_4", "prob_cat_5"]
DISCRETE_FALLBACK = {(300, 15): (200, 15)}

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


def _discrete_source_pair(items: int, anns: int) -> tuple[int, int]:
    return DISCRETE_FALLBACK.get((items, anns), (items, anns))


def _stan_metrics_path(items: int, anns: int, discrete: bool) -> Path:
    if discrete:
        items, anns = _discrete_source_pair(items, anns)
    suffix = "_DISCRETE_MISP_eval" if discrete else "_eval"
    return STAN_ROOT / f"Tensor_400_25_9_ItemAnnotTest_{items}_{anns}{suffix}" / "predictive_metrics.json"


def _stan_probs_path(items: int, anns: int, discrete: bool) -> Path:
    if discrete:
        items, anns = _discrete_source_pair(items, anns)
    suffix = "_DISCRETE_MISP_eval" if discrete else "_eval"
    return STAN_ROOT / f"Tensor_400_25_9_ItemAnnotTest_{items}_{anns}{suffix}" / "rating_probabilities.csv"


def _load_stan_missing_log_loss(items: int, anns: int, discrete: bool) -> float | None:
    path = _stan_metrics_path(items, anns, discrete)
    if not path.exists():
        return None
    data = _read_json(path)
    return float(-data["rating_missing_log_likelihood"])


def _load_stan_probs_and_labels(items: int, anns: int, discrete: bool) -> tuple[np.ndarray, np.ndarray] | None:
    source_items, source_anns = _discrete_source_pair(items, anns) if discrete else (items, anns)
    probs_path = _stan_probs_path(items, anns, discrete)
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


def _select_best_marformer_json(items: int, anns: int) -> Path | None:
    run_dir = (
        MARFORMER_ROOT
        / f"Tensor_400_25_9_ItemAnnotTest_{items}_{anns}_NOITEMDEV_TRANS_MARFORMER"
        / "TEST_RESULTS"
    )
    candidates = sorted(run_dir.glob("best-*.json"))
    if not candidates:
        return None
    return min(candidates, key=lambda p: _read_json(p)["missing"]["log_loss"])


def _load_marformer_metric(items: int, anns: int, key: str) -> float | None:
    best_json = _select_best_marformer_json(items, anns)
    if best_json is None:
        return None
    data = _read_json(best_json)
    return float(data["missing"][key])


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
    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    for label, xs, ys, style in series:
        if xs:
            ax.plot(xs, ys, label=label, **style)

    ax.set_xlabel("Train Size (items/annotators)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(X_POS)
    ax.set_xticklabels(X_LABELS)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.subplots_adjust(right=0.78)

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


def main() -> None:
    marformer_loss = _series_from_loader(lambda items, anns: _load_marformer_metric(items, anns, "log_loss"))
    stan_tensor_loss = _series_from_loader(lambda items, anns: _load_stan_missing_log_loss(items, anns, discrete=False))
    stan_discrete_loss = _series_from_loader(lambda items, anns: _load_stan_missing_log_loss(items, anns, discrete=True))

    marformer_rmse = _series_from_loader(lambda items, anns: _load_marformer_metric(items, anns, "rmse"))
    stan_tensor_rmse = _series_from_loader(
        lambda items, anns: None
        if _load_stan_probs_and_labels(items, anns, discrete=False) is None
        else _expected_rmse(*_load_stan_probs_and_labels(items, anns, discrete=False))
    )
    stan_discrete_rmse = _series_from_loader(
        lambda items, anns: None
        if _load_stan_probs_and_labels(items, anns, discrete=True) is None
        else _expected_rmse(*_load_stan_probs_and_labels(items, anns, discrete=True))
    )

    _plot_metric(
        OUT_DIR / "sparse_tensor_itemannot_test_loss_by_size.png",
        ylabel="Test Missing Log Loss",
        title="Sparse Tensor Item+Annotator Test Missing Log Loss by Training Size",
        series=[
            ("Marformer", *marformer_loss, {"color": "#1f6fba", "marker": "o"}),
            ("Stan Tensor", *stan_tensor_loss, {"color": "#d55e00", "marker": "s"}),
            ("Stan Discrete", *stan_discrete_loss, {"color": "#009e73", "marker": "^"}),
        ],
    )

    _plot_metric(
        OUT_DIR / "sparse_tensor_itemannot_test_rmse_by_size.png",
        ylabel="Test Missing RMSE",
        title="Sparse Tensor Item+Annotator Test Missing RMSE by Training Size",
        series=[
            ("Marformer", *marformer_rmse, {"color": "#1f6fba", "marker": "o"}),
            ("Stan Tensor", *stan_tensor_rmse, {"color": "#d55e00", "marker": "s"}),
            ("Stan Discrete", *stan_discrete_rmse, {"color": "#009e73", "marker": "^"}),
        ],
    )

    _plot_runtime(OUT_DIR / "sparse_tensor_itemannot_runtime_by_size.png")


if __name__ == "__main__":
    main()
