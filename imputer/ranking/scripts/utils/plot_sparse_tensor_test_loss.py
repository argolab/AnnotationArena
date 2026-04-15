#!/usr/bin/env python3
"""
Plot sparse Tensor_400_25_9 item-test results by training size.

Outputs:
  - PLOTS/TALK/sparse_tensor_test_loss_by_size.png
  - PLOTS/TALK/sparse_tensor_test_rmse_by_size.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "DATA/STAN/SPARSE/Tensor_400_25_9_ItemTest"
MARFORMER_ROOT = ROOT / "RESULTS/MARFORMER/STAN/SPARSE"
STAN_ROOT = ROOT / "RESULTS/STAN/SPARSE/Tensor-400"
MAP_ROOT = ROOT / "RESULTS/EM/STAN/SPARSE"
OUT_DIR = ROOT / "PLOTS/TALK"

SIZES = [10, 50, 100, 200, 300]
PROB_COLS = ["prob_cat_1", "prob_cat_2", "prob_cat_3", "prob_cat_4", "prob_cat_5"]
DISCRETE_FALLBACK = {300: 200}

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


def _load_bundle(size: int) -> dict:
    return _read_json(_bundle_path(size))


def _test_missing_labels(size: int) -> np.ndarray:
    bundle = _load_bundle(size)
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


def _stan_metrics_path(size: int, discrete: bool) -> Path:
    source_size = _discrete_source_size(size) if discrete else size
    suffix = "_DISCRETE_MISP_eval" if discrete else "_eval"
    return STAN_ROOT / f"Tensor_400_25_9_ItemTest_{source_size}{suffix}" / "predictive_metrics.json"


def _stan_probs_path(size: int, discrete: bool) -> Path:
    source_size = _discrete_source_size(size) if discrete else size
    suffix = "_DISCRETE_MISP_eval" if discrete else "_eval"
    return STAN_ROOT / f"Tensor_400_25_9_ItemTest_{source_size}{suffix}" / "rating_probabilities.csv"


def _load_stan_missing_log_loss(size: int, discrete: bool) -> float | None:
    path = _stan_metrics_path(size, discrete)
    if not path.exists():
        return None
    data = _read_json(path)
    return float(-data["rating_missing_log_likelihood"])


def _load_stan_probs_and_labels(size: int, discrete: bool) -> tuple[np.ndarray, np.ndarray] | None:
    probs_path = _stan_probs_path(size, discrete)
    source_size = _discrete_source_size(size) if discrete else size
    if not probs_path.exists():
        return None

    bundle = _load_bundle(source_size)
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
            f"Stan predictions/labels mismatch for size={size} discrete={discrete}: "
            f"{probs.shape[0]} vs {labels.shape[0]}"
        )
    return probs, labels


def _select_best_marformer_json(size: int) -> Path | None:
    run_dir = MARFORMER_ROOT / f"Tensor_400_25_9_ItemTest_{size}_NOITEMDEV_TRANS_MARFORMER" / "TEST_RESULTS"
    candidates = sorted(run_dir.glob("best-*.json"))
    if not candidates:
        return None
    return min(candidates, key=lambda p: _read_json(p)["missing"]["log_loss"])


def _load_marformer_metric(size: int, key: str) -> float | None:
    best_json = _select_best_marformer_json(size)
    if best_json is None:
        return None
    data = _read_json(best_json)
    return float(data["missing"][key])


def _load_map_results(size: int) -> dict | None:
    path = MAP_ROOT / f"Tensor_400_25_9_ItemTest_{size}_MAP" / "results.json"
    if not path.exists():
        return None
    return _read_json(path)


def _load_map_metric(size: int, key: str) -> float | None:
    results = _load_map_results(size)
    if results is None:
        return None
    return float(results["test_missing"][key])


def _load_unigram_probs_and_labels(size: int) -> tuple[np.ndarray, np.ndarray] | None:
    bundle = _load_bundle(size)
    observed = [
        r for r in bundle["observed_ratings"]
        if r["instance"] in {"train", "val", "test"}
    ]
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


def _plot_metric(
    output_path: Path,
    ylabel: str,
    title: str,
    series: Iterable[tuple[str, list[int], list[float], dict]],
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    for label, xs, ys, style in series:
        if xs:
            ax.plot(xs, ys, label=label, **style)

    ax.set_xlabel("Training Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(SIZES)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.subplots_adjust(right=0.78)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def main() -> None:
    marformer_loss = _series_from_loader(lambda size: _load_marformer_metric(size, "log_loss"))
    stan_tensor_loss = _series_from_loader(lambda size: _load_stan_missing_log_loss(size, discrete=False))
    stan_discrete_loss = _series_from_loader(lambda size: _load_stan_missing_log_loss(size, discrete=True))
    map_loss = _series_from_loader(lambda size: _load_map_metric(size, "xent"))
    unigram_loss = _series_from_loader(
        lambda size: None
        if _load_unigram_probs_and_labels(size) is None
        else _mean_xent(*_load_unigram_probs_and_labels(size))
    )

    marformer_rmse = _series_from_loader(lambda size: _load_marformer_metric(size, "rmse"))
    stan_tensor_rmse = _series_from_loader(
        lambda size: None
        if _load_stan_probs_and_labels(size, discrete=False) is None
        else _expected_rmse(*_load_stan_probs_and_labels(size, discrete=False))
    )
    stan_discrete_rmse = _series_from_loader(
        lambda size: None
        if _load_stan_probs_and_labels(size, discrete=True) is None
        else _expected_rmse(*_load_stan_probs_and_labels(size, discrete=True))
    )
    map_rmse = _series_from_loader(lambda size: _load_map_metric(size, "rmse"))
    unigram_rmse = _series_from_loader(
        lambda size: None
        if _load_unigram_probs_and_labels(size) is None
        else _expected_rmse(*_load_unigram_probs_and_labels(size))
    )

    _plot_metric(
        OUT_DIR / "sparse_tensor_test_loss_by_size.png",
        ylabel="Test Missing Log Loss",
        title="Sparse Tensor Test Missing Log Loss by Training Size",
        series=[
            ("Marformer", *marformer_loss, {"color": "#1f6fba", "marker": "o"}),
            ("Stan Discrete", *stan_discrete_loss, {"color": "#8e44ad", "marker": "^"}),
            ("Stan Tensor", *stan_tensor_loss, {"color": "#e67e22", "marker": "s"}),
            ("MAP", *map_loss, {"color": "#16a085", "marker": "D"}),
            ("Unigram", *unigram_loss, {"color": "#7f8c8d", "marker": "P", "linestyle": "--"}),
        ],
    )

    _plot_metric(
        OUT_DIR / "sparse_tensor_test_rmse_by_size.png",
        ylabel="Test Missing RMSE",
        title="Sparse Tensor Test Missing RMSE by Training Size",
        series=[
            ("Marformer", *marformer_rmse, {"color": "#1f6fba", "marker": "o"}),
            ("Stan Discrete", *stan_discrete_rmse, {"color": "#8e44ad", "marker": "^"}),
            ("Stan Tensor", *stan_tensor_rmse, {"color": "#e67e22", "marker": "s"}),
            ("MAP", *map_rmse, {"color": "#16a085", "marker": "D"}),
            ("Unigram", *unigram_rmse, {"color": "#7f8c8d", "marker": "P", "linestyle": "--"}),
        ],
    )


if __name__ == "__main__":
    main()
