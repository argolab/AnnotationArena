#!/usr/bin/env python3
"""
Plot real-data ranking results for LLM Rubric and SummEval.

Outputs:
  - PLOTS/TALK/LLMRubric/llm_rubric_test_loss_by_size.png
  - PLOTS/TALK/LLMRubric/llm_rubric_mbr_l2_size175.png
  - PLOTS/TALK/LLMRubric/llm_rubric_runtime_by_size.png
  - PLOTS/TALK/SummEval/summeval_test_loss_by_size.png
  - PLOTS/TALK/SummEval/summeval_mbr_l2_size1280.png
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLOTS_ROOT = ROOT / "PLOTS/TALK"

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "legend.framealpha": 0.92,
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

PROB_COL_TEMPLATE = "prob_cat_{idx}"
COLORS = {
    "Marformer": "#1f6fba",
    "CPM Stan": "#1b9e77",
    "Stan Factor": "#27ae60",
    "Stan Normal": "#e67e22",
    "REMASKER": "#8e44ad",
    "MIWAE": "#c0392b",
}
MARKERS = {
    "Marformer": "o",
    "CPM Stan": "^",
    "Stan Factor": "^",
    "Stan Normal": "D",
    "REMASKER": "s",
    "MIWAE": "P",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    pretty_name: str
    sizes: list[int]
    num_classes: int
    data_root: Path
    marformer_root: Path
    stan_root: Path
    baseline_roots: dict[str, Path]
    marformer_run: Callable[[int], str]
    stan_eval_run: Callable[[int, str], str]
    baseline_run: Callable[[int], str]
    out_dir: Path
    x_label: str
    marformer_runtime_seconds: dict[int, float] | None = None
    stan_runtime_seconds: dict[str, dict[int, float]] | None = None


def _read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _test_missing_indices_and_labels(bundle: dict) -> tuple[list[int], np.ndarray]:
    missing = bundle.get("missing_ratings", [])
    idxs = [i for i, row in enumerate(missing) if row.get("instance") == "test"]
    labels = np.asarray([missing[i]["value"] - 1 for i in idxs], dtype=np.int64)
    return idxs, labels


def _expected_score(probs: np.ndarray) -> np.ndarray:
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    return probs.astype(np.float64) @ classes


def _mse_from_probs_labels(probs: np.ndarray, labels: np.ndarray) -> float:
    truth = labels.astype(np.float64) + 1.0
    pred = _expected_score(probs)
    return float(np.mean((pred - truth) ** 2))


def _mean_nll_from_probs_labels(probs: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def _marformer_best_json(spec: DatasetSpec, size: int) -> Path | None:
    run_dir = spec.marformer_root / spec.marformer_run(size) / "TEST_RESULTS"
    preferred = run_dir / "best.json"
    if preferred.exists():
        return preferred
    candidates = sorted(run_dir.glob("best*.json"))
    return candidates[0] if candidates else None


def _marformer_missing_metrics(spec: DatasetSpec, size: int) -> dict | None:
    path = _marformer_best_json(spec, size)
    if path is None:
        return None
    return _read_json(path).get("missing")


def _stan_probs_and_labels(spec: DatasetSpec, size: int, variant: str) -> tuple[np.ndarray, np.ndarray] | None:
    eval_dir = spec.stan_root / spec.stan_eval_run(size, variant)
    probs_path = eval_dir / "rating_probabilities.csv"
    bundle_path = spec.data_root / spec.baseline_run(size) / "data_bundle.json"
    if not probs_path.exists() or not bundle_path.exists():
        return None

    bundle = _read_json(bundle_path)
    test_idxs, labels = _test_missing_indices_and_labels(bundle)
    if not test_idxs:
        return None

    df = pd.read_csv(probs_path)
    prob_cols = [PROB_COL_TEMPLATE.format(idx=i) for i in range(1, spec.num_classes + 1)]
    grouped = (
        df[df["missing_rating_idx"].isin(test_idxs)]
        .groupby("missing_rating_idx")[prob_cols]
        .mean()
        .reindex(test_idxs)
    )
    if grouped.isnull().any().any():
        return None
    probs = grouped.to_numpy(dtype=np.float64)
    if probs.shape[0] != labels.shape[0]:
        return None
    return probs, labels


def _stan_missing_log_loss(spec: DatasetSpec, size: int, variant: str) -> float | None:
    path = spec.stan_root / spec.stan_eval_run(size, variant) / "predictive_metrics.json"
    if not path.exists():
        return None
    data = _read_json(path)
    ll = data.get("rating_missing_log_likelihood")
    if ll is None:
        return None
    return float(-ll)


def _llm_rubric_cpm_probs_and_labels(spec: DatasetSpec, size: int) -> tuple[np.ndarray, np.ndarray] | None:
    eval_dir = spec.stan_root / f"LLMRubric_225_25_9_{size}_eval"
    probs_path = eval_dir / "rating_probabilities.csv"
    bundle_path = spec.data_root / spec.baseline_run(size) / "data_bundle.json"
    if not probs_path.exists() or not bundle_path.exists():
        return None

    bundle = _read_json(bundle_path)
    test_idxs, labels = _test_missing_indices_and_labels(bundle)
    if not test_idxs:
        return None

    df = pd.read_csv(probs_path)
    prob_cols = [PROB_COL_TEMPLATE.format(idx=i) for i in range(1, spec.num_classes + 1)]
    grouped = (
        df[df["missing_rating_idx"].isin(test_idxs)]
        .groupby("missing_rating_idx")[prob_cols]
        .mean()
        .reindex(test_idxs)
    )
    if grouped.isnull().any().any():
        return None
    probs = grouped.to_numpy(dtype=np.float64)
    if probs.shape[0] != labels.shape[0]:
        return None
    return probs, labels


def _llm_rubric_cpm_log_loss(spec: DatasetSpec, size: int) -> float | None:
    path = spec.stan_root / f"LLMRubric_225_25_9_{size}_eval" / "predictive_metrics.json"
    if not path.exists():
        return None
    data = _read_json(path)
    ll = data.get("rating_missing_log_likelihood")
    if ll is None:
        return None
    return float(-ll)


def _baseline_probs_and_labels(spec: DatasetSpec, method: str, size: int) -> tuple[np.ndarray, np.ndarray] | None:
    pred_path = spec.baseline_roots[method] / spec.baseline_run(size) / "test_predictions.json"
    if not pred_path.exists():
        return None
    rows = _read_json(pred_path)
    if not rows:
        return None
    labels = np.asarray([row["true_label"] for row in rows], dtype=np.int64)
    probs = np.asarray([row["probs"] for row in rows], dtype=np.float64)
    return probs, labels


def _baseline_missing_log_loss(spec: DatasetSpec, method: str, size: int) -> float | None:
    summary_path = spec.baseline_roots[method] / spec.baseline_run(size) / "summary.json"
    if summary_path.exists():
        data = _read_json(summary_path)
        metrics = data.get("metrics", {})
        if "mean_nll" in metrics:
            return float(metrics["mean_nll"])
    payload = _baseline_probs_and_labels(spec, method, size)
    if payload is None:
        return None
    probs, labels = payload
    return _mean_nll_from_probs_labels(probs, labels)


def _empirical_unigram_probs_and_labels(spec: DatasetSpec, size: int) -> tuple[np.ndarray, np.ndarray] | None:
    bundle_path = spec.data_root / spec.baseline_run(size) / "data_bundle.json"
    if not bundle_path.exists():
        return None
    bundle = _read_json(bundle_path)
    observed = bundle.get("observed_ratings", [])
    test_missing = [row for row in bundle.get("missing_ratings", []) if row.get("instance") == "test"]
    if not observed or not test_missing:
        return None

    counts = np.zeros(spec.num_classes, dtype=np.float64)
    for row in observed:
        dist = row.get("rating_dist")
        if dist is not None:
            counts += np.asarray(dist, dtype=np.float64)
        else:
            counts[int(row["value"]) - 1] += 1.0
    if counts.sum() <= 0:
        return None
    probs = counts / counts.sum()
    labels = np.asarray([row["value"] - 1 for row in test_missing], dtype=np.int64)
    tiled = np.tile(probs[None, :], (labels.shape[0], 1))
    return tiled, labels


def _empirical_unigram_log_loss(spec: DatasetSpec, size: int) -> float | None:
    payload = _empirical_unigram_probs_and_labels(spec, size)
    if payload is None:
        return None
    return _mean_nll_from_probs_labels(*payload)


def _collect_loss_series(spec: DatasetSpec) -> dict[str, tuple[list[int], list[float]]]:
    series: dict[str, tuple[list[int], list[float]]] = {}

    def collect(method: str, fn: Callable[[int], float | None]) -> None:
        xs, ys = [], []
        for size in spec.sizes:
            value = fn(size)
            if value is not None:
                xs.append(size)
                ys.append(value)
        series[method] = (xs, ys)

    collect("Marformer", lambda size: None if (m := _marformer_missing_metrics(spec, size)) is None else float(m["log_loss"]))
    if spec.name == "LLMRubric":
        collect("CPM Stan", lambda size: _llm_rubric_cpm_log_loss(spec, size))
    else:
        collect("Stan Factor", lambda size: _stan_missing_log_loss(spec, size, "Factor"))
        collect("Stan Normal", lambda size: _stan_missing_log_loss(spec, size, "Normal"))
    collect("REMASKER", lambda size: _baseline_missing_log_loss(spec, "REMASKER", size))
    collect("MIWAE", lambda size: _baseline_missing_log_loss(spec, "MIWAE", size))
    collect("Unigram", lambda size: _empirical_unigram_log_loss(spec, size))
    return series


def _collect_mse_series(spec: DatasetSpec) -> dict[str, tuple[list[int], list[float]]]:
    series: dict[str, tuple[list[int], list[float]]] = {}

    def collect(method: str, fn: Callable[[int], float | None]) -> None:
        xs, ys = [], []
        for size in spec.sizes:
            value = fn(size)
            if value is not None:
                xs.append(size)
                ys.append(value)
        series[method] = (xs, ys)

    collect("Marformer", lambda size: None if (m := _marformer_missing_metrics(spec, size)) is None else float(m["rmse"]) ** 2)
    if spec.name == "LLMRubric":
        collect("CPM Stan", lambda size: None if (payload := _llm_rubric_cpm_probs_and_labels(spec, size)) is None else _mse_from_probs_labels(*payload))
    else:
        collect("Stan Factor", lambda size: None if (payload := _stan_probs_and_labels(spec, size, "Factor")) is None else _mse_from_probs_labels(*payload))
        collect("Stan Normal", lambda size: None if (payload := _stan_probs_and_labels(spec, size, "Normal")) is None else _mse_from_probs_labels(*payload))
    collect("REMASKER", lambda size: None if (payload := _baseline_probs_and_labels(spec, "REMASKER", size)) is None else _mse_from_probs_labels(*payload))
    collect("MIWAE", lambda size: None if (payload := _baseline_probs_and_labels(spec, "MIWAE", size)) is None else _mse_from_probs_labels(*payload))
    return series


def _plot_series(
    spec: DatasetSpec,
    series: dict[str, tuple[list[int], list[float]]],
    title: str,
    y_label: str,
    out_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    order = ["Marformer", "CPM Stan", "Stan Factor", "Stan Normal", "REMASKER", "MIWAE", "Unigram"]
    for method in order:
        xs, ys = series.get(method, ([], []))
        if not xs:
            continue
        linestyle = ":" if method == "Unigram" else "-"
        color = "0.45" if method == "Unigram" else COLORS[method]
        marker = "x" if method == "Unigram" else MARKERS[method]
        ax.plot(xs, ys, label=method, color=color, marker=marker, linestyle=linestyle)

    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=14)
    ax.set_xticks(spec.sizes)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    ax.grid(True, alpha=0.35)

    out_path = spec.out_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def _plot_broken_series(
    spec: DatasetSpec,
    series: dict[str, tuple[list[int], list[float]]],
    title: str,
    y_label: str,
    out_name: str,
    lower_ylim: tuple[float, float],
    upper_ylim: tuple[float, float],
) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.6], "hspace": 0.06},
    )
    order = ["Marformer", "CPM Stan", "Stan Factor", "Stan Normal", "REMASKER", "MIWAE", "Unigram"]
    for method in order:
        xs, ys = series.get(method, ([], []))
        if not xs:
            continue
        linestyle = ":" if method == "Unigram" else "-"
        color = "0.45" if method == "Unigram" else COLORS[method]
        marker = "x" if method == "Unigram" else MARKERS[method]
        ax_top.plot(xs, ys, label=method, color=color, marker=marker, linestyle=linestyle)
        ax_bottom.plot(xs, ys, label=method, color=color, marker=marker, linestyle=linestyle)

    ax_top.set_ylim(*upper_ylim)
    ax_bottom.set_ylim(*lower_ylim)
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bottom.tick_params(top=False)

    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.2)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Faint jagged guide across the split so the discontinuity reads immediately.
    xs = np.linspace(0.0, 1.0, 33)
    top_y = np.where(np.arange(xs.size) % 2 == 0, -0.006, 0.006)
    bottom_y = np.where(np.arange(xs.size) % 2 == 0, 1.006, 0.994)
    ax_top.plot(xs, top_y, transform=ax_top.transAxes, color="0.35", alpha=0.28, linewidth=0.9, clip_on=False)
    ax_bottom.plot(xs, bottom_y, transform=ax_bottom.transAxes, color="0.35", alpha=0.28, linewidth=0.9, clip_on=False)

    ax_bottom.set_xlabel(spec.x_label)
    ax_bottom.set_ylabel(y_label)
    ax_top.set_title(title, pad=18)
    ax_bottom.set_xticks(spec.sizes)

    for ax in (ax_top, ax_bottom):
        ax.grid(True, alpha=0.35)

    ax_top.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

    out_path = spec.out_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def _plot_mbr_l2_snapshot(spec: DatasetSpec, size: int | None = None) -> None:
    size = spec.sizes[-1] if size is None else size
    method_rows: list[tuple[str, float, str]] = []

    metrics = _marformer_missing_metrics(spec, size)
    if metrics is not None:
        method_rows.append(("Marformer", float(metrics["rmse"]) ** 2, COLORS["Marformer"]))

    if spec.name == "LLMRubric":
        comparisons = [
            ("CPM Stan", _llm_rubric_cpm_probs_and_labels(spec, size)),
            ("REMASKER", _baseline_probs_and_labels(spec, "REMASKER", size)),
            ("MIWAE", _baseline_probs_and_labels(spec, "MIWAE", size)),
            ("Unigram", _empirical_unigram_probs_and_labels(spec, size)),
        ]
    else:
        comparisons = [
            ("Stan Factor", _stan_probs_and_labels(spec, size, "Factor")),
            ("Stan Normal", _stan_probs_and_labels(spec, size, "Normal")),
            ("REMASKER", _baseline_probs_and_labels(spec, "REMASKER", size)),
            ("MIWAE", _baseline_probs_and_labels(spec, "MIWAE", size)),
            ("Unigram", _empirical_unigram_probs_and_labels(spec, size)),
        ]

    for method, payload in comparisons:
        if payload is not None:
            color = "0.45" if method == "Unigram" else COLORS[method]
            method_rows.append((method, _mse_from_probs_labels(*payload), color))

    if not method_rows:
        print(f"Skip MBR-L2 plot for {spec.pretty_name}: no data")
        return

    fig, ax = plt.subplots(figsize=(11.8, 6.5))
    x = np.arange(len(method_rows), dtype=float)
    vals = [row[1] for row in method_rows]
    colors = [row[2] for row in method_rows]
    ax.bar(x, vals, color=colors, alpha=0.92)
    ax.set_ylabel("MBR-L2 (MSE)")
    ax.set_title(f"{spec.pretty_name}: MBR-L2 at Size {size}", pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels([row[0] for row in method_rows], rotation=12, ha="right")
    ax.grid(True, axis="y", alpha=0.35)

    out_path = spec.out_dir / f"{spec.name.lower()}_mbr_l2_size{size}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def _plot_runtime_series(spec: DatasetSpec) -> None:
    if spec.marformer_runtime_seconds is None or spec.stan_runtime_seconds is None:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    if spec.name == "LLMRubric":
        runtime_series = [
            ("Marformer", spec.marformer_runtime_seconds, COLORS["Marformer"], MARKERS["Marformer"]),
            ("CPM Stan", spec.stan_runtime_seconds["CPM Stan"], COLORS["CPM Stan"], MARKERS["CPM Stan"]),
        ]
    else:
        runtime_series = [
            ("Marformer", spec.marformer_runtime_seconds, "#1f6fba", "o"),
            ("Stan Factor", spec.stan_runtime_seconds["Factor"], "#27ae60", "^"),
            ("Stan Normal", spec.stan_runtime_seconds["Normal"], "#e67e22", "D"),
        ]

    for label, runtime_map, color, marker in runtime_series:
        xs = [size for size in spec.sizes if size in runtime_map]
        ys = [runtime_map[size] / 60.0 for size in xs]
        ax.plot(xs, ys, label=label, color=color, marker=marker)

    ax.set_xlabel(spec.x_label)
    ax.set_ylabel("Runtime (minutes)")
    ax.set_title(f"{spec.pretty_name}: Runtime by Training Size", pad=14)
    ax.set_xticks(spec.sizes)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    ax.grid(True, alpha=0.35)

    out_path = spec.out_dir / f"{spec.name.lower()}_runtime_by_size.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def _plot_dataset(spec: DatasetSpec) -> None:
    loss_series = _collect_loss_series(spec)

    if spec.name == "LLMRubric":
        _plot_broken_series(
            spec,
            loss_series,
            title=f"{spec.pretty_name}: Test Log Loss by Training Size",
            y_label="Test Log Loss",
            out_name=f"{spec.name.lower()}_test_loss_by_size.png",
            lower_ylim=(0.55, 2.0),
            upper_ylim=(4.0, 5.6),
        )
    else:
        _plot_series(
            spec,
            loss_series,
            title=f"{spec.pretty_name}: Test Log Loss by Training Size",
            y_label="Test Log Loss",
            out_name=f"{spec.name.lower()}_test_loss_by_size.png",
        )
    if spec.name == "LLMRubric":
        for size in spec.sizes:
            _plot_mbr_l2_snapshot(spec, size)
    else:
        _plot_mbr_l2_snapshot(spec)
    _plot_runtime_series(spec)


LLM_RUBRIC_MARFORMER_RUNTIME_SECONDS = {
    10: (4 * 60 + 47) * 70 / 300.0,
    20: (7 * 60 + 25) * 82 / 300.0,
    30: (10 * 60 + 27) * 114 / 300.0,
    40: (13 * 60 + 29) * 120 / 300.0,
    50: (18 * 60 + 4) * 124 / 300.0,
    75: (24 * 60 + 23) * 98 / 300.0,
    100: (33 * 60 + 58) * 119 / 300.0,
    125: (39 * 60 + 44) * 146 / 300.0,
    150: (50 * 60 + 24) * 142 / 300.0,
    175: (54 * 60 + 55) * 170 / 300.0,
}

LLM_RUBRIC_STAN_RUNTIME_SECONDS = {
    "CPM Stan": {
        10: 20 * 60,
        20: 23 * 60,
        75: 43 * 60,
        100: 74 * 60,
        125: 91 * 60,
        150: 153 * 60,
        175: 162 * 60,
    },
}


LLM_RUBRIC = DatasetSpec(
    name="LLMRubric",
    pretty_name="LLM Rubric",
    sizes=[10, 20, 30, 40, 50, 75, 100, 125, 150, 175],
    num_classes=4,
    data_root=ROOT / "DATA/LLM_RUBRIC",
    marformer_root=ROOT / "RESULTS/MARFORMER/LLM_RUBRIC",
    stan_root=ROOT / "RESULTS/STAN/LLM_RUBRIC_T",
    baseline_roots={
        "REMASKER": ROOT / "RESULTS/BASELINES/REMASKER/LLMRUBRIC",
        "MIWAE": ROOT / "RESULTS/BASELINES/MIWAE/LLMRUBRIC",
    },
    marformer_run=lambda size: f"LLMRubric_225_25_9_{size}",
    stan_eval_run=lambda size, variant: f"LLMRubric_225_25_9_{size}_nt_{variant}_eval",
    baseline_run=lambda size: f"LLMRubric_225_25_9_{size}",
    out_dir=PLOTS_ROOT / "LLMRubric",
    x_label="Training Items",
    marformer_runtime_seconds=LLM_RUBRIC_MARFORMER_RUNTIME_SECONDS,
    stan_runtime_seconds=LLM_RUBRIC_STAN_RUNTIME_SECONDS,
)

SUMMEVAL = DatasetSpec(
    name="SummEval",
    pretty_name="SummEval",
    sizes=[50, 100, 500, 750, 1000, 1280],
    num_classes=5,
    data_root=ROOT / "DATA/SUMMEVAL",
    marformer_root=ROOT / "RESULTS/MARFORMER/SUMMEVAL",
    stan_root=ROOT / "RESULTS/STAN/SUMMEVAL_T",
    baseline_roots={
        "REMASKER": ROOT / "RESULTS/BASELINES/REMASKER/SUMMEVAL",
        "MIWAE": ROOT / "RESULTS/BASELINES/MIWAE/SUMMEVAL",
    },
    marformer_run=lambda size: f"SummEval_1600_8_4_{size}",
    stan_eval_run=lambda size, variant: f"SummEval_1600_8_4_{size}_nt_{variant}_eval",
    baseline_run=lambda size: f"SummEval_1600_8_4_{size}",
    out_dir=PLOTS_ROOT / "SummEval",
    x_label="Training Items",
)


def main() -> None:
    for spec in [LLM_RUBRIC, SUMMEVAL]:
        _plot_dataset(spec)


if __name__ == "__main__":
    main()
