#!/usr/bin/env python3
"""
Plot LLMRubric generalization figures using the Domain 3 talk-plot style.

Outputs:
  - PLOTS/TALK/LLMRubric/ITEM_GENERALIZATION/llmrubric_generalization_log_loss.png
  - PLOTS/TALK/LLMRubric/ITEM_GENERALIZATION/llmrubric_generalization_mbr_l2.png
  - PLOTS/TALK/LLMRubric/ITEM_GENERALIZATION/llmrubric_generalization_runtime.png
  - PLOTS/TALK/LLMRubric/ITEM_GENERALIZATION/calibration/*
  - PLOTS/TALK/LLMRubric/ITEM_GENERALIZATION/llmrubric_generalization_summary.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np
import pandas as pd
import relplot as rp
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.test import _load_checkpoint, _reconstruct


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

rp.config.use_tex_fonts = False

PROB_COL_TEMPLATE = "prob_cat_{idx}"

COLORS = {
    "Marformer": "#1f6fba",
    "CPM Stan": "#7a1f3d",
    "REMASKER": "#9f2f5f",
    "MIWAE": "#b85aa0",
    "Best Unigram": "#6b7280",
}

MARKERS = {
    "Marformer": "o",
    "CPM Stan": "^",
    "REMASKER": "s",
    "MIWAE": "P",
    "Best Unigram": "D",
}

UNIGRAM_VARIANTS = {
    "global": (),
    "attribute": ("attribute",),
    "item": ("item",),
    "annotator": ("annotator",),
    "attribute-item": ("attribute", "item"),
    "attribute-annotator": ("attribute", "annotator"),
    "item-annotator": ("item", "annotator"),
    "attribute-item-annotator": ("attribute", "item", "annotator"),
}


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    title_prefix: str
    sizes: list[int]
    x_label: str
    num_classes: int
    data_root: Path
    marformer_root: Path
    stan_root: Path
    baseline_roots: dict[str, Path]
    out_dir: Path
    marformer_total_runtime_seconds: dict[int, float]
    cpm_runtime_seconds: dict[int, float]
    figure_size: tuple[float, float]


SPEC = DatasetSpec(
    slug="llmrubric",
    title_prefix="LLMRubric Item Generalization",
    sizes=[10, 20, 30, 40, 50, 75, 100, 125, 150, 175],
    x_label="Training Items",
    num_classes=4,
    data_root=ROOT / "DATA/LLM_RUBRIC",
    marformer_root=ROOT / "RESULTS/MARFORMER/LLM_RUBRIC",
    stan_root=ROOT / "RESULTS/STAN/LLM_RUBRIC_T",
    baseline_roots={
        "REMASKER": ROOT / "RESULTS/BASELINES/REMASKER/LLMRUBRIC",
        "MIWAE": ROOT / "RESULTS/BASELINES/MIWAE/LLMRUBRIC",
    },
    out_dir=ROOT / "PLOTS/TALK/LLMRubric/ITEM_GENERALIZATION",
    marformer_total_runtime_seconds={
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
    },
    cpm_runtime_seconds={
        10: 20 * 60,
        20: 23 * 60,
        75: 43 * 60,
        100: 74 * 60,
        125: 91 * 60,
        150: 153 * 60,
        175: 162 * 60,
    },
    figure_size=(15.2, 7.8),
)


def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _bundle_path(size: int) -> Path:
    return SPEC.data_root / f"LLMRubric_225_25_9_{size}" / "data_bundle.json"


@lru_cache(maxsize=None)
def _bundle(size: int) -> dict:
    return _read_json(_bundle_path(size))


def _test_missing_indices_and_labels(bundle: dict) -> tuple[list[int], np.ndarray]:
    idxs = [i for i, row in enumerate(bundle.get("missing_ratings", [])) if row.get("instance") == "test"]
    labels = np.asarray([bundle["missing_ratings"][i]["value"] - 1 for i in idxs], dtype=np.int64)
    return idxs, labels


def _series_labels() -> list[str]:
    return ["Marformer", "CPM Stan", "REMASKER", "MIWAE", "Best Unigram"]


def _series_style(label: str) -> tuple[str, str, str]:
    if label == "Best Unigram":
        return ":", COLORS[label], MARKERS[label]
    return "-", COLORS[label], MARKERS[label]


def _series_linewidth(label: str) -> float:
    if label == "Marformer":
        return 2.9
    if label == "CPM Stan":
        return 2.7
    return 2.4


def _series_alpha(label: str) -> float:
    if label == "CPM Stan":
        return 0.98
    return 1.0


def _series_fill_alpha(label: str) -> float:
    if label == "CPM Stan":
        return 0.12
    if label in {"REMASKER", "MIWAE"}:
        return 0.08
    return 0.14


def _style_legend(ax: plt.Axes, loc: str = "best") -> None:
    leg = ax.legend(
        loc=loc,
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor="white",
        edgecolor="0.75",
    )
    if leg is not None:
        leg.get_frame().set_linewidth(1.0)


def _per_example_nll(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0)
    return -np.log(clipped)


def _per_example_mse(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    expected = probs @ classes
    truth = labels.astype(np.float64) + 1.0
    return (expected - truth) ** 2


def _all_class_calibration(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.arange(probs.shape[1], dtype=np.int64)
    conf = probs.flatten()
    acc = (labels[:, None] == classes[None, :]).astype(np.float32).flatten()
    return conf, acc


def _draw_empty(ax: plt.Axes, title: str) -> None:
    ax.plot([0, 1], [0, 1], color="0.75", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.text(0.5, 0.5, "no data", ha="center", va="center", color="0.6", transform=ax.transAxes)
    ax.set_title(title)


def _plot_ece(ax: plt.Axes, probs: np.ndarray, labels: np.ndarray, title: str, color: str) -> None:
    conf, acc = _all_class_calibration(probs, labels)
    diag = rp.prepare_rel_diagram(
        conf,
        acc,
        num_bootstrap=500,
        report_CE=True,
        report_CE_std=True,
    )
    ce = diag.get("ce", float("nan"))
    ci_w = diag.get("ce_ci_width", 0.0)
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
    ax.set_title(f"{title}\nsmECE = {ce:.3f} ± {ci_w:.3f}", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)


def _marformer_run_dir(size: int) -> Path | None:
    run_dir = SPEC.marformer_root / f"LLMRubric_225_25_9_{size}"
    return run_dir if run_dir.exists() else None


def _marformer_best_json(size: int) -> Path | None:
    run_dir = _marformer_run_dir(size)
    if run_dir is None:
        return None
    preferred = run_dir / "TEST_RESULTS" / "best.json"
    if preferred.exists():
        return preferred
    candidates = sorted((run_dir / "TEST_RESULTS").glob("best*.json"))
    return candidates[0] if candidates else None


def _parse_best_epoch(checkpoint_name: str | None, total_epochs: int = 300) -> int:
    if not checkpoint_name or checkpoint_name == "last.ckpt":
        return total_epochs
    match = re.search(r"epoch=(\d+)", checkpoint_name)
    if match is None:
        return total_epochs
    return int(match.group(1)) + 1


@lru_cache(maxsize=None)
def _marformer_probs_and_labels(size: int) -> tuple[np.ndarray, np.ndarray] | None:
    run_dir = _marformer_run_dir(size)
    best_json = _marformer_best_json(size)
    if run_dir is None or best_json is None:
        return None

    payload = _read_json(best_json)
    checkpoint_name = payload.get("checkpoint")
    if checkpoint_name is None:
        return None

    ckpt_path = run_dir / "checkpoints" / checkpoint_name
    if not ckpt_path.exists():
        return None

    model, eval_vars, _train_cfg = _reconstruct(run_dir)
    device = torch.device("cpu")
    _load_checkpoint(model, ckpt_path, device)
    model.eval()

    with torch.no_grad():
        entity_graph = variable_list_to_entity_graph(eval_vars, model.types)
        params = model(entity_graph, device=device)

    probs: list[np.ndarray] = []
    labels: list[int] = []
    num_classes = model.types["rating"].num_classes
    for idx, token in enumerate(entity_graph.tokens):
        if token.type_name != "rating" or token.status != 0:
            continue
        rating_value = (token.raw_data or {}).get("rating_value")
        if rating_value is None:
            continue
        logits = params[0, idx, 1:1 + num_classes]
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labels.append(int(rating_value))

    if not probs:
        return None
    return np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.int64)


@lru_cache(maxsize=None)
def _cpm_probs_and_labels(size: int) -> tuple[np.ndarray, np.ndarray] | None:
    probs_path = SPEC.stan_root / f"LLMRubric_225_25_9_{size}_eval" / "rating_probabilities.csv"
    if not probs_path.exists():
        return None

    bundle = _bundle(size)
    test_idxs, labels = _test_missing_indices_and_labels(bundle)
    if not test_idxs:
        return None

    df = pd.read_csv(probs_path)
    prob_cols = [PROB_COL_TEMPLATE.format(idx=i) for i in range(1, SPEC.num_classes + 1)]
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


@lru_cache(maxsize=None)
def _baseline_probs_and_labels(method: str, size: int) -> tuple[np.ndarray, np.ndarray] | None:
    pred_path = SPEC.baseline_roots[method] / f"LLMRubric_225_25_9_{size}" / "test_predictions.json"
    if not pred_path.exists():
        return None
    rows = _read_json(pred_path)
    if not rows:
        return None
    probs = np.asarray([row["probs"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["true_label"] for row in rows], dtype=np.int64)
    return probs, labels


def _global_probs_from_observed(observed: list[dict], num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    for row in observed:
        dist = row.get("rating_dist")
        if dist is not None:
            counts += np.asarray(dist, dtype=np.float64)
        else:
            counts[int(row["value"]) - 1] += 1.0
    counts = np.maximum(counts, 0.0)
    total = counts.sum()
    if total <= 0:
        return np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    return counts / total


@lru_cache(maxsize=None)
def _best_unigram_result(size: int) -> dict | None:
    bundle = _bundle(size)
    observed = bundle.get("observed_ratings", [])
    test_idxs, labels = _test_missing_indices_and_labels(bundle)
    test_missing = [bundle["missing_ratings"][i] for i in test_idxs]
    if not observed or not test_missing:
        return None

    num_classes = max(max(r["value"] for r in observed), max(r["value"] for r in test_missing))
    global_probs = _global_probs_from_observed(observed, num_classes)
    prior_strength = 1.0

    counts_by_variant: dict[str, dict[tuple, np.ndarray]] = {}
    for name, fields in UNIGRAM_VARIANTS.items():
        table: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(num_classes, dtype=np.float64))
        for row in observed:
            key = tuple(row.get(field) for field in fields)
            dist = row.get("rating_dist")
            if dist is not None:
                table[key] += np.asarray(dist, dtype=np.float64)
            else:
                table[key][int(row["value"]) - 1] += 1.0
        counts_by_variant[name] = dict(table)

    best: dict | None = None
    for name, fields in UNIGRAM_VARIANTS.items():
        probs = np.zeros((len(test_missing), num_classes), dtype=np.float64)
        table = counts_by_variant[name]
        for i, row in enumerate(test_missing):
            key = tuple(row.get(field) for field in fields)
            counts = table.get(key)
            if counts is None:
                probs[i] = global_probs
            else:
                probs[i] = (counts + prior_strength * global_probs) / (counts.sum() + prior_strength)

        result = {
            "variant": name,
            "fields": list(fields),
            "probs": probs,
            "labels": labels,
            "mean_log_loss": float(np.mean(_per_example_nll(probs, labels))),
            "mean_mbr_l2": float(np.mean(_per_example_mse(probs, labels))),
        }
        if best is None or result["mean_log_loss"] < best["mean_log_loss"]:
            best = result
    return best


def _collect_metric_series(metric: str) -> tuple[dict[str, dict[str, list[float]]], list[dict]]:
    stats = {
        stat_name: {label: [np.nan] * len(SPEC.sizes) for label in _series_labels()}
        for stat_name in ("mean", "std", "q10", "q25", "median", "q75", "q90", "min", "max")
    }
    summary_rows: list[dict] = []

    for idx, size in enumerate(SPEC.sizes):
        loaders = [
            ("Marformer", lambda: _marformer_probs_and_labels(size)),
            ("CPM Stan", lambda: _cpm_probs_and_labels(size)),
            ("REMASKER", lambda: _baseline_probs_and_labels("REMASKER", size)),
            ("MIWAE", lambda: _baseline_probs_and_labels("MIWAE", size)),
            ("Best Unigram", lambda: None if _best_unigram_result(size) is None else (_best_unigram_result(size)["probs"], _best_unigram_result(size)["labels"])),
        ]
        best_unigram = _best_unigram_result(size)

        for label, loader in loaders:
            payload = loader()
            if payload is None:
                continue
            probs, labels = payload
            values = _per_example_nll(probs, labels) if metric == "log_loss" else _per_example_mse(probs, labels)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=0))
            q10, q25, median, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
            stats["mean"][label][idx] = mean
            stats["std"][label][idx] = std
            stats["q10"][label][idx] = float(q10)
            stats["q25"][label][idx] = float(q25)
            stats["median"][label][idx] = float(median)
            stats["q75"][label][idx] = float(q75)
            stats["q90"][label][idx] = float(q90)
            stats["min"][label][idx] = float(np.min(values))
            stats["max"][label][idx] = float(np.max(values))

            summary_rows.append({
                "dataset": SPEC.slug,
                "size": size,
                "model": label,
                "metric": metric,
                "mean": mean,
                "std": std,
                "q10": float(q10),
                "q25": float(q25),
                "median": float(median),
                "q75": float(q75),
                "q90": float(q90),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": int(values.size),
                "unigram_variant": best_unigram["variant"] if label == "Best Unigram" and best_unigram is not None else None,
            })

    return stats, summary_rows


def _interval_bounds(
    stats: dict[str, dict[str, list[float]]],
    label: str,
    interval_kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(stats["mean"][label], dtype=float)
    if interval_kind == "std01":
        std = np.asarray(stats["std"][label], dtype=float)
        lower = np.maximum(0.0, mean - 0.05 * std)
        upper = mean + 0.05 * std
        return mean, lower, upper
    raise ValueError(f"Unknown interval kind: {interval_kind}")


def _metric_variant_title(title: str, interval_kind: str) -> str:
    if interval_kind == "std01":
        return f"{title} (mean ± 0.05 SD)"
    raise ValueError(f"Unknown interval kind: {interval_kind}")


def _plot_metric_band_series(
    stats: dict[str, dict[str, list[float]]],
    ylabel: str,
    title: str,
    output_name: str,
    interval_kind: str,
) -> None:
    fig, ax = plt.subplots(figsize=SPEC.figure_size)
    ymins: list[float] = []
    ymaxs: list[float] = []
    for label in _series_labels():
        xs = np.asarray(SPEC.sizes, dtype=float)
        ys, lo, hi = _interval_bounds(stats, label, interval_kind)
        valid = ~np.isnan(ys)
        if not np.any(valid):
            continue
        linestyle, color, marker = _series_style(label)
        ax.plot(
            xs[valid],
            ys[valid],
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
            linewidth=_series_linewidth(label),
            alpha=_series_alpha(label),
        )
        ymins.extend(lo[valid].tolist())
        ymaxs.extend(hi[valid].tolist())
        if label != "Best Unigram":
            ax.fill_between(xs[valid], lo[valid], hi[valid], color=color, alpha=_series_fill_alpha(label), linewidth=0)

    ax.set_xlabel(SPEC.x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14)
    ax.set_xticks(SPEC.sizes)
    ax.set_xticklabels([str(size) for size in SPEC.sizes])
    if ymins and ymaxs:
        ymin = min(ymins)
        ymax = max(ymaxs)
        pad = max(0.03, 0.07 * (ymax - ymin))
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    _style_legend(ax, loc="center left")
    ax.margins(x=0.03)

    output_path = SPEC.out_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 0.98], pad=1.0)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _plot_log_loss_broken(
    stats: dict[str, dict[str, list[float]]],
    ylabel: str,
    title: str,
    output_name: str,
    interval_kind: str,
) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=SPEC.figure_size,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 4.2], "hspace": 0.05},
    )

    for label in _series_labels():
        xs = np.asarray(SPEC.sizes, dtype=float)
        ys, lo, hi = _interval_bounds(stats, label, interval_kind)
        valid = ~np.isnan(ys)
        if not np.any(valid):
            continue
        linestyle, color, marker = _series_style(label)
        ax_top.plot(
            xs[valid], ys[valid], color=color, marker=marker, linestyle=linestyle, label=label,
            linewidth=_series_linewidth(label), alpha=_series_alpha(label),
        )
        ax_bottom.plot(
            xs[valid], ys[valid], color=color, marker=marker, linestyle=linestyle, label=label,
            linewidth=_series_linewidth(label), alpha=_series_alpha(label),
        )
        if label != "Best Unigram":
            ax_top.fill_between(xs[valid], lo[valid], hi[valid], color=color, alpha=_series_fill_alpha(label), linewidth=0)
            ax_bottom.fill_between(xs[valid], lo[valid], hi[valid], color=color, alpha=_series_fill_alpha(label), linewidth=0)

    lower_ticks = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    upper_ticks = [3.0, 4.0, 5.0]
    ax_bottom.set_ylim(0.55, 1.70)
    ax_top.set_ylim(2.9, 5.55)
    ax_bottom.yaxis.set_major_locator(FixedLocator(lower_ticks))
    ax_bottom.yaxis.set_major_formatter(FixedFormatter([f"{tick:.1f}" for tick in lower_ticks]))
    ax_top.yaxis.set_major_locator(FixedLocator(upper_ticks))
    ax_top.yaxis.set_major_formatter(FixedFormatter(["3.0", "4.0", "5.0"]))

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False, labelbottom=False)
    ax_bottom.tick_params(top=False)
    ax_top.minorticks_off()
    ax_bottom.minorticks_off()

    xs_break = np.linspace(0.0, 1.0, 61)
    top_break = np.where(np.arange(xs_break.size) % 2 == 0, -0.005, 0.005)
    bottom_break = np.where(np.arange(xs_break.size) % 2 == 0, 1.005, 0.995)
    ax_top.plot(xs_break, top_break, transform=ax_top.transAxes, color="0.25", linewidth=1.0, clip_on=False)
    ax_bottom.plot(xs_break, bottom_break, transform=ax_bottom.transAxes, color="0.25", linewidth=1.0, clip_on=False)

    ax_bottom.set_xlabel(SPEC.x_label)
    ax_bottom.set_ylabel(ylabel)
    ax_top.set_title(title, pad=14)
    ax_bottom.set_xticks(SPEC.sizes)
    ax_bottom.set_xticklabels([str(size) for size in SPEC.sizes])
    ax_bottom.set_xlim(SPEC.sizes[0] - 2.5, SPEC.sizes[-1] + 2.5)

    for ax in (ax_top, ax_bottom):
        ax.grid(True, alpha=0.35)

    _style_legend(ax_bottom, loc="center left")

    output_path = SPEC.out_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.80, top=0.90, bottom=0.12, hspace=0.05)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _runtime_minutes_for_marformer(size: int) -> float | None:
    total = SPEC.marformer_total_runtime_seconds.get(size)
    best_json = _marformer_best_json(size)
    if total is None or best_json is None:
        return None
    checkpoint_name = _read_json(best_json).get("checkpoint")
    best_epoch_count = _parse_best_epoch(checkpoint_name, total_epochs=300)
    return total * (best_epoch_count / 300.0) / 60.0


def _runtime_minutes_for_cpm(size: int) -> float | None:
    total = SPEC.cpm_runtime_seconds.get(size)
    if total is None:
        return None
    return total / 60.0


def _plot_runtime() -> list[dict]:
    rows: list[dict] = []
    fig, ax = plt.subplots(figsize=SPEC.figure_size)
    for label, fn in (
        ("Marformer", _runtime_minutes_for_marformer),
        ("CPM Stan", _runtime_minutes_for_cpm),
    ):
        xs: list[int] = []
        ys: list[float] = []
        for size in SPEC.sizes:
            value = fn(size)
            if value is None:
                continue
            xs.append(size)
            ys.append(value)
            rows.append({
                "dataset": SPEC.slug,
                "size": size,
                "model": label,
                "metric": "runtime_minutes",
                "mean": float(value),
            })
        if xs:
            linestyle, color, marker = _series_style(label)
            ax.plot(xs, ys, color=color, marker=marker, linestyle=linestyle, linewidth=_series_linewidth(label), label=label)

    ax.set_xlabel(SPEC.x_label)
    ax.set_ylabel("Runtime (minutes)")
    ax.set_title(f"{SPEC.title_prefix}: Runtime by Training Size", pad=14)
    ax.set_xticks(SPEC.sizes)
    ax.set_xticklabels([str(size) for size in SPEC.sizes])
    _style_legend(ax, loc="center left")

    output_path = SPEC.out_dir / "llmrubric_generalization_runtime.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 0.98], pad=1.0)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")
    return rows


def _plot_calibration_grid() -> list[dict]:
    rows: list[dict] = []
    cal_dir = SPEC.out_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    for size in SPEC.sizes:
        best_unigram = _best_unigram_result(size)
        unigram_title = "Best Unigram"
        if best_unigram is not None:
            unigram_title = f"Best Unigram\n({best_unigram['variant'].replace('-', ' ').title()})"

        payloads = [
            ("Marformer", _marformer_probs_and_labels(size), "Marformer"),
            ("CPM Stan", _cpm_probs_and_labels(size), "CPM Stan"),
            ("REMASKER", _baseline_probs_and_labels("REMASKER", size), "REMASKER"),
            ("MIWAE", _baseline_probs_and_labels("MIWAE", size), "MIWAE"),
            (unigram_title, None if best_unigram is None else (best_unigram["probs"], best_unigram["labels"]), "Best Unigram"),
        ]

        ncols = 3
        nrows = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 4.7 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for ax, (title, payload, color_key) in zip(axes, payloads):
            if payload is None:
                _draw_empty(ax, title)
                continue
            probs, labels = payload
            _plot_ece(ax, probs, labels, title, COLORS[color_key])
            rows.append({
                "dataset": SPEC.slug,
                "size": size,
                "model": color_key,
                "metric": "calibration_panel",
                "unigram_variant": best_unigram["variant"] if color_key == "Best Unigram" and best_unigram is not None else None,
            })

        if len(axes) > len(payloads):
            for ax in axes[len(payloads):]:
                ax.axis("off")

        fig.suptitle(f"{SPEC.title_prefix}: Calibration at Size {size}", fontsize=18, y=0.98)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92], pad=1.0, w_pad=1.8, h_pad=2.0)
        output_path = cal_dir / f"llmrubric_generalization_calibration_size{size}.png"
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved -> {output_path}")

    return rows


def _save_summary(rows: list[dict]) -> None:
    unigram_summary = {}
    for size in SPEC.sizes:
        best = _best_unigram_result(size)
        if best is None:
            continue
        unigram_summary[size] = {
            "variant": best["variant"],
            "fields": best["fields"],
            "mean_log_loss": best["mean_log_loss"],
            "mean_mbr_l2": best["mean_mbr_l2"],
        }

    out = {
        "dataset": SPEC.slug,
        "title_prefix": SPEC.title_prefix,
        "unigram_best_by_size": unigram_summary,
        "rows": rows,
    }
    output_path = SPEC.out_dir / "llmrubric_generalization_summary.json"
    output_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {output_path}")


def main() -> None:
    rows: list[dict] = []

    log_loss_stats, metric_rows = _collect_metric_series("log_loss")
    rows.extend(metric_rows)
    _plot_log_loss_broken(
        log_loss_stats,
        ylabel="Test Log-Loss",
        title=_metric_variant_title(f"{SPEC.title_prefix}: Log-Loss by Training Size", "std01"),
        output_name="llmrubric_generalization_log_loss.png",
        interval_kind="std01",
    )

    mbr_stats, metric_rows = _collect_metric_series("mbr_l2")
    rows.extend(metric_rows)
    _plot_metric_band_series(
        mbr_stats,
        ylabel="MBR L2 (MSE)",
        title=_metric_variant_title(f"{SPEC.title_prefix}: MBR L2 by Training Size", "std01"),
        output_name="llmrubric_generalization_mbr_l2.png",
        interval_kind="std01",
    )

    rows.extend(_plot_runtime())
    rows.extend(_plot_calibration_grid())
    _save_summary(rows)


if __name__ == "__main__":
    main()
