#!/usr/bin/env python3
"""
Plot polished Domain 3 generalization figures for item and annotator expansion.

Outputs:
  - PLOTS/TALK/DOMAIN3/ITEM_GENERALIZATION/*
  - PLOTS/TALK/DOMAIN3/ANNOT_GENERALIZATION/*
"""

from __future__ import annotations

import json
import math
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

PROB_PREFIX = "prob_cat_"
BOOTSTRAP_SAMPLES = 400

COLORS = {
    "Marformer": "#1f6fba",
    "Stan Oracle": "#1b9e77",
    "Best Unigram": "#6b7280",
}

MARKERS = {
    "Marformer": "o",
    "Stan Oracle": "^",
    "Best Unigram": "D",
}

FIGSIZE = (10.4, 5.8)

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
    data_root: Path
    marformer_root: Path
    stan_root: Path
    marformer_run_glob: str
    stan_eval_name: str
    out_dir: Path
    marformer_total_runtime_seconds: dict[int, float]
    stan_runtime_seconds: dict[int, float]
    figure_size: tuple[float, float]
    marformer_nt_root: Path
    marformer_nt_run_glob: str
    nt_offset: int
    log_loss_break_at: float | None


ITEM_SPEC = DatasetSpec(
    slug="item",
    title_prefix="Item Generalization",
    sizes=[50, 100, 150, 200, 250, 300, 350, 400],
    x_label="Training Items",
    data_root=ROOT / "DATA/STAN/DOMAIN3/ItemSplits/Transductive",
    marformer_root=ROOT / "RESULTS/MARFORMER/DOMAIN3/ITEM",
    stan_root=ROOT / "RESULTS/STAN/TENSOR/DOMAIN3/ITEM",
    marformer_run_glob="Tensor_400_25_9_DOMAIN3_Item_T_{size}_MARFORMER*",
    stan_eval_name="Tensor_400_25_9_DOMAIN3_Item_T_{size}_TENSOR_eval",
    out_dir=ROOT / "PLOTS/TALK/DOMAIN3/ITEM_GENERALIZATION",
    marformer_total_runtime_seconds={
        50: 7 * 60 + 12,
        100: 13 * 60 + 3,
        150: 19 * 60 + 50,
        200: 27 * 60 + 51,
        250: 33 * 60 + 1,
        300: 43 * 60 + 57,
        350: 45 * 60 + 12,
        400: 51 * 60 + 20,
    },
    stan_runtime_seconds={
        50: 40 * 60,
        100: 1 * 3600 + 22 * 60 + 45,
        150: 1 * 3600 + 45 * 60 + 56,
        200: 2 * 3600 + 37 * 60 + 23,
        250: 2 * 3600 + 43 * 60 + 24,
        300: 2 * 3600 + 49 * 60 + 24,
        350: 3 * 3600 + 4 * 60 + 7,
        400: 3 * 3600 + 32 * 60 + 48,
    },
    figure_size=(13.0, 5.7),
    marformer_nt_root=ROOT / "RESULTS/MARFORMER-NT/STAN/DOMAIN3/ITEM",
    marformer_nt_run_glob="Tensor_400_25_9_DOMAIN3_Item_NT_{size}_MARFORMER*",
    nt_offset=50,
    log_loss_break_at=None,
)


ANNOT_SPEC = DatasetSpec(
    slug="annot",
    title_prefix="Annotator Generalization",
    sizes=[5, 10, 15, 20, 25],
    x_label="Training Annotators",
    data_root=ROOT / "DATA/STAN/DOMAIN3/AnnotSplits/Transductive",
    marformer_root=ROOT / "RESULTS/MARFORMER/DOMAIN3/ANNOT",
    stan_root=ROOT / "RESULTS/STAN/TENSOR/DOMAIN3_CPM_ORACLE",
    marformer_run_glob="Tensor_400_25_9_DOMAIN3_Annot_T_{size}_MARFORMER*",
    stan_eval_name="Tensor_400_25_9_DOMAIN3_Annot_T_{size}_CPM_ORACLE_eval",
    out_dir=ROOT / "PLOTS/TALK/DOMAIN3/ANNOT_GENERALIZATION",
    marformer_total_runtime_seconds={
        5: 5 * 60 + 4,
        10: 10 * 60 + 38,
        15: 21 * 60 + 39,
        20: 29 * 60 + 49,
        25: 29 * 60 + 41,
    },
    stan_runtime_seconds={
        5: 40 * 60 + 50,
        10: 1 * 3600 + 22 * 60 + 45,
        15: 2 * 3600 + 7 * 60 + 56,
        20: 2 * 3600 + 49 * 60 + 41,
        25: 3 * 3600 + 29 * 60 + 59,
    },
    figure_size=(10.4, 5.8),
    marformer_nt_root=ROOT / "RESULTS/MARFORMER-NT/STAN/DOMAIN3/ANNOT",
    marformer_nt_run_glob="Tensor_400_25_9_DOMAIN3_Annot_NT_{size}_MARFORMER*",
    nt_offset=5,
    log_loss_break_at=1.30,
)


def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _dataset_dir(spec: DatasetSpec, size: int) -> Path:
    if spec.slug == "item":
        return spec.data_root / f"Tensor_400_25_9_DOMAIN3_Item_T_{size}"
    return spec.data_root / f"Tensor_400_25_9_DOMAIN3_Annot_T_{size}"


@lru_cache(maxsize=None)
def _bundle(spec_slug: str, size: int) -> dict:
    spec = ITEM_SPEC if spec_slug == "item" else ANNOT_SPEC
    return _read_json(_dataset_dir(spec, size) / "data_bundle.json")


def _test_missing_indices_and_labels(bundle: dict) -> tuple[list[int], np.ndarray]:
    if "missing_ratings_indexes_in_test_instance" in bundle:
        idxs = list(bundle["missing_ratings_indexes_in_test_instance"])
    else:
        idxs = [i for i, row in enumerate(bundle.get("missing_ratings", [])) if row.get("instance") == "test"]
    labels = np.asarray([bundle["missing_ratings"][i]["value"] - 1 for i in idxs], dtype=np.int64)
    return idxs, labels


def _find_marformer_run_dir(spec: DatasetSpec, size: int, nontrans: bool = False) -> Path | None:
    root = spec.marformer_nt_root if nontrans else spec.marformer_root
    glob = spec.marformer_nt_run_glob if nontrans else spec.marformer_run_glob
    exact = root / glob.format(size=size).replace("*", "")
    if exact.exists():
        return exact
    candidates = sorted(root.glob(glob.format(size=size)))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _marformer_best_json(spec: DatasetSpec, size: int, nontrans: bool = False) -> Path | None:
    run_dir = _find_marformer_run_dir(spec, size, nontrans=nontrans)
    if run_dir is None:
        return None
    path = run_dir / "TEST_RESULTS" / "best.json"
    return path if path.exists() else None


def _parse_best_epoch(checkpoint_name: str | None, total_epochs: int = 300) -> int:
    if not checkpoint_name:
        return total_epochs
    if checkpoint_name == "last.ckpt":
        return total_epochs
    match = re.search(r"epoch=(\d+)", checkpoint_name)
    if match is None:
        return total_epochs
    return int(match.group(1)) + 1


@lru_cache(maxsize=None)
def _marformer_probs_and_labels(spec_slug: str, size: int, nontrans: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
    spec = ITEM_SPEC if spec_slug == "item" else ANNOT_SPEC
    run_dir = _find_marformer_run_dir(spec, size, nontrans=nontrans)
    best_json = _marformer_best_json(spec, size, nontrans=nontrans)
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

    num_classes = model.types["rating"].num_classes
    with torch.no_grad():
        entity_graph = variable_list_to_entity_graph(eval_vars, model.types)
        params = model(entity_graph, device=device)

    probs: list[np.ndarray] = []
    labels: list[int] = []
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


def _nt_x_position(spec: DatasetSpec, nt_train_size: int) -> int:
    return nt_train_size + spec.nt_offset


def _x_tick_labels(spec: DatasetSpec) -> list[str]:
    return [f"{size}/{size - spec.nt_offset}" for size in spec.sizes]


@lru_cache(maxsize=None)
def _stan_probs_and_labels(spec_slug: str, size: int) -> tuple[np.ndarray, np.ndarray] | None:
    spec = ITEM_SPEC if spec_slug == "item" else ANNOT_SPEC
    eval_dir = spec.stan_root / spec.stan_eval_name.format(size=size)
    probs_path = eval_dir / "rating_probabilities.csv"
    if not probs_path.exists():
        return None

    bundle = _bundle(spec_slug, size)
    test_idxs, labels = _test_missing_indices_and_labels(bundle)
    if not test_idxs:
        return None

    df = pd.read_csv(probs_path)
    prob_cols = sorted(col for col in df.columns if col.startswith(PROB_PREFIX))
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
def _best_unigram_result(spec_slug: str, size: int) -> dict | None:
    bundle = _bundle(spec_slug, size)
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
            key = tuple(row[field] for field in fields)
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
            key = tuple(row[field] for field in fields)
            counts = table.get(key)
            if counts is None:
                probs[i] = global_probs
            else:
                probs[i] = (counts + prior_strength * global_probs) / (counts.sum() + prior_strength)

        nll = _per_example_nll(probs, labels)
        mse = _per_example_mse(probs, labels)
        result = {
            "variant": name,
            "fields": list(fields),
            "probs": probs,
            "labels": labels,
            "mean_log_loss": float(np.mean(nll)),
            "mean_mbr_l2": float(np.mean(mse)),
        }
        if best is None or result["mean_log_loss"] < best["mean_log_loss"]:
            best = result

    return best


def _per_example_nll(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0)
    return -np.log(clipped)


def _per_example_mse(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    expected = probs @ classes
    truth = labels.astype(np.float64) + 1.0
    return (expected - truth) ** 2


def _bootstrap_mean_ci(values: np.ndarray, rng_seed: int) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean, mean
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    samples = values[idx].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return mean, float(lo), float(hi)


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


def _style_legend(ax: plt.Axes, loc: str = "best") -> None:
    leg = ax.legend(
        loc=loc,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor="white",
        edgecolor="0.75",
    )
    if leg is not None:
        leg.get_frame().set_linewidth(1.0)


def _series_style(label: str) -> tuple[str, str, str]:
    if label == "Best Unigram":
        return ":", COLORS["Best Unigram"], MARKERS["Best Unigram"]
    if label == "Marformer NT":
        return ":", COLORS["Marformer"], MARKERS["Marformer"]
    return "-", COLORS[label], MARKERS[label]


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


def _collect_metric_series(spec: DatasetSpec, metric: str) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]], list[dict]]:
    labels_in_plot = list(COLORS) + ["Marformer NT"]
    means = {label: [np.nan] * len(spec.sizes) for label in labels_in_plot}
    lowers = {label: [np.nan] * len(spec.sizes) for label in labels_in_plot}
    uppers = {label: [np.nan] * len(spec.sizes) for label in labels_in_plot}
    summary_rows: list[dict] = []

    for idx, size in enumerate(spec.sizes):
        for label, loader in (
            ("Marformer", lambda: _marformer_probs_and_labels(spec.slug, size, nontrans=False)),
            ("Stan Oracle", lambda: _stan_probs_and_labels(spec.slug, size)),
            ("Best Unigram", lambda: None if _best_unigram_result(spec.slug, size) is None else (_best_unigram_result(spec.slug, size)["probs"], _best_unigram_result(spec.slug, size)["labels"])),
        ):
            payload = loader()
            if payload is None:
                continue
            probs, labels = payload
            values = _per_example_nll(probs, labels) if metric == "log_loss" else _per_example_mse(probs, labels)
            mean, lo, hi = _bootstrap_mean_ci(values, rng_seed=hash((spec.slug, size, label, metric)) % (2**32))
            means[label][idx] = mean
            lowers[label][idx] = lo
            uppers[label][idx] = hi

            if label == "Best Unigram":
                best_uni = _best_unigram_result(spec.slug, size)
                unigram_variant = best_uni["variant"] if best_uni is not None else None
            else:
                unigram_variant = None
            summary_rows.append({
                "dataset": spec.slug,
                "size": size,
                "model": label,
                "metric": metric,
                "mean": mean,
                "ci_lower": lo,
                "ci_upper": hi,
                "unigram_variant": unigram_variant,
            })

    for nt_train_size in spec.sizes:
        total_size = _nt_x_position(spec, nt_train_size)
        if total_size not in spec.sizes:
            continue
        payload = _marformer_probs_and_labels(spec.slug, nt_train_size, nontrans=True)
        if payload is None:
            continue
        probs, labels = payload
        values = _per_example_nll(probs, labels) if metric == "log_loss" else _per_example_mse(probs, labels)
        mean, lo, hi = _bootstrap_mean_ci(values, rng_seed=hash((spec.slug, nt_train_size, "Marformer NT", metric)) % (2**32))
        idx = spec.sizes.index(total_size)
        means["Marformer NT"][idx] = mean
        lowers["Marformer NT"][idx] = lo
        uppers["Marformer NT"][idx] = hi
        summary_rows.append({
            "dataset": spec.slug,
            "size": total_size,
            "model": "Marformer NT",
            "metric": metric,
            "mean": mean,
            "ci_lower": lo,
            "ci_upper": hi,
            "unigram_variant": None,
            "nt_train_size": nt_train_size,
        })
    return means, lowers, uppers, summary_rows


def _plot_metric_series(
    spec: DatasetSpec,
    metric: str,
    ylabel: str,
    title: str,
    output_name: str,
) -> list[dict]:
    means, lowers, uppers, summary_rows = _collect_metric_series(spec, metric)

    if metric == "log_loss" and _should_use_log_loss_break(spec, means):
        _plot_log_loss_broken(spec, means, lowers, uppers, ylabel, title, output_name)
        return summary_rows

    fig, ax = plt.subplots(figsize=spec.figure_size)
    ymins: list[float] = []
    ymaxs: list[float] = []
    for label in ("Marformer", "Marformer NT", "Stan Oracle", "Best Unigram"):
        xs = np.asarray(spec.sizes, dtype=float)
        ys = np.asarray(means[label], dtype=float)
        lo = np.asarray(lowers[label], dtype=float)
        hi = np.asarray(uppers[label], dtype=float)
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
        )
        ymins.extend(lo[valid].tolist())
        ymaxs.extend(hi[valid].tolist())
        if label not in {"Best Unigram", "Marformer NT"} and np.any(~np.isnan(lo[valid])) and np.any(~np.isnan(hi[valid])):
            ax.fill_between(xs[valid], lo[valid], hi[valid], color=color, alpha=0.14, linewidth=0)

    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14)
    ax.set_xticks(spec.sizes)
    ax.set_xticklabels(_x_tick_labels(spec))
    if metric == "log_loss" and ymins and ymaxs:
        ymin = min(ymins)
        ymax = max(ymaxs)
        pad = max(0.03, 0.07 * (ymax - ymin))
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    _style_legend(ax, loc="upper right")
    ax.margins(x=0.03)

    output_path = spec.out_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98], pad=0.9)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")
    return summary_rows


def _should_use_log_loss_break(spec: DatasetSpec, means: dict[str, list[float]]) -> bool:
    if spec.log_loss_break_at is None:
        return False
    cutoff = spec.log_loss_break_at
    for label in ("Marformer", "Marformer NT", "Stan Oracle", "Best Unigram"):
        ys = np.asarray(means[label], dtype=float)
        if np.any(ys[~np.isnan(ys)] > cutoff):
            return True
    return False


def _plot_log_loss_broken(
    spec: DatasetSpec,
    means: dict[str, list[float]],
    lowers: dict[str, list[float]],
    uppers: dict[str, list[float]],
    ylabel: str,
    title: str,
    output_name: str,
) -> None:
    break_center = spec.log_loss_break_at if spec.log_loss_break_at is not None else 1.30

    all_lo = []
    all_hi = []
    below_band_hi = []
    above_band_lo = []
    for label in ("Marformer", "Marformer NT", "Stan Oracle", "Best Unigram"):
        lo = np.asarray(lowers[label], dtype=float)
        hi = np.asarray(uppers[label], dtype=float)
        y = np.asarray(means[label], dtype=float)
        valid_lo = lo[~np.isnan(lo)]
        valid_hi = hi[~np.isnan(hi)]
        valid_y = y[~np.isnan(y)]
        if valid_lo.size:
            all_lo.extend(valid_lo.tolist())
        if valid_hi.size:
            all_hi.extend(valid_hi.tolist())
        elif valid_y.size:
            all_hi.extend(valid_y.tolist())
        if valid_hi.size:
            below_band_hi.extend(valid_hi[valid_hi <= break_center].tolist())
        if valid_lo.size:
            above_band_lo.extend(valid_lo[valid_lo > break_center].tolist())

    ymin = min(all_lo) if all_lo else 0.0
    ymax = max(all_hi) if all_hi else 2.0

    lower_upper = min(
        break_center - 0.05,
        (max(below_band_hi) + 0.04) if below_band_hi else break_center - 0.08,
    )
    upper_lower = max(
        break_center + 0.05,
        (min(above_band_lo) - 0.08) if above_band_lo else break_center + 0.08,
    )
    if upper_lower - lower_upper < 0.10:
        lower_upper = break_center - 0.05
        upper_lower = break_center + 0.05

    lower_pad = max(0.03, 0.08 * max(lower_upper - ymin, 0.2))
    upper_pad = max(0.05, 0.06 * max(ymax - upper_lower, 0.5))
    lower_ylim = (max(0.0, ymin - lower_pad), lower_upper)
    upper_ylim = (upper_lower, ymax + upper_pad)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=spec.figure_size,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.5], "hspace": 0.06},
    )

    for label in ("Marformer", "Marformer NT", "Stan Oracle", "Best Unigram"):
        xs = np.asarray(spec.sizes, dtype=float)
        ys = np.asarray(means[label], dtype=float)
        lo = np.asarray(lowers[label], dtype=float)
        hi = np.asarray(uppers[label], dtype=float)
        valid = ~np.isnan(ys)
        if not np.any(valid):
            continue
        linestyle, color, marker = _series_style(label)
        ax_top.plot(xs[valid], ys[valid], color=color, marker=marker, linestyle=linestyle, label=label)
        ax_bottom.plot(xs[valid], ys[valid], color=color, marker=marker, linestyle=linestyle, label=label)
        if label not in {"Best Unigram", "Marformer NT"} and np.any(~np.isnan(lo[valid])) and np.any(~np.isnan(hi[valid])):
            ax_top.fill_between(xs[valid], lo[valid], hi[valid], color=color, alpha=0.14, linewidth=0)
            ax_bottom.fill_between(xs[valid], lo[valid], hi[valid], color=color, alpha=0.14, linewidth=0)

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

    xs_diag = np.linspace(0.0, 1.0, 33)
    top_y = np.where(np.arange(xs_diag.size) % 2 == 0, -0.006, 0.006)
    bottom_y = np.where(np.arange(xs_diag.size) % 2 == 0, 1.006, 0.994)
    ax_top.plot(xs_diag, top_y, transform=ax_top.transAxes, color="0.35", alpha=0.28, linewidth=0.9, clip_on=False)
    ax_bottom.plot(xs_diag, bottom_y, transform=ax_bottom.transAxes, color="0.35", alpha=0.28, linewidth=0.9, clip_on=False)

    ax_bottom.set_xlabel(spec.x_label)
    ax_bottom.set_ylabel(ylabel)
    ax_top.set_title(title, pad=18)
    ax_bottom.set_xticks(spec.sizes)
    ax_bottom.set_xticklabels(_x_tick_labels(spec))

    for ax in (ax_top, ax_bottom):
        ax.grid(True, alpha=0.35)
        ax.margins(x=0.03)

    _style_legend(ax_top, loc="upper right")

    output_path = spec.out_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98], pad=0.9)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _runtime_minutes_for_marformer(spec: DatasetSpec, size: int) -> float | None:
    total = spec.marformer_total_runtime_seconds.get(size)
    best_json = _marformer_best_json(spec, size)
    if total is None or best_json is None:
        return None
    checkpoint_name = _read_json(best_json).get("checkpoint")
    best_epoch_count = _parse_best_epoch(checkpoint_name, total_epochs=300)
    return total * (best_epoch_count / 300.0) / 60.0


def _runtime_minutes_for_stan(spec: DatasetSpec, size: int) -> float | None:
    total = spec.stan_runtime_seconds.get(size)
    if total is None:
        return None
    return total / 60.0


def _plot_runtime(spec: DatasetSpec) -> list[dict]:
    summary_rows: list[dict] = []
    fig, ax = plt.subplots(figsize=spec.figure_size)
    for label, fn in (
        ("Marformer", lambda size: _runtime_minutes_for_marformer(spec, size)),
        ("Stan Oracle", lambda size: _runtime_minutes_for_stan(spec, size)),
    ):
        xs: list[int] = []
        ys: list[float] = []
        for size in spec.sizes:
            value = fn(size)
            if value is None:
                continue
            xs.append(size)
            ys.append(value)
            summary_rows.append({
                "dataset": spec.slug,
                "size": size,
                "model": label,
                "metric": "runtime_minutes",
                "mean": float(value),
            })
        if xs:
            ax.plot(xs, ys, color=COLORS[label], marker=MARKERS[label], linestyle="-", label=label)

    ax.set_xlabel(spec.x_label)
    ax.set_ylabel("Runtime (minutes)")
    ax.set_title(f"{spec.title_prefix}: Runtime by {spec.x_label}", pad=14)
    ax.set_xticks(spec.sizes)
    ax.set_xticklabels(_x_tick_labels(spec))
    _style_legend(ax, loc="upper left")

    output_path = spec.out_dir / f"{spec.slug}_generalization_runtime.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98], pad=0.9)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")
    return summary_rows


def _plot_calibration_grid(spec: DatasetSpec) -> list[dict]:
    summary_rows: list[dict] = []
    cal_dir = spec.out_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    for size in spec.sizes:
        nt_payload = None
        nt_variant = None
        for nt_train_size in spec.sizes:
            if _nt_x_position(spec, nt_train_size) == size:
                nt_payload = _marformer_probs_and_labels(spec.slug, nt_train_size, nontrans=True)
                nt_variant = nt_train_size
                break

        has_nt = nt_payload is not None
        if has_nt:
            fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.0))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.3))
            axes = np.atleast_1d(axes)
        payloads = [
            ("Marformer", _marformer_probs_and_labels(spec.slug, size), "Marformer"),
        ]
        if has_nt:
            payloads.append((f"Marformer NT\n(train={nt_variant})", nt_payload, "Marformer"))
        payloads.extend([
            ("Stan Oracle", _stan_probs_and_labels(spec.slug, size), "Stan Oracle"),
        ])
        best_unigram = _best_unigram_result(spec.slug, size)
        unigram_title = "Best Unigram"
        if best_unigram is not None:
            pretty = best_unigram["variant"].replace("-", " ").title()
            unigram_title = f"Best Unigram\n({pretty})"
        payloads.append((unigram_title, None if best_unigram is None else (best_unigram["probs"], best_unigram["labels"]), "Best Unigram"))

        for ax, (title, payload, color_key) in zip(axes, payloads):
            if payload is None:
                _draw_empty(ax, title)
                continue
            probs, labels = payload
            _plot_ece(ax, probs, labels, title, COLORS[color_key])
            summary_rows.append({
                "dataset": spec.slug,
                "size": size,
                "model": color_key if color_key != "Best Unigram" else "Best Unigram",
                "metric": "calibration_panel",
                "unigram_variant": None if best_unigram is None or color_key != "Best Unigram" else best_unigram["variant"],
            })

        if len(axes) > len(payloads):
            for ax in axes[len(payloads):]:
                ax.axis("off")

        fig.suptitle(
            f"{spec.title_prefix}: Calibration at {size} {spec.x_label}",
            fontsize=18,
            y=0.98,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92], pad=1.0, w_pad=1.8, h_pad=2.0)
        output_path = cal_dir / f"{spec.slug}_generalization_calibration_size{size}.png"
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved -> {output_path}")

    return summary_rows


def _save_summary(spec: DatasetSpec, rows: list[dict]) -> None:
    unigram_summary = {}
    for size in spec.sizes:
        best = _best_unigram_result(spec.slug, size)
        if best is None:
            continue
        unigram_summary[size] = {
            "variant": best["variant"],
            "fields": best["fields"],
            "mean_log_loss": best["mean_log_loss"],
            "mean_mbr_l2": best["mean_mbr_l2"],
        }

    out = {
        "dataset": spec.slug,
        "title_prefix": spec.title_prefix,
        "unigram_best_by_size": unigram_summary,
        "rows": rows,
    }
    output_path = spec.out_dir / f"{spec.slug}_generalization_summary.json"
    output_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {output_path}")


def _plot_dataset(spec: DatasetSpec) -> None:
    rows: list[dict] = []
    rows.extend(_plot_metric_series(
        spec,
        metric="log_loss",
        ylabel="Test Log Loss",
        title=f"{spec.title_prefix}: Log Loss on Novel {'Items' if spec.slug == 'item' else 'Annotators'}",
        output_name=f"{spec.slug}_generalization_log_loss.png",
    ))
    rows.extend(_plot_metric_series(
        spec,
        metric="mbr_l2",
        ylabel="MBR L2 (MSE)",
        title=f"{spec.title_prefix}: MBR L2 on Novel {'Items' if spec.slug == 'item' else 'Annotators'}",
        output_name=f"{spec.slug}_generalization_mbr_l2.png",
    ))
    rows.extend(_plot_runtime(spec))
    rows.extend(_plot_calibration_grid(spec))
    _save_summary(spec, rows)


def plot_item() -> None:
    _plot_dataset(ITEM_SPEC)


def plot_annot() -> None:
    _plot_dataset(ANNOT_SPEC)


def main() -> None:
    plot_item()
    plot_annot()


if __name__ == "__main__":
    main()
