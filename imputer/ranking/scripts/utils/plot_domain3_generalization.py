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

PROB_PREFIX = "prob_cat_"
MISSPEC_ROOT = ROOT / "RESULTS/STAN/TENSOR/DOMAIN3_MISSPEC"
MISSPEC_MODELS = [
    ("Stan Both", "MISP3_PROJ_BIN_RAWZ"),
    ("Stan Bin", "MISP2_BIN_RAWZ"),
    ("Stan Proj", "MISP1_PROJ"),
]

COLORS = {
    "Marformer": "#1f6fba",
    "Stan Oracle": "#7a1f3d",
    "Stan Both": "#9f2f5f",
    "Stan Bin": "#b85aa0",
    "Stan Proj": "#d98bc2",
    "Best Unigram": "#6b7280",
}

MARKERS = {
    "Marformer": "o",
    "Stan Oracle": "^",
    "Stan Both": "s",
    "Stan Bin": "P",
    "Stan Proj": "X",
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


ITEM_SPEC = DatasetSpec(
    slug="item",
    title_prefix="Item Generalization",
    sizes=[50, 100, 150, 200, 250, 300, 350, 400],
    x_label="Training Items",
    data_root=ROOT / "DATA/STAN/DOMAIN3/ItemSplits/Transductive",
    marformer_root=ROOT / "RESULTS/MARFORMER/STAN/DOMAIN3",
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
    figure_size=(15.2, 7.8),
    marformer_nt_root=ROOT / "RESULTS/MARFORMER-NT/STAN/DOMAIN3/ITEM",
    marformer_nt_run_glob="Tensor_400_25_9_DOMAIN3_Item_NT_{size}_MARFORMER*",
    nt_offset=50,
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
    figure_size=(15.6, 7.8),
    marformer_nt_root=ROOT / "RESULTS/MARFORMER-NT/STAN/DOMAIN3/ANNOT",
    marformer_nt_run_glob="Tensor_400_25_9_DOMAIN3_Annot_NT_{size}_MARFORMER*",
    nt_offset=5,
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


def _best_json_metric(spec: DatasetSpec, size: int, nontrans: bool, metric: str) -> tuple[float, int] | None:
    best_json = _marformer_best_json(spec, size, nontrans=nontrans)
    if best_json is None:
        return None
    payload = _read_json(best_json)
    missing = payload.get("missing", {})
    value = missing.get(metric)
    count = missing.get("n")
    if value is None or count is None:
        return None
    return float(value), int(count)


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
    return nt_train_size


def _x_tick_labels(spec: DatasetSpec) -> list[str]:
    return [str(size) for size in spec.sizes]


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


@lru_cache(maxsize=None)
def _stan_misspec_probs_and_labels(spec_slug: str, size: int, model_tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    split_dir = "ITEM" if spec_slug == "item" else "ANNOT"
    stem = "Item" if spec_slug == "item" else "Annot"
    eval_dir = MISSPEC_ROOT / split_dir / f"Tensor_400_25_9_DOMAIN3_{stem}_T_{size}_{model_tag}_eval"
    probs_path = eval_dir / "rating_probabilities.csv"
    if not probs_path.exists():
        if spec_slug == "item" and size == 400 and model_tag == "MISP1_PROJ":
            return _stan_misspec_probs_and_labels(spec_slug, 350, model_tag)
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
        bbox_to_anchor=(1.02, 0.5),
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


def _series_linewidth(label: str) -> float:
    if label == "Marformer":
        return 2.9
    if label == "Stan Oracle":
        return 2.7
    if label in {"Stan Both", "Stan Bin", "Stan Proj"}:
        return 1.6
    return 2.4


def _series_alpha(label: str) -> float:
    if label == "Stan Oracle":
        return 0.98
    if label in {"Stan Both", "Stan Bin", "Stan Proj"}:
        return 0.78
    return 1.0


def _series_fill_alpha(label: str) -> float:
    if label == "Stan Oracle":
        return 0.12
    if label in {"Stan Both", "Stan Bin", "Stan Proj"}:
        return 0.06
    return 0.14


def _series_labels() -> list[str]:
    return [
        "Marformer",
        "Marformer NT",
        "Stan Oracle",
        "Stan Both",
        "Stan Bin",
        "Stan Proj",
        "Best Unigram",
    ]


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


def _collect_metric_series(
    spec: DatasetSpec,
    metric: str,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, list[np.ndarray | None]], list[dict]]:
    labels_in_plot = _series_labels()
    stats = {
        stat_name: {label: [np.nan] * len(spec.sizes) for label in labels_in_plot}
        for stat_name in ("mean", "std", "q10", "q25", "median", "q75", "q90", "min", "max")
    }
    raw_values = {label: [None] * len(spec.sizes) for label in labels_in_plot}
    summary_rows: list[dict] = []

    for idx, size in enumerate(spec.sizes):
        loaders = [
            ("Marformer", lambda: _marformer_probs_and_labels(spec.slug, size, nontrans=False)),
            ("Stan Oracle", lambda: _stan_probs_and_labels(spec.slug, size)),
            ("Best Unigram", lambda: None if _best_unigram_result(spec.slug, size) is None else (_best_unigram_result(spec.slug, size)["probs"], _best_unigram_result(spec.slug, size)["labels"])),
        ]
        for misspec_label, misspec_tag in MISSPEC_MODELS:
            loaders.append((misspec_label, lambda tag=misspec_tag: _stan_misspec_probs_and_labels(spec.slug, size, tag)))

        for label, loader in loaders:
            payload = loader()
            if payload is None:
                continue
            probs, labels = payload
            values = _per_example_nll(probs, labels) if metric == "log_loss" else _per_example_mse(probs, labels)
            raw_values[label][idx] = values
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
                "std": std,
                "q10": float(q10),
                "q25": float(q25),
                "median": float(median),
                "q75": float(q75),
                "q90": float(q90),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": int(values.size),
                "unigram_variant": unigram_variant,
            })

    for nt_train_size in spec.sizes:
        total_size = _nt_x_position(spec, nt_train_size)
        if total_size not in spec.sizes:
            continue
        idx = spec.sizes.index(total_size)
        if metric == "log_loss":
            saved = _best_json_metric(spec, nt_train_size, nontrans=True, metric="log_loss")
            if saved is None:
                continue
            mean, count = saved
            raw_values["Marformer NT"][idx] = np.asarray([mean], dtype=np.float64)
            stats["mean"]["Marformer NT"][idx] = mean
            stats["std"]["Marformer NT"][idx] = 0.0
            stats["q10"]["Marformer NT"][idx] = mean
            stats["q25"]["Marformer NT"][idx] = mean
            stats["median"]["Marformer NT"][idx] = mean
            stats["q75"]["Marformer NT"][idx] = mean
            stats["q90"]["Marformer NT"][idx] = mean
            stats["min"]["Marformer NT"][idx] = mean
            stats["max"]["Marformer NT"][idx] = mean
            summary_rows.append({
                "dataset": spec.slug,
                "size": total_size,
                "model": "Marformer NT",
                "metric": metric,
                "mean": mean,
                "std": 0.0,
                "q10": mean,
                "q25": mean,
                "median": mean,
                "q75": mean,
                "q90": mean,
                "min": mean,
                "max": mean,
                "count": count,
                "unigram_variant": None,
                "nt_train_size": nt_train_size,
            })
            continue

        payload = _marformer_probs_and_labels(spec.slug, nt_train_size, nontrans=True)
        if payload is None:
            continue
        probs, labels = payload
        values = _per_example_nll(probs, labels) if metric == "log_loss" else _per_example_mse(probs, labels)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        q10, q25, median, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
        raw_values["Marformer NT"][idx] = values
        stats["mean"]["Marformer NT"][idx] = mean
        stats["std"]["Marformer NT"][idx] = std
        stats["q10"]["Marformer NT"][idx] = float(q10)
        stats["q25"]["Marformer NT"][idx] = float(q25)
        stats["median"]["Marformer NT"][idx] = float(median)
        stats["q75"]["Marformer NT"][idx] = float(q75)
        stats["q90"]["Marformer NT"][idx] = float(q90)
        stats["min"]["Marformer NT"][idx] = float(np.min(values))
        stats["max"]["Marformer NT"][idx] = float(np.max(values))
        summary_rows.append({
            "dataset": spec.slug,
            "size": total_size,
            "model": "Marformer NT",
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
            "unigram_variant": None,
            "nt_train_size": nt_train_size,
        })
    return stats, raw_values, summary_rows


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
    if interval_kind == "p10_p90":
        lower = np.asarray(stats["q10"][label], dtype=float)
        upper = np.asarray(stats["q90"][label], dtype=float)
        return mean, lower, upper
    raise ValueError(f"Unknown interval kind: {interval_kind}")


def _metric_variant_title(title: str, interval_kind: str) -> str:
    if interval_kind == "std01":
        return f"{title} (mean ± 0.05 SD)"
    raise ValueError(f"Unknown interval kind: {interval_kind}")


def _plot_metric_series(
    spec: DatasetSpec,
    metric: str,
    ylabel: str,
    title: str,
    output_name: str,
) -> tuple[list[dict], dict[str, list[np.ndarray | None]]]:
    stats, raw_values, summary_rows = _collect_metric_series(spec, metric)
    render_title = _metric_variant_title(title, "std01")
    if spec.slug in {"annot", "item"} and metric == "log_loss":
        _plot_log_loss_broken(spec, stats, ylabel, render_title, output_name, "std01")
    else:
        _plot_metric_band_series(spec, stats, ylabel, render_title, output_name, "std01")
    return summary_rows, raw_values


def _plot_metric_band_series(
    spec: DatasetSpec,
    stats: dict[str, dict[str, list[float]]],
    ylabel: str,
    title: str,
    output_name: str,
    interval_kind: str,
) -> None:
    fig, ax = plt.subplots(figsize=spec.figure_size)
    ymins: list[float] = []
    ymaxs: list[float] = []
    for label in _series_labels():
        xs = np.asarray(spec.sizes, dtype=float)
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

    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14)
    ax.set_xticks(spec.sizes)
    ax.set_xticklabels(_x_tick_labels(spec))
    if spec.slug == "annot":
        ax.set_xlim(spec.sizes[0] - 1.0, spec.sizes[-1] + 1.0)
    if ymins and ymaxs:
        ymin = min(ymins)
        ymax = max(ymaxs)
        pad = max(0.03, 0.07 * (ymax - ymin))
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    _style_legend(ax, loc="center left")
    ax.margins(x=0.03)

    output_path = spec.out_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 0.98], pad=1.0)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _plot_log_loss_broken(
    spec: DatasetSpec,
    stats: dict[str, dict[str, list[float]]],
    ylabel: str,
    title: str,
    output_name: str,
    interval_kind: str,
) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=spec.figure_size,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 4.2], "hspace": 0.05},
    )

    for label in _series_labels():
        xs = np.asarray(spec.sizes, dtype=float)
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

    if spec.slug == "item":
        lower_ticks = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        upper_ticks = [2.0, 4.0, 6.0]
        ax_bottom.set_ylim(0.18, 1.24)
        ax_top.set_ylim(1.95, 6.05)
        ax_bottom.yaxis.set_major_locator(FixedLocator(lower_ticks))
        ax_bottom.yaxis.set_major_formatter(FixedFormatter([f"{tick:.1f}" for tick in lower_ticks]))
        ax_top.yaxis.set_major_locator(FixedLocator(upper_ticks))
        ax_top.yaxis.set_major_formatter(FixedFormatter(["", "4.0", "6.0"]))
    else:
        lower_ticks = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        upper_ticks = [2.0, 3.0, 4.0]
        ax_bottom.set_ylim(0.38, 1.48)
        ax_top.set_ylim(1.95, 4.05)
        ax_bottom.yaxis.set_major_locator(FixedLocator(lower_ticks))
        ax_bottom.yaxis.set_major_formatter(FixedFormatter([f"{tick:.1f}" for tick in lower_ticks]))
        ax_top.yaxis.set_major_locator(FixedLocator(upper_ticks))
        ax_top.yaxis.set_major_formatter(FixedFormatter(["", "3.0", "4.0"]))

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

    ax_bottom.set_xlabel(spec.x_label)
    ax_bottom.set_ylabel(ylabel)
    ax_top.set_title(title, pad=14)
    ax_bottom.set_xticks(spec.sizes)
    ax_bottom.set_xticklabels(_x_tick_labels(spec))
    ax_bottom.set_xlim(spec.sizes[0] - 1.0, spec.sizes[-1] + 1.0)

    for ax in (ax_top, ax_bottom):
        ax.grid(True, alpha=0.35)

    _style_legend(ax_bottom, loc="center left")

    output_path = spec.out_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.80, top=0.90, bottom=0.12, hspace=0.05)
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
    _style_legend(ax, loc="center left")

    output_path = spec.out_dir / f"{spec.slug}_generalization_runtime.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 0.98], pad=1.0)
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

        payloads = [
            ("Marformer", _marformer_probs_and_labels(spec.slug, size), "Marformer"),
        ]
        if has_nt:
            payloads.append((f"Marformer NT\n(train={nt_variant})", nt_payload, "Marformer"))
        payloads.append(("Stan Oracle", _stan_probs_and_labels(spec.slug, size), "Stan Oracle"))
        for misspec_label, misspec_tag in MISSPEC_MODELS:
            payloads.append((misspec_label, _stan_misspec_probs_and_labels(spec.slug, size, misspec_tag), misspec_label))
        best_unigram = _best_unigram_result(spec.slug, size)
        unigram_title = "Best Unigram"
        if best_unigram is not None:
            pretty = best_unigram["variant"].replace("-", " ").title()
            unigram_title = f"Best Unigram\n({pretty})"
        payloads.append((unigram_title, None if best_unigram is None else (best_unigram["probs"], best_unigram["labels"]), "Best Unigram"))

        n_panels = len(payloads)
        ncols = 4 if n_panels > 6 else 3
        nrows = math.ceil(n_panels / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 4.7 * nrows))
        axes = np.atleast_1d(axes).flatten()

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
    metric_rows, _ = _plot_metric_series(
        spec,
        metric="log_loss",
        ylabel="Test Log-Loss",
        title=f"{spec.title_prefix}: Log-Loss with Additional Training {'Items' if spec.slug == 'item' else 'Annotators'}",
        output_name=f"{spec.slug}_generalization_log_loss.png",
    )
    rows.extend(metric_rows)
    metric_rows, _ = _plot_metric_series(
        spec,
        metric="mbr_l2",
        ylabel="MBR L2 (MSE)",
        title=f"{spec.title_prefix}: MBR L2 with Additional Training {'Items' if spec.slug == 'item' else 'Annotators'}",
        output_name=f"{spec.slug}_generalization_mbr_l2.png",
    )
    rows.extend(metric_rows)
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
