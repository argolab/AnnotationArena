#!/usr/bin/env python3
"""
Temporary plotting utility for DOMAIN3-SOFTPLUS item-expansion limit tests.

Goal:
  Compare large-data Marformer item-generalization runs against the Stan oracle
  reference (run at 400 items) and structured baselines fit on the 4000-item
  split to assess whether Marformer converges in:
    - log loss
    - MBR L2
    - calibration

Outputs:
  - PLOTS/LimitTest/domain3_softplus_limit_log_loss_bar.png
  - PLOTS/LimitTest/domain3_softplus_limit_mbr_l2_bar.png
  - PLOTS/LimitTest/domain3_softplus_limit_calibration.png
  - PLOTS/LimitTest/domain3_softplus_limit_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
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

from BASELINES.structured_baselines.dataset_adapter import (
    build_test_examples,
    build_training_examples,
    bundle_dims,
)
from BASELINES.structured_baselines.log_linear_structured import StructuredLogLinear
from BASELINES.structured_baselines.naive_bayes_ijk import NaiveBayesIJK
from BASELINES.structured_baselines.naive_bayes_structured import StructuredNaiveBayes
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.model import EntityMarformer
from imputer.entity_mf.test import _load_checkpoint, _reconstruct


mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.framealpha": 0.96,
    "legend.edgecolor": "0.75",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.6,
    "lines.markersize": 8,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

rp.config.use_tex_fonts = False

MARFORMER_ROOT = ROOT / "RESULTS/MARFORMER/DOMAIN3-SOFTPLUS/ITEM"
STAN_EVAL_ROOT = ROOT / "RESULTS/STAN/DOMAIN3-SOFTPLUS/ORACLE/ITEM"
DATA_ROOT = ROOT / "DATA/STAN/DOMAIN3-SOFTPLUS/ItemSplits/Transductive"
OUT_DIR = ROOT / "PLOTS/LimitTest"
SUMMARY_PATH = OUT_DIR / "domain3_softplus_limit_summary.json"
CACHE_PATH = OUT_DIR / "domain3_softplus_limit_cache.npz"

STAN_SIZE = 400
BAR_SIZE = 4000
MARFORMER_SIZES = [4000]
MARFORMER_MAX_ITEM = 100
PROB_PREFIX = "prob_cat_"

COLORS = {
    "Marformer": "#1f6fba",
    "Stan Oracle": "#7a1f3d",
    "Naive Bayes IJK": "#1f7a4d",
    "Structured NB": "#2fa36b",
    "Structured Log-Linear": "#73c69d",
}

MARKERS = {
    "Marformer": "o",
    "Stan Oracle": "^",
    "Naive Bayes IJK": "v",
    "Structured NB": ">",
    "Structured Log-Linear": "<",
}

STRUCTURED_BASELINES = [
    "Naive Bayes IJK",
    "Structured NB",
    "Structured Log-Linear",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DOMAIN3-SOFTPLUS limit-test figures.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for Marformer evaluation. Default: auto-detect cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--use-summary",
        action="store_true",
        help="Reuse cached Marformer checkpoint selections from the saved summary when available.",
    )
    return parser.parse_args()


def _choose_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _marformer_run_dir(size: int) -> Path:
    return MARFORMER_ROOT / f"DOMAIN3-SOFTPLUS_Item_T_{size}_MARFORMER"


def _bundle_path(size: int) -> Path:
    return DATA_ROOT / f"DOMAIN3-SOFTPLUS_Item_T_{size}" / "data_bundle.json"


def _stan_eval_dir(size: int) -> Path:
    return STAN_EVAL_ROOT / f"DOMAIN3-SOFTPLUS_Item_T_{size}_TENSOR_eval"


def _test_missing_indices_and_labels(bundle: dict) -> tuple[list[int], np.ndarray]:
    idxs = list(bundle.get("missing_ratings_indexes_in_test_instance", []))
    labels = np.asarray([bundle["missing_ratings"][idx]["value"] - 1 for idx in idxs], dtype=np.int64)
    return idxs, labels


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


def _bundle_num_items(bundle: dict) -> int:
    rows = bundle.get("observed_ratings", []) + bundle.get("missing_ratings", [])
    if not rows:
        return 0
    return max(int(row["item"]) for row in rows)


def _load_bundle(size: int) -> dict:
    return _read_json(_bundle_path(size))


def _global_probs_from_observed(observed: list[dict], num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    for row in observed:
        dist = row.get("rating_dist")
        if dist is not None:
            counts += np.asarray(dist, dtype=np.float64)
        else:
            counts[int(row["value"]) - 1] += 1.0
    total = counts.sum()
    if total <= 0:
        return np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    return counts / total


def _structured_baseline_probs_and_labels(size: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    bundle = _load_bundle(size)
    bundle_path = _bundle_path(size)
    test_examples = build_test_examples(bundle)
    if not test_examples:
        raise RuntimeError(f"No test examples found in {bundle_path}")

    num_attrs, num_anns, num_classes = bundle_dims(bundle, bundle_path)
    num_items = _bundle_num_items(bundle)
    labels = np.asarray([ex.y for ex in test_examples], dtype=np.int64)

    if method == "Naive Bayes IJK":
        model = NaiveBayesIJK.fit_from_bundle(bundle, transductive=True)
        probs = model.predict_proba(test_examples)
    else:
        train_examples = build_training_examples(bundle, instances={"train"})
        if method == "Structured NB":
            model = StructuredNaiveBayes.fit(
                train_examples,
                num_attrs=num_attrs,
                num_classes=num_classes,
                num_anns=num_anns,
                num_items=num_items,
            )
            probs = model.predict_proba(test_examples)
        elif method == "Structured Log-Linear":
            model = StructuredLogLinear.fit(
                train_examples,
                num_attrs=num_attrs,
                num_classes=num_classes,
                epochs=40,
                lr=0.05,
                batch_size=256,
                device=None,
                verbose=False,
                show_progress=False,
            )
            probs = model.predict_proba(test_examples)
        else:
            raise ValueError(f"Unknown baseline method: {method}")

    return probs, labels


def _stan_probs_and_labels(size: int) -> tuple[np.ndarray, np.ndarray]:
    eval_dir = _stan_eval_dir(size)
    probs_path = eval_dir / "rating_probabilities.csv"
    if not probs_path.exists():
        raise FileNotFoundError(f"Missing Stan probabilities: {probs_path}")

    bundle = _load_bundle(size)
    test_idxs, labels = _test_missing_indices_and_labels(bundle)
    df = pd.read_csv(probs_path)
    prob_cols = sorted(col for col in df.columns if col.startswith(PROB_PREFIX))
    grouped = (
        df[df["missing_rating_idx"].isin(test_idxs)]
        .groupby("missing_rating_idx")[prob_cols]
        .mean()
        .reindex(test_idxs)
    )
    if grouped.isnull().any().any():
        raise RuntimeError(f"Incomplete Stan predictions in {probs_path}")
    return grouped.to_numpy(dtype=np.float64), labels


def _load_summary() -> dict | None:
    if not SUMMARY_PATH.exists():
        return None
    return _read_json(SUMMARY_PATH)


def _save_cache(
    marformer_payloads: dict[int, tuple[str, np.ndarray, np.ndarray]],
    stan_payload: tuple[np.ndarray, np.ndarray],
    baseline_payloads: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    payload: dict[str, np.ndarray] = {}
    for size, (checkpoint_name, probs, labels) in marformer_payloads.items():
        payload[f"marformer_probs_{size}"] = probs
        payload[f"marformer_labels_{size}"] = labels
        payload[f"marformer_checkpoint_{size}"] = np.asarray([checkpoint_name])
    payload["stan_probs"] = stan_payload[0]
    payload["stan_labels"] = stan_payload[1]
    for model_name, (probs, labels) in baseline_payloads.items():
        safe = model_name.lower().replace(" ", "_").replace("-", "_")
        payload[f"baseline_probs_{safe}"] = probs
        payload[f"baseline_labels_{safe}"] = labels
    np.savez_compressed(CACHE_PATH, **payload)


def _load_cache() -> tuple[
    dict[int, tuple[str, np.ndarray, np.ndarray]],
    tuple[np.ndarray, np.ndarray],
    dict[str, tuple[np.ndarray, np.ndarray]],
] | None:
    if not CACHE_PATH.exists():
        return None

    data = np.load(CACHE_PATH, allow_pickle=False)
    marformer_payloads: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    for size in MARFORMER_SIZES:
        probs_key = f"marformer_probs_{size}"
        labels_key = f"marformer_labels_{size}"
        ckpt_key = f"marformer_checkpoint_{size}"
        if probs_key not in data or labels_key not in data or ckpt_key not in data:
            return None
        checkpoint_name = str(data[ckpt_key][0])
        marformer_payloads[size] = (checkpoint_name, data[probs_key], data[labels_key])

    if "stan_probs" not in data or "stan_labels" not in data:
        return None
    stan_payload = (data["stan_probs"], data["stan_labels"])

    baseline_payloads: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name in STRUCTURED_BASELINES:
        safe = model_name.lower().replace(" ", "_").replace("-", "_")
        probs_key = f"baseline_probs_{safe}"
        labels_key = f"baseline_labels_{safe}"
        if probs_key not in data or labels_key not in data:
            return None
        baseline_payloads[model_name] = (data[probs_key], data[labels_key])

    return marformer_payloads, stan_payload, baseline_payloads


def _chunk_variables_by_item_limit(variables: list, max_item: int) -> list[list]:
    item_ids = sorted({iid for var in variables for iid in var.item_ids})
    if len(item_ids) <= max_item:
        return [variables]

    chunks: list[list] = []
    for start in range(0, len(item_ids), max_item):
        item_set = set(item_ids[start : start + max_item])
        chunk_vars = [var for var in variables if all(iid in item_set for iid in var.item_ids)]
        if chunk_vars:
            chunks.append(chunk_vars)
    return chunks


@torch.no_grad()
def _marformer_probs_for_checkpoint(
    model: EntityMarformer,
    variables: list,
    ckpt_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    _load_checkpoint(model, ckpt_path, device)
    model.to(device)
    model.eval()

    num_classes = model.types["rating"].num_classes
    labels: list[int] = []
    prob_rows: list[np.ndarray] = []

    for chunk_vars in _chunk_variables_by_item_limit(variables, MARFORMER_MAX_ITEM):
        graph = variable_list_to_entity_graph(chunk_vars, model.types)
        params = model(graph, device=device)
        for idx, _var in enumerate(chunk_vars):
            tok = graph.tokens[idx]
            if tok.type_name != "rating" or tok.status != 0:
                continue
            raw = tok.raw_data or {}
            rating_value = raw.get("rating_value")
            if rating_value is None:
                continue
            logits = params[0, idx, 1 : 1 + num_classes]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            prob_rows.append(probs.astype(np.float64))
            labels.append(int(rating_value))

    if not prob_rows:
        raise RuntimeError(f"No missing test predictions produced for checkpoint {ckpt_path}")
    return np.vstack(prob_rows), np.asarray(labels, dtype=np.int64)


def _write_best_json_from_test_results(run_dir: Path) -> str:
    test_dir = run_dir / "TEST_RESULTS"
    candidates: list[tuple[float, float, str, dict]] = []
    for path in sorted(test_dir.glob("*.json")):
        if path.name in {"best.json", "launcher_metadata.json"}:
            continue
        data = _read_json(path)
        missing = data.get("missing", {})
        log_loss = missing.get("log_loss")
        rmse = missing.get("rmse")
        count = missing.get("n", 0)
        if log_loss is None or count == 0:
            continue
        candidates.append((
            float(log_loss),
            float(rmse) if rmse is not None else math.inf,
            path.name,
            data,
        ))

    if not candidates:
        raise RuntimeError(f"No valid checkpoint evaluation JSON found in {test_dir}")

    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    best_log_loss, best_rmse, best_name, best_data = candidates[0]
    best_data["selected_from"] = best_name
    best_data["selection_metric"] = "missing.log_loss"

    best_path = test_dir / "best.json"
    _write_json(best_path, best_data)
    print(f"  selected best checkpoint from TEST_RESULTS: {best_name}")
    print(f"  selected missing log_loss={best_log_loss:.6f} rmse={best_rmse:.6f}")
    return str(best_data.get("checkpoint", "") or best_name.replace(".json", ""))


def _ensure_marformer_best_json(run_dir: Path, device_name: str) -> Path:
    test_dir = run_dir / "TEST_RESULTS"
    best_json = test_dir / "best.json"
    if best_json.exists():
        return best_json

    print("  TEST_RESULTS/best.json missing; running full Marformer checkpoint evaluation")
    test_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint_mode in ("all", "last"):
        cmd = [
            sys.executable,
            "-m",
            "imputer.entity_mf.test",
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            checkpoint_mode,
            "--device",
            device_name,
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True)

    checkpoint_name = _write_best_json_from_test_results(run_dir)
    if not checkpoint_name:
        raise RuntimeError(f"Failed to determine best checkpoint for {run_dir}")
    return best_json


def _evaluate_marformer_run(
    size: int,
    device_name: str,
    selected_checkpoint_from_summary: str | None = None,
) -> tuple[str, np.ndarray, np.ndarray]:
    run_dir = _marformer_run_dir(size)
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise RuntimeError(f"Checkpoint directory missing: {ckpt_dir}")

    if selected_checkpoint_from_summary:
        candidate = ckpt_dir / selected_checkpoint_from_summary
        if candidate.exists():
            print(f"  using cached summary checkpoint selection: {candidate.name}")
            model, eval_vars, _train_cfg = _reconstruct(run_dir)
            device = torch.device(device_name)
            probs, labels = _marformer_probs_for_checkpoint(model, eval_vars, candidate, device)
            return candidate.name, probs, labels

    best_json = _ensure_marformer_best_json(run_dir, device_name)
    checkpoint_name = _read_json(best_json).get("checkpoint")
    if not checkpoint_name:
        raise RuntimeError(f"Checkpoint field missing in {best_json}")
    selected_ckpt = ckpt_dir / checkpoint_name
    if not selected_ckpt.exists():
        raise RuntimeError(f"Selected checkpoint missing: {selected_ckpt}")

    model, eval_vars, _train_cfg = _reconstruct(run_dir)
    device = torch.device(device_name)
    probs, labels = _marformer_probs_for_checkpoint(model, eval_vars, selected_ckpt, device)
    return selected_ckpt.name, probs, labels


def _metric_row(model: str, display_size: int, source_size: int, metric: str, values: np.ndarray) -> dict:
    q10, q25, q50, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "model": model,
        "display_size": int(display_size),
        "source_size": int(source_size),
        "metric": metric,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "q10": float(q10),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "q90": float(q90),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "count": int(values.size),
    }


def _build_summary(
    marformer_payloads: dict[int, tuple[str, np.ndarray, np.ndarray]],
    stan_payload: tuple[np.ndarray, np.ndarray],
    baseline_payloads: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    rows: list[dict] = []
    selection = {
        str(size): {"checkpoint": ckpt_name}
        for size, (ckpt_name, _probs, _labels) in marformer_payloads.items()
    }

    for size, (_ckpt_name, probs, labels) in marformer_payloads.items():
        rows.append(_metric_row("Marformer", size, size, "log_loss", _per_example_nll(probs, labels)))
        rows.append(_metric_row("Marformer", size, size, "mbr_l2", _per_example_mse(probs, labels)))

    stan_probs, stan_labels = stan_payload
    rows.append(_metric_row("Stan Oracle", STAN_SIZE, STAN_SIZE, "log_loss", _per_example_nll(stan_probs, stan_labels)))
    rows.append(_metric_row("Stan Oracle", STAN_SIZE, STAN_SIZE, "mbr_l2", _per_example_mse(stan_probs, stan_labels)))

    for model_name, (probs, labels) in baseline_payloads.items():
        rows.append(_metric_row(model_name, BAR_SIZE, BAR_SIZE, "log_loss", _per_example_nll(probs, labels)))
        rows.append(_metric_row(model_name, BAR_SIZE, BAR_SIZE, "mbr_l2", _per_example_mse(probs, labels)))

    return {
        "dataset": "DOMAIN3-SOFTPLUS",
        "stan_reference_size": STAN_SIZE,
        "bar_reference_size": BAR_SIZE,
        "marformer_sizes": sorted(marformer_payloads.keys()),
        "marformer_selection": selection,
        "rows": rows,
    }


def _summary_rows_by_metric(summary: dict, metric: str) -> list[dict]:
    return [row for row in summary.get("rows", []) if row.get("metric") == metric]


def _summary_lookup(summary: dict, metric: str, model: str, display_size: int) -> dict | None:
    for row in summary.get("rows", []):
        if row.get("metric") == metric and row.get("model") == model and row.get("display_size") == display_size:
            return row
    return None


def _metric_arrays(
    marformer_payloads: dict[int, tuple[str, np.ndarray, np.ndarray]],
    stan_payload: tuple[np.ndarray, np.ndarray],
    baseline_payloads: dict[str, tuple[np.ndarray, np.ndarray]],
    metric: str,
) -> dict[str, np.ndarray]:
    metric_fn = _per_example_nll if metric == "log_loss" else _per_example_mse
    out: dict[str, np.ndarray] = {}

    marformer_probs, marformer_labels = marformer_payloads[BAR_SIZE][1], marformer_payloads[BAR_SIZE][2]
    out["Stan Oracle"] = metric_fn(*stan_payload)
    out["Marformer"] = metric_fn(marformer_probs, marformer_labels)
    for model_name, payload in baseline_payloads.items():
        out[model_name] = metric_fn(*payload)
    return out


def _bar_plot(metric: str, ylabel: str, out_path: Path, raw_values: dict[str, np.ndarray]) -> None:
    ordered_models = ["Stan Oracle", "Marformer", "Naive Bayes IJK", "Structured Log-Linear", "Structured NB"]
    labels = [label for label in ordered_models if label in raw_values]
    arrays = [raw_values[label] for label in labels]
    means = [float(np.mean(values)) for values in arrays]
    errs = [0.05 * float(np.std(values, ddof=0)) for values in arrays]

    fig = plt.figure(figsize=(11.8, 6.0))
    ax = fig.add_subplot(1, 1, 1)
    xpos = np.arange(len(labels))
    bars = ax.bar(
        xpos,
        means,
        yerr=np.asarray(errs),
        color=[COLORS[label] for label in labels],
        edgecolor="#374151",
        linewidth=1.0,
        capsize=4,
        alpha=0.9,
    )
    for bar in bars:
        bar.set_zorder(3)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f"Best {ylabel} Comparison at {BAR_SIZE} Items\n(Stan Oracle uses {STAN_SIZE} items)",
        pad=10,
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    if metric == "log_loss":
        ax.set_ylim(-0.06, 1.6)
        ax.set_yticks(np.arange(0.0, 1.61, 0.2))
    else:
        ymax = max(max(mean + err for mean, err in zip(means, errs)), 0.4)
        ymax = max(0.4, math.ceil((ymax + 0.05) / 0.2) * 0.2)
        ax.set_ylim(-0.06, ymax)
        ax.set_yticks(np.arange(0.0, ymax + 1e-9, 0.2))
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.35)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _calibration_plot(
    marformer_payload: tuple[np.ndarray, np.ndarray],
    stan_payload: tuple[np.ndarray, np.ndarray],
    baseline_payloads: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    panels = [
        ("Marformer", marformer_payload, "Marformer"),
        ("Stan Oracle", stan_payload, "Stan Oracle"),
        ("Naive Bayes IJK", baseline_payloads.get("Naive Bayes IJK"), "Naive Bayes IJK"),
        ("Structured NB", baseline_payloads.get("Structured NB"), "Structured NB"),
        ("Structured Log-Linear", baseline_payloads.get("Structured Log-Linear"), "Structured Log-Linear"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.0))
    flat_axes = axes.ravel()
    for ax, (title, payload, color_key) in zip(flat_axes, panels):
        if payload is None:
            ax.axis("off")
            continue
        probs, labels = payload
        _plot_ece(ax, probs, labels, title, COLORS[color_key])
    for ax in flat_axes[len(panels):]:
        ax.axis("off")

    fig.suptitle("Calibration Plots", fontsize=18, y=0.98)
    fig.subplots_adjust(wspace=0.26, hspace=0.42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device_name = _choose_device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cached_summary = _load_summary() if args.use_summary else None
    cached_selection = (
        cached_summary.get("marformer_selection", {})
        if cached_summary is not None else {}
    )
    cached_payloads = _load_cache() if args.use_summary else None

    marformer_sizes = MARFORMER_SIZES
    missing_runs = [size for size in marformer_sizes if not _marformer_run_dir(size).exists()]
    if missing_runs:
        raise RuntimeError(
            f"Required Marformer run(s) missing under {MARFORMER_ROOT}: {missing_runs}"
        )

    print("============================================================")
    print(" DOMAIN3-SOFTPLUS LIMIT TEST")
    print("============================================================")
    print(f" REPO_ROOT        : {ROOT}")
    print(f" DEVICE           : {device_name}")
    print(f" MARFORMER_SIZES  : {marformer_sizes}")
    print(f" STAN_REFERENCE   : {STAN_SIZE}")
    print(f" BAR_REFERENCE    : {BAR_SIZE}")
    print(f" OUTPUT_DIR       : {OUT_DIR}")
    if cached_payloads is not None and cached_summary is not None:
        print(" Using cached payloads and summary; skipping recomputation.")
        marformer_payloads, stan_payload, baseline_payloads = cached_payloads
        summary = cached_summary
    else:
        marformer_payloads: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
        for size in marformer_sizes:
            selected_ckpt = None
            if str(size) in cached_selection:
                selected_ckpt = cached_selection[str(size)].get("checkpoint")
            print("")
            print(f"[Marformer] size={size}")
            ckpt_name, probs, labels = _evaluate_marformer_run(size, device_name, selected_ckpt)
            print(f"  checkpoint : {ckpt_name}")
            print(f"  log_loss   : {np.mean(_per_example_nll(probs, labels)):.6f}")
            print(f"  mbr_l2     : {np.mean(_per_example_mse(probs, labels)):.6f}")
            marformer_payloads[size] = (ckpt_name, probs, labels)

        print("")
        print(f"[Stan Oracle] size={STAN_SIZE}")
        stan_payload = _stan_probs_and_labels(STAN_SIZE)
        print(f"  log_loss   : {np.mean(_per_example_nll(*stan_payload)):.6f}")
        print(f"  mbr_l2     : {np.mean(_per_example_mse(*stan_payload)):.6f}")

        baseline_payloads: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        print("")
        print(f"[Baselines] size={BAR_SIZE}")

        for model_name in STRUCTURED_BASELINES:
            payload = _structured_baseline_probs_and_labels(BAR_SIZE, model_name)
            baseline_payloads[model_name] = payload
            print(f"  {model_name:<22} log_loss={np.mean(_per_example_nll(*payload)):.6f}  mbr_l2={np.mean(_per_example_mse(*payload)):.6f}")

        summary = _build_summary(marformer_payloads, stan_payload, baseline_payloads)
        _write_json(SUMMARY_PATH, summary)
        _save_cache(marformer_payloads, stan_payload, baseline_payloads)

    log_loss_values = _metric_arrays(marformer_payloads, stan_payload, baseline_payloads, "log_loss")
    mbr_values = _metric_arrays(marformer_payloads, stan_payload, baseline_payloads, "mbr_l2")

    _bar_plot("log_loss", "Test Log Loss", OUT_DIR / "domain3_softplus_limit_log_loss_bar.png", log_loss_values)
    _bar_plot("mbr_l2", "MBR L2 (MSE)", OUT_DIR / "domain3_softplus_limit_mbr_l2_bar.png", mbr_values)
    _calibration_plot(
        (marformer_payloads[BAR_SIZE][1], marformer_payloads[BAR_SIZE][2]),
        stan_payload,
        baseline_payloads,
        OUT_DIR / "domain3_softplus_limit_calibration.png",
    )

    print("")
    print("Wrote:")
    print(f"  {OUT_DIR / 'domain3_softplus_limit_log_loss_bar.png'}")
    print(f"  {OUT_DIR / 'domain3_softplus_limit_mbr_l2_bar.png'}")
    print(f"  {OUT_DIR / 'domain3_softplus_limit_calibration.png'}")
    print(f"  {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
