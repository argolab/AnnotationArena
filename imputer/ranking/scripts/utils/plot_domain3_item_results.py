#!/usr/bin/env python3
"""
Plot Domain 3 item-expansion talk figures.

Outputs:
  - PLOTS/TALK/DOMAIN3/domain3_item_test_loss_by_size.png
  - PLOTS/TALK/DOMAIN3/domain3_item_mbr_l2_by_size.png
  - PLOTS/TALK/DOMAIN3/domain3_item_mbr_l2_size400.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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

DATA_ROOT = ROOT / "DATA/STAN/DOMAIN3/ItemSplits/Transductive"
MARFORMER_ROOT = ROOT / "RESULTS/MARFORMER/DOMAIN3/ITEM"
STAN_ROOT = ROOT / "RESULTS/STAN/TENSOR/DOMAIN3/ITEM"
STAN_MISP_ROOT = ROOT / "RESULTS/STAN/TENSOR/DOMAIN3_MISP/ITEM"
OUT_DIR = ROOT / "PLOTS/TALK/DOMAIN3"

SIZES = [50, 100, 150, 200, 250, 300, 350, 400]
PROB_PREFIX = "prob_cat_"

COLORS = {
    "Marformer": "#1f6fba",
    "Oracle Tensor Stan": "#1b9e77",
    "Stan Misspecified": "#e67e22",
}

MARKERS = {
    "Marformer": "o",
    "Oracle Tensor Stan": "^",
    "Stan Misspecified": "D",
}

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


def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _bundle_path(size: int) -> Path:
    return DATA_ROOT / f"Tensor_400_25_9_DOMAIN3_Item_T_{size}" / "data_bundle.json"


def _marformer_best_json(size: int) -> Path:
    return (
        MARFORMER_ROOT
        / f"Tensor_400_25_9_DOMAIN3_Item_T_{size}_MARFORMER"
        / "TEST_RESULTS"
        / "best.json"
    )


def _stan_eval_dir(size: int) -> Path:
    return STAN_ROOT / f"Tensor_400_25_9_DOMAIN3_Item_T_{size}_TENSOR_eval"


def _stan_misp_eval_dir(size: int) -> Path:
    return STAN_MISP_ROOT / f"Tensor_400_25_9_DOMAIN3_Item_T_{size}_DISCRETE_MISP_DD_eval"


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


def _load_marformer_metric(size: int, key: str) -> float | None:
    path = _marformer_best_json(size)
    if not path.exists():
        return None
    return float(_read_json(path).get("missing", {}).get(key))


def _load_stan_missing_log_loss(size: int, misspecified: bool) -> float | None:
    eval_dir = _stan_misp_eval_dir(size) if misspecified else _stan_eval_dir(size)
    path = eval_dir / "predictive_metrics.json"
    if not path.exists():
        return None
    ll = _read_json(path).get("rating_missing_log_likelihood")
    if ll is None:
        return None
    return float(-ll)


def _load_stan_probs_and_labels(size: int, misspecified: bool) -> tuple[np.ndarray, np.ndarray] | None:
    bundle_path = _bundle_path(size)
    eval_dir = _stan_misp_eval_dir(size) if misspecified else _stan_eval_dir(size)
    probs_path = eval_dir / "rating_probabilities.csv"
    if not bundle_path.exists() or not probs_path.exists():
        return None

    bundle = _read_json(bundle_path)
    test_idxs, labels = _test_missing_indices_and_labels(bundle)
    if not test_idxs:
        return None

    df = pd.read_csv(probs_path)
    prob_cols = sorted([col for col in df.columns if col.startswith(PROB_PREFIX)])
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


def _load_marformer_probs_and_labels(size: int) -> tuple[np.ndarray, np.ndarray] | None:
    run_dir = MARFORMER_ROOT / f"Tensor_400_25_9_DOMAIN3_Item_T_{size}_MARFORMER"
    best_json = _marformer_best_json(size)
    if not best_json.exists():
        return None

    best = _read_json(best_json)
    ckpt_name = best.get("checkpoint")
    if ckpt_name is None:
        return None
    ckpt_path = run_dir / "checkpoints" / ckpt_name
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
    return np.asarray(probs, dtype=np.float32), np.asarray(labels, dtype=np.int64)


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
    ci_w = diag.get("ce_ci_width", 0.0)
    ax.set_title(f"{title}\nsmECE = {ce:.3f} ± {ci_w:.3f}", fontsize=15, fontweight="bold", pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)


def _plot_calibration(size: int, output_path: Path, exclude_misspecified: bool) -> None:
    panels = [
        ("Marformer", _load_marformer_probs_and_labels(size), COLORS["Marformer"]),
        ("Oracle Tensor Stan", _load_stan_probs_and_labels(size, misspecified=False), COLORS["Oracle Tensor Stan"]),
    ]
    if not exclude_misspecified:
        panels.append(("Stan Misspecified", _load_stan_probs_and_labels(size, misspecified=True), COLORS["Stan Misspecified"]))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 5.4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, payload, color) in zip(axes, panels):
        if payload is None:
            _draw_empty(ax, title)
            continue
        probs, labels = payload
        _plot_ece(ax, probs, labels, title, color)

    fig.suptitle(f"Domain 3: Item Expansion Reliability at Size {size}", fontsize=18, y=0.97)
    fig.subplots_adjust(top=0.74, wspace=0.30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _plot_metric_series(
    output_path: Path,
    ylabel: str,
    title: str,
    series: list[tuple[str, list[int], list[float]]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for label, xs, ys in series:
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            label=label,
            color=COLORS[label],
            marker=MARKERS[label],
        )

    ax.set_xlabel("Training Items")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14)
    ax.set_xticks(SIZES)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _plot_mbr_snapshot(size: int, output_path: Path) -> None:
    method_rows: list[tuple[str, float]] = []

    marformer_rmse = _load_marformer_metric(size, "rmse")
    if marformer_rmse is not None:
        method_rows.append(("Marformer", float(marformer_rmse) ** 2))

    stan_payload = _load_stan_probs_and_labels(size, misspecified=False)
    if stan_payload is not None:
        method_rows.append(("Oracle Tensor Stan", _mse_from_probs_labels(*stan_payload)))

    stan_misp_payload = _load_stan_probs_and_labels(size, misspecified=True)
    if stan_misp_payload is not None:
        method_rows.append(("Stan Misspecified", _mse_from_probs_labels(*stan_misp_payload)))

    if not method_rows:
        print(f"Skip snapshot plot at size {size}: no data")
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    x = np.arange(len(method_rows), dtype=float)
    vals = [row[1] for row in method_rows]
    colors = [COLORS[row[0]] for row in method_rows]
    ax.bar(x, vals, color=colors, alpha=0.92)
    ax.set_ylabel("MBR-L2 (MSE)")
    ax.set_title(f"Domain 3: Item Expansion MBR-L2 at Size {size}", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels([row[0] for row in method_rows], rotation=12, ha="right")
    ax.grid(True, axis="y", alpha=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Domain 3 item-expansion talk figures.")
    parser.add_argument(
        "--exclude-misspecified",
        action="store_true",
        help="Omit the Stan misspecified series from all plots.",
    )
    args = parser.parse_args()

    marformer_log_loss = _series_from_loader(lambda size: _load_marformer_metric(size, "log_loss"))
    oracle_log_loss = _series_from_loader(lambda size: _load_stan_missing_log_loss(size, misspecified=False))
    misp_log_loss = _series_from_loader(lambda size: _load_stan_missing_log_loss(size, misspecified=True))

    marformer_mbr = _series_from_loader(
        lambda size: None if _load_marformer_metric(size, "rmse") is None else _load_marformer_metric(size, "rmse") ** 2
    )
    oracle_mbr = _series_from_loader(
        lambda size: None if _load_stan_probs_and_labels(size, misspecified=False) is None else _mse_from_probs_labels(*_load_stan_probs_and_labels(size, misspecified=False))
    )
    misp_mbr = _series_from_loader(
        lambda size: None if _load_stan_probs_and_labels(size, misspecified=True) is None else _mse_from_probs_labels(*_load_stan_probs_and_labels(size, misspecified=True))
    )

    log_loss_series = [
        ("Marformer", *marformer_log_loss),
        ("Oracle Tensor Stan", *oracle_log_loss),
    ]
    mbr_series = [
        ("Marformer", *marformer_mbr),
        ("Oracle Tensor Stan", *oracle_mbr),
    ]

    if not args.exclude_misspecified:
        log_loss_series.append(("Stan Misspecified", *misp_log_loss))
        mbr_series.append(("Stan Misspecified", *misp_mbr))

    _plot_metric_series(
        OUT_DIR / "domain3_item_test_loss_by_size.png",
        ylabel="Test Missing Log Loss",
        title="Domain 3: Item Expansion Test Loss by Training Size",
        series=log_loss_series,
    )

    _plot_metric_series(
        OUT_DIR / "domain3_item_mbr_l2_by_size.png",
        ylabel="MBR-L2 (MSE)",
        title="Domain 3: Item Expansion MBR-L2 by Training Size",
        series=mbr_series,
    )

    snapshot_size = 400
    if args.exclude_misspecified:
        original = COLORS.pop("Stan Misspecified", None), MARKERS.pop("Stan Misspecified", None)
        try:
            fig_rows: list[tuple[str, float]] = []
            marformer_rmse = _load_marformer_metric(snapshot_size, "rmse")
            if marformer_rmse is not None:
                fig_rows.append(("Marformer", float(marformer_rmse) ** 2))
            stan_payload = _load_stan_probs_and_labels(snapshot_size, misspecified=False)
            if stan_payload is not None:
                fig_rows.append(("Oracle Tensor Stan", _mse_from_probs_labels(*stan_payload)))

            if fig_rows:
                fig, ax = plt.subplots(figsize=(8.8, 5.0))
                x = np.arange(len(fig_rows), dtype=float)
                vals = [row[1] for row in fig_rows]
                colors = [COLORS[row[0]] for row in fig_rows]
                ax.bar(x, vals, color=colors, alpha=0.92)
                ax.set_ylabel("MBR-L2 (MSE)")
                ax.set_title(f"Domain 3: Item Expansion MBR-L2 at Size {snapshot_size}", pad=14)
                ax.set_xticks(x)
                ax.set_xticklabels([row[0] for row in fig_rows], rotation=12, ha="right")
                ax.grid(True, axis="y", alpha=0.35)
                output_path = OUT_DIR / "domain3_item_mbr_l2_size400.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path)
                plt.close(fig)
                print(f"Saved -> {output_path}")
            else:
                print(f"Skip snapshot plot at size {snapshot_size}: no data")
        finally:
            color, marker = original
            if color is not None:
                COLORS["Stan Misspecified"] = color
            if marker is not None:
                MARKERS["Stan Misspecified"] = marker
    else:
        _plot_mbr_snapshot(snapshot_size, OUT_DIR / "domain3_item_mbr_l2_size400.png")

    _plot_calibration(
        snapshot_size,
        OUT_DIR / "ece_reliability_domain3_item_size400.png",
        exclude_misspecified=args.exclude_misspecified,
    )


if __name__ == "__main__":
    main()
