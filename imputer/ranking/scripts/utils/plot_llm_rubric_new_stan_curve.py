#!/usr/bin/env python3
"""
Plot performance curves for the new STAN SharedThreshold model on LLM Rubric.

Metrics plotted (separate files):
  - Test missing log loss = -rating_missing_log_likelihood
  - Test missing RMSE
  - Test missing L2 (MSE)

Run from imputer/ranking:
  python scripts/utils/plot_llm_rubric_new_stan_curve.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIZE_RE = re.compile(r"LLMRubric_225_25_9_(\d+)_eval$")


def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _extract_size(eval_dir_name: str) -> int | None:
    match = SIZE_RE.match(eval_dir_name)
    return int(match.group(1)) if match else None


def _load_bundle(data_root: Path, size: int) -> dict | None:
    bundle_path = data_root / f"LLMRubric_225_25_9_{size}" / "data_bundle.json"
    if not bundle_path.exists():
        return None
    return _read_json(bundle_path)


def _load_unigram_global_log_loss(data_root: Path, size: int) -> float | None:
    bundle = _load_bundle(data_root, size)
    if bundle is None:
        return None
    observed = bundle.get("observed_ratings", [])
    test_missing = [r for r in bundle.get("missing_ratings", []) if r.get("instance") == "test"]
    if not observed or not test_missing:
        return None

    c = max(int(r["value"]) for r in observed + test_missing)
    counts = [0.0] * c
    for r in observed:
        counts[int(r["value"]) - 1] += 1.0
    total = sum(counts)
    probs = [(cnt + 1.0) / (total + c) for cnt in counts]  # add-one smoothing

    xent = 0.0
    for r in test_missing:
        idx = int(r["value"]) - 1
        xent -= math.log(probs[idx] + 1e-12)
    return xent / len(test_missing)


def _load_unigram_subset_log_loss(data_root: Path, size: int, pool_by: str) -> float | None:
    bundle = _load_bundle(data_root, size)
    if bundle is None:
        return None
    observed = bundle.get("observed_ratings", [])
    test_missing = [r for r in bundle.get("missing_ratings", []) if r.get("instance") == "test"]
    if not observed or not test_missing:
        return None

    c = max(int(r["value"]) for r in observed + test_missing)
    pool_counts: dict[tuple[int, ...], list[float]] = {}

    def key_fn(r: dict) -> tuple[int, ...]:
        i = int(r["attribute"])
        j = int(r["annotator"])
        k = int(r["item"])
        if pool_by == "i":
            return (i,)
        if pool_by == "j":
            return (j,)
        if pool_by == "k":
            return (k,)
        if pool_by == "ij":
            return (i, j)
        if pool_by == "ik":
            return (i, k)
        if pool_by == "jk":
            return (j, k)
        raise ValueError(pool_by)

    for r in observed:
        key = key_fn(r)
        if key not in pool_counts:
            pool_counts[key] = [0.0] * c
        pool_counts[key][int(r["value"]) - 1] += 1.0

    xent = 0.0
    for r in test_missing:
        counts = pool_counts.get(key_fn(r), [0.0] * c)
        denom = sum(counts) + c
        idx = int(r["value"]) - 1
        prob = (counts[idx] + 1.0) / denom
        xent -= math.log(prob + 1e-12)
    return xent / len(test_missing)


def _load_nb_ijk_log_loss(data_root: Path, size: int) -> float | None:
    bundle = _load_bundle(data_root, size)
    if bundle is None:
        return None
    observed = [r for r in bundle.get("observed_ratings", []) if r.get("instance") in {"train", "val", "test"}]
    test_missing = [r for r in bundle.get("missing_ratings", []) if r.get("instance") == "test"]
    if not observed or not test_missing:
        return None

    c = max(int(r["value"]) for r in observed + test_missing)
    max_i = max(int(r["attribute"]) for r in observed + test_missing)
    max_j = max(int(r["annotator"]) for r in observed + test_missing)
    max_k = max(int(r["item"]) for r in observed + test_missing)

    class_counts = [0.0] * c
    i_counts = [[0.0] * max_i for _ in range(c)]
    j_counts = [[0.0] * max_j for _ in range(c)]
    k_counts = [[0.0] * max_k for _ in range(c)]
    for r in observed:
        y = int(r["value"]) - 1
        i = int(r["attribute"]) - 1
        j = int(r["annotator"]) - 1
        k = int(r["item"]) - 1
        class_counts[y] += 1.0
        i_counts[y][i] += 1.0
        j_counts[y][j] += 1.0
        k_counts[y][k] += 1.0

    n = sum(class_counts)
    log_py = [math.log((class_counts[y] + 1.0) / (n + c)) for y in range(c)]
    log_pi = [[math.log((cnt + 1.0) / (class_counts[y] + max_i)) for cnt in i_counts[y]] for y in range(c)]
    log_pj = [[math.log((cnt + 1.0) / (class_counts[y] + max_j)) for cnt in j_counts[y]] for y in range(c)]
    log_pk = [[math.log((cnt + 1.0) / (class_counts[y] + max_k)) for cnt in k_counts[y]] for y in range(c)]

    xent = 0.0
    for r in test_missing:
        y_true = int(r["value"]) - 1
        i = int(r["attribute"]) - 1
        j = int(r["annotator"]) - 1
        k = int(r["item"]) - 1
        scores = [log_py[y] + log_pi[y][i] + log_pj[y][j] + log_pk[y][k] for y in range(c)]
        m = max(scores)
        log_norm = m + math.log(sum(math.exp(s - m) for s in scores))
        xent -= (scores[y_true] - log_norm)
    return xent / len(test_missing)


def main() -> None:
    root = Path("RESULTS/STAN/LLM_RUBRIC/CPM_SHARED_THRESHOLD")
    data_root = Path("DATA/LLM_RUBRIC")
    output_logloss = Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_cpm_shared_threshold_log_loss_curve.png")
    output_rmse = Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_cpm_shared_threshold_rmse_curve.png")
    output_l2 = Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_cpm_shared_threshold_l2_curve.png")

    logloss_points: list[tuple[int, float]] = []
    unigram_global_points: list[tuple[int, float]] = []
    nb_points: list[tuple[int, float]] = []
    subset_keys = ["i", "j", "k", "ij", "ik", "jk"]
    unigram_subset_points: dict[str, list[tuple[int, float]]] = {k: [] for k in subset_keys}
    rmse_points: list[tuple[int, float]] = []
    l2_points: list[tuple[int, float]] = []
    for metrics_path in root.glob("LLMRubric_225_25_9_*_eval/predictive_metrics.json"):
        size = _extract_size(metrics_path.parent.name)
        if size is None:
            continue
        metrics = _read_json(metrics_path)
        ll = metrics.get("rating_missing_log_likelihood")
        if ll is None:
            continue
        logloss_points.append((size, float(-ll)))
        ug = _load_unigram_global_log_loss(data_root, size)
        if ug is not None:
            unigram_global_points.append((size, ug))
        nb = _load_nb_ijk_log_loss(data_root, size)
        if nb is not None:
            nb_points.append((size, nb))
        for key in subset_keys:
            us = _load_unigram_subset_log_loss(data_root, size, key)
            if us is not None:
                unigram_subset_points[key].append((size, us))

        probs_path = metrics_path.parent / "rating_probabilities.csv"
        bundle_path = data_root / f"LLMRubric_225_25_9_{size}" / "data_bundle.json"
        if probs_path.exists() and bundle_path.exists():
            bundle = _read_json(bundle_path)
            missing = bundle.get("missing_ratings", [])
            test_idxs = [i for i, row in enumerate(missing) if row.get("instance") == "test"]
            if test_idxs:
                labels = np.asarray([missing[i]["value"] - 1 for i in test_idxs], dtype=np.int64)
                df = pd.read_csv(probs_path)
                prob_cols = ["prob_cat_1", "prob_cat_2", "prob_cat_3", "prob_cat_4"]
                grouped = (
                    df[df["missing_rating_idx"].isin(test_idxs)]
                    .groupby("missing_rating_idx")[prob_cols]
                    .mean()
                    .reindex(test_idxs)
                )
                if not grouped.isnull().any().any():
                    probs = grouped.to_numpy(dtype=np.float64)
                    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
                    pred_expected = probs @ classes
                    truth = labels.astype(np.float64) + 1.0
                    mse = float(np.mean((pred_expected - truth) ** 2))
                    rmse = float(np.sqrt(mse))
                    rmse_points.append((size, rmse))
                    l2_points.append((size, mse))

    if not logloss_points:
        raise SystemExit(f"No LLM Rubric predictive_metrics.json found under {root}")

    logloss_points.sort(key=lambda x: x[0])
    xs = [p[0] for p in logloss_points]
    ys = [p[1] for p in logloss_points]

    plt.figure(figsize=(8.8, 5.2))
    plt.plot(xs, ys, marker="o", color="#1b9e77", linewidth=2.2, label="CPM SharedThreshold STAN")
    if unigram_global_points:
        unigram_global_points.sort(key=lambda x: x[0])
        plt.plot(
            [p[0] for p in unigram_global_points],
            [p[1] for p in unigram_global_points],
            marker="D",
            linestyle=":",
            color="#7a7a7a",
            linewidth=2.0,
            label="Unigram (Global)",
        )
    subset_style = {
        "i": ("#2e8b57", "x"),
        "j": ("#3b5bdb", "P"),
        "k": ("#e67700", "v"),
        "ij": ("#0b7285", ">"),
        "ik": ("#5f3dc4", "<"),
        "jk": ("#c2255c", "^"),
    }
    for key in subset_keys:
        pts = unigram_subset_points[key]
        if not pts:
            continue
        pts.sort(key=lambda x: x[0])
        color, marker = subset_style[key]
        plt.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker=marker,
            linestyle=":",
            color=color,
            linewidth=1.8,
            label=f"Unigram (Pool {key})",
        )
    if nb_points:
        nb_points.sort(key=lambda x: x[0])
        plt.plot(
            [p[0] for p in nb_points],
            [p[1] for p in nb_points],
            marker="*",
            linestyle="-.",
            color="#111111",
            linewidth=2.0,
            label="Naive Bayes (i,j,k, T)",
        )
    plt.xlabel("Training Items")
    plt.ylabel("Test Missing Log Loss")
    plt.title("LLM Rubric: New STAN Model Performance Curve")
    plt.xticks(xs)
    plt.grid(alpha=0.3)
    plt.legend()

    output_logloss.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_logloss, dpi=300)
    plt.close()

    if rmse_points:
        rmse_points.sort(key=lambda x: x[0])
        xs_rmse = [p[0] for p in rmse_points]
        ys_rmse = [p[1] for p in rmse_points]
        plt.figure(figsize=(8.8, 5.2))
        plt.plot(xs_rmse, ys_rmse, marker="o", color="#d55e00", linewidth=2.2, label="CPM SharedThreshold STAN")
        plt.xlabel("Training Items")
        plt.ylabel("Test Missing RMSE")
        plt.title("LLM Rubric: New STAN RMSE Curve")
        plt.xticks(xs_rmse)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_rmse, dpi=300)
        plt.close()

    if l2_points:
        l2_points.sort(key=lambda x: x[0])
        xs_l2 = [p[0] for p in l2_points]
        ys_l2 = [p[1] for p in l2_points]
        plt.figure(figsize=(8.8, 5.2))
        plt.plot(xs_l2, ys_l2, marker="o", color="#8e44ad", linewidth=2.2, label="CPM SharedThreshold STAN")
        plt.xlabel("Training Items")
        plt.ylabel("Test Missing L2 (MSE)")
        plt.title("LLM Rubric: New STAN L2 Curve")
        plt.xticks(xs_l2)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_l2, dpi=300)
        plt.close()

    print(f"Saved plot to: {output_logloss}")
    if rmse_points:
        print(f"Saved plot to: {output_rmse}")
    if l2_points:
        print(f"Saved plot to: {output_l2}")
    print(f"Log-loss sizes: {xs}")


if __name__ == "__main__":
    main()
