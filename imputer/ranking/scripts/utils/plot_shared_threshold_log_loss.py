#!/usr/bin/env python3
"""
Plot test-missing log loss for SharedThreshold experiments:
  - MARFORMER transductive
  - MARFORMER non-transductive
  - STAN (true model)

Run from imputer/ranking:
  python scripts/utils/plot_shared_threshold_log_loss.py

Notes:
- For MARFORMER, this script first tries TEST_RESULTS/best*.json (true test log loss).
- If TEST_RESULTS is missing, it falls back to the final epoch's missing xent in
  training_history.json (typically val/combined/train depending on run config).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt


SIZE_RE = re.compile(r"Tensor_400_25_9_ItemTest_SharedThreshold_(\d+)_")


def _read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _extract_size_from_name(name: str) -> int | None:
    match = SIZE_RE.search(name)
    return int(match.group(1)) if match else None


def _discover_stan_sizes(stan_root: Path, dataset_tag: str = "", old_tensor: bool = False) -> list[int]:
    sizes: set[int] = set()
    suffix = f"{dataset_tag}_OLDTENSOR_eval" if (dataset_tag and old_tensor) else (
        "OLDTENSOR_eval" if old_tensor else (f"{dataset_tag}_eval" if dataset_tag else "eval")
    )
    pattern = f"Tensor_400_25_9_ItemTest_SharedThreshold_*{suffix}/predictive_metrics.json"
    for metrics_path in stan_root.glob(pattern):
        size = _extract_size_from_name(metrics_path.parent.name + "_")
        if size is not None:
            sizes.add(size)
    return sorted(sizes)


def _discover_marformer_sizes(marformer_root: Path, suffix: str) -> list[int]:
    sizes: set[int] = set()
    for run_dir in marformer_root.glob(f"Tensor_400_25_9_ItemTest_SharedThreshold_*_{suffix}"):
        size = _extract_size_from_name(run_dir.name + "_")
        if size is not None:
            sizes.add(size)
    return sorted(sizes)


def _load_stan_log_loss(stan_root: Path, size: int, dataset_tag: str = "", old_tensor: bool = False) -> float | None:
    run_suffix = f"{dataset_tag}_OLDTENSOR_eval" if (dataset_tag and old_tensor) else (
        "OLDTENSOR_eval" if old_tensor else (f"{dataset_tag}_eval" if dataset_tag else "eval")
    )
    metrics_path = (
        stan_root
        / f"Tensor_400_25_9_ItemTest_SharedThreshold_{size}_{run_suffix}"
        / "predictive_metrics.json"
    )
    if not metrics_path.exists():
        return None
    data = _read_json(metrics_path)
    # Convert log-likelihood to log loss.
    return float(-data["rating_missing_log_likelihood"])


def _load_marformer_test_log_loss(run_dir: Path) -> float | None:
    test_dir = run_dir / "TEST_RESULTS"
    if not test_dir.exists():
        return None

    # Prefer best*.json if present.
    best_candidates = sorted(test_dir.glob("best*.json"))
    if best_candidates:
        best_data = _read_json(best_candidates[0])
        missing = best_data.get("missing", {})
        if "log_loss" in missing:
            return float(missing["log_loss"])
    return None


def _load_marformer_training_history_fallback(run_dir: Path) -> float | None:
    hist_path = run_dir / "training_history.json"
    if not hist_path.exists():
        return None

    history = _read_json(hist_path)
    if not history:
        return None

    # Use the last epoch. Prefer test/val/combined/train in that order.
    last = history[-1]
    for eval_key in ("test_eval", "val_eval", "combined_eval", "train_eval"):
        eval_block = last.get(eval_key, {})
        value = (
            eval_block
            .get("metrics", {})
            .get("missing", {})
            .get("rating", {})
            .get("xent")
        )
        if value is not None:
            return float(value)
    return None


def _load_marformer_log_loss(marformer_root: Path, size: int, suffix: str) -> float | None:
    run_dir = marformer_root / f"Tensor_400_25_9_ItemTest_SharedThreshold_{size}_{suffix}"
    if not run_dir.exists():
        return None

    test_ll = _load_marformer_test_log_loss(run_dir)
    if test_ll is not None:
        return test_ll
    return _load_marformer_training_history_fallback(run_dir)


def _load_unigram_log_loss(size: int, dataset_tag: str = "") -> float | None:
    data_root = (
        Path("DATA/STAN/DOMAIN3-ITEM/Tensor_400_25_9_ItemTest_SharedThreshold")
        if not dataset_tag
        else Path(f"DATA/STAN/DOMAIN3-ITEM/Tensor_400_25_9_ItemTest_SharedThreshold_{dataset_tag}")
    )
    run_name = (
        f"Tensor_400_25_9_ItemTest_SharedThreshold_{size}"
        if not dataset_tag
        else f"Tensor_400_25_9_ItemTest_SharedThreshold_{size}_{dataset_tag}"
    )
    bundle_path = data_root / run_name / "data_bundle.json"
    if not bundle_path.exists():
        return None

    bundle = _read_json(bundle_path)
    observed = bundle.get("observed_ratings", [])
    test_missing = [r for r in bundle.get("missing_ratings", []) if r.get("instance") == "test"]
    if not observed or not test_missing:
        return None

    # Build a global unigram class distribution from all observed ratings.
    max_class = 0
    for row in observed + test_missing:
        if "value" in row:
            max_class = max(max_class, int(row["value"]))
    if max_class <= 0:
        return None

    counts = [0.0] * max_class
    for row in observed:
        dist = row.get("rating_dist")
        if dist is not None:
            for i, p in enumerate(dist):
                if i < max_class:
                    counts[i] += float(p)
        else:
            v = int(row["value"]) - 1
            if 0 <= v < max_class:
                counts[v] += 1.0

    total = sum(counts)
    if total <= 0:
        return None
    # Add-one smoothing for consistency with pooled unigram baselines.
    probs = [(c + 1.0) / (total + max_class) for c in counts]

    xent = 0.0
    for row in test_missing:
        idx = int(row["value"]) - 1
        if idx < 0 or idx >= len(probs):
            return None
        xent -= math.log(probs[idx] + 1e-12)
    return xent / len(test_missing)


def _load_unigram_pooled_log_loss(size: int, dataset_tag: str = "", pool_by: str = "ik") -> float | None:
    """
    Pooled unigram baselines with add-one smoothing.
      - pool_by in {"i","j","k","ij","ik","jk"}
    """
    data_root = (
        Path("DATA/STAN/DOMAIN3-ITEM/Tensor_400_25_9_ItemTest_SharedThreshold")
        if not dataset_tag
        else Path(f"DATA/STAN/DOMAIN3-ITEM/Tensor_400_25_9_ItemTest_SharedThreshold_{dataset_tag}")
    )
    run_name = (
        f"Tensor_400_25_9_ItemTest_SharedThreshold_{size}"
        if not dataset_tag
        else f"Tensor_400_25_9_ItemTest_SharedThreshold_{size}_{dataset_tag}"
    )
    bundle_path = data_root / run_name / "data_bundle.json"
    if not bundle_path.exists():
        return None

    bundle = _read_json(bundle_path)
    observed = bundle.get("observed_ratings", [])
    test_missing = [r for r in bundle.get("missing_ratings", []) if r.get("instance") == "test"]
    if not observed or not test_missing:
        return None

    max_class = 0
    for row in observed + test_missing:
        if "value" in row:
            max_class = max(max_class, int(row["value"]))
    if max_class <= 0:
        return None

    # Build pooled class counts from observed ratings.
    pool_counts: dict[tuple, list[float]] = {}

    def _pool_key(row: dict):
        i = int(row["attribute"])
        j = int(row["annotator"])
        k = int(row["item"])
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
        raise ValueError(f"Unsupported pool_by={pool_by}")

    for row in observed:
        key = _pool_key(row)
        if key not in pool_counts:
            pool_counts[key] = [0.0] * max_class
        dist = row.get("rating_dist")
        if dist is not None:
            for i, p in enumerate(dist):
                if i < max_class:
                    pool_counts[key][i] += float(p)
        else:
            idx = int(row["value"]) - 1
            if 0 <= idx < max_class:
                pool_counts[key][idx] += 1.0

    # Evaluate with add-one smoothing in each pool.
    xent = 0.0
    for row in test_missing:
        key = _pool_key(row)
        counts = pool_counts.get(key, [0.0] * max_class)
        smoothed_total = sum(counts) + max_class  # add-one per class
        idx = int(row["value"]) - 1
        if idx < 0 or idx >= max_class:
            return None
        prob = (counts[idx] + 1.0) / smoothed_total
        xent -= math.log(prob + 1e-12)
    return xent / len(test_missing)


def _load_naive_bayes_ijk_log_loss(size: int, dataset_tag: str = "", transductive: bool = True) -> float | None:
    """
    Naive Bayes baseline with categorical features i, j, k:
      P(y | i,j,k) ∝ P(y) P(i|y) P(j|y) P(k|y)
    Uses add-one smoothing for all categorical distributions.
    """
    data_root = (
        Path("DATA/STAN/DOMAIN3-ITEM/Tensor_400_25_9_ItemTest_SharedThreshold")
        if not dataset_tag
        else Path(f"DATA/STAN/DOMAIN3-ITEM/Tensor_400_25_9_ItemTest_SharedThreshold_{dataset_tag}")
    )
    run_name = (
        f"Tensor_400_25_9_ItemTest_SharedThreshold_{size}"
        if not dataset_tag
        else f"Tensor_400_25_9_ItemTest_SharedThreshold_{size}_{dataset_tag}"
    )
    bundle_path = data_root / run_name / "data_bundle.json"
    if not bundle_path.exists():
        return None

    bundle = _read_json(bundle_path)
    observed_all = bundle.get("observed_ratings", [])
    if transductive:
        # Transductive NB uses observed ratings from train/val/test.
        observed = [r for r in observed_all if r.get("instance") in {"train", "val", "test"}]
    else:
        # Non-transductive NB excludes test-observed ratings from the fit set.
        observed = [r for r in observed_all if r.get("instance") in {"train", "val"}]
    test_missing = [r for r in bundle.get("missing_ratings", []) if r.get("instance") == "test"]
    if not observed or not test_missing:
        return None

    max_class = 0
    max_i = 0
    max_j = 0
    max_k = 0
    for row in observed + test_missing:
        max_class = max(max_class, int(row.get("value", 0)))
        max_i = max(max_i, int(row.get("attribute", 0)))
        max_j = max(max_j, int(row.get("annotator", 0)))
        max_k = max(max_k, int(row.get("item", 0)))
    if min(max_class, max_i, max_j, max_k) <= 0:
        return None

    c = max_class
    class_counts = [0.0] * c
    i_counts = [[0.0] * max_i for _ in range(c)]
    j_counts = [[0.0] * max_j for _ in range(c)]
    k_counts = [[0.0] * max_k for _ in range(c)]

    # Fit from observed ratings.
    for row in observed:
        y = int(row["value"]) - 1
        i_idx = int(row["attribute"]) - 1
        j_idx = int(row["annotator"]) - 1
        k_idx = int(row["item"]) - 1
        if not (0 <= y < c and 0 <= i_idx < max_i and 0 <= j_idx < max_j and 0 <= k_idx < max_k):
            continue
        class_counts[y] += 1.0
        i_counts[y][i_idx] += 1.0
        j_counts[y][j_idx] += 1.0
        k_counts[y][k_idx] += 1.0

    n_obs = sum(class_counts)
    if n_obs <= 0:
        return None

    # Add-one smoothed log-probabilities.
    log_py = [
        math.log((class_counts[y] + 1.0) / (n_obs + c))
        for y in range(c)
    ]
    log_pi_given_y = []
    log_pj_given_y = []
    log_pk_given_y = []
    for y in range(c):
        denom_i = class_counts[y] + max_i
        denom_j = class_counts[y] + max_j
        denom_k = class_counts[y] + max_k
        log_pi_given_y.append([math.log((cnt + 1.0) / denom_i) for cnt in i_counts[y]])
        log_pj_given_y.append([math.log((cnt + 1.0) / denom_j) for cnt in j_counts[y]])
        log_pk_given_y.append([math.log((cnt + 1.0) / denom_k) for cnt in k_counts[y]])

    # Evaluate test-missing cross-entropy.
    xent = 0.0
    for row in test_missing:
        y_true = int(row["value"]) - 1
        i_idx = int(row["attribute"]) - 1
        j_idx = int(row["annotator"]) - 1
        k_idx = int(row["item"]) - 1
        if not (0 <= y_true < c and 0 <= i_idx < max_i and 0 <= j_idx < max_j and 0 <= k_idx < max_k):
            return None

        log_scores = [
            log_py[y]
            + log_pi_given_y[y][i_idx]
            + log_pj_given_y[y][j_idx]
            + log_pk_given_y[y][k_idx]
            for y in range(c)
        ]
        m = max(log_scores)
        log_norm = m + math.log(sum(math.exp(s - m) for s in log_scores))
        log_p_true = log_scores[y_true] - log_norm
        xent -= log_p_true

    return xent / len(test_missing)


def _collect_series(sizes: list[int], loader: Callable[[int], float | None]) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for size in sizes:
        value = loader(size)
        if value is None:
            continue
        xs.append(size)
        ys.append(float(value))
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stan-root",
        type=Path,
        default=Path("RESULTS/STAN/SPARSE"),
        help="Root containing STAN SharedThreshold *_eval directories.",
    )
    parser.add_argument(
        "--marformer-root",
        type=Path,
        default=Path("RESULTS/MARFORMER/STAN/SPARSE"),
        help="Root containing MARFORMER SharedThreshold run directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("PLOTS/TALK/Item/shared_threshold_test_missing_log_loss.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dataset-tag",
        type=str,
        default="",
        help="Optional run-name tag inserted before eval/method suffixes (e.g., C4).",
    )
    parser.add_argument(
        "--include-stan-old",
        action="store_true",
        help="Include old-tensor STAN curve if available.",
    )
    parser.add_argument(
        "--nb-nontransductive",
        action="store_true",
        help="Use train+val observed only for Naive Bayes fit (default uses transductive train+val+test observed).",
    )
    args = parser.parse_args()

    tag = f"{args.dataset_tag}_" if args.dataset_tag else ""
    trans_suffix = f"{tag}NOITEMDEV_TRANS_MARFORMER"
    nontrans_suffix = f"{tag}NOITEMDEV_NONTRANS_MARFORMER"

    sizes = sorted(
        set(_discover_stan_sizes(args.stan_root, dataset_tag=args.dataset_tag, old_tensor=False))
        | (
            set(_discover_stan_sizes(args.stan_root, dataset_tag=args.dataset_tag, old_tensor=True))
            if args.include_stan_old
            else set()
        )
        | set(_discover_marformer_sizes(args.marformer_root, trans_suffix))
        | set(_discover_marformer_sizes(args.marformer_root, nontrans_suffix))
    )
    if not sizes:
        raise SystemExit("No matching SharedThreshold result directories found.")

    stan_x, stan_y = _collect_series(
        sizes,
        lambda s: _load_stan_log_loss(args.stan_root, s, dataset_tag=args.dataset_tag, old_tensor=False),
    )
    stan_old_x, stan_old_y = _collect_series(
        sizes,
        lambda s: _load_stan_log_loss(args.stan_root, s, dataset_tag=args.dataset_tag, old_tensor=True),
    ) if args.include_stan_old else ([], [])
    mt_x, mt_y = _collect_series(
        sizes,
        lambda s: _load_marformer_log_loss(args.marformer_root, s, trans_suffix),
    )
    mnt_x, mnt_y = _collect_series(
        sizes,
        lambda s: _load_marformer_log_loss(args.marformer_root, s, nontrans_suffix),
    )
    subset_specs = [
        ("i", "Unigram (Pool i)", "#2e8b57", "x"),
        ("j", "Unigram (Pool j)", "#3b5bdb", "P"),
        ("k", "Unigram (Pool k)", "#e67700", "v"),
        ("ij", "Unigram (Pool i,j)", "#0b7285", ">"),
        ("ik", "Unigram (Pool i,k)", "#5f3dc4", "<"),
        ("jk", "Unigram (Pool j,k)", "#c2255c", "^"),
    ]
    subset_series: list[tuple[str, str, str, str, list[int], list[float]]] = []
    for key, label, color, marker in subset_specs:
        xs, ys = _collect_series(
            sizes,
            lambda s, kk=key: _load_unigram_pooled_log_loss(s, dataset_tag=args.dataset_tag, pool_by=kk),
        )
        subset_series.append((key, label, color, marker, xs, ys))
    nb_x, nb_y = _collect_series(
        sizes,
        lambda s: _load_naive_bayes_ijk_log_loss(
            s,
            dataset_tag=args.dataset_tag,
            transductive=not args.nb_nontransductive,
        ),
    )

    plt.figure(figsize=(9.5, 5.8))
    if mt_x:
        plt.plot(mt_x, mt_y, marker="o", color="#1f6fba", label="MARFORMER (Transductive)")
    if mnt_x:
        plt.plot(mnt_x, mnt_y, marker="o", linestyle="--", color="#1f6fba", label="MARFORMER (Non-Transductive)")
    if stan_x:
        plt.plot(stan_x, stan_y, marker="s", color="#d55e00", label="STAN (True Model)")
    if args.include_stan_old and stan_old_x:
        plt.plot(
            stan_old_x,
            stan_old_y,
            marker="^",
            linestyle="--",
            color="#8e44ad",
            label="STAN (Old Tensor Model)",
        )
    for _, label, color, marker, xs, ys in subset_series:
        if xs:
            plt.plot(
                xs,
                ys,
                marker=marker,
                linestyle=":",
                color=color,
                label=label,
            )
    if nb_x:
        nb_label = "Naive Bayes (i,j,k, T)" if not args.nb_nontransductive else "Naive Bayes (i,j,k, NT)"
        plt.plot(
            nb_x,
            nb_y,
            marker="*",
            linestyle="-.",
            color="#111111",
            label=nb_label,
        )

    plt.xlabel("Training Size")
    plt.ylabel("Test Missing Log Loss")
    title_tag = f" ({args.dataset_tag})" if args.dataset_tag else ""
    plt.title(f"SharedThreshold{title_tag}: Test Missing Log Loss by Training Size")
    plt.xticks(sizes)
    plt.grid(alpha=0.25)
    plt.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

    print(f"Saved plot to: {args.output}")
    print(f"Sizes considered: {sizes}")
    if args.include_stan_old:
        print(
            f"STAN points: {len(stan_x)} | STAN old-tensor points: {len(stan_old_x)} | "
            f"MARFORMER trans points: {len(mt_x)} | MARFORMER non-trans points: {len(mnt_x)} | "
            f"Unigram subset points: "
            + ", ".join(f"{k}:{len(xs)}" for k, _, _, _, xs, _ in subset_series)
            + f" | NB points: {len(nb_x)}"
        )
    else:
        print(
            f"STAN points: {len(stan_x)} | "
            f"MARFORMER trans points: {len(mt_x)} | MARFORMER non-trans points: {len(mnt_x)} | "
            f"Unigram subset points: "
            + ", ".join(f"{k}:{len(xs)}" for k, _, _, _, xs, _ in subset_series)
            + f" | NB points: {len(nb_x)}"
        )


if __name__ == "__main__":
    main()
