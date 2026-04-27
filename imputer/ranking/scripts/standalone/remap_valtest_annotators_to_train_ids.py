#!/usr/bin/env python3
"""
For joint item+annotator split bundles, remap validation/test *annotator* ids into
1..J_train while keeping rating rows in the same train/val/test *instance*.

This removes annotator cold-start on val/test for embedding models: every
annotator index that appears under instance \"val\" or \"test\" is a training
annotator id. Item cold-start (val/test items) is unchanged.

Requires injective maps from val and test annotators into disjoint training ids:
  J_train_split >= J_val_split + J_test_split

Also permutes rows of base_scores / rating_probs / rating_cumprobs /
rating_thresholds_z (layouts [I*J, ...]) consistently.

Usage:
  python scripts/standalone/remap_valtest_annotators_to_train_ids.py \\
    --input-dir DATA/STAN/SPARSE/Tensor_400_25_9_ItemAnnotTest/Tensor_400_25_9_ItemAnnotTest_200_15 \\
    --output-dir DATA/STAN/SPARSE/Tensor_400_25_9_ItemAnnotTest/Tensor_400_25_9_ItemAnnotTest_200_15_annremap
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def _annotator_perm_local(
    *,
    j_train: int,
    j_val: int,
    j_test: int,
    val_map: Dict[int, int],
    test_map: Dict[int, int],
) -> np.ndarray:
    """Permutation p of length J such that new_mat[:, k] = old_mat[:, p[k]]."""
    j_total = j_train + j_val + j_test
    val_ids = list(range(j_train + 1, j_train + j_val + 1))
    test_ids = list(range(j_train + j_val + 1, j_total + 1))

    image = set(val_map.values()) | set(test_map.values())
    if len(image) != j_val + j_test:
        raise ValueError("val/test annotator maps must have disjoint image ids in 1..J_train")
    if any(t < 1 or t > j_train for t in image):
        raise ValueError("Mapped annotator ids must lie in 1..J_train")

    inv: Dict[int, int] = {}
    for old, new in val_map.items():
        inv[int(new)] = int(old)
    for old, new in test_map.items():
        if int(new) in inv:
            raise ValueError("Duplicate inverse for mapped train annotator slot")
        inv[int(new)] = int(old)

    f = np.empty(j_total, dtype=int)
    for t in range(1, j_train + 1):
        if t in image:
            f[t - 1] = inv[t] - 1
        else:
            f[t - 1] = t - 1
    for old in val_ids:
        f[old - 1] = val_map[old] - 1
    for old in test_ids:
        f[old - 1] = test_map[old] - 1

    if f.min() < 0 or f.max() >= j_total:
        raise RuntimeError("Annotator permutation out of range")
    if np.unique(f).size != j_total:
        raise RuntimeError("Annotator forward map is not bijective")

    return f


def _global_row_perm(i_dim: int, p_local: np.ndarray) -> np.ndarray:
    j_total = int(p_local.shape[0])
    out = np.empty(i_dim * j_total, dtype=int)
    for a in range(i_dim):
        base = a * j_total
        out[base : base + j_total] = base + p_local
    return out


def _remap_rating_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    val_map: Dict[int, int],
    test_map: Dict[int, int],
    j_train: int,
    j_val: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        ann = int(row["annotator"])
        new_row = dict(row)
        if j_train < ann <= j_train + j_val:
            new_row["annotator"] = val_map[ann]
        elif ann > j_train + j_val:
            new_row["annotator"] = test_map[ann]
        out.append(new_row)
    return out


def _remap_pairwise_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    val_map: Dict[int, int],
    test_map: Dict[int, int],
    j_train: int,
    j_val: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        ann = int(row["annotator"])
        new_row = dict(row)
        if j_train < ann <= j_train + j_val:
            new_row["annotator"] = val_map[ann]
        elif ann > j_train + j_val:
            new_row["annotator"] = test_map[ann]
        out.append(new_row)
    return out


def _permute_ij_rows(arr: np.ndarray, row_perm: np.ndarray) -> np.ndarray:
    if arr.shape[0] != row_perm.shape[0]:
        raise ValueError(f"Row count {arr.shape[0]} != perm length {row_perm.shape[0]}")
    return arr[row_perm, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    bundle = load_json(in_dir / "data_bundle.json")
    cfg = load_json(in_dir / "configs.json")
    dg = cfg.get("datagen", cfg)

    stats = bundle.get("stats", {})
    j_train = int(stats.get("J_train_split", dg.get("J_train_split")))
    j_val = int(stats.get("J_val_split", dg.get("J_val_split")))
    j_test = int(stats.get("J_test_split", dg.get("J_test_split")))
    i_dim = int(dg["I"])

    if j_train < j_val + j_test:
        raise ValueError(
            f"Need J_train_split >= J_val_split + J_test_split for disjoint remap targets; "
            f"got J_train={j_train}, J_val={j_val}, J_test={j_test}"
        )

    rng = np.random.default_rng(args.seed)
    train_pool = np.arange(1, j_train + 1)

    val_ids = list(range(j_train + 1, j_train + j_val + 1))
    test_ids = list(range(j_train + j_val + 1, j_train + j_val + j_test + 1))

    val_targets = rng.choice(train_pool, size=len(val_ids), replace=False)
    remaining = np.setdiff1d(train_pool, val_targets, assume_unique=False)
    if len(remaining) < len(test_ids):
        raise RuntimeError("Not enough unused train annotator slots for test remap")
    test_targets = rng.choice(remaining, size=len(test_ids), replace=False)

    val_map = {old: int(t) for old, t in zip(val_ids, val_targets)}
    test_map = {old: int(t) for old, t in zip(test_ids, test_targets)}

    p_local = _annotator_perm_local(
        j_train=j_train, j_val=j_val, j_test=j_test, val_map=val_map, test_map=test_map
    )
    row_perm = _global_row_perm(i_dim, p_local)

    new_bundle: Dict[str, Any] = dict(bundle)
    for key in (
        "all_ratings",
        "observed_ratings",
        "missing_ratings",
    ):
        if key in new_bundle:
            new_bundle[key] = _remap_rating_rows(
                new_bundle[key], val_map=val_map, test_map=test_map, j_train=j_train, j_val=j_val
            )

    for key in ("all_pairwise", "observed_pairwise", "missing_pairwise"):
        if key in new_bundle and new_bundle[key]:
            new_bundle[key] = _remap_pairwise_rows(
                new_bundle[key], val_map=val_map, test_map=test_map, j_train=j_train, j_val=j_val
            )

    for key in ("base_scores", "rating_probs", "rating_cumprobs", "rating_thresholds_z"):
        if key in new_bundle and new_bundle[key] is not None:
            arr = np.asarray(new_bundle[key], dtype=float)
            new_bundle[key] = _permute_ij_rows(arr, row_perm).tolist()

    new_stats = dict(stats)
    new_stats["valtest_annotator_remap"] = True
    new_stats["valtest_annotator_remap_seed"] = args.seed
    new_bundle["stats"] = new_stats

    new_cfg = dict(cfg)
    new_dg = dict(dg)
    new_dg["valtest_annotator_remap"] = True
    new_dg["valtest_annotator_remap_seed"] = args.seed
    new_cfg["datagen"] = new_dg

    save_json(new_bundle, out_dir / "data_bundle.json")
    save_json(new_cfg, out_dir / "configs.json")
    save_json(
        {
            "J_train_split": j_train,
            "J_val_split": j_val,
            "J_test_split": j_test,
            "seed": args.seed,
            "val_map": {str(k): v for k, v in val_map.items()},
            "test_map": {str(k): v for k, v in test_map.items()},
        },
        out_dir / "annotator_remap_meta.json",
    )

    stan_path = in_dir / "stan_data.json"
    if stan_path.exists():
        shutil.copy2(stan_path, out_dir / "stan_data.json")

    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
