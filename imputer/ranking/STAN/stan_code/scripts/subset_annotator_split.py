#!/usr/bin/env python3
"""
Subset an annotator-split generated data directory by J_train.

Given a directory produced by generate_data_annotator_split.py (train/val/test
split by annotator group, shared items), creates a new directory containing only
the first `train_num` training annotators while keeping all val and test data
unchanged.

Items are shared (IDs 1..K), so no item remapping is needed.
Val and test annotator IDs are shifted down by (J_train_orig - train_num) to
remain contiguous after train annotators shrink:

  original train annotators: 1..J_train
  original val annotators:   J_train+1..J_train+J_val
  original test annotators:  J_train+J_val+1..J

  subset train annotators: 1..train_num
  subset val annotators:   train_num+1..train_num+J_val   (shifted)
  subset test annotators:  train_num+J_val+1..train_num+J_val+J_test

Usage:
    python subset_annotator_split.py \\
        --input-dir DATA/STAN/annotator_run_001 \\
        --output-dir DATA/STAN/annotator_run_001_subset \\
        --train-num 10
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def remap_annotator_id(annotator_id: int, J_train_orig: int, train_num: int) -> int:
    """
    Remap a val/test annotator ID after shrinking train from J_train_orig to train_num.
    Val/test annotators shift down by (J_train_orig - train_num).
    """
    shift = J_train_orig - train_num
    return annotator_id - shift


def subset_ratings(
    ratings: List[Dict],
    J_train_orig: int,
    train_num: int,
) -> List[Dict]:
    """
    Keep train ratings for annotators 1..train_num.
    Remap val/test annotator IDs to stay contiguous.
    Item IDs are shared and unchanged.
    """
    out = []
    for r in ratings:
        ann = r["annotator"]
        instance = r["instance"]
        if instance == "train":
            if ann > train_num:
                continue  # drop this training annotator's ratings
            out.append(dict(r))
        else:
            # val or test: remap annotator ID
            new_r = dict(r)
            new_r["annotator"] = remap_annotator_id(ann, J_train_orig, train_num)
            out.append(new_r)
    return out


def subset_numpy_annotator_rows(
    arr: np.ndarray,
    I: int,
    J_train_orig: int,
    J_val: int,
    J_test: int,
    train_num: int,
) -> np.ndarray:
    """
    Subset an array with shape [I*J_total, ...] along the first axis.

    The first axis is laid out as I blocks of J_total annotators:
      block i: rows i*J_total .. i*J_total + J_total - 1
        within each block:
          train:  0..J_train_orig-1
          val:    J_train_orig..J_train_orig+J_val-1
          test:   J_train_orig+J_val..J_total-1

    After subsetting, the layout becomes I blocks of (train_num+J_val+J_test).
    """
    J_total_orig = J_train_orig + J_val + J_test
    J_total_new  = train_num + J_val + J_test
    n_rows_new   = I * J_total_new
    rest_shape   = arr.shape[1:]  # everything after the first axis

    out = np.empty((n_rows_new,) + rest_shape, dtype=arr.dtype)
    for i in range(I):
        block_start = i * J_total_orig
        new_block_start = i * J_total_new

        # train annotators: first train_num rows of this block
        out[new_block_start : new_block_start + train_num] = \
            arr[block_start : block_start + train_num]

        # val annotators
        val_src_start = block_start + J_train_orig
        val_dst_start = new_block_start + train_num
        out[val_dst_start : val_dst_start + J_val] = \
            arr[val_src_start : val_src_start + J_val]

        # test annotators
        test_src_start = block_start + J_train_orig + J_val
        test_dst_start = new_block_start + train_num + J_val
        out[test_dst_start : test_dst_start + J_test] = \
            arr[test_src_start : test_src_start + J_test]

    return out


def subset_posterior_probs(
    probs: np.ndarray,
    I: int,
    J_train_orig: int,
    train_num: int,
) -> np.ndarray:
    """
    train_posterior_rating_probs shape: [I*J_train_orig, K, C]
    Subset to [I*train_num, K, C] by keeping the first train_num annotator
    rows within each attribute block.
    """
    K = probs.shape[1]
    C = probs.shape[2]
    out = np.empty((I * train_num, K, C), dtype=probs.dtype)
    for i in range(I):
        src_start = i * J_train_orig
        dst_start = i * train_num
        out[dst_start : dst_start + train_num] = probs[src_start : src_start + train_num]
    return out


def recompute_missing_indexes_in_test(missing_ratings: List[Dict]) -> List[int]:
    return [i for i, r in enumerate(missing_ratings) if r["instance"] == "test"]


def recompute_stats(
    orig_stats: Dict,
    J_train_orig: int,
    train_num: int,
    all_ratings: List[Dict],
    observed_ratings: List[Dict],
    missing_ratings: List[Dict],
) -> Dict:
    train_r  = [r for r in all_ratings if r["instance"] == "train"]
    val_r    = [r for r in all_ratings if r["instance"] == "val"]
    test_r   = [r for r in all_ratings if r["instance"] == "test"]
    train_ob = [r for r in observed_ratings if r["instance"] == "train"]
    val_ob   = [r for r in observed_ratings if r["instance"] == "val"]
    test_ob  = [r for r in observed_ratings if r["instance"] == "test"]

    J_val  = orig_stats.get("J_val", 0)
    J_test = orig_stats.get("J_test", 0)
    K      = orig_stats.get("K", orig_stats.get("total_items", 0))
    I      = orig_stats.get("I", 0)

    J_total_new = train_num + J_val + J_test

    return {
        **orig_stats,
        "J_train": train_num,
        "J": J_total_new,
        "total_possible_ratings": I * J_total_new * K,
        "total_ratings": len(all_ratings),
        "observed_ratings": len(observed_ratings),
        "missing_ratings": len(missing_ratings),
        "train_ratings": len(train_r),
        "val_ratings": len(val_r),
        "test_ratings": len(test_r),
        "train_observed": len(train_ob),
        "val_observed": len(val_ob),
        "test_observed": len(test_ob),
        "total_pairwise": 0,
        "observed_pairwise": 0,
        "missing_pairwise": 0,
        "observation_rate": len(observed_ratings) / len(all_ratings) if all_ratings else 0,
        "train_observation_rate": len(train_ob) / len(train_r) if train_r else 0,
        "val_observation_rate":   len(val_ob) / len(val_r) if val_r else 0,
        "test_observation_rate":  len(test_ob) / len(test_r) if test_r else 0,
        "subset_train_num": train_num,
        "subset_J_train_orig": J_train_orig,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Subset an annotator-split data directory by J_train."
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to the original generated data run directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to write the subsetted data directory")
    parser.add_argument("--train-num", type=int, required=True,
                        help="Number of training annotators to keep (must be <= J_train)")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    train_num  = args.train_num

    # ===== Load =====
    bundle_path = input_dir / "data_bundle.json"
    config_path = input_dir / "configs.json"

    print(f"Loading bundle from {bundle_path}")
    bundle = load_json(bundle_path)

    orig_stats   = bundle["stats"]
    J_train_orig = orig_stats["J_train"]
    J_val        = orig_stats["J_val"]
    J_test       = orig_stats["J_test"]
    K            = orig_stats.get("K", orig_stats.get("total_items", 0))

    # Load I from configs if not in stats
    I = orig_stats.get("I", None)
    if I is None and config_path.exists():
        cfg = load_json(config_path)
        I = cfg.get("datagen", cfg).get("I")
    if I is None:
        raise ValueError("Could not determine I from stats or configs.json")

    if train_num > J_train_orig:
        raise ValueError(f"--train-num {train_num} exceeds J_train={J_train_orig}")
    if train_num < 1:
        raise ValueError("--train-num must be >= 1")

    print(f"Original J_train={J_train_orig}, J_val={J_val}, J_test={J_test}, K={K}, I={I}")
    print(f"Subsetting to J_train={train_num}")

    # ===== Subset ratings =====
    all_ratings      = subset_ratings(bundle["all_ratings"],      J_train_orig, train_num)
    observed_ratings = subset_ratings(bundle["observed_ratings"], J_train_orig, train_num)
    missing_ratings  = subset_ratings(bundle["missing_ratings"],  J_train_orig, train_num)

    missing_indexes_in_test = recompute_missing_indexes_in_test(missing_ratings)

    # ===== Rebuild stats =====
    stats = recompute_stats(
        orig_stats, J_train_orig, train_num,
        all_ratings, observed_ratings, missing_ratings,
    )

    # ===== Build output bundle dict =====
    out_bundle: Dict[str, Any] = {
        "all_ratings": all_ratings,
        "all_pairwise": [],
        "observed_ratings": observed_ratings,
        "missing_ratings": missing_ratings,
        "missing_ratings_indexes_in_test_instance": missing_indexes_in_test,
        "observed_pairwise": [],
        "missing_pairwise": [],
        "stats": stats,
    }

    # ===== Subset numpy arrays =====
    J_total_orig = J_train_orig + J_val + J_test

    # Arrays shaped [I*J_total, ...]: annotator_preferences, rating_probs, rating_cumprobs,
    # rating_thresholds_z, base_scores
    for key in ("annotator_preferences", "rating_probs", "rating_cumprobs",
                "rating_thresholds_z", "base_scores"):
        if key in bundle and bundle[key] is not None:
            arr = np.array(bundle[key])
            out_bundle[key] = subset_numpy_annotator_rows(
                arr, I, J_train_orig, J_val, J_test, train_num
            ).tolist()

    # mean_preferences shape [I, D]: unchanged
    if "mean_preferences" in bundle and bundle["mean_preferences"] is not None:
        out_bundle["mean_preferences"] = bundle["mean_preferences"]

    # embeddings shape [K, D]: shared items, unchanged
    if "embeddings" in bundle and bundle["embeddings"] is not None:
        out_bundle["embeddings"] = bundle["embeddings"]

    # train_posterior_rating_probs shape [I*J_train_orig, K, C]: subset annotator axis
    if "train_posterior_rating_probs" in bundle and bundle["train_posterior_rating_probs"] is not None:
        probs = np.array(bundle["train_posterior_rating_probs"])
        out_bundle["train_posterior_rating_probs"] = subset_posterior_probs(
            probs, I, J_train_orig, train_num
        ).tolist()

    # val/test posterior probs unchanged
    for key in ("val_posterior_rating_probs", "test_posterior_rating_probs"):
        if key in bundle and bundle[key] is not None:
            out_bundle[key] = bundle[key]

    # extra ground truth: pass through unchanged
    standard_keys = {
        "all_ratings", "all_pairwise", "observed_ratings", "missing_ratings",
        "observed_pairwise", "missing_pairwise", "missing_ratings_indexes_in_test_instance",
        "stats", "embeddings", "base_scores", "mean_preferences", "annotator_preferences",
        "rating_probs", "rating_cumprobs", "rating_thresholds_z",
        "train_posterior_rating_probs", "val_posterior_rating_probs", "test_posterior_rating_probs",
    }
    for key, value in bundle.items():
        if key not in standard_keys and value is not None:
            out_bundle[key] = value

    # ===== Write output =====
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing subsetted bundle to {output_dir}")

    save_json(out_bundle, output_dir / "data_bundle.json")

    # Copy configs.json, adjusting J_train and J
    if config_path.exists():
        cfg = load_json(config_path)
        section = cfg.get("datagen", cfg)
        section["J_train"] = train_num
        section["J"] = train_num + J_val + J_test
        section["subset_J_train_orig"] = J_train_orig
        save_json(cfg, output_dir / "configs.json")

    # Copy stan_data.json if present, adjusting J_train and J
    stan_data_path = input_dir / "stan_data.json"
    if stan_data_path.exists():
        stan_data = load_json(stan_data_path)
        stan_data["J_train"] = train_num
        stan_data["J"] = train_num + J_val + J_test
        save_json(stan_data, output_dir / "stan_data.json")

    print(f"\nDone.")
    print(f"  Original train ratings: {orig_stats['train_ratings']}")
    print(f"  Subset   train ratings: {stats['train_ratings']}")
    print(f"  Val ratings (unchanged): {stats['val_ratings']}")
    print(f"  Test ratings (unchanged): {stats['test_ratings']}")
    print(f"  Val annotator IDs now start at: {train_num + 1}")
    print(f"  Test annotator IDs now start at: {train_num + J_val + 1}")


if __name__ == "__main__":
    main()