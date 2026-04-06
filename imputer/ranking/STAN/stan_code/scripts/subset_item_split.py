#!/usr/bin/env python3
"""
Subset an item-split generated data directory by K_train.

Given a directory produced by generate_data.py (train/val/test split by items),
creates a new directory containing only the first `train_num` training items
while keeping all val and test data unchanged.

Val and test item IDs are shifted down to remain contiguous after train shrinks:
  original train: 1..K_train
  original val:   K_train+1..K_train+K_val
  original test:  K_train+K_val+1..K_train+K_val+K_test

  subset train: 1..train_num
  subset val:   train_num+1..train_num+K_val       (shifted down by K_train-train_num)
  subset test:  train_num+K_val+1..train_num+K_val+K_test

Usage:
    python subset_item_split.py \\
        --input-dir DATA/STAN/run_001 \\
        --output-dir DATA/STAN/run_001_subset \\
        --train-num 100
"""

import argparse
import json
import shutil
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


def remap_item_id(item_id: int, K_train_orig: int, train_num: int) -> int:
    """
    Remap a val/test item ID after shrinking train from K_train_orig to train_num.
    Train items (1..K_train_orig) are unchanged in value but filtered.
    Val/test items shift down by (K_train_orig - train_num).
    """
    shift = K_train_orig - train_num
    return item_id - shift


def subset_ratings(
    ratings: List[Dict],
    K_train_orig: int,
    train_num: int,
) -> List[Dict]:
    """Keep train ratings for items 1..train_num; remap val/test item IDs."""
    out = []
    for r in ratings:
        item = r["item"]
        instance = r["instance"]
        if instance == "train":
            if item > train_num:
                continue  # drop this training item
            out.append(dict(r))
        else:
            # val or test: remap item ID
            new_r = dict(r)
            new_r["item"] = remap_item_id(item, K_train_orig, train_num)
            out.append(new_r)
    return out


def subset_pairwise(
    pairwise: List[Dict],
    K_train_orig: int,
    train_num: int,
) -> List[Dict]:
    """Keep train pairwise for items within 1..train_num; remap val/test item IDs."""
    out = []
    for p in pairwise:
        instance = p["instance"]
        if instance == "train":
            if any(item > train_num for item in p["items"]):
                continue
            out.append(dict(p))
        else:
            new_p = dict(p)
            new_p["items"] = [remap_item_id(i, K_train_orig, train_num) for i in p["items"]]
            out.append(new_p)
    return out


def subset_numpy_embeddings(
    embeddings: np.ndarray,
    K_train_orig: int,
    K_val: int,
    K_test: int,
    train_num: int,
) -> np.ndarray:
    """
    embeddings shape: [K_train + K_val + K_test, D]
    Layout:  train rows 0..K_train_orig-1 | val rows | test rows
    Subset:  train rows 0..train_num-1    | val rows | test rows
    """
    train_emb = embeddings[:train_num]
    val_emb   = embeddings[K_train_orig : K_train_orig + K_val]
    test_emb  = embeddings[K_train_orig + K_val : K_train_orig + K_val + K_test]
    return np.vstack([train_emb, val_emb, test_emb])


def subset_numpy_base_scores(
    base_scores: np.ndarray,
    K_train_orig: int,
    K_val: int,
    K_test: int,
    train_num: int,
) -> np.ndarray:
    """
    base_scores shape: [I*J, K_train + K_val + K_test]
    Columns layout mirrors embeddings rows.
    """
    train_cols = base_scores[:, :train_num]
    val_cols   = base_scores[:, K_train_orig : K_train_orig + K_val]
    test_cols  = base_scores[:, K_train_orig + K_val : K_train_orig + K_val + K_test]
    return np.hstack([train_cols, val_cols, test_cols])


def subset_posterior_probs(
    probs: np.ndarray,
    K_train_orig: int,
    train_num: int,
) -> np.ndarray:
    """
    train_posterior_rating_probs shape: [I*J, K_train, C]
    Slice the K_train axis to train_num.
    """
    return probs[:, :train_num, :]


def recompute_missing_indexes_in_test(missing_ratings: List[Dict]) -> List[int]:
    return [i for i, r in enumerate(missing_ratings) if r["instance"] == "test"]


def recompute_stats(
    bundle: Dict,
    K_train_orig: int,
    train_num: int,
    all_ratings: List[Dict],
    observed_ratings: List[Dict],
    missing_ratings: List[Dict],
    all_pairwise: List[Dict],
    observed_pairwise: List[Dict],
    missing_pairwise: List[Dict],
) -> Dict:
    orig = bundle["stats"]
    K_val  = orig.get("K_val", 0)
    K_test = orig.get("K_test", 0)

    train_r  = [r for r in all_ratings if r["instance"] == "train"]
    val_r    = [r for r in all_ratings if r["instance"] == "val"]
    test_r   = [r for r in all_ratings if r["instance"] == "test"]
    train_ob = [r for r in observed_ratings if r["instance"] == "train"]
    val_ob   = [r for r in observed_ratings if r["instance"] == "val"]
    test_ob  = [r for r in observed_ratings if r["instance"] == "test"]

    total_items = train_num + K_val + K_test
    return {
        **orig,
        "K_train": train_num,
        "total_items": total_items,
        "total_possible_ratings": orig.get("total_possible_ratings", 0)
            * total_items // (K_train_orig + K_val + K_test)
            if (K_train_orig + K_val + K_test) > 0 else 0,
        "total_ratings": len(all_ratings),
        "observed_ratings": len(observed_ratings),
        "missing_ratings": len(missing_ratings),
        "train_ratings": len(train_r),
        "val_ratings": len(val_r),
        "test_ratings": len(test_r),
        "train_observed": len(train_ob),
        "val_observed": len(val_ob),
        "test_observed": len(test_ob),
        "total_pairwise": len(all_pairwise),
        "observed_pairwise": len(observed_pairwise),
        "missing_pairwise": len(missing_pairwise),
        "train_pairwise": len([p for p in all_pairwise if p["instance"] == "train"]),
        "val_pairwise":   len([p for p in all_pairwise if p["instance"] == "val"]),
        "test_pairwise":  len([p for p in all_pairwise if p["instance"] == "test"]),
        "observation_rate": len(observed_ratings) / len(all_ratings) if all_ratings else 0,
        "train_observation_rate": len(train_ob) / len(train_r) if train_r else 0,
        "val_observation_rate":   len(val_ob) / len(val_r) if val_r else 0,
        "test_observation_rate":  len(test_ob) / len(test_r) if test_r else 0,
        "subset_train_num": train_num,
        "subset_K_train_orig": K_train_orig,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Subset an item-split data directory by K_train."
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to the original generated data run directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to write the subsetted data directory")
    parser.add_argument("--train-num", type=int, required=True,
                        help="Number of training items to keep (must be <= K_train)")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    train_num  = args.train_num

    # ===== Load =====
    bundle_path = input_dir / "data_bundle.json"
    config_path = input_dir / "configs.json"

    print(f"Loading bundle from {bundle_path}")
    bundle = load_json(bundle_path)

    K_train_orig = bundle["stats"]["K_train"]
    K_val        = bundle["stats"].get("K_val", 0)
    K_test       = bundle["stats"].get("K_test", 0)

    if train_num > K_train_orig:
        raise ValueError(f"--train-num {train_num} exceeds K_train={K_train_orig}")
    if train_num < 1:
        raise ValueError(f"--train-num must be >= 1")

    print(f"Original K_train={K_train_orig}, K_val={K_val}, K_test={K_test}")
    print(f"Subsetting to K_train={train_num}")

    # ===== Subset ratings and pairwise =====
    all_ratings      = subset_ratings(bundle["all_ratings"],      K_train_orig, train_num)
    observed_ratings = subset_ratings(bundle["observed_ratings"], K_train_orig, train_num)
    missing_ratings  = subset_ratings(bundle["missing_ratings"],  K_train_orig, train_num)

    all_pairwise      = subset_pairwise(bundle.get("all_pairwise", []),      K_train_orig, train_num)
    observed_pairwise = subset_pairwise(bundle.get("observed_pairwise", []), K_train_orig, train_num)
    missing_pairwise  = subset_pairwise(bundle.get("missing_pairwise", []),  K_train_orig, train_num)

    missing_indexes_in_test = recompute_missing_indexes_in_test(missing_ratings)

    # ===== Rebuild stats =====
    stats = recompute_stats(
        bundle, K_train_orig, train_num,
        all_ratings, observed_ratings, missing_ratings,
        all_pairwise, observed_pairwise, missing_pairwise,
    )

    # ===== Build output bundle dict =====
    out_bundle: Dict[str, Any] = {
        "all_ratings": all_ratings,
        "all_pairwise": all_pairwise,
        "observed_ratings": observed_ratings,
        "missing_ratings": missing_ratings,
        "missing_ratings_indexes_in_test_instance": missing_indexes_in_test,
        "observed_pairwise": observed_pairwise,
        "missing_pairwise": missing_pairwise,
        "stats": stats,
    }

    # ===== Subset numpy arrays =====
    if "embeddings" in bundle and bundle["embeddings"] is not None:
        emb = np.array(bundle["embeddings"])
        out_bundle["embeddings"] = subset_numpy_embeddings(
            emb, K_train_orig, K_val, K_test, train_num
        ).tolist()

    if "base_scores" in bundle and bundle["base_scores"] is not None:
        bs = np.array(bundle["base_scores"])
        out_bundle["base_scores"] = subset_numpy_base_scores(
            bs, K_train_orig, K_val, K_test, train_num
        ).tolist()

    if "train_posterior_rating_probs" in bundle and bundle["train_posterior_rating_probs"] is not None:
        probs = np.array(bundle["train_posterior_rating_probs"])
        out_bundle["train_posterior_rating_probs"] = subset_posterior_probs(
            probs, K_train_orig, train_num
        ).tolist()

    # val/test posterior probs are unchanged
    for key in ("val_posterior_rating_probs", "test_posterior_rating_probs",
                "mean_preferences", "annotator_preferences",
                "rating_probs", "rating_cumprobs", "rating_thresholds_z"):
        if key in bundle and bundle[key] is not None:
            out_bundle[key] = bundle[key]

    # extra ground truth: pass through unchanged
    standard_keys = {
        "all_ratings", "all_pairwise", "observed_ratings", "missing_ratings",
        "observed_pairwise", "missing_pairwise", "missing_ratings_indexes_in_test_instance",
        "stats", "embeddings", "base_scores", "train_posterior_rating_probs",
        "val_posterior_rating_probs", "test_posterior_rating_probs",
        "mean_preferences", "annotator_preferences",
        "rating_probs", "rating_cumprobs", "rating_thresholds_z",
    }
    for key, value in bundle.items():
        if key not in standard_keys and value is not None:
            out_bundle[key] = value

    # ===== Write output =====
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing subsetted bundle to {output_dir}")

    save_json(out_bundle, output_dir / "data_bundle.json")

    # Copy configs.json, adjusting K_train
    if config_path.exists():
        cfg = load_json(config_path)
        if "datagen" in cfg:
            cfg["datagen"]["K_train"] = train_num
            cfg["datagen"]["subset_K_train_orig"] = K_train_orig
        save_json(cfg, output_dir / "configs.json")

    # Copy stan_data.json if present, adjusting K_train
    stan_data_path = input_dir / "stan_data.json"
    if stan_data_path.exists():
        stan_data = load_json(stan_data_path)
        stan_data["K_train"] = train_num
        save_json(stan_data, output_dir / "stan_data.json")

    print(f"\nDone.")
    print(f"  Original train ratings: {bundle['stats']['train_ratings']}")
    print(f"  Subset   train ratings: {stats['train_ratings']}")
    print(f"  Val ratings (unchanged): {stats['val_ratings']}")
    print(f"  Test ratings (unchanged): {stats['test_ratings']}")
    print(f"  Val item IDs now start at: {train_num + 1}")
    print(f"  Test item IDs now start at: {train_num + K_val + 1}")


if __name__ == "__main__":
    main()