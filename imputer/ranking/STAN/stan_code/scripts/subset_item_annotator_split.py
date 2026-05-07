#!/usr/bin/env python3
"""Subset a joint item+annotator split bundle by paired train sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def item_group(item_id: int, k_train: int, k_val: int) -> str:
    if item_id <= k_train:
        return "train"
    if item_id <= k_train + k_val:
        return "val"
    return "test"


def annotator_group(annotator_id: int, j_train: int, j_val: int) -> str:
    if annotator_id <= j_train:
        return "train"
    if annotator_id <= j_train + j_val:
        return "val"
    return "test"


def combine_instance(item_g: str, ann_g: str) -> str:
    if item_g == "test" or ann_g == "test":
        return "test"
    if item_g == "val" or ann_g == "val":
        return "val"
    return "train"


def remap_item_id(item_id: int, k_train_orig: int, train_num: int) -> int:
    return item_id - (k_train_orig - train_num)


def remap_annotator_id(annotator_id: int, j_train_orig: int, train_num: int) -> int:
    return annotator_id - (j_train_orig - train_num)


def subset_ratings(
    ratings: List[Dict[str, Any]],
    *,
    k_train_orig: int,
    k_val: int,
    train_items: int,
    j_train_orig: int,
    j_val: int,
    train_annotators: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in ratings:
        item = int(row["item"])
        ann = int(row["annotator"])
        item_g = item_group(item, k_train_orig, k_val)
        ann_g = annotator_group(ann, j_train_orig, j_val)

        if item_g == "train" and item > train_items:
            continue
        if ann_g == "train" and ann > train_annotators:
            continue

        new_row = dict(row)
        if item_g != "train":
            new_row["item"] = remap_item_id(item, k_train_orig, train_items)
        if ann_g != "train":
            new_row["annotator"] = remap_annotator_id(ann, j_train_orig, train_annotators)

        new_item_g = item_group(int(new_row["item"]), train_items, k_val)
        new_ann_g = annotator_group(int(new_row["annotator"]), train_annotators, j_val)
        new_row["instance"] = combine_instance(new_item_g, new_ann_g)
        out.append(new_row)
    return out


def subset_pairwise(
    pairwise: List[Dict[str, Any]],
    *,
    k_train_orig: int,
    k_val: int,
    train_items: int,
    j_train_orig: int,
    j_val: int,
    train_annotators: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in pairwise:
        ann = int(row["annotator"])
        ann_g = annotator_group(ann, j_train_orig, j_val)
        if ann_g == "train" and ann > train_annotators:
            continue
        keep = True
        new_items: List[int] = []
        for item in row["items"]:
            item = int(item)
            item_g = item_group(item, k_train_orig, k_val)
            if item_g == "train" and item > train_items:
                keep = False
                break
            if item_g == "train":
                new_items.append(item)
            else:
                new_items.append(remap_item_id(item, k_train_orig, train_items))
        if not keep:
            continue
        new_row = dict(row)
        if ann_g != "train":
            new_row["annotator"] = remap_annotator_id(ann, j_train_orig, train_annotators)
        new_row["items"] = new_items
        label = annotator_group(int(new_row["annotator"]), train_annotators, j_val)
        for item in new_items:
            label = combine_instance(item_group(int(item), train_items, k_val), label)
        new_row["instance"] = label
        out.append(new_row)
    return out


def subset_numpy_embeddings(embeddings: np.ndarray, k_train_orig: int, k_val: int, k_test: int, train_num: int) -> np.ndarray:
    train_emb = embeddings[:train_num]
    val_emb = embeddings[k_train_orig:k_train_orig + k_val]
    test_emb = embeddings[k_train_orig + k_val:k_train_orig + k_val + k_test]
    return np.vstack([train_emb, val_emb, test_emb])


def subset_numpy_annotator_rows(arr: np.ndarray, i_dim: int, j_train_orig: int, j_val: int, j_test: int, train_num: int) -> np.ndarray:
    j_total_orig = j_train_orig + j_val + j_test
    j_total_new = train_num + j_val + j_test
    rest_shape = arr.shape[1:]
    out = np.empty((i_dim * j_total_new,) + rest_shape, dtype=arr.dtype)
    for i in range(i_dim):
        src = i * j_total_orig
        dst = i * j_total_new
        out[dst:dst + train_num] = arr[src:src + train_num]
        val_src = src + j_train_orig
        val_dst = dst + train_num
        out[val_dst:val_dst + j_val] = arr[val_src:val_src + j_val]
        test_src = src + j_train_orig + j_val
        test_dst = dst + train_num + j_val
        out[test_dst:test_dst + j_test] = arr[test_src:test_src + j_test]
    return out


def subset_numpy_base_scores(
    base_scores: np.ndarray,
    *,
    i_dim: int,
    j_train_orig: int,
    j_val: int,
    j_test: int,
    train_annotators: int,
    k_train_orig: int,
    k_val: int,
    k_test: int,
    train_items: int,
) -> np.ndarray:
    row_subset = subset_numpy_annotator_rows(base_scores, i_dim, j_train_orig, j_val, j_test, train_annotators)
    return np.hstack([
        row_subset[:, :train_items],
        row_subset[:, k_train_orig:k_train_orig + k_val],
        row_subset[:, k_train_orig + k_val:k_train_orig + k_val + k_test],
    ])


def subset_train_posterior_probs(
    probs: np.ndarray,
    *,
    i_dim: int,
    j_train_orig: int,
    j_val: int,
    j_test: int,
    train_annotators: int,
    k_train_orig: int,
    train_items: int,
) -> np.ndarray:
    row_subset = subset_numpy_annotator_rows(probs, i_dim, j_train_orig, j_val, j_test, train_annotators)
    return row_subset[:, :train_items, :]


def recompute_stats(
    *,
    bundle_stats: Dict[str, Any],
    train_items: int,
    train_annotators: int,
    all_ratings: List[Dict[str, Any]],
    observed_ratings: List[Dict[str, Any]],
    missing_ratings: List[Dict[str, Any]],
    all_pairwise: List[Dict[str, Any]],
    observed_pairwise: List[Dict[str, Any]],
    missing_pairwise: List[Dict[str, Any]],
) -> Dict[str, Any]:
    def _count(rows: List[Dict[str, Any]], instance: str) -> int:
        return sum(1 for row in rows if row["instance"] == instance)

    train_ratings = _count(all_ratings, "train")
    val_ratings = _count(all_ratings, "val")
    test_ratings = _count(all_ratings, "test")
    train_obs = _count(observed_ratings, "train")
    val_obs = _count(observed_ratings, "val")
    test_obs = _count(observed_ratings, "test")

    return {
        **bundle_stats,
        "K_train": train_items,
        "J": train_annotators + int(bundle_stats["J_val_split"]) + int(bundle_stats["J_test_split"]),
        "J_train_split": train_annotators,
        "total_items": train_items + int(bundle_stats["K_val"]) + int(bundle_stats["K_test"]),
        "total_possible_ratings": len(all_ratings),
        "total_ratings": len(all_ratings),
        "observed_ratings": len(observed_ratings),
        "missing_ratings": len(missing_ratings),
        "train_ratings": train_ratings,
        "val_ratings": val_ratings,
        "test_ratings": test_ratings,
        "train_observed": train_obs,
        "val_observed": val_obs,
        "test_observed": test_obs,
        "total_pairwise": len(all_pairwise),
        "observed_pairwise": len(observed_pairwise),
        "missing_pairwise": len(missing_pairwise),
        "train_pairwise": _count(all_pairwise, "train"),
        "val_pairwise": _count(all_pairwise, "val"),
        "test_pairwise": _count(all_pairwise, "test"),
        "observation_rate": len(observed_ratings) / len(all_ratings) if all_ratings else 0.0,
        "train_observation_rate": train_obs / train_ratings if train_ratings else 0.0,
        "val_observation_rate": val_obs / val_ratings if val_ratings else 0.0,
        "test_observation_rate": test_obs / test_ratings if test_ratings else 0.0,
        "subset_K_train_orig": bundle_stats["K_train"],
        "subset_J_train_orig": bundle_stats["J_train_split"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Subset a joint item+annotator split data directory")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-items", type=int, required=True)
    parser.add_argument("--train-annotators", type=int, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    bundle = load_json(input_dir / "data_bundle.json")
    cfg = load_json(input_dir / "configs.json")
    dg = cfg.get("datagen", cfg)

    k_train_orig = int(bundle["stats"]["K_train"])
    k_val = int(bundle["stats"]["K_val"])
    k_test = int(bundle["stats"]["K_test"])
    j_train_orig = int(bundle["stats"]["J_train_split"])
    j_val = int(bundle["stats"]["J_val_split"])
    j_test = int(bundle["stats"]["J_test_split"])
    i_dim = int(dg["I"])

    if args.train_items < 1 or args.train_items > k_train_orig:
        raise ValueError(f"--train-items must be in [1, {k_train_orig}]")
    if args.train_annotators < 1 or args.train_annotators > j_train_orig:
        raise ValueError(f"--train-annotators must be in [1, {j_train_orig}]")

    all_ratings = subset_ratings(bundle["all_ratings"], k_train_orig=k_train_orig, k_val=k_val, train_items=args.train_items,
                                 j_train_orig=j_train_orig, j_val=j_val, train_annotators=args.train_annotators)
    observed_ratings = subset_ratings(bundle["observed_ratings"], k_train_orig=k_train_orig, k_val=k_val, train_items=args.train_items,
                                      j_train_orig=j_train_orig, j_val=j_val, train_annotators=args.train_annotators)
    missing_ratings = subset_ratings(bundle["missing_ratings"], k_train_orig=k_train_orig, k_val=k_val, train_items=args.train_items,
                                     j_train_orig=j_train_orig, j_val=j_val, train_annotators=args.train_annotators)
    all_pairwise = subset_pairwise(bundle.get("all_pairwise", []), k_train_orig=k_train_orig, k_val=k_val, train_items=args.train_items,
                                   j_train_orig=j_train_orig, j_val=j_val, train_annotators=args.train_annotators)
    observed_pairwise = subset_pairwise(bundle.get("observed_pairwise", []), k_train_orig=k_train_orig, k_val=k_val, train_items=args.train_items,
                                        j_train_orig=j_train_orig, j_val=j_val, train_annotators=args.train_annotators)
    missing_pairwise = subset_pairwise(bundle.get("missing_pairwise", []), k_train_orig=k_train_orig, k_val=k_val, train_items=args.train_items,
                                       j_train_orig=j_train_orig, j_val=j_val, train_annotators=args.train_annotators)
    missing_indexes = [i for i, row in enumerate(missing_ratings) if row["instance"] == "test"]

    out_bundle: Dict[str, Any] = {
        "all_ratings": all_ratings,
        "all_pairwise": all_pairwise,
        "observed_ratings": observed_ratings,
        "missing_ratings": missing_ratings,
        "missing_ratings_indexes_in_test_instance": missing_indexes,
        "observed_pairwise": observed_pairwise,
        "missing_pairwise": missing_pairwise,
        "stats": recompute_stats(
            bundle_stats=bundle["stats"],
            train_items=args.train_items,
            train_annotators=args.train_annotators,
            all_ratings=all_ratings,
            observed_ratings=observed_ratings,
            missing_ratings=missing_ratings,
            all_pairwise=all_pairwise,
            observed_pairwise=observed_pairwise,
            missing_pairwise=missing_pairwise,
        ),
    }

    if "embeddings" in bundle and bundle["embeddings"] is not None:
        out_bundle["embeddings"] = subset_numpy_embeddings(np.array(bundle["embeddings"]), k_train_orig, k_val, k_test, args.train_items).tolist()

    for key in ("annotator_preferences", "rating_probs", "rating_cumprobs", "rating_thresholds_z"):
        if key in bundle and bundle[key] is not None:
            out_bundle[key] = subset_numpy_annotator_rows(np.array(bundle[key]), i_dim, j_train_orig, j_val, j_test, args.train_annotators).tolist()

    if "base_scores" in bundle and bundle["base_scores"] is not None:
        out_bundle["base_scores"] = subset_numpy_base_scores(
            np.array(bundle["base_scores"]),
            i_dim=i_dim,
            j_train_orig=j_train_orig,
            j_val=j_val,
            j_test=j_test,
            train_annotators=args.train_annotators,
            k_train_orig=k_train_orig,
            k_val=k_val,
            k_test=k_test,
            train_items=args.train_items,
        ).tolist()

    if "train_posterior_rating_probs" in bundle and bundle["train_posterior_rating_probs"] is not None:
        out_bundle["train_posterior_rating_probs"] = subset_train_posterior_probs(
            np.array(bundle["train_posterior_rating_probs"]),
            i_dim=i_dim,
            j_train_orig=j_train_orig,
            j_val=j_val,
            j_test=j_test,
            train_annotators=args.train_annotators,
            k_train_orig=k_train_orig,
            train_items=args.train_items,
        ).tolist()

    for key in ("val_posterior_rating_probs", "test_posterior_rating_probs"):
        if key in bundle and bundle[key] is not None:
            out_bundle[key] = subset_numpy_annotator_rows(np.array(bundle[key]), i_dim, j_train_orig, j_val, j_test, args.train_annotators).tolist()

    for key in ("mean_preferences",):
        if key in bundle and bundle[key] is not None:
            out_bundle[key] = bundle[key]

    standard_keys = {
        "all_ratings", "all_pairwise", "observed_ratings", "missing_ratings",
        "missing_ratings_indexes_in_test_instance", "observed_pairwise", "missing_pairwise",
        "stats", "embeddings", "annotator_preferences", "rating_probs", "rating_cumprobs",
        "rating_thresholds_z", "base_scores", "train_posterior_rating_probs",
        "val_posterior_rating_probs", "test_posterior_rating_probs", "mean_preferences",
    }
    for key, value in bundle.items():
        if key not in standard_keys and value is not None:
            out_bundle[key] = value

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_bundle, output_dir / "data_bundle.json")

    section = cfg.get("datagen", cfg)
    section["K_train"] = args.train_items
    section["J"] = args.train_annotators + j_val + j_test
    section["J_train_split"] = args.train_annotators
    section["subset_K_train_orig"] = k_train_orig
    section["subset_J_train_orig"] = j_train_orig
    save_json(cfg, output_dir / "configs.json")

    stan_data_path = input_dir / "stan_data.json"
    if stan_data_path.exists():
        stan_data = load_json(stan_data_path)
        stan_data["K_train"] = args.train_items
        stan_data["J"] = args.train_annotators + j_val + j_test
        stan_data["J_train_split"] = args.train_annotators
        save_json(stan_data, output_dir / "stan_data.json")

    print("Done.")
    print(f"  Train items: {k_train_orig} -> {args.train_items}")
    print(f"  Train annotators: {j_train_orig} -> {args.train_annotators}")
    print(f"  Train ratings: {out_bundle['stats']['train_ratings']}")
    print(f"  Val ratings:   {out_bundle['stats']['val_ratings']}")
    print(f"  Test ratings:  {out_bundle['stats']['test_ratings']}")


if __name__ == "__main__":
    main()
