#!/usr/bin/env python3
"""
Remap non-train item IDs into the train item ID pool (1..K_train).

Use this on item-split bundles (e.g., Tensor_400_25_9_ItemTest_*), so validation
and test rows do not introduce unseen item IDs.

Behavior:
  - Keeps train rows unchanged.
  - Remaps any item id > K_train to an id in [1, K_train].
  - Mapping is deterministic given --seed.
  - If K_val or K_test is larger than K_train, remap uses replacement.
  - Optionally drops latent arrays (embeddings/base_scores/etc.) since they are
    no longer aligned with the remapped item IDs.
"""

from __future__ import annotations

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


def _remap_rating_rows(rows: List[Dict[str, Any]], item_map: Dict[int, int], k_train: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = int(row["item"])
        new_row = dict(row)
        if item > k_train:
            new_row["item"] = int(item_map[item])
        out.append(new_row)
    return out


def _remap_pairwise_rows(rows: List[Dict[str, Any]], item_map: Dict[int, int], k_train: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        items = []
        for item in row.get("items", []):
            it = int(item)
            if it > k_train:
                it = int(item_map[it])
            items.append(it)
        new_row["items"] = items
        out.append(new_row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-latent-arrays",
        action="store_true",
        help="Keep embeddings/base_scores/rating_probs arrays even though they no longer match remapped item IDs.",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    bundle = load_json(in_dir / "data_bundle.json")
    cfg = load_json(in_dir / "configs.json")
    dg = cfg.get("datagen", cfg)
    stats = bundle.get("stats", {})

    k_train = int(stats.get("K_train", dg["K_train"]))
    k_val = int(stats.get("K_val", dg.get("K_val", 0)))
    k_test = int(stats.get("K_test", dg.get("K_test", 0)))
    total = k_train + k_val + k_test

    val_ids = list(range(k_train + 1, k_train + k_val + 1))
    test_ids = list(range(k_train + k_val + 1, total + 1))

    rng = np.random.default_rng(args.seed)
    train_pool = np.arange(1, k_train + 1)

    val_replace = len(val_ids) > k_train
    test_replace = len(test_ids) > k_train
    val_targets = rng.choice(train_pool, size=len(val_ids), replace=val_replace)
    test_targets = rng.choice(train_pool, size=len(test_ids), replace=test_replace)

    item_map: Dict[int, int] = {}
    for old, new in zip(val_ids, val_targets):
        item_map[int(old)] = int(new)
    for old, new in zip(test_ids, test_targets):
        item_map[int(old)] = int(new)

    new_bundle: Dict[str, Any] = dict(bundle)
    for key in ("all_ratings", "observed_ratings", "missing_ratings"):
        if key in new_bundle:
            new_bundle[key] = _remap_rating_rows(new_bundle[key], item_map=item_map, k_train=k_train)
    for key in ("all_pairwise", "observed_pairwise", "missing_pairwise"):
        if key in new_bundle and new_bundle[key]:
            new_bundle[key] = _remap_pairwise_rows(new_bundle[key], item_map=item_map, k_train=k_train)

    if not args.keep_latent_arrays:
        for key in (
            "embeddings",
            "base_scores",
            "rating_probs",
            "rating_cumprobs",
            "rating_thresholds_z",
            "train_posterior_rating_probs",
            "val_posterior_rating_probs",
            "test_posterior_rating_probs",
        ):
            if key in new_bundle:
                new_bundle[key] = None

    new_stats = dict(stats)
    new_stats["item_id_pool"] = "shared_train_ids_only"
    new_stats["item_id_remap_seed"] = args.seed
    new_stats["item_id_remap_val_replace"] = bool(val_replace)
    new_stats["item_id_remap_test_replace"] = bool(test_replace)
    new_bundle["stats"] = new_stats

    new_cfg = dict(cfg)
    new_dg = dict(dg)
    new_dg["item_id_pool"] = "shared_train_ids_only"
    new_dg["item_id_remap_seed"] = args.seed
    new_cfg["datagen"] = new_dg

    save_json(new_bundle, out_dir / "data_bundle.json")
    save_json(new_cfg, out_dir / "configs.json")
    save_json(
        {
            "K_train": k_train,
            "K_val": k_val,
            "K_test": k_test,
            "seed": args.seed,
            "val_replace": bool(val_replace),
            "test_replace": bool(test_replace),
            "val_old_to_new": {str(k): int(v) for k, v in zip(val_ids, val_targets)},
            "test_old_to_new": {str(k): int(v) for k, v in zip(test_ids, test_targets)},
        },
        out_dir / "item_remap_meta.json",
    )

    stan_path = in_dir / "stan_data.json"
    if stan_path.exists():
        shutil.copy2(stan_path, out_dir / "stan_data.json")

    print(f"Wrote {out_dir}")
    print(
        "Remap settings: "
        f"K_train={k_train}, K_val={k_val}, K_test={k_test}, "
        f"val_replace={val_replace}, test_replace={test_replace}"
    )


if __name__ == "__main__":
    main()
