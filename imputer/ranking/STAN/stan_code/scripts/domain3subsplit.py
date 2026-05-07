#!/usr/bin/env python3
"""Create DOMAIN3 expansion splits from one full 400x25x9 tensor bundle.

DOMAIN3 fixes the lower-right 50 x 5 block as the evaluation block:
- test items:      last 50 items
- test annotators: last 5 annotators

We then create two families of bundles:
1. Item expansion:
   - Transductive: use only the bottom 5 annotators and expand left over items.
   - Non-transductive: train on items immediately left of the test block, evaluate on
     the fixed test block, still using only the bottom 5 annotators.
2. Annotator expansion:
   - Transductive: use only the rightmost 50 items and expand upward over annotators.
   - Non-transductive: train on annotators immediately above the test block,
     evaluate on the fixed test block, still using only the rightmost 50 items.

The output bundles are written in formats already supported by the codebase:
- item expansion bundles use item-split style configs (K_train/K_test/K_val).
- annotator expansion bundles use annotator-split style configs (K/J_train/J_test).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from STAN.stan_code.pipeline.bundle import GroundTruthBundle
from STAN.stan_code.pipeline.configs import AnnotatorSplitConfig, DataGenConfig, STAN_TYPE_REQUIRED
from STAN.stan_code.pipeline.io import save_bundle, save_configs, save_json


def _parse_sizes(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _bundle_to_dict(bundle: GroundTruthBundle) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "all_ratings": bundle.all_ratings,
        "all_pairwise": bundle.all_pairwise,
        "observed_ratings": bundle.observed_ratings,
        "missing_ratings": bundle.missing_ratings,
        "observed_pairwise": bundle.observed_pairwise,
        "missing_pairwise": bundle.missing_pairwise,
        "missing_ratings_indexes_in_test_instance": bundle.missing_ratings_indexes_in_test_instance,
        "stats": bundle.stats,
    }
    for key in (
        "embeddings",
        "mean_preferences",
        "annotator_preferences",
        "rating_probs",
        "rating_cumprobs",
        "rating_thresholds_z",
        "base_scores",
        "train_posterior_rating_probs",
        "val_posterior_rating_probs",
        "test_posterior_rating_probs",
    ):
        value = getattr(bundle, key, None)
        if value is not None:
            out[key] = value.tolist() if isinstance(value, np.ndarray) else value
    if bundle.extra_ground_truth:
        for key, value in bundle.extra_ground_truth.items():
            out[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return out


def _build_type_kwargs(dg: Dict[str, Any]) -> Dict[str, Any]:
    required = STAN_TYPE_REQUIRED[dg["stan_type"]]
    out = {key: dg.get(key) for key in required}
    missing = [key for key, value in out.items() if value is None]
    if missing:
        raise ValueError(f"Missing required Stan fields in base config: {missing}")
    return out


def _subset_flat_attr_annot(arr: np.ndarray, i_dim: int, annot_orig_ids: Sequence[int]) -> np.ndarray:
    j_orig = arr.shape[0] // i_dim
    j_new = len(annot_orig_ids)
    rest_shape = arr.shape[1:]
    out = np.empty((i_dim * j_new,) + rest_shape, dtype=arr.dtype)
    orig_zero = [a - 1 for a in annot_orig_ids]
    for i in range(i_dim):
        src = i * j_orig
        for new_j, old_j0 in enumerate(orig_zero):
            out[i * j_new + new_j] = arr[src + old_j0]
    return out


def _subset_base_scores(arr: np.ndarray, i_dim: int, annot_orig_ids: Sequence[int], item_orig_ids: Sequence[int]) -> np.ndarray:
    row_subset = _subset_flat_attr_annot(arr, i_dim, annot_orig_ids)
    col_idx = [item_id - 1 for item_id in item_orig_ids]
    return row_subset[:, col_idx]


def _subset_embeddings(arr: np.ndarray, item_orig_ids: Sequence[int]) -> np.ndarray:
    return arr[[item_id - 1 for item_id in item_orig_ids]]


def _collect_extra_arrays(
    bundle: GroundTruthBundle,
    item_orig_ids: Sequence[int],
    annot_orig_ids: Sequence[int],
    i_dim: int,
    j_orig: int,
    total_items_orig: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if bundle.embeddings is not None:
        out["embeddings"] = _subset_embeddings(np.asarray(bundle.embeddings), item_orig_ids).tolist()
    if bundle.mean_preferences is not None:
        out["mean_preferences"] = np.asarray(bundle.mean_preferences).tolist()
    for key in ("annotator_preferences", "rating_probs", "rating_cumprobs", "rating_thresholds_z"):
        value = getattr(bundle, key, None)
        if value is not None:
            out[key] = _subset_flat_attr_annot(np.asarray(value), i_dim, annot_orig_ids).tolist()
    if bundle.base_scores is not None:
        out["base_scores"] = _subset_base_scores(np.asarray(bundle.base_scores), i_dim, annot_orig_ids, item_orig_ids).tolist()

    extra = bundle.extra_ground_truth or {}
    for key, value in extra.items():
        arr = np.asarray(value) if isinstance(value, list) else value
        if isinstance(arr, np.ndarray):
            if arr.ndim >= 2 and arr.shape[0] == total_items_orig:
                out[key] = _subset_embeddings(arr, item_orig_ids).tolist()
            elif arr.ndim >= 1 and arr.shape[0] == i_dim * j_orig:
                out[key] = _subset_flat_attr_annot(arr, i_dim, annot_orig_ids).tolist()
            else:
                out[key] = arr.tolist()
        else:
            out[key] = value
    return out


def _recompute_stats(rows_all: List[Dict[str, Any]], rows_obs: List[Dict[str, Any]], rows_missing: List[Dict[str, Any]], *, k_train: int | None, k_val: int | None, k_test: int | None, k_shared: int | None, j_total: int, j_train: int | None, j_val: int | None, j_test: int | None, i_dim: int, c_dim: int) -> Dict[str, Any]:
    def count(rows: Sequence[Dict[str, Any]], instance: str) -> int:
        return sum(1 for row in rows if row["instance"] == instance)

    train_ratings = count(rows_all, "train")
    val_ratings = count(rows_all, "val")
    test_ratings = count(rows_all, "test")
    train_obs = count(rows_obs, "train")
    val_obs = count(rows_obs, "val")
    test_obs = count(rows_obs, "test")

    stats = {
        "I": i_dim,
        "J": j_total,
        "C": c_dim,
        "total_possible_ratings": len(rows_all),
        "total_ratings": len(rows_all),
        "observed_ratings": len(rows_obs),
        "missing_ratings": len(rows_missing),
        "train_ratings": train_ratings,
        "val_ratings": val_ratings,
        "test_ratings": test_ratings,
        "train_observed": train_obs,
        "val_observed": val_obs,
        "test_observed": test_obs,
        "total_pairwise": 0,
        "observed_pairwise": 0,
        "missing_pairwise": 0,
        "train_pairwise": 0,
        "val_pairwise": 0,
        "test_pairwise": 0,
        "observation_rate": len(rows_obs) / len(rows_all) if rows_all else 0.0,
        "train_observation_rate": train_obs / train_ratings if train_ratings else 0.0,
        "val_observation_rate": val_obs / val_ratings if val_ratings else 0.0,
        "test_observation_rate": test_obs / test_ratings if test_ratings else 0.0,
    }
    if k_shared is not None:
        stats.update({"K": k_shared})
    else:
        stats.update({
            "K_train": int(k_train or 0),
            "K_val": int(k_val or 0),
            "K_test": int(k_test or 0),
            "total_items": int((k_train or 0) + (k_val or 0) + (k_test or 0)),
        })
    if j_train is not None:
        stats.update({
            "J_train_split": int(j_train),
            "J_val_split": int(j_val or 0),
            "J_test_split": int(j_test or 0),
        })
    return stats


def _remap_rows(rows: Iterable[Dict[str, Any]], *, item_map: Dict[int, int], annot_map: Dict[int, int], item_instance_fn, annot_instance_fn) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        item_orig = int(row["item"])
        annot_orig = int(row["annotator"])
        if item_orig not in item_map or annot_orig not in annot_map:
            continue
        new_row = dict(row)
        new_row["item"] = item_map[item_orig]
        new_row["annotator"] = annot_map[annot_orig]
        item_inst = item_instance_fn(item_orig)
        annot_inst = annot_instance_fn(annot_orig)
        if annot_inst is None:
            new_row["instance"] = item_inst
        elif item_inst is None:
            new_row["instance"] = annot_inst
        elif item_inst == "test" or annot_inst == "test":
            new_row["instance"] = "test"
        elif item_inst == "val" or annot_inst == "val":
            new_row["instance"] = "val"
        else:
            new_row["instance"] = "train"
        out.append(new_row)
    return out


def _save_item_split(
    bundle: GroundTruthBundle,
    dg: Dict[str, Any],
    output_dir: Path,
    run_name: str,
    *,
    train_items_orig: Sequence[int],
    test_items_orig: Sequence[int],
    annot_orig_ids: Sequence[int],
    protocol: str,
) -> None:
    item_orig_ids = list(train_items_orig) + list(test_items_orig)
    item_map = {orig: idx + 1 for idx, orig in enumerate(item_orig_ids)}
    annot_map = {orig: idx + 1 for idx, orig in enumerate(annot_orig_ids)}
    train_set = set(train_items_orig)
    test_set = set(test_items_orig)

    def item_instance(item_orig: int) -> str:
        if item_orig in train_set:
            return "train"
        if item_orig in test_set:
            return "test"
        raise KeyError(item_orig)

    rows_all = _remap_rows(bundle.all_ratings, item_map=item_map, annot_map=annot_map, item_instance_fn=item_instance, annot_instance_fn=lambda _a: None)
    rows_obs = _remap_rows(bundle.observed_ratings, item_map=item_map, annot_map=annot_map, item_instance_fn=item_instance, annot_instance_fn=lambda _a: None)
    rows_missing = _remap_rows(bundle.missing_ratings, item_map=item_map, annot_map=annot_map, item_instance_fn=item_instance, annot_instance_fn=lambda _a: None)
    missing_test_idx = [i for i, row in enumerate(rows_missing) if row["instance"] == "test"]

    stats = _recompute_stats(
        rows_all,
        rows_obs,
        rows_missing,
        k_train=len(train_items_orig),
        k_val=0,
        k_test=len(test_items_orig),
        k_shared=None,
        j_total=len(annot_orig_ids),
        j_train=None,
        j_val=None,
        j_test=None,
        i_dim=int(dg["I"]),
        c_dim=int(dg["C"]),
    )

    type_kwargs = _build_type_kwargs(dg)
    config = DataGenConfig(
        K_train=len(train_items_orig),
        K_val=0,
        K_test=len(test_items_orig),
        I=int(dg["I"]),
        J=len(annot_orig_ids),
        C=int(dg["C"]),
        enable_pairwise_rankings=False,
        pairwise_cap_per_item=int(dg.get("pairwise_cap_per_item", 10)),
        observation_protocol=str(dg.get("observation_protocol", "mcar")),
        mcar_missing_rate=float(dg.get("mcar_missing_rate", 0.5)),
        pairwise_observation_rate=float(dg.get("pairwise_observation_rate", 1.0)),
        seed=int(dg.get("seed", 42)) if dg.get("seed") is not None else None,
        stan_type=str(dg["stan_type"]),
        **type_kwargs,
    )

    bundle_dict = {
        "all_ratings": rows_all,
        "all_pairwise": [],
        "observed_ratings": rows_obs,
        "missing_ratings": rows_missing,
        "missing_ratings_indexes_in_test_instance": missing_test_idx,
        "observed_pairwise": [],
        "missing_pairwise": [],
        "stats": stats,
        "domain3_metadata": {
            "experiment_axis": "item",
            "protocol": protocol,
            "selected_item_orig_ids": item_orig_ids,
            "selected_annotator_orig_ids": list(annot_orig_ids),
            "test_block_item_orig_ids": list(test_items_orig),
            "test_block_annotator_orig_ids": list(annot_orig_ids),
        },
    }
    bundle_dict.update(
        _collect_extra_arrays(
            bundle,
            item_orig_ids,
            annot_orig_ids,
            int(dg["I"]),
            int(dg["J"]),
            int(dg["K_train"]) + int(dg.get("K_val", 0)) + int(dg["K_test"]),
        )
    )
    bundle_dict["missing_ratings_indexes_in_test_instance"] = missing_test_idx

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_bundle(run_dir, bundle_dict)

    cfg_dict = asdict(config)
    cfg_dict["domain3_metadata"] = bundle_dict["domain3_metadata"]
    save_configs(run_dir, datagen=cfg_dict)
    save_json(config.to_stan_data(), run_dir / "stan_data.json")


def _save_annot_split(
    bundle: GroundTruthBundle,
    dg: Dict[str, Any],
    output_dir: Path,
    run_name: str,
    *,
    item_orig_ids: Sequence[int],
    train_annot_orig: Sequence[int],
    test_annot_orig: Sequence[int],
    protocol: str,
) -> None:
    annot_orig_ids = list(train_annot_orig) + list(test_annot_orig)
    item_map = {orig: idx + 1 for idx, orig in enumerate(item_orig_ids)}
    annot_map = {orig: idx + 1 for idx, orig in enumerate(annot_orig_ids)}
    train_set = set(train_annot_orig)
    test_set = set(test_annot_orig)

    def annot_instance(annot_orig: int) -> str:
        if annot_orig in train_set:
            return "train"
        if annot_orig in test_set:
            return "test"
        raise KeyError(annot_orig)

    rows_all = _remap_rows(bundle.all_ratings, item_map=item_map, annot_map=annot_map, item_instance_fn=lambda _i: None, annot_instance_fn=annot_instance)
    rows_obs = _remap_rows(bundle.observed_ratings, item_map=item_map, annot_map=annot_map, item_instance_fn=lambda _i: None, annot_instance_fn=annot_instance)
    rows_missing = _remap_rows(bundle.missing_ratings, item_map=item_map, annot_map=annot_map, item_instance_fn=lambda _i: None, annot_instance_fn=annot_instance)
    missing_test_idx = [i for i, row in enumerate(rows_missing) if row["instance"] == "test"]

    stats = _recompute_stats(
        rows_all,
        rows_obs,
        rows_missing,
        k_train=None,
        k_val=None,
        k_test=None,
        k_shared=len(item_orig_ids),
        j_total=len(annot_orig_ids),
        j_train=len(train_annot_orig),
        j_val=0,
        j_test=len(test_annot_orig),
        i_dim=int(dg["I"]),
        c_dim=int(dg["C"]),
    )

    type_kwargs = _build_type_kwargs(dg)
    config = AnnotatorSplitConfig(
        K=len(item_orig_ids),
        J_train=len(train_annot_orig),
        J_val=0,
        J_test=len(test_annot_orig),
        I=int(dg["I"]),
        C=int(dg["C"]),
        observation_protocol=str(dg.get("observation_protocol", "mcar")),
        mcar_missing_rate=float(dg.get("mcar_missing_rate", 0.5)),
        enable_pairwise_rankings=False,
        pairwise_cap_per_item=int(dg.get("pairwise_cap_per_item", 10)),
        seed=int(dg.get("seed", 42)) if dg.get("seed") is not None else None,
        stan_type=str(dg["stan_type"]),
        **type_kwargs,
    )

    bundle_dict = {
        "all_ratings": rows_all,
        "all_pairwise": [],
        "observed_ratings": rows_obs,
        "missing_ratings": rows_missing,
        "missing_ratings_indexes_in_test_instance": missing_test_idx,
        "observed_pairwise": [],
        "missing_pairwise": [],
        "stats": stats,
        "domain3_metadata": {
            "experiment_axis": "annotator",
            "protocol": protocol,
            "selected_item_orig_ids": list(item_orig_ids),
            "selected_annotator_orig_ids": annot_orig_ids,
            "test_block_item_orig_ids": list(item_orig_ids),
            "test_block_annotator_orig_ids": list(test_annot_orig),
        },
    }
    bundle_dict.update(
        _collect_extra_arrays(
            bundle,
            item_orig_ids,
            annot_orig_ids,
            int(dg["I"]),
            int(dg["J"]),
            int(dg["K_train"]) + int(dg.get("K_val", 0)) + int(dg["K_test"]),
        )
    )
    bundle_dict["missing_ratings_indexes_in_test_instance"] = missing_test_idx

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_bundle(run_dir, bundle_dict)

    cfg_dict = asdict(config)
    cfg_dict["J"] = len(annot_orig_ids)
    cfg_dict["domain3_metadata"] = bundle_dict["domain3_metadata"]
    save_configs(run_dir, datagen=cfg_dict)
    stan_data = config.to_stan_data()
    stan_data["J"] = len(annot_orig_ids)
    save_json(stan_data, run_dir / "stan_data.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create DOMAIN3 item/annotator expansion splits")
    parser.add_argument("--input-dir", required=True, help="Directory containing the full generated base bundle")
    parser.add_argument("--output-root", required=True, help="Root output directory, e.g. DATA/STAN/DOMAIN3")
    parser.add_argument("--run-prefix", default="Tensor_400_25_9_DOMAIN3")
    parser.add_argument("--item-trans-sizes", default="50,100,150,200,250,300,350,400")
    parser.add_argument("--item-nt-sizes", default="50,100,150,200,250,300,350")
    parser.add_argument("--annot-trans-sizes", default="5,10,15,20,25")
    parser.add_argument("--annot-nt-sizes", default="5,10,15,20")
    parser.add_argument("--test-items", type=int, default=50)
    parser.add_argument("--test-annotators", type=int, default=5)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    bundle = GroundTruthBundle.from_dict(_load_json(input_dir / "data_bundle.json"))
    cfg = _load_json(input_dir / "configs.json")
    dg = cfg.get("datagen", cfg)

    total_items = int(dg["K_train"]) + int(dg.get("K_val", 0)) + int(dg["K_test"])
    total_annots = int(dg["J"])
    test_items_orig = list(range(total_items - args.test_items + 1, total_items + 1))
    test_annots_orig = list(range(total_annots - args.test_annotators + 1, total_annots + 1))

    item_t_root = output_root / "ItemSplits" / "Transductive"
    item_nt_root = output_root / "ItemSplits" / "NonTransductive"
    annot_t_root = output_root / "AnnotSplits" / "Transductive"
    annot_nt_root = output_root / "AnnotSplits" / "NonTransductive"
    for root in (item_t_root, item_nt_root, annot_t_root, annot_nt_root):
        root.mkdir(parents=True, exist_ok=True)

    for total_selected in _parse_sizes(args.item_trans_sizes):
        if total_selected < args.test_items or total_selected > total_items:
            raise ValueError(f"Invalid item transductive size: {total_selected}")
        selected_items = list(range(total_items - total_selected + 1, total_items + 1))
        train_items_orig = selected_items[:-args.test_items]
        run_name = f"{args.run_prefix}_Item_T_{total_selected}"
        _save_item_split(
            bundle,
            dg,
            item_t_root,
            run_name,
            train_items_orig=train_items_orig,
            test_items_orig=test_items_orig,
            annot_orig_ids=test_annots_orig,
            protocol="transductive",
        )

    for train_size in _parse_sizes(args.item_nt_sizes):
        if train_size < 1 or train_size > total_items - args.test_items:
            raise ValueError(f"Invalid item non-transductive size: {train_size}")
        train_items_orig = list(range(total_items - args.test_items - train_size + 1, total_items - args.test_items + 1))
        run_name = f"{args.run_prefix}_Item_NT_{train_size}"
        _save_item_split(
            bundle,
            dg,
            item_nt_root,
            run_name,
            train_items_orig=train_items_orig,
            test_items_orig=test_items_orig,
            annot_orig_ids=test_annots_orig,
            protocol="nontransductive",
        )

    fixed_items_orig = test_items_orig
    for total_selected in _parse_sizes(args.annot_trans_sizes):
        if total_selected < args.test_annotators or total_selected > total_annots:
            raise ValueError(f"Invalid annotator transductive size: {total_selected}")
        selected_annots = list(range(total_annots - total_selected + 1, total_annots + 1))
        train_annots = selected_annots[:-args.test_annotators]
        run_name = f"{args.run_prefix}_Annot_T_{total_selected}"
        _save_annot_split(
            bundle,
            dg,
            annot_t_root,
            run_name,
            item_orig_ids=fixed_items_orig,
            train_annot_orig=train_annots,
            test_annot_orig=test_annots_orig,
            protocol="transductive",
        )

    for train_size in _parse_sizes(args.annot_nt_sizes):
        if train_size < 1 or train_size > total_annots - args.test_annotators:
            raise ValueError(f"Invalid annotator non-transductive size: {train_size}")
        train_annots = list(range(total_annots - args.test_annotators - train_size + 1, total_annots - args.test_annotators + 1))
        run_name = f"{args.run_prefix}_Annot_NT_{train_size}"
        _save_annot_split(
            bundle,
            dg,
            annot_nt_root,
            run_name,
            item_orig_ids=fixed_items_orig,
            train_annot_orig=train_annots,
            test_annot_orig=test_annots_orig,
            protocol="nontransductive",
        )

    metadata = {
        "run_prefix": args.run_prefix,
        "input_dir": str(input_dir),
        "total_items": total_items,
        "total_annotators": total_annots,
        "attributes": int(dg["I"]),
        "categories": int(dg["C"]),
        "test_item_orig_ids": test_items_orig,
        "test_annotator_orig_ids": test_annots_orig,
        "item_trans_sizes": _parse_sizes(args.item_trans_sizes),
        "item_nt_sizes": _parse_sizes(args.item_nt_sizes),
        "annot_trans_sizes": _parse_sizes(args.annot_trans_sizes),
        "annot_nt_sizes": _parse_sizes(args.annot_nt_sizes),
    }
    save_json(metadata, output_root / "domain3_metadata.json")

    print("DOMAIN3 splits written to:")
    print(f"  {output_root}")
    print(f"  Fixed test block items: {test_items_orig[0]}..{test_items_orig[-1]}")
    print(f"  Fixed test block annotators: {test_annots_orig[0]}..{test_annots_orig[-1]}")


if __name__ == "__main__":
    main()
