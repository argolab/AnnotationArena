#!/usr/bin/env python3
"""
Re-split rating rows into train/val/test by relabeling `instance` only.

This keeps each observation's (attribute, annotator, item, value) unchanged and
does NOT remap IDs. It is useful when you want train/val/test to share the same
item and annotator pools, avoiding entity cold-start due to split definition.

By default, the script preserves original split sizes from `all_ratings`:
  n_train, n_val, n_test.
It then shuffles all rating rows and assigns new instance labels with those
counts (MCAR-style row split).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


RatingKey = Tuple[int, int, int]


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def rating_key(row: Dict[str, Any]) -> RatingKey:
    return (int(row["attribute"]), int(row["annotator"]), int(row["item"]))


def count_by_instance(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    out = {"train": 0, "val": 0, "test": 0}
    for r in rows:
        inst = str(r.get("instance", ""))
        if inst in out:
            out[inst] += 1
    return out


def items_by_instance(rows: Iterable[Dict[str, Any]]) -> Dict[str, set]:
    out = {"train": set(), "val": set(), "test": set()}
    for r in rows:
        inst = str(r.get("instance", ""))
        if inst in out:
            out[inst].add(int(r["item"]))
    return out


def annotators_by_instance(rows: Iterable[Dict[str, Any]]) -> Dict[str, set]:
    out = {"train": set(), "val": set(), "test": set()}
    for r in rows:
        inst = str(r.get("instance", ""))
        if inst in out:
            out[inst].add(int(r["annotator"]))
    return out


def recompute_stats(bundle: Dict[str, Any]) -> Dict[str, Any]:
    all_ratings = bundle.get("all_ratings", [])
    observed_ratings = bundle.get("observed_ratings", [])
    missing_ratings = bundle.get("missing_ratings", [])
    all_pairwise = bundle.get("all_pairwise", [])
    observed_pairwise = bundle.get("observed_pairwise", [])
    missing_pairwise = bundle.get("missing_pairwise", [])

    c_all = count_by_instance(all_ratings)
    c_obs = count_by_instance(observed_ratings)
    c_mis = count_by_instance(missing_ratings)
    c_pw_all = count_by_instance(all_pairwise)
    c_pw_obs = count_by_instance(observed_pairwise)
    c_pw_mis = count_by_instance(missing_pairwise)

    base = dict(bundle.get("stats", {}))
    base.update(
        {
            "total_ratings": len(all_ratings),
            "observed_ratings": len(observed_ratings),
            "missing_ratings": len(missing_ratings),
            "train_ratings": c_all["train"],
            "val_ratings": c_all["val"],
            "test_ratings": c_all["test"],
            "train_observed": c_obs["train"],
            "val_observed": c_obs["val"],
            "test_observed": c_obs["test"],
            "total_pairwise": len(all_pairwise),
            "observed_pairwise": len(observed_pairwise),
            "missing_pairwise": len(missing_pairwise),
            "train_pairwise": c_pw_all["train"],
            "val_pairwise": c_pw_all["val"],
            "test_pairwise": c_pw_all["test"],
            "observation_rate": len(observed_ratings) / len(all_ratings) if all_ratings else 0.0,
            "train_observation_rate": c_obs["train"] / c_all["train"] if c_all["train"] else 0.0,
            "val_observation_rate": c_obs["val"] / c_all["val"] if c_all["val"] else 0.0,
            "test_observation_rate": c_obs["test"] / c_all["test"] if c_all["test"] else 0.0,
            "split_mode": "row_mcar_shared_entities",
        }
    )
    return base


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

    all_ratings = list(bundle.get("all_ratings", []))
    if not all_ratings:
        raise ValueError("No all_ratings found in bundle.")

    # Preserve original split sizes, but randomize assignment over all rows.
    counts = count_by_instance(all_ratings)
    n_train, n_val, n_test = counts["train"], counts["val"], counts["test"]
    n_total = len(all_ratings)
    if n_train + n_val + n_test != n_total:
        raise ValueError("Found ratings with invalid instance labels.")

    rng = random.Random(args.seed)
    indices = list(range(n_total))
    rng.shuffle(indices)

    new_labels = [""] * n_total
    for idx in indices[:n_train]:
        new_labels[idx] = "train"
    for idx in indices[n_train : n_train + n_val]:
        new_labels[idx] = "val"
    for idx in indices[n_train + n_val :]:
        new_labels[idx] = "test"

    key_to_instance: Dict[RatingKey, str] = {}
    for idx, row in enumerate(all_ratings):
        k = rating_key(row)
        if k in key_to_instance:
            raise ValueError(f"Duplicate (attribute,annotator,item) key in all_ratings: {k}")
        key_to_instance[k] = new_labels[idx]

    def _rewrite_rating_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for row in rows:
            new_row = dict(row)
            new_row["instance"] = key_to_instance[rating_key(row)]
            out.append(new_row)
        return out

    new_bundle = dict(bundle)
    new_bundle["all_ratings"] = _rewrite_rating_rows(bundle.get("all_ratings", []))
    new_bundle["observed_ratings"] = _rewrite_rating_rows(bundle.get("observed_ratings", []))
    new_bundle["missing_ratings"] = _rewrite_rating_rows(bundle.get("missing_ratings", []))

    # Pairwise can remain unchanged for standalone rating training (often disabled),
    # but we still keep its split labels untouched for now.

    new_bundle["missing_ratings_indexes_in_test_instance"] = [
        i for i, row in enumerate(new_bundle["missing_ratings"]) if row.get("instance") == "test"
    ]

    new_bundle["stats"] = recompute_stats(new_bundle)

    new_cfg = dict(cfg)
    dg = dict(new_cfg.get("datagen", {}))
    dg["split_mode"] = "row_mcar_shared_entities"
    dg["resplit_seed"] = args.seed
    new_cfg["datagen"] = dg

    save_json(new_bundle, out_dir / "data_bundle.json")
    save_json(new_cfg, out_dir / "configs.json")
    save_json(
        {
            "seed": args.seed,
            "counts_preserved": {"train": n_train, "val": n_val, "test": n_test},
            "note": "Only instance labels were changed; (attribute, annotator, item, value) unchanged.",
        },
        out_dir / "resplit_meta.json",
    )

    stan_path = in_dir / "stan_data.json"
    if stan_path.exists():
        shutil.copy2(stan_path, out_dir / "stan_data.json")

    # Quick audit summary
    item_sets = items_by_instance(new_bundle["all_ratings"])
    ann_sets = annotators_by_instance(new_bundle["all_ratings"])
    print(f"Wrote {out_dir}")
    print(
        "Counts preserved: "
        f"train={n_train}, val={n_val}, test={n_test}; "
        f"item coverage | train={len(item_sets['train'])}, val={len(item_sets['val'])}, test={len(item_sets['test'])}; "
        f"annotator coverage | train={len(ann_sets['train'])}, val={len(ann_sets['val'])}, test={len(ann_sets['test'])}"
    )


if __name__ == "__main__":
    main()
