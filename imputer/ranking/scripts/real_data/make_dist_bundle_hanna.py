#!/usr/bin/env python3
"""
Build data_bundle.json for hanna_dist from hanna/data_bundle.json.

Adds 'rating_dist' (length-5 probability vector) to every rating record:
  - Human annotators (1-18) → one-hot of the integer rating value
  - GPT annotator (19)      → actual 5-class softmax distribution from
                              gpt-3.5-turbo-data-new.json

Usage (from Marformer repo root):
    python scripts/real_data/make_dist_bundle_hanna.py
"""

import copy
import json
import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
MARFORMER_ROOT = Path(__file__).resolve().parents[2]

BUNDLE_IN   = MARFORMER_ROOT / "OUTPUT/generated_data/hanna/data_bundle.json"
CONFIGS_IN  = MARFORMER_ROOT / "OUTPUT/generated_data/hanna/configs.json"
BUNDLE_OUT  = MARFORMER_ROOT / "OUTPUT/generated_data/hanna_dist/data_bundle.json"
CONFIGS_OUT = MARFORMER_ROOT / "OUTPUT/generated_data/hanna_dist/configs.json"

GPT_JSON    = Path("/Users/prabhavsingh/Documents/JHU/JHUResearch/Marformer/src/input/fixed/gpt-3.5-turbo-data-new.json")

GPT_ANN_ID  = 19   # 1-indexed
C           = 5
ATTRIBUTES  = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_item_index_to_str(all_item_ids_sorted_numerically: list[str]) -> dict[int, str]:
    """Map 1-indexed bundle item index → original JSON item_id string."""
    return {idx + 1: iid for idx, iid in enumerate(all_item_ids_sorted_numerically)}


def build_gpt_dist_lookup(gpt_data: dict, item_idx_to_str: dict[int, str]) -> dict[tuple, list]:
    lookup = {}
    for item_idx, item_id_str in item_idx_to_str.items():
        qs = gpt_data[item_id_str]
        for q_name in ATTRIBUTES:
            attr_idx = int(q_name[1:])   # Q1→1, ..., Q6→6
            lookup[(item_idx, attr_idx)] = qs[q_name]
    return lookup


def one_hot(value_1indexed: int, length: int = C) -> list:
    dist = [0.0] * length
    idx  = value_1indexed - 1
    if 0 <= idx < length:
        dist[idx] = 1.0
    return dist


def annotate_records(records: list, gpt_lookup: dict) -> list:
    annotated = []
    for rec in records:
        r = dict(rec)
        if r["annotator"] == GPT_ANN_ID:
            key = (r["item"], r["attribute"])
            r["rating_dist"] = gpt_lookup.get(key, one_hot(r["value"]))
        else:
            r["rating_dist"] = one_hot(r["value"])
        annotated.append(r)
    return annotated


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading HANNA bundle...")
    with open(BUNDLE_IN) as f:
        bundle = json.load(f)

    print("Loading GPT distributions...")
    with open(GPT_JSON) as f:
        gpt_data = json.load(f)

    # Reconstruct item ordering (same as convert_hanna.py)
    all_item_ids_sorted = sorted(gpt_data.keys(), key=int)
    item_idx_to_str = build_item_index_to_str(all_item_ids_sorted)

    gpt_lookup = build_gpt_dist_lookup(gpt_data, item_idx_to_str)
    print(f"  GPT dist entries: {len(gpt_lookup)}")

    dist_bundle = copy.deepcopy(bundle)
    dist_bundle["all_ratings"]      = annotate_records(bundle["all_ratings"],      gpt_lookup)
    dist_bundle["observed_ratings"] = annotate_records(bundle["observed_ratings"], gpt_lookup)
    dist_bundle["missing_ratings"]  = annotate_records(bundle["missing_ratings"],  gpt_lookup)

    # Sanity: every record has rating_dist
    for lst_name in ["all_ratings", "observed_ratings", "missing_ratings"]:
        n_missing = sum(1 for r in dist_bundle[lst_name] if "rating_dist" not in r)
        assert n_missing == 0, f"{n_missing} records in {lst_name} missing rating_dist"

    gpt_sample = next(
        (r for r in dist_bundle["observed_ratings"] if r["annotator"] == GPT_ANN_ID), None
    )
    if gpt_sample:
        print(f"  Sample GPT dist (item={gpt_sample['item']}, attr={gpt_sample['attribute']}): "
              f"{[round(p, 4) for p in gpt_sample['rating_dist']]}")

    BUNDLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(BUNDLE_OUT, "w") as f:
        json.dump(dist_bundle, f)
    print(f"\nSaved dist bundle → {BUNDLE_OUT}")

    shutil.copy(CONFIGS_IN, CONFIGS_OUT)
    print(f"Copied configs    → {CONFIGS_OUT}")

    total    = len(dist_bundle["all_ratings"])
    gpt_ct   = sum(1 for r in dist_bundle["all_ratings"] if r["annotator"] == GPT_ANN_ID)
    human_ct = total - gpt_ct
    print(f"\nRecords: {total} total  ({human_ct} human one-hot, {gpt_ct} GPT soft-dist)")


if __name__ == "__main__":
    main()
