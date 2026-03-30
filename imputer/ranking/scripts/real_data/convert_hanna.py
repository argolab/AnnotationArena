#!/usr/bin/env python3
"""
Convert human-data-new.json + gpt-3.5-turbo-data-new.json (HANNA dataset) into
data_bundle.json + configs.json for use with run_imputer.py.

Dataset: HANNA
  K = 1200 items (0-1199 in source, re-mapped to 1-indexed)
  I = 6 attributes (Q1-Q6; Q0 excluded as it is a derived average)
  J = 19 annotators (1-18 human, 19 = GPT-3.5-turbo)
  C = 5 rating categories (1-5 scale)

Train/test split (seed=42):
  - Items sorted numerically, first 1100 = train, last 100 = test
  - observed_ratings: train (human+LLM) + test (LLM only)   [instance='train'/'test']
  - missing_ratings:  test (human only)                      [instance='test']

GPT scalar value: argmax(dist) + 1  (mode, 1-indexed)

Usage (from Marformer repo root):
    python scripts/real_data/convert_hanna.py
"""

import json
import random
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
MARFORMER_ROOT = Path(__file__).resolve().parents[2]

# Raw HANNA JSONs now live in the repo under real-data/hanna/
HUMAN_JSON = MARFORMER_ROOT / "real-data/hanna/human-data-new.json"
GPT_JSON   = MARFORMER_ROOT / "real-data/hanna/gpt-3.5-turbo-data-new.json"

OUTPUT_DIR = MARFORMER_ROOT / "OUTPUT/generated_data/hanna"

# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
N_TRAIN      = 1100
N_TEST       = 100
ATTRIBUTES   = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]  # Q0 excluded
GPT_ANN_ID   = 19   # 1-indexed annotator ID for GPT in bundle
C            = 5


def build_item_map(all_item_ids: list[str]) -> dict[str, int]:
    """
    Sort item IDs numerically, first N_TRAIN → train (1..N_TRAIN),
    last N_TEST → test (N_TRAIN+1..N_TRAIN+N_TEST).
    Returns {item_id_str: 1-indexed_int}.
    """
    sorted_ids = sorted(all_item_ids, key=int)
    assert len(sorted_ids) == N_TRAIN + N_TEST, f"Expected {N_TRAIN+N_TEST} items, got {len(sorted_ids)}"
    return {iid: idx + 1 for idx, iid in enumerate(sorted_ids)}


def gpt_mode(dist: list[float]) -> int:
    """Return argmax(dist) + 1 (1-indexed mode)."""
    return int(max(range(len(dist)), key=lambda i: dist[i])) + 1


def make_rating(attribute: int, annotator: int, item: int, value: int, instance: str) -> dict:
    return {
        "attribute": attribute,
        "annotator": annotator,
        "item": item,
        "value": value,
        "instance": instance,
    }


def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading HANNA data...")
    with open(HUMAN_JSON) as f:
        human = json.load(f)
    with open(GPT_JSON) as f:
        gpt = json.load(f)

    all_item_ids = list(human.keys())
    assert set(all_item_ids) == set(gpt.keys()), "Human and GPT item sets differ"

    # ── Build item map ─────────────────────────────────────────────────────────
    item_map = build_item_map(all_item_ids)
    train_item_ids = {iid for iid, idx in item_map.items() if idx <= N_TRAIN}
    test_item_ids  = {iid for iid, idx in item_map.items() if idx > N_TRAIN}

    print(f"  Items: {len(all_item_ids)} total  ({len(train_item_ids)} train, {len(test_item_ids)} test)")

    # ── Build annotator map: human string IDs '1'-'18' → int 1-18 ─────────────
    all_human_annotators = sorted(
        {ann_id for ann_dict in human.values() for ann_id in ann_dict.keys()},
        key=int
    )
    ann_map = {ann_id: int(ann_id) for ann_id in all_human_annotators}  # '1'→1, ..., '18'→18
    print(f"  Human annotators: {len(ann_map)}  ({sorted(ann_map.values())})")
    print(f"  GPT annotator ID: {GPT_ANN_ID}")

    # ── Build rating lists ─────────────────────────────────────────────────────
    all_ratings:      list[dict] = []
    observed_ratings: list[dict] = []
    missing_ratings:  list[dict] = []

    for item_id in sorted(all_item_ids, key=int):
        item_idx = item_map[item_id]
        instance = "train" if item_id in train_item_ids else "test"

        # -- Human ratings --
        ann_dict = human[item_id]
        assert len(ann_dict) == 1, f"Expected 1 annotator per item, got {len(ann_dict)} for item {item_id}"
        for ann_id_str, qs in ann_dict.items():
            ann_idx = ann_map[ann_id_str]
            for q_name in ATTRIBUTES:
                value = int(qs[q_name])
                assert 1 <= value <= C, f"Invalid value {value} for {item_id}/{ann_id_str}/{q_name}"
                attr_idx = int(q_name[1:])   # Q1→1, ..., Q6→6
                rec = make_rating(attr_idx, ann_idx, item_idx, value, instance)
                all_ratings.append(rec)
                if instance == "train":
                    observed_ratings.append(rec)
                else:
                    missing_ratings.append(rec)   # held-out test human

        # -- GPT ratings --
        gpt_qs = gpt[item_id]
        for q_name in ATTRIBUTES:
            dist = gpt_qs[q_name]
            assert len(dist) == C, f"GPT dist length {len(dist)} != {C} for {item_id}/{q_name}"
            value = gpt_mode(dist)
            attr_idx = int(q_name[1:])
            rec = make_rating(attr_idx, GPT_ANN_ID, item_idx, value, instance)
            all_ratings.append(rec)
            observed_ratings.append(rec)   # LLM always in observed (train+test)

    # ── Sanity checks ──────────────────────────────────────────────────────────
    expected_total = (N_TRAIN + N_TEST) * (1 + 1) * len(ATTRIBUTES)   # 1 human + 1 LLM per item
    assert len(all_ratings) == expected_total, f"Expected {expected_total} total, got {len(all_ratings)}"

    expected_obs = N_TRAIN * 2 * len(ATTRIBUTES) + N_TEST * 1 * len(ATTRIBUTES)  # train h+LLM + test LLM
    assert len(observed_ratings) == expected_obs, f"Expected {expected_obs} observed, got {len(observed_ratings)}"

    expected_miss = N_TEST * 1 * len(ATTRIBUTES)   # test human only
    assert len(missing_ratings) == expected_miss, f"Expected {expected_miss} missing, got {len(missing_ratings)}"

    obs_set  = {(r["attribute"], r["annotator"], r["item"]) for r in observed_ratings}
    miss_set = {(r["attribute"], r["annotator"], r["item"]) for r in missing_ratings}
    all_set  = {(r["attribute"], r["annotator"], r["item"]) for r in all_ratings}
    assert obs_set | miss_set == all_set, "observed ∪ missing ≠ all_ratings"
    assert obs_set & miss_set == set(),   "observed ∩ missing is non-empty"

    print(f"\n  all_ratings:      {len(all_ratings)}")
    print(f"  observed_ratings: {len(observed_ratings)}  (train human+LLM + test LLM)")
    print(f"  missing_ratings:  {len(missing_ratings)}   (test human, held-out ground truth)")

    # ── Assemble bundle ────────────────────────────────────────────────────────
    bundle = {
        "embeddings":             None,
        "mean_preferences":       None,
        "annotator_preferences":  None,
        "rating_probs":           None,
        "rating_cumprobs":        None,
        "rating_thresholds_z":    None,
        "base_scores":            None,
        "all_ratings":      all_ratings,
        "all_pairwise":     [],
        "observed_ratings": observed_ratings,
        "missing_ratings":  missing_ratings,
        "observed_pairwise": [],
        "missing_pairwise":  [],
        "stats": {
            "K":            N_TRAIN + N_TEST,
            "K_train":      N_TRAIN,
            "K_test":       N_TEST,
            "I":            len(ATTRIBUTES),
            "J":            len(ann_map) + 1,   # humans + GPT
            "C":            C,
            "total_ratings":          len(all_ratings),
            "train_ratings":          N_TRAIN * 2 * len(ATTRIBUTES),
            "test_ratings":           N_TEST  * 2 * len(ATTRIBUTES),
            "observed_ratings":       len(observed_ratings),
            "missing_ratings":        len(missing_ratings),
            "train_observation_rate": 1.0,
        },
    }

    # ── Configs ────────────────────────────────────────────────────────────────
    configs = {
        "datagen": {
            "K_train": N_TRAIN,
            "K_test":  N_TEST,
            "I":       len(ATTRIBUTES),
            "J":       len(ann_map) + 1,
            "C":       C,
        }
    }

    # ── Save ───────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle_path  = OUTPUT_DIR / "data_bundle.json"
    configs_path = OUTPUT_DIR / "configs.json"

    with open(bundle_path, "w") as f:
        json.dump(bundle, f)
    with open(configs_path, "w") as f:
        json.dump(configs, f, indent=2)

    print(f"\nSaved → {bundle_path}")
    print(f"Saved → {configs_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
