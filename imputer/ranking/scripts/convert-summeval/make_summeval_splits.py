#!/usr/bin/env python3
"""
Generate SummEval data bundles with train/val/test splits and multiple training sizes.

Split (seed=42, by source doc to avoid leakage):
  - 80 train docs → 1280 train items (16 models × 80 docs)
  - 10 val  docs →  160 val  items  (same shuffle as original, docs 80-89)
  - 10 test docs →  160 test items  (docs 90-99)
  Train docs are identical to original convert_summeval.py.

Observed/missing split:
  observed_ratings : all turker ratings (train + val + test) + expert ratings (train only)
  missing_ratings  : expert ratings (val + test) — held-out ground truth

Outputs (all under DATA/SUMMEVAL/):
  ORIGINAL_SPLIT/              — full bundle (1280 + 160 + 160)
  SummEval_1600_8_4_{size}/    — size in {50, 100, 500, 750, 1000, 1280}

Training subsets are nested: size=50 ⊆ size=100 ⊆ ... ⊆ size=1280 (SUBSAMPLE_SEED=2024).
Subsampling is at item level (not doc level) within the 1280 train items.

NOTE: configs.json now includes K_val. utils.py in the Marformer codebase will need
      to be updated to handle K_val before running training on these bundles.

Usage (from repo root):
    python scripts/convert-summeval/make_summeval_splits.py
"""

import json
import random
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
RANKING_ROOT = Path(__file__).resolve().parents[2]
JSONL_PATH   = RANKING_ROOT / "DATA" / "OLD_DATA" / "summeval" / "model_annotations.aligned.jsonl"
OUTPUT_ROOT  = RANKING_ROOT / "DATA" / "SUMMEVAL"

# ── Constants ─────────────────────────────────────────────────────────────────
SPLIT_SEED     = 42      # controls doc-level train/val/test split — do not change
SUBSAMPLE_SEED = 2024    # controls nested training item subsets — do not change
N_TRAIN_DOCS   = 80
N_VAL_DOCS     = 10
N_TEST_DOCS    = 10
N_DOCS         = N_TRAIN_DOCS + N_VAL_DOCS + N_TEST_DOCS   # 100
ATTRIBUTES     = ["coherence", "consistency", "fluency", "relevance"]
N_EXPERTS      = 3
N_TURKERS      = 5
C              = 5
EXPERT_IDS     = [1, 2, 3]
TURKER_IDS     = [4, 5, 6, 7, 8]
J              = N_EXPERTS + N_TURKERS    # 8
TRAIN_SIZES    = [50, 100, 500, 750, 1000, 1280]

# Dataset-level identifiers used in folder names
TOTAL_ITEMS        = 1600
N_ANNOTATORS_NAME  = 8
N_ATTRIBUTES_NAME  = 4


# ── Rating helper ─────────────────────────────────────────────────────────────

def make_rating(attribute, annotator, item, value, instance):
    rating_dist = [0.0] * C
    rating_dist[value - 1] = 1.0
    return {
        "attribute":   attribute,
        "annotator":   annotator,
        "item":        item,
        "value":       value,
        "instance":    instance,
        "rating_dist": rating_dist,
    }


# ── Bundle builder ────────────────────────────────────────────────────────────

def build_bundle(records, item_map, instance_of):
    """
    Build data_bundle.json + configs.json from a list of records with
    pre-assigned item_map and instance_of mappings.

    observed_ratings: all turker ratings + expert ratings for train items.
    missing_ratings:  expert ratings for val + test items.
    """
    all_ratings      = []
    observed_ratings = []
    missing_ratings  = []

    for r in records:
        key = (r["id"], r["model_id"])
        if key not in item_map:
            continue
        item_id  = item_map[key]
        instance = instance_of[key]

        assert len(r["turker_annotations"]) == N_TURKERS
        assert len(r["expert_annotations"]) == N_EXPERTS

        # Turker ratings — always observed
        for slot_idx, ann in enumerate(r["turker_annotations"]):
            ann_id = TURKER_IDS[slot_idx]
            for attr_idx, attr_name in enumerate(ATTRIBUTES):
                value = int(ann[attr_name])
                assert 1 <= value <= C
                rec = make_rating(attr_idx + 1, ann_id, item_id, value, instance)
                all_ratings.append(rec)
                observed_ratings.append(rec)

        # Expert ratings — observed for train only, missing for val + test
        for exp_idx, ann in enumerate(r["expert_annotations"]):
            ann_id = EXPERT_IDS[exp_idx]
            for attr_idx, attr_name in enumerate(ATTRIBUTES):
                value = int(ann[attr_name])
                assert 1 <= value <= C
                rec = make_rating(attr_idx + 1, ann_id, item_id, value, instance)
                all_ratings.append(rec)
                if instance == "train":
                    observed_ratings.append(rec)
                else:
                    missing_ratings.append(rec)

    K_train = sum(1 for v in instance_of.values() if v == "train")
    K_val   = sum(1 for v in instance_of.values() if v == "val")
    K_test  = sum(1 for v in instance_of.values() if v == "test")

    # Sanity checks
    obs_set  = {(r["attribute"], r["annotator"], r["item"]) for r in observed_ratings}
    miss_set = {(r["attribute"], r["annotator"], r["item"]) for r in missing_ratings}
    assert obs_set & miss_set == set(), "observed ∩ missing is non-empty"

    exp_obs  = (K_train * N_TURKERS + K_train * N_EXPERTS
                + K_val  * N_TURKERS
                + K_test * N_TURKERS) * len(ATTRIBUTES)
    exp_miss = (K_val + K_test) * N_EXPERTS * len(ATTRIBUTES)
    assert len(observed_ratings) == exp_obs,  \
        f"Expected {exp_obs} observed, got {len(observed_ratings)}"
    assert len(missing_ratings)  == exp_miss, \
        f"Expected {exp_miss} missing, got {len(missing_ratings)}"

    bundle = {
        "embeddings":            None,
        "mean_preferences":      None,
        "annotator_preferences": None,
        "rating_probs":          None,
        "rating_cumprobs":       None,
        "rating_thresholds_z":   None,
        "base_scores":           None,
        "all_ratings":           all_ratings,
        "all_pairwise":          [],
        "observed_ratings":      observed_ratings,
        "missing_ratings":       missing_ratings,
        "observed_pairwise":     [],
        "missing_pairwise":      [],
        "stats": {
            "K":                K_train + K_val + K_test,
            "K_train":          K_train,
            "K_val":            K_val,
            "K_test":           K_test,
            "I":                len(ATTRIBUTES),
            "J":                J,
            "C":                C,
            "total_ratings":    len(all_ratings),
            "observed_ratings": len(observed_ratings),
            "missing_ratings":  len(missing_ratings),
        },
    }
    configs = {
        "datagen": {
            "K_train": K_train,
            "K_val":   K_val,
            "K_test":  K_test,
            "I":       len(ATTRIBUTES),
            "J":       J,
            "C":       C,
        }
    }
    return bundle, configs


# ── Save ──────────────────────────────────────────────────────────────────────

def save_bundle(bundle, configs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "data_bundle.json", "w") as f:
        json.dump(bundle, f)
    with open(out_dir / "configs.json", "w") as f:
        json.dump(configs, f, indent=2)
    dg = configs["datagen"]
    print(f"  Saved → {out_dir.name}  "
          f"(train={dg['K_train']}, val={dg['K_val']}, test={dg['K_test']}, "
          f"obs={bundle['stats']['observed_ratings']}, miss={bundle['stats']['missing_ratings']})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"SummEval JSONL not found: {JSONL_PATH}")

    print("Loading SummEval JSONL...")
    records = [json.loads(line) for line in open(JSONL_PATH)]
    print(f"  {len(records)} records loaded")

    source_doc_ids = sorted(set(r["id"] for r in records))
    assert len(source_doc_ids) == N_DOCS, \
        f"Expected {N_DOCS} source docs, got {len(source_doc_ids)}"

    # ── Doc-level split ───────────────────────────────────────────────────────
    # Same shuffle as original convert_summeval.py → train docs are identical.
    rng = random.Random(SPLIT_SEED)
    shuffled = list(source_doc_ids)
    rng.shuffle(shuffled)
    train_docs = set(shuffled[:N_TRAIN_DOCS])
    val_docs   = set(shuffled[N_TRAIN_DOCS : N_TRAIN_DOCS + N_VAL_DOCS])
    test_docs  = set(shuffled[N_TRAIN_DOCS + N_VAL_DOCS :])
    print(f"  train={len(train_docs)} docs, val={len(val_docs)} docs, test={len(test_docs)} docs")

    # ── Sort records; assign item IDs (train first, val next, test last) ──────
    sorted_records = sorted(records, key=lambda r: (r["id"], r["model_id"]))
    train_records  = [r for r in sorted_records if r["id"] in train_docs]
    val_records    = [r for r in sorted_records if r["id"] in val_docs]
    test_records   = [r for r in sorted_records if r["id"] in test_docs]

    K_train_full = len(train_records)
    K_val_full   = len(val_records)
    K_test_full  = len(test_records)
    print(f"  train={K_train_full} items, val={K_val_full} items, test={K_test_full} items")

    def make_item_map_and_instance(subset_train, val_recs, test_recs):
        item_map    = {}
        instance_of = {}
        for idx, r in enumerate(subset_train):
            key = (r["id"], r["model_id"])
            item_map[key]    = idx + 1
            instance_of[key] = "train"
        offset = len(subset_train)
        for idx, r in enumerate(val_recs):
            key = (r["id"], r["model_id"])
            item_map[key]    = offset + idx + 1
            instance_of[key] = "val"
        offset += len(val_recs)
        for idx, r in enumerate(test_recs):
            key = (r["id"], r["model_id"])
            item_map[key]    = offset + idx + 1
            instance_of[key] = "test"
        return item_map, instance_of

    # ── Nested training subset index order ────────────────────────────────────
    rng_sub      = np.random.default_rng(SUBSAMPLE_SEED)
    perm_indices = rng_sub.permutation(K_train_full).tolist()
    # perm_indices[i] is the index into train_records for the i-th item in subset order.
    # For size N: use perm_indices[:N] (sorted by original position for deterministic IDs).

    # ── ORIGINAL_SPLIT ────────────────────────────────────────────────────────
    print("\nGenerating ORIGINAL_SPLIT...")
    item_map, instance_of = make_item_map_and_instance(
        train_records, val_records, test_records
    )
    bundle, configs = build_bundle(
        sorted_records, item_map, instance_of
    )
    save_bundle(bundle, configs, OUTPUT_ROOT / "ORIGINAL_SPLIT")

    # ── Sized subsets ─────────────────────────────────────────────────────────
    print("\nGenerating sized subsets (nested, SUBSAMPLE_SEED=2024)...")
    for size in sorted(TRAIN_SIZES):
        subset_indices = sorted(perm_indices[:size])
        subset_train   = [train_records[i] for i in subset_indices]
        all_subset     = subset_train + val_records + test_records

        item_map, instance_of = make_item_map_and_instance(
            subset_train, val_records, test_records
        )
        bundle, configs = build_bundle(all_subset, item_map, instance_of)

        folder_name = (f"SummEval_{TOTAL_ITEMS}_{N_ANNOTATORS_NAME}"
                       f"_{N_ATTRIBUTES_NAME}_{size}")
        save_bundle(bundle, configs, OUTPUT_ROOT / folder_name)

    print(f"\nAll done. Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
