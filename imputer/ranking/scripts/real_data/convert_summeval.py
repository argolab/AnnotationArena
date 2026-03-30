#!/usr/bin/env python3
"""
Convert model_annotations.aligned.jsonl (SummEval) into data_bundle.json + configs.json.

Dataset: SummEval
  Items: 1600 summaries = 100 source docs x 16 summarization models
  Attributes: 4 (coherence, consistency, fluency, relevance), Likert 1-5
  Annotators (J=8):
    Experts  1-3: same 3 people rated all 1600 summaries (IDs consistent across items)
    Turkers  4-8: positional slots 0-4 per summary (different AMT workers per item)

Train/test split (seed=42):
  80 source docs → 1280 train items (80 docs x 16 models)
  20 source docs →  320 test  items (20 docs x 16 models)
  Split is by source doc to avoid leakage across model variants of the same source.

Bundle split:
  observed_ratings: all turker ratings (train+test) + expert ratings (train only)
  missing_ratings:  expert ratings (test only) — held-out ground truth

Training note:
  No LLM annotator ID → MCAR masking at training time.
  Turker ratings in train will occasionally be masked (15% rate) alongside experts.
  This is acceptable: model still learns to predict expert ratings from turker context.

Usage (from repo root):
    python scripts/real_data/convert_summeval.py
"""

import json
import random
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
MARFORMER_ROOT = Path(__file__).resolve().parents[2]
JSONL_PATH     = MARFORMER_ROOT / "OUTPUT/generated_data/model_annotations.aligned.jsonl"
OUTPUT_DIR     = MARFORMER_ROOT / "OUTPUT/generated_data/summeval"

# ── Constants ──────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
N_TRAIN_DOCS   = 80
N_TEST_DOCS    = 20
N_DOCS         = N_TRAIN_DOCS + N_TEST_DOCS   # 100
ATTRIBUTES     = ["coherence", "consistency", "fluency", "relevance"]
N_EXPERTS      = 3
N_TURKERS      = 5
C              = 5  # Likert scale 1-5

# Annotator IDs (1-indexed, matching HANNA convention)
EXPERT_IDS = [1, 2, 3]        # expert slots 0-2 → annotator IDs 1-3
TURKER_IDS = [4, 5, 6, 7, 8]  # turker slots 0-4 → annotator IDs 4-8
J = N_EXPERTS + N_TURKERS      # 8 annotators total


def make_rating(attribute: int, annotator: int, item: int, value: int, instance: str) -> dict:
    # One-hot rating_dist (same convention as LLM-Rubric human annotations).
    # When --llm-input-dist is passed, build_param checks _is_one_hot and falls
    # through to hard-label encoding — so loss is identical to not having rating_dist.
    # Including it keeps the bundle format consistent with llm_rubric_dist.
    rating_dist = [0.0] * C
    rating_dist[value - 1] = 1.0  # value is 1-indexed; dist is 0-indexed
    return {
        "attribute":   attribute,
        "annotator":   annotator,
        "item":        item,
        "value":       value,
        "instance":    instance,
        "rating_dist": rating_dist,
    }


def main():
    print("Loading SummEval JSONL...")
    records = [json.loads(line) for line in open(JSONL_PATH)]
    print(f"  {len(records)} records loaded")

    # ── Source doc split ───────────────────────────────────────────────────────
    source_doc_ids = sorted(set(r["id"] for r in records))
    assert len(source_doc_ids) == N_DOCS, f"Expected {N_DOCS} source docs, got {len(source_doc_ids)}"

    rng = random.Random(RANDOM_SEED)
    shuffled_docs = list(source_doc_ids)
    rng.shuffle(shuffled_docs)
    train_docs = set(shuffled_docs[:N_TRAIN_DOCS])
    test_docs  = set(shuffled_docs[N_TRAIN_DOCS:])

    # ── Item ID assignment ─────────────────────────────────────────────────────
    # Sort deterministically: (doc_id, model_id); train items first, then test.
    sorted_records = sorted(records, key=lambda r: (r["id"], r["model_id"]))
    train_records  = [r for r in sorted_records if r["id"] in train_docs]
    test_records   = [r for r in sorted_records if r["id"] in test_docs]

    item_map: dict[tuple, int] = {}
    for idx, r in enumerate(train_records):
        item_map[(r["id"], r["model_id"])] = idx + 1
    offset = len(train_records)
    for idx, r in enumerate(test_records):
        item_map[(r["id"], r["model_id"])] = offset + idx + 1

    K_train = len(train_records)
    K_test  = len(test_records)
    K_total = K_train + K_test

    print(f"  Train: {K_train} items ({N_TRAIN_DOCS} docs x {K_train // N_TRAIN_DOCS} models)")
    print(f"  Test:  {K_test} items  ({N_TEST_DOCS} docs x {K_test // N_TEST_DOCS} models)")
    print(f"  Attributes: {len(ATTRIBUTES)}  ({', '.join(ATTRIBUTES)})")
    print(f"  Annotators: {J}  (experts {EXPERT_IDS}, turker slots {TURKER_IDS})")

    # ── Build rating lists ─────────────────────────────────────────────────────
    all_ratings:      list[dict] = []
    observed_ratings: list[dict] = []
    missing_ratings:  list[dict] = []

    for r in sorted_records:
        item_id  = item_map[(r["id"], r["model_id"])]
        instance = "train" if r["id"] in train_docs else "test"

        assert len(r["turker_annotations"]) == N_TURKERS, \
            f"Expected {N_TURKERS} turker annotations, got {len(r['turker_annotations'])}"
        assert len(r["expert_annotations"]) == N_EXPERTS, \
            f"Expected {N_EXPERTS} expert annotations, got {len(r['expert_annotations'])}"

        # Turker ratings — always observed (train and test)
        for slot_idx, ann in enumerate(r["turker_annotations"]):
            ann_id = TURKER_IDS[slot_idx]
            for attr_idx, attr_name in enumerate(ATTRIBUTES):
                value = int(ann[attr_name])
                assert 1 <= value <= C, f"Turker value {value} out of range"
                rec = make_rating(attr_idx + 1, ann_id, item_id, value, instance)
                all_ratings.append(rec)
                observed_ratings.append(rec)

        # Expert ratings — observed for train, missing (held-out) for test
        for exp_idx, ann in enumerate(r["expert_annotations"]):
            ann_id = EXPERT_IDS[exp_idx]
            for attr_idx, attr_name in enumerate(ATTRIBUTES):
                value = int(ann[attr_name])
                assert 1 <= value <= C, f"Expert value {value} out of range"
                rec = make_rating(attr_idx + 1, ann_id, item_id, value, instance)
                all_ratings.append(rec)
                if instance == "train":
                    observed_ratings.append(rec)
                else:
                    missing_ratings.append(rec)

    # ── Sanity checks ──────────────────────────────────────────────────────────
    expected_total = K_total * (N_TURKERS + N_EXPERTS) * len(ATTRIBUTES)
    assert len(all_ratings) == expected_total, \
        f"Expected {expected_total} total ratings, got {len(all_ratings)}"

    expected_obs = (K_total * N_TURKERS + K_train * N_EXPERTS) * len(ATTRIBUTES)
    assert len(observed_ratings) == expected_obs, \
        f"Expected {expected_obs} observed, got {len(observed_ratings)}"

    expected_miss = K_test * N_EXPERTS * len(ATTRIBUTES)
    assert len(missing_ratings) == expected_miss, \
        f"Expected {expected_miss} missing, got {len(missing_ratings)}"

    obs_set  = {(r["attribute"], r["annotator"], r["item"]) for r in observed_ratings}
    miss_set = {(r["attribute"], r["annotator"], r["item"]) for r in missing_ratings}
    all_set  = {(r["attribute"], r["annotator"], r["item"]) for r in all_ratings}
    assert obs_set | miss_set == all_set, "observed ∪ missing ≠ all_ratings"
    assert obs_set & miss_set == set(),   "observed ∩ missing is non-empty"

    print(f"\n  all_ratings:      {len(all_ratings)}")
    print(f"  observed_ratings: {len(observed_ratings)}  (all turkers + train experts)")
    print(f"  missing_ratings:  {len(missing_ratings)}   (test experts, held-out)")

    # ── Assemble bundle ────────────────────────────────────────────────────────
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
            "K":                K_total,
            "K_train":          K_train,
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
            "K_test":  K_test,
            "I":       len(ATTRIBUTES),
            "J":       J,
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
