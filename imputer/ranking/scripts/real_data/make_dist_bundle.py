#!/usr/bin/env python3
"""
Build data_bundle_dist.json from the converted data_bundle.json.

Adds a 'rating_dist' field to every rating record:
  - Human annotators  → one-hot distribution of length C over value (1-indexed)
  - LLM annotator (25)→ actual 4-class probability vector from the original TSV

The resulting data_bundle_dist.json is saved alongside configs.json in
OUTPUT/generated_data/llm_rubric_dist/ so run_imputer.py can use it as-is.

Usage (from Marformer repo root):
    python scripts/real_data/make_dist_bundle.py
"""

import copy
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
MARFORMER_ROOT = Path(__file__).resolve().parents[2]

BUNDLE_IN   = MARFORMER_ROOT / "OUTPUT/generated_data/llm_rubric/data_bundle.json"
CONFIGS_IN  = MARFORMER_ROOT / "OUTPUT/generated_data/llm_rubric/configs.json"
BUNDLE_OUT  = MARFORMER_ROOT / "OUTPUT/generated_data/llm_rubric_dist/data_bundle.json"
CONFIGS_OUT = MARFORMER_ROOT / "OUTPUT/generated_data/llm_rubric_dist/configs.json"

# Original raw TSVs (stay in RealData folder — not moved)
HUMAN_TSV = Path("/Users/prabhavsingh/Documents/JHU/JHUResearch/RealData/imputer/ranking/OUTPUT/real_data/human_judges_real_convs_FIXED_ANON.tsv")
LLM_TSV   = Path("/Users/prabhavsingh/Documents/JHU/JHUResearch/RealData/imputer/ranking/OUTPUT/real_data/gpt-3.5-turbo-16k_real_evaluations_FIXED.tsv")

RANDOM_SEED = 42
N_TEST      = 25
C           = 4          # number of rating categories (1..4)
LLM_ANN_ID  = 25         # 1-indexed annotator ID for the LLM in the bundle
PROB_COLS   = ["answer1_prob", "answer2_prob", "answer3_prob", "answer4_prob"]


# ── Rebuild item_map (same logic as convert.py) ───────────────────────────────

def build_item_map(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> dict:
    """Reconstruct {text_id: 1-based item index} identically to convert.py."""
    llm_ids  = set(llm_df["text_id"].unique())
    all_ids  = sorted(set(human_df["text_id"].unique()) & llm_ids)

    rng      = np.random.default_rng(RANDOM_SEED)
    test_ids = set(rng.choice(all_ids, size=N_TEST, replace=False).tolist())

    train_sorted = sorted(tid for tid in all_ids if tid not in test_ids)
    test_sorted  = sorted(tid for tid in all_ids if tid in test_ids)
    ordered      = train_sorted + test_sorted
    return {tid: idx + 1 for idx, tid in enumerate(ordered)}


# ── Build LLM distribution lookup ────────────────────────────────────────────

def build_llm_dist_lookup(llm_df: pd.DataFrame, item_map: dict) -> dict:
    """Return {(item_id, attribute_id): [p1, p2, p3, p4]} for the LLM annotator."""
    lookup = {}
    for _, row in llm_df.iterrows():
        text_id   = row["text_id"]
        if text_id not in item_map:
            continue
        item_id   = item_map[text_id]
        attr_id   = int(row["criterion"][1:]) + 1   # Q0→1, ..., Q8→9
        probs     = [float(row[c]) for c in PROB_COLS]
        lookup[(item_id, attr_id)] = probs
    return lookup


# ── One-hot helper ────────────────────────────────────────────────────────────

def one_hot(value_1indexed: int, length: int = C) -> list:
    """Return a one-hot list of given length at position value_1indexed - 1."""
    dist = [0.0] * length
    idx  = value_1indexed - 1
    if 0 <= idx < length:
        dist[idx] = 1.0
    return dist


# ── Add rating_dist to a list of rating records ───────────────────────────────

def annotate_records(records: list, llm_lookup: dict) -> list:
    """Return a new list of records with 'rating_dist' added."""
    annotated = []
    for rec in records:
        r = dict(rec)   # shallow copy — safe since values are scalars/strings
        ann_id  = r["annotator"]
        item_id = r["item"]
        attr_id = r["attribute"]
        value   = r["value"]

        if ann_id == LLM_ANN_ID:
            key = (item_id, attr_id)
            if key in llm_lookup:
                r["rating_dist"] = llm_lookup[key]
            else:
                # Fallback: one-hot from argmax (shouldn't happen for covered items)
                r["rating_dist"] = one_hot(value)
        else:
            # Human annotator: one-hot
            r["rating_dist"] = one_hot(value)

        annotated.append(r)
    return annotated


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    with open(BUNDLE_IN) as f:
        bundle = json.load(f)

    human_df = pd.read_csv(HUMAN_TSV, sep="\t")
    human_df = human_df.drop(columns=[c for c in human_df.columns if c.startswith("DQ")])
    human_df = human_df.drop_duplicates(subset=["text_id", "annotator_id"], keep="first")

    llm_df   = pd.read_csv(LLM_TSV, sep="\t")
    q_cols   = [f"Q{i}" for i in range(9)]
    llm_df   = llm_df[llm_df["criterion"].isin(q_cols)].copy()

    item_map    = build_item_map(human_df, llm_df)
    llm_lookup  = build_llm_dist_lookup(llm_df, item_map)

    print(f"  Items in map:       {len(item_map)}")
    print(f"  LLM dist entries:   {len(llm_lookup)}")

    # Annotate all rating lists
    dist_bundle = copy.deepcopy(bundle)
    dist_bundle["all_ratings"]      = annotate_records(bundle["all_ratings"],      llm_lookup)
    dist_bundle["observed_ratings"] = annotate_records(bundle["observed_ratings"], llm_lookup)
    dist_bundle["missing_ratings"]  = annotate_records(bundle["missing_ratings"],  llm_lookup)

    # Sanity check: every record now has rating_dist
    for lst_name in ["all_ratings", "observed_ratings", "missing_ratings"]:
        missing = sum(1 for r in dist_bundle[lst_name] if "rating_dist" not in r)
        assert missing == 0, f"{missing} records in {lst_name} missing rating_dist"

    # Check a sample LLM record has non-one-hot dist
    llm_sample = next(
        (r for r in dist_bundle["observed_ratings"] if r["annotator"] == LLM_ANN_ID), None
    )
    if llm_sample:
        print(f"  Sample LLM dist:  {llm_sample['rating_dist']}  (item={llm_sample['item']}, attr={llm_sample['attribute']})")

    # Save
    BUNDLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(BUNDLE_OUT, "w") as f:
        json.dump(dist_bundle, f)
    print(f"\nSaved dist bundle → {BUNDLE_OUT}")

    # Copy configs.json to dist dir
    import shutil
    shutil.copy(CONFIGS_IN, CONFIGS_OUT)
    print(f"Copied configs     → {CONFIGS_OUT}")

    total   = len(dist_bundle["all_ratings"])
    llm_ct  = sum(1 for r in dist_bundle["all_ratings"] if r["annotator"] == LLM_ANN_ID)
    human_ct = total - llm_ct
    print(f"\nRecords: {total} total  ({human_ct} human one-hot, {llm_ct} LLM soft)")


if __name__ == "__main__":
    main()
