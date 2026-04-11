#!/usr/bin/env python3
"""
Generate additional LLM-Rubric training-size splits for finer granularity.

New sizes: 20, 30, 40, 75, 125
Same seeds and logic as make_llmrubric_splits.py — subsets are nested within
the existing shuffled order (SUBSAMPLE_SEED=2024). Val/test sets are identical.

Existing folders are NEVER touched — skipped if already present.

Usage (from repo root):
    python scripts/convert-llm-rubric/make_llmrubric_granular.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
RANKING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR   = Path(__file__).resolve().parent

HUMAN_TSV   = SCRIPT_DIR / "human_judges_real_convs_FIXED_ANON.tsv"
LLM_TSV     = SCRIPT_DIR / "gpt-3.5-turbo-16k_real_evaluations_FIXED.tsv"
OUTPUT_ROOT = RANKING_ROOT / "DATA" / "LLM_RUBRIC"

# ── Constants — must match make_llmrubric_splits.py exactly ──────────────────
SPLIT_SEED     = 42
SUBSAMPLE_SEED = 2024
N_VAL          = 25
N_TEST         = 25
C              = 4
Q_COLS         = [f"Q{i}" for i in range(9)]
PROB_COLS      = ["answer1_prob", "answer2_prob", "answer3_prob", "answer4_prob"]

TOTAL_ITEMS  = 225
N_ANNOTATORS = 25
N_ATTRIBUTES = 9

NEW_SIZES = [20, 30, 40, 75, 125]


# ── Reuse build logic from make_llmrubric_splits.py ──────────────────────────

def load_tsvs():
    human_df = pd.read_csv(HUMAN_TSV, sep="\t")
    human_df = human_df.drop(columns=[c for c in human_df.columns if c.startswith("DQ")])
    human_df = human_df.drop_duplicates(subset=["text_id", "annotator_id"], keep="first")
    llm_df   = pd.read_csv(LLM_TSV, sep="\t")
    llm_df   = llm_df[llm_df["criterion"].isin(Q_COLS)].copy()
    return human_df, llm_df


def build_splits(human_df, llm_df):
    llm_ids  = set(llm_df["text_id"].unique())
    all_ids  = sorted(set(human_df["text_id"].unique()) & llm_ids)
    rng      = np.random.default_rng(SPLIT_SEED)
    test_ids = set(rng.choice(all_ids, size=N_TEST, replace=False).tolist())
    remaining = sorted(tid for tid in all_ids if tid not in test_ids)
    val_ids  = set(rng.choice(remaining, size=N_VAL, replace=False).tolist())
    train_ids = sorted(
        tid for tid in all_ids if tid not in test_ids and tid not in val_ids
    )
    return sorted(test_ids), sorted(val_ids), train_ids


def build_llm_dist_lookup(llm_df):
    lookup = {}
    for _, row in llm_df.iterrows():
        attr_id = int(row["criterion"][1:]) + 1
        probs   = [float(row[c]) for c in PROB_COLS]
        lookup[(row["text_id"], attr_id)] = probs
    return lookup


def get_llm_annotator_id(human_df):
    return int(human_df["annotator_id"].max()) + 2


def build_bundle(human_df, llm_df, llm_lookup, llm_ann_id,
                 train_text_ids, val_text_ids, test_text_ids):
    train_sorted = sorted(train_text_ids)
    val_sorted   = sorted(val_text_ids)
    test_sorted  = sorted(test_text_ids)

    ordered  = train_sorted + val_sorted + test_sorted
    item_map = {tid: idx + 1 for idx, tid in enumerate(ordered)}
    instance_of = (
        {tid: "train" for tid in train_sorted}
        | {tid: "val"   for tid in val_sorted}
        | {tid: "test"  for tid in test_sorted}
    )
    all_ids_set = set(ordered)
    K_train = len(train_sorted)
    K_val   = len(val_sorted)
    K_test  = len(test_sorted)

    all_ratings = []
    observed_ratings = []
    missing_ratings  = []

    for _, row in human_df.iterrows():
        text_id = row["text_id"]
        if text_id not in all_ids_set:
            continue
        item_id  = item_map[text_id]
        instance = instance_of[text_id]
        ann_id   = int(row["annotator_id"]) + 1
        for q_col in Q_COLS:
            attr_id = int(q_col[1:]) + 1
            raw_val = row[q_col]
            raw_int = 0 if pd.isna(raw_val) else int(raw_val)
            value   = raw_int if raw_int != 0 else 1
            one_hot = [0.0] * C
            one_hot[value - 1] = 1.0
            rec = {
                "attribute":   attr_id,
                "annotator":   ann_id,
                "item":        item_map[text_id],
                "value":       value,
                "instance":    instance,
                "rating_dist": one_hot,
            }
            all_ratings.append(rec)
            if instance == "train":
                observed_ratings.append(rec)
            else:
                missing_ratings.append(rec)

    for _, row in llm_df.iterrows():
        text_id = row["text_id"]
        if text_id not in all_ids_set:
            continue
        item_id  = item_map[text_id]
        instance = instance_of[text_id]
        attr_id  = int(row["criterion"][1:]) + 1
        probs = llm_lookup.get((text_id, attr_id))
        if probs is None:
            value = 1
            probs = [0.0] * C
            probs[0] = 1.0
        else:
            value = int(np.argmax(probs)) + 1
        rec = {
            "attribute":   attr_id,
            "annotator":   llm_ann_id,
            "item":        item_id,
            "value":       value,
            "instance":    instance,
            "rating_dist": probs,
        }
        all_ratings.append(rec)
        observed_ratings.append(rec)

    obs_set  = {(r["attribute"], r["annotator"], r["item"]) for r in observed_ratings}
    miss_set = {(r["attribute"], r["annotator"], r["item"]) for r in missing_ratings}
    assert obs_set & miss_set == set(), "observed ∩ missing is non-empty"

    bundle = {
        "embeddings": None, "mean_preferences": None,
        "annotator_preferences": None, "rating_probs": None,
        "rating_cumprobs": None, "rating_thresholds_z": None,
        "base_scores": None,
        "all_ratings":       all_ratings,
        "all_pairwise":      [],
        "observed_ratings":  observed_ratings,
        "missing_ratings":   missing_ratings,
        "observed_pairwise": [],
        "missing_pairwise":  [],
        "stats": {
            "K": K_train + K_val + K_test,
            "K_train": K_train, "K_val": K_val, "K_test": K_test,
            "I": N_ATTRIBUTES, "J": N_ANNOTATORS, "C": C,
            "total_ratings":    len(all_ratings),
            "observed_ratings": len(observed_ratings),
            "missing_ratings":  len(missing_ratings),
        },
    }
    configs = {
        "datagen": {
            "K_train": K_train, "K_val": K_val, "K_test": K_test,
            "I": N_ATTRIBUTES, "J": N_ANNOTATORS, "C": C,
        }
    }
    return bundle, configs


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


def main():
    print("Loading TSVs...")
    human_df, llm_df = load_tsvs()
    llm_lookup = build_llm_dist_lookup(llm_df)
    llm_ann_id = get_llm_annotator_id(human_df)

    print("\nReconstructing splits (seed=42)...")
    test_ids, val_ids, train_ids = build_splits(human_df, llm_df)
    print(f"  train pool={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    rng_sub        = np.random.default_rng(SUBSAMPLE_SEED)
    train_shuffled = rng_sub.permutation(train_ids).tolist()

    print(f"\nGenerating new sizes {NEW_SIZES} (skipping existing folders)...")
    for size in sorted(NEW_SIZES):
        folder_name = f"LLMRubric_{TOTAL_ITEMS}_{N_ANNOTATORS}_{N_ATTRIBUTES}_{size}"
        out_dir     = OUTPUT_ROOT / folder_name
        if out_dir.exists():
            print(f"  [skip] {folder_name} already exists")
            continue
        subset_train = sorted(train_shuffled[:size])
        bundle, configs = build_bundle(
            human_df, llm_df, llm_lookup, llm_ann_id,
            subset_train, val_ids, test_ids,
        )
        save_bundle(bundle, configs, out_dir)

    print(f"\nDone. Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
