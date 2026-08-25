#!/usr/bin/env python3
"""
Generate LLM-Rubric data bundles with train/val/test splits and multiple training sizes.

Split (seed=42):
  - 25 test items  (same selection as original convert.py — backward compatible)
  - 25 val  items  (next selection from same RNG, from remaining items)
  - 175 train items (remainder)

Outputs (all under DATA/LLM_RUBRIC/):
  ORIGINAL_SPLIT/              — full bundle (175 train + 25 val + 25 test)
  LLMRubric_225_25_9_{size}/   — size in {10, 50, 100, 150, 175}

Each folder contains data_bundle.json + configs.json.
Training subsets are nested: size=10 ⊆ size=50 ⊆ ... ⊆ size=175 (SUBSAMPLE_SEED=2024).
LLM annotator always observed everywhere; human annotations observed only for train items.
LLM ratings encoded as soft probability distributions; human ratings as one-hot.

NOTE: configs.json now includes K_val. utils.py in the Marformer codebase will need
      to be updated to handle K_val before running training on these bundles.

Usage (from repo root):
    python scripts/convert-llm-rubric/make_llmrubric_splits.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
RANKING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR   = Path(__file__).resolve().parent

HUMAN_TSV    = SCRIPT_DIR / "human_judges_real_convs_FIXED_ANON.tsv"
LLM_TSV      = SCRIPT_DIR / "gpt-3.5-turbo-16k_real_evaluations_FIXED.tsv"
OUTPUT_ROOT  = RANKING_ROOT / "DATA" / "LLM_RUBRIC"

# ── Constants ─────────────────────────────────────────────────────────────────
SPLIT_SEED     = 42      # controls train/val/test split — do not change
SUBSAMPLE_SEED = 2024    # controls nested training subsets — do not change
N_VAL          = 25
N_TEST         = 25
C              = 4
Q_COLS         = [f"Q{i}" for i in range(9)]
PROB_COLS      = ["answer1_prob", "answer2_prob", "answer3_prob", "answer4_prob"]
TRAIN_SIZES    = [10, 50, 100, 150, 175]

# Dataset-level identifiers used in folder names
TOTAL_ITEMS  = 225   # 175 train + 25 val + 25 test
N_ANNOTATORS = 25    # 24 human + 1 LLM
N_ATTRIBUTES = 9     # Q0..Q8


# ── Data loading ──────────────────────────────────────────────────────────────

def load_tsvs():
    if not HUMAN_TSV.exists():
        raise FileNotFoundError(f"Human TSV not found: {HUMAN_TSV}")
    if not LLM_TSV.exists():
        raise FileNotFoundError(f"LLM TSV not found: {LLM_TSV}")
    human_df = pd.read_csv(HUMAN_TSV, sep="\t")
    human_df = human_df.drop(columns=[c for c in human_df.columns if c.startswith("DQ")])
    human_df = human_df.drop_duplicates(subset=["text_id", "annotator_id"], keep="first")
    llm_df   = pd.read_csv(LLM_TSV, sep="\t")
    llm_df   = llm_df[llm_df["criterion"].isin(Q_COLS)].copy()
    return human_df, llm_df


# ── Split construction ────────────────────────────────────────────────────────

def build_splits(human_df, llm_df):
    """
    Return (test_ids, val_ids, train_ids) as sorted lists.

    Test selection is identical to original convert.py (seed=42, n=25).
    Val is selected from remaining items with the same RNG continuing.
    """
    llm_ids  = set(llm_df["text_id"].unique())
    all_ids  = sorted(set(human_df["text_id"].unique()) & llm_ids)

    rng = np.random.default_rng(SPLIT_SEED)
    test_ids  = set(rng.choice(all_ids, size=N_TEST, replace=False).tolist())
    remaining = sorted(tid for tid in all_ids if tid not in test_ids)
    val_ids   = set(rng.choice(remaining, size=N_VAL, replace=False).tolist())
    train_ids = sorted(
        tid for tid in all_ids if tid not in test_ids and tid not in val_ids
    )
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(all_ids)
    return sorted(test_ids), sorted(val_ids), train_ids


# ── LLM helpers ───────────────────────────────────────────────────────────────

def build_llm_dist_lookup(llm_df):
    """Return {(text_id, attr_1indexed): [p1, p2, p3, p4]}."""
    lookup = {}
    for _, row in llm_df.iterrows():
        attr_id = int(row["criterion"][1:]) + 1   # Q0→1, ..., Q8→9
        probs   = [float(row[c]) for c in PROB_COLS]
        lookup[(row["text_id"], attr_id)] = probs
    return lookup


def get_llm_annotator_id(human_df):
    """1-indexed LLM annotator ID (one above the highest human annotator)."""
    return int(human_df["annotator_id"].max()) + 2   # +1 for 0→1 indexing, +1 for LLM


# ── Bundle builder ────────────────────────────────────────────────────────────

def build_bundle(human_df, llm_df, llm_lookup, llm_ann_id,
                 train_text_ids, val_text_ids, test_text_ids):
    """
    Build data_bundle.json + configs.json for the given item sets.

    Item IDs (1-indexed): train items 1..K_train,
                          val   items K_train+1..K_train+K_val,
                          test  items K_train+K_val+1..K_total.
    observed_ratings : LLM ratings (all items) + human ratings (train only).
    missing_ratings  : human ratings (val + test).
    """
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

    all_ratings      = []
    observed_ratings = []
    missing_ratings  = []

    # ── Human ratings ─────────────────────────────────────────────────────────
    for _, row in human_df.iterrows():
        text_id = row["text_id"]
        if text_id not in all_ids_set:
            continue
        item_id  = item_map[text_id]
        instance = instance_of[text_id]
        ann_id   = int(row["annotator_id"]) + 1   # 0→1-indexed

        for q_col in Q_COLS:
            attr_id  = int(q_col[1:]) + 1          # Q0→1, ..., Q8→9
            raw_val  = row[q_col]
            raw_int  = 0 if pd.isna(raw_val) else int(raw_val)
            value    = raw_int if raw_int != 0 else 1
            one_hot  = [0.0] * C
            one_hot[value - 1] = 1.0
            rec = {
                "attribute":   attr_id,
                "annotator":   ann_id,
                "item":        item_id,
                "value":       value,
                "instance":    instance,
                "rating_dist": one_hot,
            }
            all_ratings.append(rec)
            if instance == "train":
                observed_ratings.append(rec)
            else:
                missing_ratings.append(rec)

    # ── LLM ratings (always observed) ─────────────────────────────────────────
    for _, row in llm_df.iterrows():
        text_id = row["text_id"]
        if text_id not in all_ids_set:
            continue
        item_id  = item_map[text_id]
        instance = instance_of[text_id]
        attr_id  = int(row["criterion"][1:]) + 1

        probs = llm_lookup.get((text_id, attr_id))
        if probs is None:
            # Fallback: one-hot from argmax (shouldn't happen for covered items)
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

    # ── Sanity checks ──────────────────────────────────────────────────────────
    obs_set  = {(r["attribute"], r["annotator"], r["item"]) for r in observed_ratings}
    miss_set = {(r["attribute"], r["annotator"], r["item"]) for r in missing_ratings}
    assert obs_set & miss_set == set(), "observed ∩ missing is non-empty"

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
            "I":                N_ATTRIBUTES,
            "J":                N_ANNOTATORS,
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
            "I":       N_ATTRIBUTES,
            "J":       N_ANNOTATORS,
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
    print("Loading TSVs...")
    human_df, llm_df = load_tsvs()
    llm_lookup = build_llm_dist_lookup(llm_df)
    llm_ann_id = get_llm_annotator_id(human_df)
    print(f"  Human annotators: 1–{llm_ann_id - 1}  |  LLM annotator ID (1-indexed): {llm_ann_id}")

    print("\nBuilding splits (seed=42)...")
    test_ids, val_ids, train_ids = build_splits(human_df, llm_df)
    print(f"  train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}  "
          f"(total={len(train_ids)+len(val_ids)+len(test_ids)})")

    # Build nested training subsets (largest first → size N uses first N of shuffle)
    rng_sub        = np.random.default_rng(SUBSAMPLE_SEED)
    train_shuffled = rng_sub.permutation(train_ids).tolist()

    # ── ORIGINAL_SPLIT ────────────────────────────────────────────────────────
    print("\nGenerating ORIGINAL_SPLIT...")
    bundle, configs = build_bundle(
        human_df, llm_df, llm_lookup, llm_ann_id,
        train_ids, val_ids, test_ids,
    )
    save_bundle(bundle, configs, OUTPUT_ROOT / "ORIGINAL_SPLIT")

    # ── Sized subsets ─────────────────────────────────────────────────────────
    print("\nGenerating sized subsets (nested, SUBSAMPLE_SEED=2024)...")
    for size in sorted(TRAIN_SIZES):
        subset_train = sorted(train_shuffled[:size])
        folder_name  = f"LLMRubric_{TOTAL_ITEMS}_{N_ANNOTATORS}_{N_ATTRIBUTES}_{size}"
        bundle, configs = build_bundle(
            human_df, llm_df, llm_lookup, llm_ann_id,
            subset_train, val_ids, test_ids,
        )
        save_bundle(bundle, configs, OUTPUT_ROOT / folder_name)

    print(f"\nAll done. Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
