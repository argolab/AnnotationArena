"""
Convert raw TSV annotation files → data_bundle.json format.

Inputs:
  - human_tsv:  human_judges_real_convs_FIXED_ANON.tsv
      Columns: round, text_id, DQQ0, DQQ1, DQQ2, Q0-Q8, annotator_id
      One row per (annotator, text_id). Q values are integers 1-4; 0 = DQ'd → treated as 1.
      Duplicate (text_id, annotator_id) pairs across rounds → keep first.

  - llm_tsv:    gpt-3_5-turbo-16k_real_evaluations_FIXED.tsv
      Columns: text_id, criterion, sample_llm, answer1_prob..answer4_prob
      One row per (text_id, criterion). criterion ∈ {Q0..Q8}.

Mapping to data_bundle.json:
  - item      ← text_id  (globally sorted, 1-indexed)
  - attribute ← Q index  (Q0→1, Q1→2, ..., Q8→9)
  - annotator ← annotator_id + 1  (human, 1-indexed)
               OR  max_human_annotator + 1  (LLM, always last)
  - value     ← integer rating (human, 0→1) or argmax(answer_probs)+1 (LLM)
  - observed  ← all ratings are observed (human 0 treated as 1, LLM always observed)
  - instance  ← random 10% of items → "test", remainder → "train"

Model-learned fields (embeddings, preferences, posteriors, etc.) are set to null;
populate them after running AnnotationArena inference.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# CONFIGURATION — repo-relative paths
# ============================================================================
ROOT = Path(__file__).resolve().parents[1]  # imputer/ranking

HUMAN_TSV   = ROOT / "real-data/human_judges_real_convs_FIXED_ANON.tsv"
LLM_TSV     = ROOT / "real-data/gpt-3.5-turbo-16k_real_evaluations_FIXED.tsv"
OUTPUT_PATH = ROOT / "OUTPUT/generated_data/llm_rubric/data_bundle.json"

RANDOM_SEED   = 42
TEST_FRACTION = 0.10
# ============================================================================

Q_COLS = [f"Q{i}" for i in range(9)]   # Q0..Q8  (DQ columns are ignored)


def load_human(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.drop(columns=[c for c in df.columns if c.startswith("DQ")])
    df = df.drop_duplicates(subset=["text_id", "annotator_id"], keep="first")
    return df


def load_llm(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df[df["criterion"].isin(Q_COLS)].copy()


def build_item_map(human_df: pd.DataFrame, llm_df: pd.DataFrame, test_ids: set) -> dict:
    """Map text_id → 1-based item integer.
    Only includes items that have LLM coverage; items missing from the LLM TSV are excluded.
    Train items are indexed first (1..K_train), test items last (K_train+1..N).
    """
    llm_ids = set(llm_df["text_id"].unique())
    all_ids = sorted(set(human_df["text_id"].unique()) & llm_ids)  # intersection only
    train_ids = sorted(tid for tid in all_ids if tid not in test_ids)
    test_ids_sorted = sorted(tid for tid in all_ids if tid in test_ids)
    ordered = train_ids + test_ids_sorted
    return {tid: idx + 1 for idx, tid in enumerate(ordered)}


def split_test_items(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> set:
    """Randomly select 25 text_ids as test instance (from the 225 LLM-covered items)."""
    llm_ids = set(llm_df["text_id"].unique())
    all_ids = sorted(set(human_df["text_id"].unique()) & llm_ids)
    rng = np.random.default_rng(RANDOM_SEED)
    n_test = 25
    return set(rng.choice(all_ids, size=n_test, replace=False).tolist())


def convert():
    human_df = load_human(HUMAN_TSV)
    llm_df   = load_llm(LLM_TSV)

    test_ids = split_test_items(human_df, llm_df)
    item_map = build_item_map(human_df, llm_df, test_ids)

    # Human annotator_id is 0-indexed in file → shift to 1-indexed
    max_human_ann    = int(human_df["annotator_id"].max()) + 1   # 1-indexed max
    llm_annotator_id = max_human_ann + 1                          # LLM is last

    print(f"Total items:        {len(item_map)}")
    print(f"Test items (10%):   {len(test_ids)}")
    print(f"Human annotators:   1–{max_human_ann}  (original 0–{max_human_ann - 1})")
    print(f"LLM annotator ID:   {llm_annotator_id}")

    all_ratings      = []
    observed_ratings = []
    missing_ratings  = []

    # ------------------------------------------------------------------
    # Human ratings — all observed; value 0 (DQ) treated as 1
    # ------------------------------------------------------------------
    for _, row in human_df.iterrows():
        text_id  = row["text_id"]
        if text_id not in item_map:   # skip items excluded (no LLM coverage)
            continue
        item_id  = item_map[text_id]
        instance = "test" if text_id in test_ids else "train"
        ann_id   = int(row["annotator_id"]) + 1

        for q_col in Q_COLS:
            attribute = int(q_col[1:]) + 1
            raw_val   = row[q_col]
            raw_int   = 0 if pd.isna(raw_val) else int(raw_val)
            value     = raw_int if raw_int != 0 else 1   # DQ / NaN → treat as 1

            record = {
                "attribute": attribute,
                "annotator":  ann_id,
                "item":       item_id,
                "value":      value,
                "instance":   instance,
            }
            all_ratings.append(record)
            # Human ratings in test items are missing; train items are observed
            if instance == "test":
                missing_ratings.append(record)
            else:
                observed_ratings.append(record)

    # ------------------------------------------------------------------
    # LLM ratings — always observed
    # ------------------------------------------------------------------
    prob_cols = ["answer1_prob", "answer2_prob", "answer3_prob", "answer4_prob"]
    for _, row in llm_df.iterrows():
        text_id   = row["text_id"]
        item_id   = item_map[text_id]
        instance  = "test" if text_id in test_ids else "train"
        attribute = int(row["criterion"][1:]) + 1   # Q0→1, ..., Q8→9
        probs     = [row[c] for c in prob_cols]
        value     = int(np.argmax(probs)) + 1        # 1-based

        record = {
            "attribute": attribute,
            "annotator":  llm_annotator_id,
            "item":       item_id,
            "value":      value,
            "instance":   instance,
        }
        all_ratings.append(record)
        observed_ratings.append(record)

    # ------------------------------------------------------------------
    # missing_ratings_indexes_in_test_instance
    # (indices into all_ratings that are in test instance and missing)
    # Since all ratings here are observed, this will be empty — but kept
    # for format compatibility in case missing ratings are added later.
    # ------------------------------------------------------------------
    missing_keys = {
        (r["attribute"], r["annotator"], r["item"])
        for r in missing_ratings
    }
    missing_ratings_indexes_in_test_instance = [
        i for i, rec in enumerate(all_ratings)
        if rec["instance"] == "test"
        and (rec["attribute"], rec["annotator"], rec["item"]) in missing_keys
    ]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    K_train = sum(1 for tid in item_map if tid not in test_ids)
    K_test  = len(test_ids)
    total_ratings = len(all_ratings)

    train_ratings  = [r for r in all_ratings      if r["instance"] == "train"]
    test_ratings   = [r for r in all_ratings      if r["instance"] == "test"]
    train_observed = [r for r in observed_ratings if r["instance"] == "train"]
    test_observed  = [r for r in observed_ratings if r["instance"] == "test"]

    stats = {
        "K_train":                K_train,
        "K_test":                 K_test,
        "total_items":            len(item_map),
        "total_possible_ratings": total_ratings,
        "total_ratings":          total_ratings,
        "observed_ratings":       len(observed_ratings),
        "missing_ratings":        len(missing_ratings),
        "train_ratings":          len(train_ratings),
        "test_ratings":           len(test_ratings),
        "train_observed":         len(train_observed),
        "test_observed":          len(test_observed),
        "total_pairwise":         0,
        "observed_pairwise":      0,
        "missing_pairwise":       0,
        "train_pairwise":         0,
        "test_pairwise":          0,
        "observation_rate":       len(observed_ratings) / total_ratings if total_ratings else 0.0,
        "train_observation_rate": len(train_observed) / len(train_ratings) if train_ratings else 0.0,
        "test_observation_rate":  len(test_observed)  / len(test_ratings)  if test_ratings  else 0.0,
        "protocol":               "random_split",
        "mcar_missing_rate":      None,
    }

    # ------------------------------------------------------------------
    # Assemble bundle
    # ------------------------------------------------------------------
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
        "missing_ratings_indexes_in_test_instance": missing_ratings_indexes_in_test_instance,
        "stats":                 stats,
        "train_posterior_rating_probs": None,
        "test_posterior_rating_probs":  None,
    }

    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  Train items:    {K_train}")
    print(f"  Test items:     {K_test}")
    print(f"  Total ratings:  {total_ratings}")
    print(f"  Observed:       {len(observed_ratings)}")
    print(f"  Missing:        {len(missing_ratings)}  (human ratings in test items)")
    print(f"  Missing in test instance: {len(missing_ratings_indexes_in_test_instance)}")


if __name__ == "__main__":
    convert()