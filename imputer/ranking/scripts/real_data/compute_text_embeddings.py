#!/usr/bin/env python3
"""
Precompute SentenceBERT text embeddings for LLMRubric dialogue items.

Each item (dialogue) is embedded with all-MiniLM-L6-v2 and stored as a
[K, 768] float32 tensor in item_text_embeddings.pt, indexed 0-based to
match Marformer's 0-indexed item_ids (bundle item 1 → row 0, item K → row K-1).

The item ordering exactly mirrors the build_item_map() logic in
make_dist_bundle.py (seed=42, N_TEST=25) so the tensor rows align with the
bundle's item indices.

Usage (from Marformer repo root):
    python scripts/real_data/compute_text_embeddings.py
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
MARFORMER_ROOT = Path(__file__).resolve().parents[2]
DATA_JSON   = MARFORMER_ROOT / "OUTPUT/generated_data/llm_rubric_dist/data.json"
OUT_PT      = MARFORMER_ROOT / "OUTPUT/generated_data/llm_rubric_dist/item_text_embeddings.pt"
OUT_ORDER   = MARFORMER_ROOT / "OUTPUT/generated_data/llm_rubric_dist/item_id_order.json"

# ── Must match make_dist_bundle.py exactly ────────────────────────────────────
RANDOM_SEED = 42
N_TEST      = 25
SBERT_MODEL = "all-MiniLM-L6-v2"

# Same TSV paths as make_dist_bundle.py
HUMAN_TSV = Path("/Users/prabhavsingh/Documents/JHU/JHUResearch/RealData/imputer/ranking/OUTPUT/real_data/human_judges_real_convs_FIXED_ANON.tsv")
LLM_TSV   = Path("/Users/prabhavsingh/Documents/JHU/JHUResearch/RealData/imputer/ranking/OUTPUT/real_data/gpt-3.5-turbo-16k_real_evaluations_FIXED.tsv")

# ── Rebuild item_map identically to make_dist_bundle.py ──────────────────────

def build_item_map(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> dict:
    """Return {text_id: 1-based item_idx} — identical to make_dist_bundle.py."""
    llm_ids  = set(llm_df["text_id"].unique())
    all_ids  = sorted(set(human_df["text_id"].unique()) & llm_ids)

    rng      = np.random.default_rng(RANDOM_SEED)
    test_ids = set(rng.choice(all_ids, size=N_TEST, replace=False).tolist())

    train_sorted = sorted(tid for tid in all_ids if tid not in test_ids)
    test_sorted  = sorted(tid for tid in all_ids if tid in test_ids)
    ordered      = train_sorted + test_sorted
    return {tid: idx + 1 for idx, tid in enumerate(ordered)}   # 1-indexed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence_transformers not installed. "
            "Run: pip install sentence-transformers"
        )

    print(f"Loading data from {DATA_JSON}")
    with open(DATA_JSON) as f:
        data = json.load(f)
    print(f"  {len(data)} dialogue entries in data.json")

    # Build dialogue lookup: {_id: dialogue_text}
    id_to_text = {entry["_id"]: entry["dialogue"] for entry in data}

    # Build item ordering from TSVs — identical to make_dist_bundle.py
    # This is the authoritative item_map used when building the bundle.
    # data.json may have fewer entries (2 items are in TSVs but not data.json).
    print(f"Loading TSVs to reconstruct item_map...")
    human_df = pd.read_csv(HUMAN_TSV, sep="\t")
    human_df = human_df.drop(columns=[c for c in human_df.columns if c.startswith("DQ")])
    human_df = human_df.drop_duplicates(subset=["text_id", "annotator_id"], keep="first")
    llm_df   = pd.read_csv(LLM_TSV, sep="\t")
    q_cols   = [f"Q{i}" for i in range(9)]
    llm_df   = llm_df[llm_df["criterion"].isin(q_cols)].copy()

    item_map = build_item_map(human_df, llm_df)  # {text_id: 1-indexed item_idx}
    K        = len(item_map)
    print(f"  item_map covers {K} items  (train={K - N_TEST}  test={N_TEST})")

    # ordered_ids[i] = _id for bundle item i+1 (0-indexed into ordered list)
    ordered_ids = sorted(item_map.keys(), key=lambda tid: item_map[tid])

    # Collect dialogue texts in item order; zero-fill if missing from data.json
    texts = []
    zero_count = 0
    for _id in ordered_ids:
        if _id in id_to_text:
            texts.append(id_to_text[_id])
        else:
            texts.append("")   # will produce near-zero embedding
            zero_count += 1

    if zero_count:
        print(f"  WARNING: {zero_count} items have no dialogue in data.json — zero vectors will be used")

    # Load SBERT and encode
    print(f"\nLoading SentenceBERT model: {SBERT_MODEL}")
    model = SentenceTransformer(SBERT_MODEL)

    print(f"Encoding {K} dialogues...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )   # shape [K, 768]

    # Zero-out rows that had no text (empty string → near-zero but not exact zero)
    for i, _id in enumerate(ordered_ids):
        if _id not in id_to_text:
            embeddings[i] = 0.0

    tensor = torch.tensor(embeddings, dtype=torch.float32)
    print(f"\nEmbedding tensor shape: {tensor.shape}")

    # Save
    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, OUT_PT)
    print(f"Saved embeddings → {OUT_PT}")

    with open(OUT_ORDER, "w") as f:
        json.dump(ordered_ids, f, indent=2)
    print(f"Saved item order  → {OUT_ORDER}")

    # Quick sanity: norms
    norms = tensor.norm(dim=1)
    print(f"\nNorm stats:  mean={norms.mean():.3f}  min={norms.min():.3f}  max={norms.max():.3f}")
    print(f"Zero rows: {(norms < 1e-6).sum().item()}")


if __name__ == "__main__":
    main()
