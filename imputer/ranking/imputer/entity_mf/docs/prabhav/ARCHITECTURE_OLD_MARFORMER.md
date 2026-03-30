# Old Marformer (Marformer-Test): Architecture Reference

This document describes the original Marformer (`Marformer-Test` repo) that Entity Marformer is being
compared against and partly inspired by. When Prabhav says "old code" or "old Marformer", this is it.

**Repo location:** `/Users/prabhavsingh/Documents/JHU/JHUResearch/Marformer-Test/imputer/ranking/`

**Key files:**
```
imputer/
    transformer.py         — TransformerBlock (dual-stream attention)
    ranking_imputer.py     — MultiVariableImputer (full model)
    embedding.py           — AtomCompositonalEmbeddingProvider
    run_imputer.py         — CLI entry point + ImputerLightningModule
    lightning_trainer.py   — Full training loop (ImputerLightningModule)
    eval.py                — Evaluation helpers
```

---

## Core idea: flat sequence + compositional embedding

The old Marformer operates on a **flat sequence of N observation tokens** only — no separate entity tokens.
Instead, entity identity is baked directly into each token's initial embedding via learned weight matrices.

**No entity tokens.** There are no attribute/annotator/item tokens in the sequence.
The sequence length is just the number of observations (masked + observed + missing).

---

## Embedding: AtomCompositonalEmbeddingProvider

For each observation `(i, j, k)` — attribute i, annotator j, item k — the initial feature embedding is:

```python
feature = W_I @ attr_vec(i) + W_J @ annot_vec(j) + W_K @ item_vec(k)
```

Where:
- `attr_vec(i)`: one-hot for attribute i (length I) → projected to `embedding_dim` by `W_I [D, I]`
- `annot_vec(j)`: one-hot for annotator j + per-annotator deviation
- `item_vec(k)`: mean-pool of `[features | params]` from item k's observed observations → projected by `W_K_init`

This gives each observation token an **immediate, direct identity** — no multi-hop needed.
The model knows from the first layer which attribute, annotator, and item this token belongs to.

An alternative `use_concat_embedding=True` mode concatenates instead of sums, but the sum mode is standard.

### Item embedding details

Items are particularly important. In the default setup:
- `item_vec(k)` is computed as mean-pooled `[features | params]` from item k's observed tokens,
  projected by `W_K_init [embedding_dim, embedding_dim]`.
- This means item embeddings are **data-dependent**: items with more observed ratings get richer
  embeddings. Items with zero observations get a zero vector.
- `w_init` flag controls `W_K_init` initialization: `random`, `xavier`, or `identity`.
- `item_embedding_dropout`: dropout rate on item embeddings (independent of per-annotator dropout).

---

## Param stream

Same concept as Entity Marformer. Each token has a `param_stream` vector:

```
[mask_bit | rating_logits (num_classes) | ranking_logits (max_rank_size)]
```

- `param_dim = 1 + max(num_classes, max_rank_size)` — shared across all tokens (zero-padded for type mismatches)
- Observed rating: mask bit = 0, one-hot spike at `logit_high` for true class
- Masked/missing rating: mask bit = 1, rest zeros
- `llm_input_dist=True`: LLM soft distributions encoded as `log(clamp(p_c, min=exp(-logit_high)))`

---

## TransformerBlock (dual-stream)

**File:** `transformer.py`

The block operates on separate `features [B, N, feature_dim]` and `params [B, N, param_dim]` tensors
and returns updated versions of both.

### Architecture per block

```
1.  proj_in(features)                   → [B, N, feat_proj_dim]   (linear, no activation)
2.  combined = cat([proj_in(feat), params])  → [B, N, model_dim]
3.  norm_1(combined)                    → normalize before attention
4.  Q, K, V projections on combined
5.  Multi-head attention + residual     → z [B, N, model_dim]
6.  norm_2(z) → FFN → residual         → z [B, N, model_dim]
7.  proj_out(z) + residual on features  → feature stream outer residual
8.  W_param(z) + residual on params    → param stream outer residual
```

Where:
- `feat_proj_dim = model_dim - param_dim`
- `model_dim = feature_dim + param_dim` (= `embedding_dim + param_dim`)

The outer residuals in steps 7-8 are the key design choice: the FFN on the combined stream mixes
feature and param information, then the output is projected back to each stream separately.
This is identical to Entity Marformer's `proj_out` and `W_param` design.

### Normalize parameter mode

`normalize_parameter=True`: applies LayerNorm to the full combined `[features | params]` before attention.
`normalize_parameter=False` (default): normalizes only the feature portion; scales param portion by
a learned scalar `param_scale` (initialized to 0.01) to prevent the param stream's large logit spikes
from dominating the attention keys.

**Note:** With `logit_high=20.0` and `normalize_parameter=True`, the LayerNorm distorts the feature
stream because the param stream's ±20 spikes dominate the mean and variance.
Use `normalize_parameter=False` (default) to avoid this.

---

## Pointer mechanism (K_aug)

**File:** `transformer.py`, `ranking_imputer.py`

The core feature that lets old Marformer perform well.

### K_aug: obs-obs shared-identity indicators

For each pair of observation tokens (i, j), builds 3 binary indicators:
- Channel 0: same attribute? (`attr_ids[i] == attr_ids[j]`)
- Channel 1: same annotator? (`annot_ids[i] == annot_ids[j]`)
- Channel 2: same item? (`item_ids[i] == item_ids[j]`)

Shape: `K_aug [B, N, N, 3]` (Entity MF uses `[L, L, 3]` without batch dim).

**Note:** Old Marformer does NOT check for ID validity (no `id >= 0` guard), so two tokens with
`item_id=-1` would spuriously match. Entity MF fixed this.

### How it's used in attention

Q projects to `model_dim + 3`:
```python
Q_full = self.Q(combined)          # [B, N, model_dim + 3]
Q_base = Q_full[:, :, :model_dim]  # [B, N, model_dim]   — normal Q
Q_ptr  = Q_full[:, :, model_dim:]  # [B, N, 3]           — pointer weights
```

Pointer bias:
```python
ptr_additions = (Q_ptr.unsqueeze(2) * K_aug).sum(dim=-1)  # [B, N, N]
raw_scores    = raw_scores + ptr_additions
```

This adds a per-query learned bias to each key based on shared identity.
The model learns: "if this key shares my annotator, boost the attention score by Q_ptr[1]".

This is applied **per head** in old Marformer (the pointer bias is added inside the head loop,
so each head gets the same `ptr_additions` but different `raw_scores`). Effectively shared across heads.
Entity MF does the same: pointer bias is broadcast over all heads.

### Enable/disable

`enable_pointer_mechanism=True` by default in old Marformer. It is always on unless explicitly disabled.
This is one of the most important components — disabling it significantly hurts performance.

---

## Multi-head attention implementation

**File:** `transformer.py`, `_multihead_attention`

Notably: old Marformer handles **unequal head sizes** when `model_dim % num_heads != 0`.
It distributes the remainder across the first few heads (`head_dims = [base + 1 if i < remainder else base]`).
This makes the implementation head-by-head in a Python loop (not batched), but allows non-divisible dims.

Standard batched matmul is used in Entity MF (requires divisible dims, assertion enforced).

The add-one attention option (`use_addone_attn`) was also added to old Marformer. See `CHANGES.md`.

---

## Training loop

**File:** `lightning_trainer.py` (ImputerLightningModule)

### Loss

```
total_loss = masked_loss_weight * CE_masked + observed_loss_weight * CE_observed
```

Defaults: `masked_loss_weight=8.0`, `observed_loss_weight=1.0` in `run_imputer.py`.
(Entity MF scripts use `masked_loss_weight=15.0` to match the emphasis on masked-token accuracy.)

### Mask augmentations

Old Marformer also uses a dummy dataloader approach for mask augmentations.
`mask_augmentations` (default 1 in `run_imputer.py`) controls how many masking draws per epoch.
Setting it to 5 matches Entity MF's default.

### Logged metrics

Old Marformer logs: `obj/total_loss`, `obj/rating_loss`, `obj/ranking_loss`, and breakdowns
like `obj/masked_total_loss`, `obj/observed_total_loss` per type.
The logged `obj/total_loss` is the weighted sum (not raw CE), same magnitude as Entity MF.

### Item chunking

`max_item` controls how many unique items are processed per forward pass.
Variables whose items don't all fall in the current chunk are skipped for that chunk.
This is used for real data where the full item set is too large for one pass.

---

## Evaluation

**File:** `utils/evaluate_checkpoint.py`

Loads a saved `model.pt` (which stores `state_dict` + `model_config`), reconstructs
`MultiVariableImputer` from the saved config, runs evaluation on the full bundle.

Important: when loading a checkpoint, all config flags (including `use_addone_attn`, `logit_high`, etc.)
must be read from `model_config` in the `.pt` file, not from CLI defaults, to match the saved architecture.

---

## Performance reference

On LLMRubric (I=25 attributes, J=25 annotators, K~50 items, MCAR-ish missingness):
- **Stan** (ground truth Bayesian model): ~95% accuracy on missing train, ~97% on missing test.
- **Old Marformer** (with pointer, 6-8 layers, embedding_dim=128): close to Stan-level.
- **Entity MF** (without pointer): test CE barely above random (~1.3); adding `--use-pointer` is critical.

---

## Run script for old Marformer

`scripts/real_data/run_real_data.sh` — trains old Marformer on LLMRubric.

Key CLI flags for old Marformer (from `run_imputer.py`):

| Flag | Default | Notes |
|---|---|---|
| `--encoder-layers` | 6 | Transformer depth |
| `--attention-heads` | 8 | Must divide `embedding_dim + param_dim` |
| `--embedding-dim` | 128 | Feature stream dimension |
| `--num-ffn-layers` | 4 | FFN depth |
| `--d-ff` | 512 | FFN hidden dim |
| `--dropout` | 0.1 | |
| `--masked-loss-weight` | 8.0 | |
| `--observed-loss-weight` | 1.0 | |
| `--mask-augmentations` | 1 | Steps per epoch |
| `--normalize-parameter` | False | Normalize full combined or just features |
| `--logit-high` | 20.0 | Observed param spike magnitude |
| `--item-embedding-dropout` | 0.0 | Item embedding dropout |
| `--llm-input-dist` | False | Soft dist input encoding |
| `--llm-annotator-id` | None | Index of LLM annotator for structured masking |
| `--human-observed-rate` | 0.0 | Fraction of human obs kept as observed |
| `--max-item` | 1 | Items per forward pass chunk |
| `--use-addone-attn` | False | Add-one attention |
