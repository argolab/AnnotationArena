# Entity Marformer: Architecture

**Key files:**
```
imputer/entity_mf/
    config.py       — EntityMarformerConfig dataclass
    types.py        — EntityType hierarchy (RatingVariableType, ItemEntityType, …)
    data.py         — variable_list_to_entity_graph → EntityGraph
    model.py        — RelationalAttentionBlock, EntityMarformer
    eval.py         — evaluate_entity_marformer_split, compute_trainable_loss
    train.py        — EntityMarformerLightningModule, main() CLI
```

---

## Core Idea

Entity Marformer treats annotation data as a **typed entity graph**. All entities (attributes,
annotators, items) and all observations (ratings) become tokens in a single flat sequence. A
relational transformer processes this sequence, allowing each token to attend to any other token
with edge-type-aware attention scores.

This gives the model a direct representation of the annotation structure: which annotator gave
which rating on which item for which attribute, and how all observations relate to each other
through shared entities.

---

## Token Sequence Layout

For a single forward pass the sequence has `L` tokens:

```
[ rating_0, ..., rating_{N-1} | attr_0, ..., attr_{I-1} | annot_0, ..., annot_{J-1} | item_0, ..., item_{K'-1} ]
  ← N variable tokens →         ← I attr tokens →          ← J annotator tokens →      ← K' item tokens →
```

`K'` = items actually present in this forward pass (chunked by `--max-item`); entity_id is
preserved so deviation tables are indexed correctly even across chunks.

---

## Two-Stream Representation

Each token carries two concatenated streams:

```
token vector [model_dim] = [ feature_stream (feature_dim) | param_stream (param_dim) ]
```

- `model_dim = config.embedding_dim` (scripts use 80)
- `param_dim = 1 + num_classes` (e.g., 5 for 4-class Likert; 1 mask bit + 4 class logits)
- `feature_dim = model_dim − param_dim` (e.g., 75 for dim=80, C=4)

### Feature Stream

Built from two components:

1. **Type centroid** — `type_embeddings[type_name]`: a learned `[feature_dim]` vector, one per
   entity/variable type (attribute, annotator, item, rating, ranking_pairwise).

2. **Deviation** — `deviation_tables[type_name][entity_id]`: per-entity offset, zero-initialized.
   Controlled by `VariationConfig`:
   - Attributes: enabled, optional L2 reg
   - Annotators: enabled, optional L2 reg, optional dropout (`--annotator-dropout-rate`)
   - Items: enabled, but **always dropped during training** (`dropout_rate=1.0` default) because
     test items are unseen; the model must impute from context alone

   When `--use-deviation-norm` is set, a LayerNorm is applied to the deviation before adding
   it to the type centroid.

### Param Stream

Built by `EntityType.build_param()`:

| Token type | Param encoding |
|---|---|
| Attribute / Annotator / Item | all zeros (entity identity comes from feature stream) |
| Rating — missing or masked | `[1.0, 0, …, 0]` (mask bit set) |
| Rating — observed, hard label | `[0.0, …, logit_high at class c, …]` (one-hot spike, default logit_high=20) |
| Rating — observed, soft dist | `[0.0, log p_0, …, log p_{C-1}]` (when `--llm-input-dist` and dist is not one-hot) |
| Pairwise — missing or masked | `[1.0, 0, 0]` |
| Pairwise — observed | `[0.0, logit_high at winner, 0]` |

The `logit_high=20` spike makes the observed class near-certain in softmax, giving a reliable
"this is what I saw" signal without gradient interference from the mask bit.

For `--llm-input-dist`: the soft encoding is only applied when `rating_dist` is present **and**
not one-hot (checked by `_is_one_hot`). One-hot distributions fall back to the hard-label spike.
This means for STAN synthetic data (all `rating_dist` entries are one-hot or absent), `--llm-input-dist`
is a no-op. It is meaningful only for LLMRubric, where the LLM annotator provides a true soft distribution.

---

## Transformer Blocks

`config.num_layers` identical Pre-LN blocks (scripts use 8):

```python
combined  = cat([features, params], dim=-1)              # [1, L, model_dim]
attn_out  = RelationalAttentionBlock(norm_1(combined))   # [1, L, model_dim]
combined  = combined + attn_out                          # residual

z_ff      = FFN(norm_2(combined))                        # [1, L, model_dim]
combined  = combined + z_ff                              # residual

# Outer stream-specific residuals
features  = features + dropout(proj_out(combined))       # [1, L, feature_dim]
params    = params   + dropout(W_param(combined))        # [1, L, param_dim]
```

The outer residuals allow the FFN to mix feature and param information, then project back to
each stream separately. Pre-LN (normalize before each sub-layer) ensures stable training at depth.

---

## Relational Attention

### Edge Types

`variable_list_to_entity_graph` builds a binary edge mask `edge_mask [L, L, 6]`:

| Index | Name | Meaning |
|---|---|---|
| 0 | ATTR | rating → its attribute entity |
| 1 | ATTR_INV | attribute entity → rating |
| 2 | ANNOT | rating → its annotator entity |
| 3 | ANNOT_INV | annotator entity → rating |
| 4 | ITEM | rating → its item entity |
| 5 | ITEM_INV | item entity → rating |

No edges between two entity tokens. No direct entity-entity attention.

When `--use-graph-mask` is enabled, attention between tokens with no edge is set to −∞
(hard graph mask over the softmax). When disabled, all tokens can attend to all other tokens
(relational bias still steers attention).

### Per-Head Relational Attention (default, `--no-per-head-rel` to disable)

Each head has `head_dim = model_dim / num_heads` total, split into:
- `k_content_dim = head_dim − R` (content dims, R=6)
- `R` relational dims

```
Q → [D + 3 with pointer, else D]  — split per head: [content (kc) | rel (R)]
K → H × k_content_dim             — content only
V → D

content_scores = Q_content @ K_content^T                       [B, H, L, L]
rel_scores     = einsum('bhir, ijr -> bhij', Q_rel, edge_mask) [B, H, L, L]
scores         = (content_scores + rel_scores) / sqrt(head_dim)
```

Each head learns independent relational weights, allowing specialization (one head attends along
ATTR edges, another along ANNOT edges, etc.).

**Shared-bias mode** (`--no-per-head-rel`): single relational bias added identically to all heads.
All experiments currently use shared-bias + `--scale-shared-rel`.

### K_aug Pointer (`--use-pointer`)

Adds direct obs-obs connections based on shared entity identity, without requiring a 2-hop path
through entity tokens. Builds `K_aug [L, L, 3]` — channels: same-attribute, same-annotator,
same-item. Both positions must have valid IDs (≥ 0); entity tokens are assigned −1 and never match.

Q gets 3 extra output dimensions:
```python
ptr_bias = (Q_ptr.unsqueeze(2) * K_aug.unsqueeze(0)).sum(-1)   # [B, L, L]
scores   = scores + ptr_bias.unsqueeze(1)                       # broadcast over heads
```

This is the single most important architectural addition — it provides the same signal as the
original Marformer's K_aug without requiring item/annotator embeddings at the token level.

### Relational Value Augmentation (`--use-rel-value`)

After attention weights are computed, adds a per-relationship correction to the aggregated output:

```python
attn_r_mass = einsum('bhij, ijr -> bhir', attn, edge_mask)      # [B, H, L, R]
rel_aug     = einsum('bhir, hrd -> bhid', attn_r_mass, rel_value_emb)
out         = out + rel_aug
```

`rel_value_emb [H, R, head_dim]` initialized to zero (no-op at init, learned from data).

### Add-One Attention (`--use-addone-attn`)

```
attn_ij = exp(s_ij) / (1 + sum_k exp(s_ik))
```

Sum of attention weights ≤ 1. A token can "opt out" of attending to anyone. Useful for entity
tokens with few relevant neighbors in a given chunk.

---

## Loss Function

### Per-Token CE

`RatingVariableType.compute_loss_breakdown` computes CE for **all** rating tokens
(observed + masked + missing), then splits by status.

- **Hard labels** (`rating_dist` absent or one-hot): `CrossEntropyLoss`.
- **Soft labels** (`rating_dist` present, non-one-hot): `−sum(p_c · log_softmax(logit_c))`.
  If any token in the batch has a soft dist, all tokens use the soft path; tokens without
  `rating_dist` use one-hot as soft target (numerically equivalent to hard CE).

Logits extracted from param stream: `params[:, 1 : 1 + num_classes]` (skipping the mask bit).

### Training Objective

```
total_loss = masked_loss_weight × mean_CE_masked + observed_loss_weight × mean_CE_observed
```

Defaults: `MASKED_LOSS_WEIGHT=15.0`, `OBSERVED_LOSS_WEIGHT=1.0`. The high masked weight
focuses training on the imputation task.

### Mask Augmentations

Each epoch runs `MASK_AUGMENTATIONS` (default 5) training steps, each with a fresh independent
masking draw. This multiplies training signal diversity without increasing data size.

---

## Config Reference

Current script hyperparameters (see `scripts/STAN/MARFORMER/*/run_train.sh`):

| Parameter | Value | Notes |
|---|---|---|
| `embedding_dim` | 80 | model_dim; must be divisible by `attention_heads` |
| `num_layers` | 8 | transformer depth |
| `attention_heads` | 4 | head_dim = 80/4 = 20; must be > R=6 for per-head-rel |
| `d_ff` | 128 | FFN hidden dimension |
| `num_ffn_layers` | 1 | single Linear+ReLU inside FFN |
| `dropout` | 0.1 | applied after attention and FFN |
| `logit_high` | 20.0 | observed hard-label spike magnitude |
| `masking_rate` | 0.15–0.5 | fraction of observed ratings masked per step |
| `mask_augmentations` | 5 | masking draws per epoch |
| `masked_loss_weight` | 15.0 | |
| `observed_loss_weight` | 1.0 | |
| `lr` | 2e-4 | Adam learning rate |
| `weight_decay` | 0.01 | AdamW weight decay |
| `epochs` | 200–300 | |
| `item_dropout_rate` | 0.7–1.0 | item deviation dropout during training |
| `use_pointer` | true | K_aug obs-obs pointer |
| `use_graph_mask` | varies | hard attention mask on entity-absent pairs |
| `scale_shared_rel` | true | scale shared relational bias |
