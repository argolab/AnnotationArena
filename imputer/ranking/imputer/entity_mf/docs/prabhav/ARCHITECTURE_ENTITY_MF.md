# Entity Marformer: Current Architecture

This document describes the Entity Marformer as it stands *after* all changes documented in `CHANGES.md`.
It is meant to give a new session enough context to understand the model without re-reading all the code.

**Key files:**
```
imputer/entity_mf/
    config.py       — EntityMarformerConfig dataclass
    types.py        — EntityType hierarchy: RatingVariableType, ItemEntityType, etc.
    data.py         — variable_list_to_entity_graph → EntityGraph
    model.py        — RelationalAttentionBlock, EntityMarformer
    eval.py         — compute_trainable_loss, evaluate_entity_marformer_split
    train.py        — EntityMarformerLightningModule, main() CLI
```

---

## High-level concept

Entity Marformer treats annotation data as a **typed entity graph**:
- Entities: attributes (I), annotators (J), items (K) — known, fixed
- Variables: one per observation (rating or pairwise ranking) — variable count

All entities and variable-observations become tokens in a single flat sequence.
A relational attention mechanism allows tokens to attend to each other weighted by labeled edge types
(which attribute/annotator/item they share). This makes the model inherently aware of the data structure.

---

## Token sequence layout

For a given forward pass, the graph contains `L` tokens:

```
[ rating_0, rating_1, ..., pairwise_0, ...,  |  attr_0, ..., attr_{I-1}, annot_0, ..., item_0, ... ]
  ← variable tokens (N_var total) →          ←       entity tokens (I + J + K total)       →
```

Variable tokens appear first in the order given by `variable_list_to_entity_graph`.
Entity tokens appear after, one per unique entity.

---

## Two-stream representation

Each token carries two concatenated streams:

```
token vector [1, model_dim] = [ features (feature_dim) | params (param_dim) ]
```

- `model_dim = config.embedding_dim` (e.g. 80)
- `param_dim = max(t.param_dim for t in types)` = `1 + num_classes` for ratings (e.g. 5 for 4-class Likert)
- `feature_dim = model_dim - param_dim`

### Feature stream

Built from:
1. **Type centroid**: `type_embeddings[type_name]` — learned `[1, feature_dim]` vector, one per type.
2. **Deviation**: `deviation_tables[type_name][entity_id]` — per-entity deviation, zero-initialized.
   - Attributes: deviation enabled
   - Annotators: deviation enabled, optional L2 reg via `annotator_reg_weight`
   - Items: deviation enabled but **always dropped during training** (`dropout_rate=1.0`) because
     items are unseen at test time; the model must learn to impute from context alone.

### Param stream

Built by `EntityType.build_param()`:

| Token type | Param encoding |
|---|---|
| Attribute, Annotator, Item | all zeros (entity tokens carry no observation) |
| Rating — masked or missing | `[1.0, 0, 0, ..., 0]` (mask bit set, no logit info) |
| Rating — observed, hard label | `[0.0, ..., logit_high at class c, ...]` (one-hot spike) |
| Rating — observed, soft dist | `[0.0, log(clamp(p_0)), ..., log(clamp(p_{C-1}))]` (when `--llm-input-dist`) |
| Pairwise — masked/missing | `[1.0, 0, 0, ...]` |
| Pairwise — observed | `[0.0, logit_high at winner pos, 0]` |

`logit_high` default: 20.0. Rationale: a spike of 20 makes the observed class near-certain in softmax,
giving the model a reliable "this is what I saw" signal in the param stream.

---

## Transformer blocks

`config.num_layers` blocks (default: 6), each a `nn.ModuleDict`:

```
norm_1   → RelationalAttentionBlock   (Pre-LN before attention)
norm_2   → FeedForward                (Pre-LN before FFN)
proj_out → Linear(model_dim, feature_dim)   (outer residual for feature stream)
W_param  → Linear(model_dim, param_dim)     (outer residual for param stream)
dropout_2
```

### Forward pass per block

```python
combined = cat([features, params], dim=-1)           # [1, L, model_dim]
attn_out  = attn(norm_1(combined), ...)               # [1, L, model_dim]
combined  = combined + attn_out                       # residual

z_ff      = ff(norm_2(combined))                      # [1, L, model_dim]
combined  = combined + z_ff                           # residual

# Outer residuals — project back to each stream independently
features = features + dropout(proj_out(combined))     # [1, L, feature_dim]
params   = params   + dropout(W_param(combined))      # [1, L, param_dim]
```

The outer residual design allows information to flow cleanly between streams in the FFN while keeping
stream-specific projections for the final update.

---

## Relational attention (`RelationalAttentionBlock`)

### Edge types

`variable_list_to_entity_graph` builds a binary edge mask `[L, L, R]` where R=6 relationship types:

| Index | Name | Meaning |
|---|---|---|
| 0 | ATTR | variable → its attribute entity |
| 1 | ATTR_INV | attribute entity → variable |
| 2 | ANNOT | variable → its annotator entity |
| 3 | ANNOT_INV | annotator entity → variable |
| 4 | ITEM | variable → its item entity |
| 5 | ITEM_INV | item entity → variable |

No edges between two entity tokens, no edges between two variable tokens via this mechanism
(obs-obs edges are handled separately by K_aug pointer).

### Per-head relational attention (`use_per_head_rel=True`, default)

```
head_dim     = model_dim // num_heads          (e.g. 80/4 = 20)
k_content_dim = head_dim - R                   (e.g. 20 - 6 = 14)

Q: D → D  (or D+3 with pointer)   — split per head: [content (kc) | rel (R)]
K: D → H * k_content_dim          — content only
V: D → D                          — full head_dim

Q_content [B, H, L, kc],  Q_rel [B, H, L, R]

content_scores = Q_content @ K_content^T                      [B, H, L, L]
rel_scores     = einsum('bhir, ijr -> bhij', Q_rel, edge_mask) [B, H, L, L]
scores         = (content_scores + rel_scores) / sqrt(head_dim)
```

Each head learns its own relational weights independently.

### Shared-bias relational attention (`--no-per-head-rel`)

Old design (pre-change, available via flag):
```
Q: D → D + R  — content D dims + R shared relational dims
K: D → D      — content only

scores = (Q_content @ K_content^T) / sqrt(head_dim)
       + (Q_rel_shared @ edge_mask^T)   [broadcast over heads]
```
Single shared relational bias added identically to every head. Easier to train (fewer params) but
all heads must share the same relational attention pattern.

### K_aug pointer (`--use-pointer`)

Adds direct obs-obs connections independent of entity edges.
Builds `K_aug [L, L, 3]` — channels: same-attribute, same-annotator, same-item.
Only variable tokens get valid IDs; entity tokens stay at -1 (excluded via validity mask).

Three extra Q dims (`Q_ptr [B, L, 3]`):
```python
ptr_bias = (Q_ptr.unsqueeze(2) * K_aug.unsqueeze(0)).sum(-1)   # [B, L, L]
scores   = scores + ptr_bias.unsqueeze(1)                       # broadcast over heads
```

### Relational value augmentation (`--use-rel-value`)

Adds a learned per-relation correction to the output of each head:
```python
attn_r_mass = einsum('bhij, ijr -> bhir', attn, edge_mask)      # [B, H, L, R]
rel_aug     = einsum('bhir, hrd -> bhid', attn_r_mass, rel_value_emb)
out         = out + rel_aug
```
`rel_value_emb [H, R, head_dim]` initialized to zero (no-op at start).

### Add-one attention (`--use-addone-attn`)

```
attn_ij = exp(s_ij) / (1 + sum_k exp(s_ik))
```
Sum ≤ 1. A token can abstain from attending, directing probability mass to a virtual null token.
Numerically stable via max-shift before exp.

---

## Loss function

### Per-token loss

`RatingVariableType.compute_loss_breakdown` computes CE for ALL rating tokens (observed + masked + missing)
then splits by status. The training loss only backprops through masked (and optionally observed) tokens.

- **Hard labels** (`rating_dist=None`): standard `CrossEntropyLoss`.
- **Soft labels** (`rating_dist` present): `-(soft_targets * log_softmax(logits)).sum(-1)`.
  If any token in the batch has a soft dist, all tokens use the soft path
  (hard-labeled tokens use one-hot as their soft target, numerically equivalent).

Logits extracted from param stream: `params[:, 1 : 1+num_classes]` — skipping the mask bit.

### Training objective

```
total_loss = masked_loss_weight * mean_CE_masked + observed_loss_weight * mean_CE_observed
```

Defaults: `masked_loss_weight=15.0`, `observed_loss_weight=1.0`.

The high masked weight emphasizes the imputation task (predicting masked values from context)
over reconstruction of observed values.

### Mask augmentations

Each epoch runs `mask_augmentations` training steps (default: 5), each with a fresh independent masking
draw over the training observations. This increases the diversity of training signal without changing
the dataset size.

---

## Key config parameters

See `EntityMarformerConfig` in `config.py`:

| Param | Default | Notes |
|---|---|---|
| `embedding_dim` | 72 | Total model_dim. Must be divisible by `attention_heads`. Current script: 80. |
| `num_layers` | 4 | Transformer depth. Current script: 6. |
| `attention_heads` | 4 | Heads. `head_dim = embedding_dim / heads`. With dim=80, heads=4: head_dim=20. |
| `d_ff` | 128 | FFN hidden dimension. |
| `num_ffn_layers` | 1 | FFN depth (1 = single Linear+ReLU layer inside FFN). |
| `dropout` | 0.1 | Applied after attention and FFN. |
| `logit_high` | 20.0 | Logit spike magnitude for observed hard-label inputs. |
| `use_per_head_rel` | True | Per-head relational bias (new design). |
| `use_pointer` | False | K_aug obs-obs pointer mechanism. |
| `use_rel_value` | False | Relational value augmentation. |
| `use_addone_attn` | False | Add-one attention. |

---

## Why Entity Marformer underperforms old Marformer (diagnostic)

With default flags (no pointer, no rel-value, no addone), Entity MF struggles because:

1. **Multi-hop bottleneck**: Observation tokens start with zero feature identity. To learn "I am
   annotator J on item K for attribute I", they must attend to entity tokens and then propagate
   that information back. This requires at minimum 2 attention hops, whereas old Marformer
   directly computes `W_I@attr + W_J@annot + W_K@item` as the embedding.

2. **No direct obs-obs signal**: Two observations that share the same annotator are only connected
   via a 2-hop path through the annotator entity token. Old Marformer had direct obs-obs edges
   via K_aug. Fix: `--use-pointer`.

3. **Attention dilution**: The sequence contains entity tokens (I+J+K extra tokens) that don't
   carry observation information, diluting attention capacity.

4. **Item representation collapse**: With `item_dropout_rate=1.0`, all items have identical
   feature vectors (type centroid only). The model cannot distinguish items at train time.
   This is intentional (items are new at test time) but may slow convergence.

Enabling `--use-pointer` is the most important fix for closing the performance gap.

---

## Script reference

Main run script: `scripts/real_data/run_real_with_entity_mf.sh`

All hyperparameters are set as named variables at the top with inline comments.
Run name is auto-generated from key hyperparams for easy identification.
Boolean flags (`USE_POINTER`, `USE_REL_VALUE`, etc.) control which extensions are active.
