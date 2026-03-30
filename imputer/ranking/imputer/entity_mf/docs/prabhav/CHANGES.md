# Entity Marformer: Changes Log

This document records all changes made to the `entity_mf` module relative to the
baseline commit on the `entity-marformer` branch. Use this to orient a new session
without re-deriving what was done and why.

**Files changed:**
- `entity_mf/config.py`
- `entity_mf/types.py`
- `entity_mf/data.py`
- `entity_mf/model.py`
- `entity_mf/eval.py`
- `entity_mf/train.py`
- `scripts/real_data/run_real_with_entity_mf.sh`

---

## 1. Per-head relational attention + flag for old shared-bias design

**File:** `model.py` (`RelationalAttentionBlock`), `config.py`

**Before (old shared-bias design):**
- Q projected to `D + R`, K projected to `D + R` (but K's extra R dims were unused).
- The last R dims of Q formed a single shared relational bias vector — identical for all heads.
- Relational score: `(Q_rel.unsqueeze(2) * edge_mask.unsqueeze(0)).sum(-1)` → `[B, L, L]`, broadcast over heads.
- Content score: `(Q_content @ K_content^T) / sqrt(head_dim)`, using full `head_dim` for content.

**After (per-head design, now default):**
- Q projected to `D` (or `D+3` with pointer). Each head gets `head_dim` total, split into:
  - `k_content_dim = head_dim - R` content dims
  - `R` relational dims
- K projected to `H * k_content_dim` (content only, no wasted relational dims).
- Relational score: `einsum('bhir, ijr -> bhij', Q_rel, edge_mask)` — **each head has independent relational weights**.
- Both content and relational scores divided together: `(content + rel) / sqrt(head_dim)`.
- Requires `head_dim > R` (i.e. `embedding_dim / num_heads > num_relationships`).

**Flag:** `config.use_per_head_rel = True` (default). Pass `--no-per-head-rel` to revert to old shared-bias design.

**Why:** Per-head relational attention lets each head specialize on different relationship types (e.g., one head attends along ATTR edges, another along ANNOT edges). The old shared bias applied the same relational scaling to every head.

---

## 2. K_aug pointer mechanism (`--use-pointer`)

**Files:** `model.py`, `data.py`, `config.py`, `train.py`

**What it does:**
Adds direct obs-obs connections capturing shared identity (same attribute, same annotator, same item).
This mimics the old Marformer's core strength: obs tokens that share I/J/K directly know about each other,
without requiring a 2-hop path through entity tokens.

**Implementation:**
- `data.py`: Variable tokens now store `attribute_id`, `annotator_id`, `item_ids` in `raw_data`
  (needed at forward time to build K_aug).
- `model.py` forward pass: builds `K_aug [L, L, 3]` — a binary float tensor where channel 0/1/2
  is 1 if position i and j share the same attribute/annotator/item respectively.
  Both positions must have valid IDs (>= 0); entity tokens stay at -1 so they never spuriously match.
- `RelationalAttentionBlock`: Q projection outputs 3 extra dimensions (`Q_ptr [B, L, 3]`) when pointer enabled.
  Pointer bias: `ptr_bias = (Q_ptr.unsqueeze(2) * K_aug.unsqueeze(0)).sum(-1)` → `[B, L, L]`.
  Added to scores after scaling, broadcast over all heads.

**Note:** This is functionally identical to old Marformer's K_aug, with one improvement:
old Marformer did not validity-check IDs, so two tokens with `item_id=-1` would spuriously match.

**Flag:** `--use-pointer` (default: off).

---

## 3. Relational value augmentation (`--use-rel-value`)

**File:** `model.py`, `config.py`, `train.py`

**What it does:**
Standard attention aggregates `V(x_j)` regardless of *which relationship type* the attended mass
came through. This extension adds a learned correction term per relationship:

```
V_effective(i,j) = V(x_j) + sum_r  e_r * edge_mask[i, j, r]
```

After attention weights are computed:
```python
attn_r_mass = einsum('bhij, ijr -> bhir', attn, edge_mask)   # [B, H, L, R]
rel_aug      = einsum('bhir, hrd -> bhid', attn_r_mass, rel_value_emb)  # [B, H, L, hd]
out = out + rel_aug
```

`rel_value_emb [H, R, head_dim]` is initialized to **zero** (no-op at start, learned from there).

**Flag:** `--use-rel-value` (default: off).

---

## 4. Add-one attention (`--use-addone-attn`)

**Files:** `model.py`, `config.py`, `train.py`

**What it does:**
Replaces standard softmax normalization with:
```
attn_{ij} = exp(s_{ij}) / (1 + sum_k exp(s_{ik}))
```
Sum of attention weights ≤ 1. A token can "opt out" of attending to anyone — the remaining mass goes
to a virtual null token. Useful for entity tokens with no meaningful neighbors in the current chunk.

**Implementation:**
- Max-shift for numerical stability before exp.
- Masked positions zeroed in `exp_s` before the denominator sum.

**Flag:** `--use-addone-attn` (default: off, uses standard softmax).

---

## 5. Pre-LN transformer block

**File:** `model.py`

**Before:** Block had `norm_2` before FFN but no norm before attention (Post-LN-style for attention).

**After:** Added `norm_1` applied to `combined` before it enters attention. Both sub-layers now use Pre-LN:
```
combined = combined + attn(norm_1(combined))
combined = combined + ff(norm_2(combined))
```
This is the standard Pre-LN pattern from "On Layer Normalization in the Transformer Architecture" and
is critical for stable training of deep networks.

---

## 6. Deviation dropout

**Files:** `model.py`, `types.py`

**What it does:**
During training, entity deviation embeddings are randomly zeroed with probability `dropout_rate`.

- `VariationConfig.dropout_rate` — new field (default: 0.0, no dropout).
- `ItemEntityType(dropout_rate=1.0)` — items **always** drop their deviation during training.
  Rationale: items are unseen at test time (new items must be imputed from scratch with no item-specific
  prior), so training with item deviations would overfit to training items and mislead evaluation.
- `AnnotatorEntityType` — `dropout_rate=0.0` by default (annotators are known at test time).

**In model.py forward:**
```python
if self.training and t.variation.dropout_rate > 0:
    if torch.rand(1).item() < t.variation.dropout_rate:
        dev = torch.zeros_like(dev)
feat_vec = feat_vec + dev
```

**Flag:** Controlled via `--item-dropout-rate` CLI arg (default: 1.0).

---

## 7. Soft distribution input encoding (`--llm-input-dist`)

**File:** `types.py` (`RatingVariableType.build_param`)

**What it does:**
The LLM annotator in the LLMRubric dataset can provide a full distribution over classes (not just a
hard vote). When `llm_input_dist=True`, observed tokens with a soft `rating_dist` encode their param
stream as `log(clamp(p_c, min=exp(-logit_high)))` instead of a one-hot spike at `logit_high`.

- Tokens with truly one-hot distributions (checked via `_is_one_hot`) still use the hard spike.
- Tokens that are masked or missing still use the mask bit regardless.

**Flag:** `--llm-input-dist` (default: off). Only meaningful with `BUNDLE=dist`.

---

## 8. Soft distribution loss

**File:** `types.py` (`RatingVariableType.compute_loss_breakdown`)

**What it does:**
If any token in the batch has a `rating_dist` (regardless of `--llm-input-dist`), all tokens use
the soft CE loss path:
```
loss_j = -sum_c  soft_target_{j,c} * log_softmax(logit_{j,c})
```
For tokens without `rating_dist`, `soft_target` is the one-hot at `rating_value` — numerically
identical to standard hard CE.

This means: when running with `BUNDLE=dist`, soft targets are used for the loss even if
`--llm-input-dist` is not set (the input encoding and loss target are independent choices).

---

## 9. Observed loss in training objective

**Files:** `types.py`, `eval.py`, `train.py`

**Before:** Training loss = mean CE over masked tokens only.

**After:** Training loss = `masked_loss_weight * mean_CE_masked + observed_loss_weight * mean_CE_observed`.

- `LossBreakdown.observed_loss_tensor` — new field; grad-enabled tensor for observed CE.
- `_aggregate_loss_from_breakdowns` accumulates both `weighted_masked_sum` and `weighted_observed_sum`.
- `compute_trainable_loss(masked_loss_weight, observed_loss_weight)` — defaults: 1.0 and 0.0.

**Default in script:** `MASKED_LOSS_WEIGHT=15.0`, `OBSERVED_LOSS_WEIGHT=1.0` (matches old Marformer defaults).

**Flags:** `--masked-loss-weight`, `--observed-loss-weight`.

---

## 10. Mask augmentations

**Files:** `train.py`

**Before:** Dummy dataloader had 1 entry → 1 training step per epoch with one masking draw.

**After:** Dataloader has `mask_augmentations` entries → that many training steps per epoch, each with
an independent fresh masking draw. Default: 5. This gives the model more diverse training signal per
epoch without changing the data, equivalent to old Marformer's behavior.

**Flag:** `--mask-augmentations` (default: 5).

---

## 11. New architecture/training CLI flags

**File:** `train.py`

The following flags were missing from `main()` and are now added:

| Flag | Controls |
|---|---|
| `--embedding-dim` | `config.embedding_dim` (was already there) |
| `--num-layers` | `config.num_layers` (was already there) |
| `--attention-heads` | `config.attention_heads` |
| `--d-ff` | `config.d_ff` |
| `--num-ffn-layers` | `config.num_ffn_layers` |
| `--dropout` | `config.dropout` |
| `--use-per-head-rel` / `--no-per-head-rel` | `config.use_per_head_rel` |
| `--use-pointer` | `config.use_pointer` |
| `--use-rel-value` | `config.use_rel_value` |
| `--use-addone-attn` | `config.use_addone_attn` |
| `--mask-augmentations` | training steps per epoch |
| `--masked-loss-weight` | weight on masked CE |
| `--observed-loss-weight` | weight on observed CE |
| `--llm-input-dist` | soft dist input encoding |
| `--item-dropout-rate` | item deviation dropout |
| `--annotator-reg-weight` | L2 reg on annotator deviations |

---

## Open items / TODOs

- **Type embedding initialization:** `kaiming_normal_` is used for type centroid embeddings.
  This was flagged with a `TODO` — it may be better to use a smaller-scale normal init.
  Not changed yet.
- **`compute_loss` (old method):** `RatingVariableType.compute_loss` is now dead code
  (superseded by `compute_loss_breakdown`). Can be removed once stable.
