# Entity Marformer: Architecture Analysis & Comparison Report

**Date**: 2026-03-13
**Scope**: Full comparison of EntityMarformer (`entity_mf`) vs. dual-stream Marformer (`Marformer-Test`), architectural issue identification, training methodology gaps, and discussion of open questions from advisor/collaborator chat.

---

## 1. Architecture Overview

### 1.1 Marformer-Test (Dual-Stream, `ranking_imputer.py`)

The original Marformer operates on a flat sequence of **observation tokens only** (no entity tokens). Each token represents one (annotator j, attribute i, item k) triplet.

**Feature stream (per token at layer 0)**:
```
attr_vec  = [1,0,0] ++ attr_embedding[i]      # [D]
annot_vec = [0,1,0] ++ annot_embedding[j]     # [D]
item_vec  = [0,0,1] ++ item_embedding[k]      # [D]

feature = W_I @ attr_vec + W_J @ annot_vec + W_K @ item_vec  # [D]
```

**Param stream (per token)**:
```
param[0]   = mask bit (1 if missing/masked, 0 if observed)
param[1:6] = rating logits (LOGIT_HIGH spike at rating_value, or log(p_c) for LLM dist)
param[1:3] = ranking logits (LOGIT_HIGH at winner position)
```

**Relational knowledge (pointer mechanism, `K_aug`)**:
```
K_aug[i, j, :] = [same_attr(i,j), same_annot(i,j), same_item(i,j)]  # [N, N, 3]
```
This is an additive bias term in attention scores. It tells the transformer "token i and token j share attribute/annotator/item".

**Transformer block**: `TransformerBlock` — attention over combined (feature ++ param) stream, then two-stream FFN (one on features, one on params), pre-LN.

---

### 1.2 Entity Marformer (`entity_mf/model.py`)

The Entity Marformer introduces **entity tokens** alongside observation tokens. All tokens — variables (ratings/rankings) and entities (attributes, annotators, items) — sit in a single flat sequence.

**Sequence layout (per forward pass)**:
```
[rating/ranking tokens] | [attribute tokens (I)] | [annotator tokens (J)] | [item tokens (K)]
```

**Feature stream at layer 0**:
- Entity tokens: `type_base_embedding[type_name] + deviation_table[entity_id]`
- Variable tokens: `type_base_embedding["rating"]` or `type_base_embedding["ranking_pairwise"]`

**Param stream at layer 0**: built by `EntityType.build_param()`. For rating: `[mask_bit, logit_0, ..., logit_C-1]`. Entity tokens: zeros.

**Relational attention** (`RelationalAttentionBlock`): An edge mask `[L, L, R]` (6 relationship types: ATTR, ATTR_INV, ANNOT, ANNOT_INV, ITEM, ITEM_INV) encodes graph structure. An extra R-dimensional component in Q and K projections enables the model to learn per-relationship attention biases.

---

## 2. Critical Architectural Bugs

### 2.1 Relational Bias is NOT Head-Specific (CONFIRMED BUG)

**Jason explicitly flagged this on March 3, 2026.** Tom Wang confirmed it: "I'm currently broadcasting the relational score across heads."

**Current implementation** (`model.py:RelationalAttentionBlock.forward`):
```python
Q_rel = Q_full[..., D:]           # [B, L, R] — one shared R-vector per token
Q_rel_exp = Q_rel.unsqueeze(2)    # [B, L, 1, R]
edge_mask_exp = edge_mask.unsqueeze(0)  # [1, L, L, R]
rel_scores = (Q_rel_exp * edge_mask_exp).sum(-1)  # [B, L, L]
rel_scores = rel_scores.unsqueeze(1)  # [B, 1, L, L] — BROADCAST to all heads
scores = base_scores + rel_scores
```

**The problem**: The relational bias is identical for all heads. Head 1 and Head 4 receive the same `+r` boost for the same (query, key) pair regardless of the edge type. Heads cannot specialize to different relationship types (e.g., one head attending over same-item tokens, another over same-annotator tokens).

**Jason's intended design** (from the discussion post, Feb 2 + March 3):
- Q matrix projects to `d/H` per head (standard)
- K matrix projects to `(d/H - r)` content dims **per head**, then concatenates the r-dimensional multi-hot edge vector to get `d/H` total per-head key
- So: `K_projected = [W_K_h @ x ... (d/H - r dims) | edge_mask[i→j, :] ... (r bits)]`
- Each head's Q_h learns to weight those last r bits differently → head-specific relational attention

**Current implementation vs. intended**:

| Aspect | Current | Intended |
|--------|---------|----------|
| Q per head | `D/H` content | `D/H` content |
| K per head | `D/H` content | `(D/H - R)` content + `R` edge bits |
| Relational bias | Shared scalar added to all heads | Per-head via Q-K dot product last R dims |
| Head specialization | None (all heads see same bias) | Each head can weight edges differently |

This is the single highest-priority architectural fix. Without it, all heads collapse to attending the same relationship pattern.

---

### 2.2 Item Deviations Should Be Disabled (or 100% Dropout)

**Current code** (`types.py:ItemEntityType`):
```python
class ItemEntityType(NullEntityType):
    """Item entity: always new at test time, no deviation, no direct prediction."""
    def __init__(self, num_items: int):
        super().__init__(
            name="item",
            variation=VariationConfig(enabled=True, num_entities=num_items, reg_weight=0.0),
        )
```

**The comment says "no deviation"** — but `enabled=True` with `num_entities=num_items` means the model IS learning a per-item deviation table (`[num_items, feature_dim]`).

**Why this is wrong**: Jason's proposal states:
> "We don't want to learn specific layer-0 embeddings for each item, because the test-time items will always be new — the dropout rate is just 100% in this case."

With `enabled=True` and no regularization and no dropout:
1. The model can memorize per-item embeddings on training items.
2. At test time, test items have indices within `[0, K)` (they were seen during training in transductive mode), but if non-transductive, test items will share indices — leading to incorrect initialization.
3. Even in transductive mode, learning item deviations without dropout means the model can "cheat" by encoding the correct answer in the item embedding rather than learning to propagate information through the graph.

**Fix options**:
- `VariationConfig(enabled=False, ...)` — disable entirely (cleanest, matches Jason's recommendation)
- `VariationConfig(enabled=True, reg_weight=<large>)` — heavy L2 regularization to keep near zero
- Add item embedding dropout (100% during training) — Jason's Option 3 but for items

---

### 2.3 Missing Observed Loss Contribution

**Marformer** uses `masked_loss_weight=15, observed_loss_weight=1`. Observed tokens — including LLM annotator tokens — contribute to the gradient even when observed.

**Entity Marformer** only backprops on masked tokens (from `compute_trainable_loss → trainable_loss = weighted_masked_sum / n_masked`). Observed tokens contribute zero to the training objective.

Impact: LLM annotator tokens (status=2) are completely wasted during training in Entity Marformer. The LLM provides rich distributional information that the Marformer uses explicitly. This is a significant source of training signal that's being discarded.

---

## 3. Training Methodology Gaps

### 3.1 Mask Augmentations: 1x vs 5x

| Setting | Marformer | Entity Marformer |
|---------|-----------|-----------------|
| `mask_augmentations` | **5** | **1** (hardcoded dummy loader) |
| Steps per epoch | 5 | 1 |

With 5 augmentations, Marformer sees **5 different masking draws per epoch**, effectively getting 5x the gradient signal. Entity Marformer's training is substantially more sample-inefficient per epoch. With the same number of epochs (150 vs 180), Marformer does 750 gradient steps vs. Entity Marformer's 180.

This may partially explain why the Entity Marformer underperforms even when controlling for architecture — it's simply seeing far less data in the same wall-clock time.

**Fix**: Add mask augmentations to `EntityMarformerLightningModule` (expand the dummy dataset to `mask_augmentations` items).

---

### 3.2 No Cosine LR Schedule / Warmup

| Setting | Marformer | Entity Marformer |
|---------|-----------|-----------------|
| LR schedule | **Cosine with warmup (5 steps)** | **Flat** |
| LR | `2e-4` | `2e-4` |
| Weight decay | `0.01` | `0.01` |

The cosine schedule with warmup is critical for transformer stability, especially at the start of training when gradient norms can be large. Without warmup, early training instability can permanently damage initial embeddings.

---

### 3.3 Transductive Learning Not Used in Run Script

The `run_real_with_entity_mf.sh` script does **not** pass `--transductive-learning`. Marformer explicitly uses it. In transductive mode, test item embeddings receive gradient signal from observed test data — this is especially important since items are all different between train/test in this setup.

Without transductive learning, the entity tokens for test items remain at type-base-embedding + zero-deviation throughout training, meaning they're effectively random at eval time.

---

### 3.4 Missing Gradient Clipping

Marformer script uses `--gradient-clip-val 0.0` (disabled in the current run script, but the option exists). Entity Marformer has no gradient clipping at all. With larger sequence lengths (all N variable tokens + I + J + K entity tokens), gradient explosions are more likely.

---

## 4. Detailed Comparison Table

| Feature | Marformer-Test | Entity Marformer |
|---------|---------------|-----------------|
| **Sequence contents** | N observation tokens | N variable tokens + I+J+K entity tokens |
| **Layer-0 features** | `W_I@attr + W_J@annot + W_K@item` (entity info at layer 0) | Pure type embedding (entity info flows through attention) |
| **Relational signal** | Additive K_aug indicators [N,N,3] | Edge mask [L,L,R] + R-dim Q/K extension |
| **Head-specific relations** | N/A (no heads-by-relation) | Intended but NOT implemented (bug 2.1) |
| **Entity tokens** | No | Yes (I+J+K extra tokens) |
| **Per-entity deviations** | `item_embedding[k]` (full), `attribute_embedding[i]`, `annotator_embedding[j]` | Deviation tables per type (items incorrectly enabled) |
| **Item variation dropout** | `item_embedding_dropout=1.0` (full during train) | None (bug 2.2) |
| **Loss: masked** | Yes, weight=15 | Yes (sole objective) |
| **Loss: observed** | Yes, weight=1 | No (bug 2.3) |
| **Mask augmentations/epoch** | 5 | 1 (gap 3.1) |
| **LR schedule** | Cosine + warmup | Flat (gap 3.2) |
| **Transductive** | Yes (run script) | No (gap 3.3) |
| **Sequence length** | N (≈ I×J×K observed subset) | N + I + J + K |
| **FFN structure** | Two-stream (feature FFN + param FFN) | Single FFN on combined, then project back |
| **Embedding dim** | 72 | 72 |
| **Layers** | 4 | 4 |
| **Heads** | 4 | 4 |
| **d_ff** | 128 | 128 |
| **normalize_parameter** | True | No equivalent (LayerNorm on combined stream) |
| **Temperature** | 1.0 | N/A (no explicit temperature) |
| **Gradient clip** | 0.0 (off) | Off |
| **w_init** | `identity` (W_J, W_K near identity) | No W matrices (replaced by type embeddings) |

---

## 5. Architectural Discussion Points

### 5.1 Layer-0 Initialization: Entity MF vs. Marformer

**Marformer** injects entity identity directly into the layer-0 feature stream via `W_I@attr + W_J@annot + W_K@item`. This gives the transformer a 1-hop head start: even at layer 0, the token "knows" which attribute/annotator/item it belongs to. Tom Wang noted this as a potential inductive bias question.

**Entity Marformer** starts all variable tokens with just a `<rating>` or `<ranking>` type embedding. The entity identity must flow in via relational attention over the first layers. This is **more principled** (supports generalization to new entities) but **requires deeper layers** to replicate what Marformer gets for free at layer 0.

Jason's comment (Feb 27): "the layer-0 embedding of a rating or ranking is just a learned representation of the type `<rating>` or `<ranking>`... it's not crazy to also add [attribute embeddings] into layer zero of the RVs that point to that entity. I'm just not sure it's needed and it may be better to keep it simple."

Current Entity Marformer is correct per the design. But with only 2-4 layers, the information may not propagate effectively. Consider testing with 6-8 layers.

---

### 5.2 Per-Entity Deviations: What Should Be Enabled?

Per Jason's proposal:
- **Attributes (I)**: Closed class, enable per-entity deviation (no dropout). Currently: enabled, reg_weight=0 — OK, but should consider non-zero reg.
- **Annotators (J)**: Enable deviation + regularization. Option 3 (dropout) is preferred. Currently: enabled, reg_weight=0, no dropout — missing regularization and dropout.
- **Items (K)**: Disable entirely (or 100% dropout). Currently: enabled, no reg, no dropout — this is wrong.

The annotator dropout (Option 3 from Jason) is the most important: "Train with dropout so that we still predict well when we back off to the type embedding." This is how the model generalizes to new annotators. Without it, the model just memorizes annotator embeddings and fails on new annotators.

---

### 5.3 Attention Head Specialization: The Right Relational Fix

Tom Wang's question: "Do we want to add a different attention boost to different heads?"

The correct formulation (Jason, March 3):
```
q = [head_1 | head_2 | head_3 | head_4]       ∈ R^d
k = [head_1+r_bits | head_2+r_bits | ...]      ∈ R^d  (each head gets R extra dims)
v = [head_1 | head_2 | head_3 | head_4]       ∈ R^d
```

For each head h:
- Q_h: `Linear(D, D/H)` — standard
- K_h: `Linear(D, D/H - R)` — projects to fewer content dims
- Concatenate R edge-mask bits → K_h_extended ∈ R^{D/H}
- Score_h = Q_h @ K_h_extended^T / sqrt(D/H)
- The last R dims of Q_h learn to attend differently to different edge types

This should NOT require a change to the Q matrix projection dim (Q still projects to D total = H × D/H). The change is only in K: `K` projects to `D - H*R` content dims total, and R bits are **appended per head** rather than added once globally.

Tom Wang's alternative (Q extended): `Q projects to (D/H + R) per head`. Jason said Q should NOT have special dimensions — "Q can learn how much to care about those bits" via the Q's projection of the normal D dims. The K side is where the edge bits live.

---

### 5.4 SHARED_I/SHARED_J/SHARED_K vs. Entity Tokens

Tom Wang asked: "Do we still need SHARED-I, SHARED-J, SHARED-K?"

The old Marformer K_aug indicators `[same_attr, same_annot, same_item]` served the same role as the entity tokens: telling the model which observations share structure.

In Entity Marformer, this information is implicitly available: if variable token v1 and v2 both point to attribute token i via ATTR edges, and the attribute token i integrates information from both, then at layer 2+, v1 and v2 can attend to i and get back the integrated representation. This is a 2-hop path (v1→i→v2 effectively), vs. Marformer's 1-hop direct SHARED_I edge.

Jason: "I don't think we need either of those things... I could imagine them either helping (by doing in one layer what would otherwise take two) or hurting."

Entity MF currently has no direct obs-to-obs edges of the "SAME_ATTR" type — only obs→entity and entity→obs edges. This is correct per design, but is another reason why more layers help.

---

### 5.5 Why max_item=10 Helps (Confirmed in Both Systems)

Both systems use `max_item=10` and it empirically helps. Analysis from prior discussion:
1. **Attention focus**: With 10 items, the fraction of relevant context is much higher — every token has a meaningful neighbor.
2. **Gradient concentration**: Item embedding gradients aren't diluted across K=225 items. With 10 items per chunk, each item gets ~22.5x more gradient per epoch.
3. **Implicit regularization**: The model must generalize across chunks rather than memorizing a single monolithic context.
4. **Consistent masked/observed ratio**: Each chunk has a similar density of masked vs. observed tokens.

In Entity Marformer, this is especially important because entity tokens scale with K (adding 225 item entity tokens to the sequence). With max_item=10, the sequence is roughly `N_vars/22 + 8 + J + 10` instead of `N_vars + 8 + J + 225`.

---

### 5.6 Sanity Check: Graph Message Passing

The discussion notes: "Need to run training with more layers. Sanity check with some simple graph inference task to test relational system works (passing messages along edge relationship chain of r1→r2→r3, count subtree number, etc.)"

This is important because the relational attention bug (2.1) means the current system is not actually doing correct relational message passing. A simple synthetic sanity check would be:
- Generate a bipartite graph where A-nodes have a hidden label
- B-nodes connect to exactly one A-node
- Task: predict A's label from attending to B's, using CONTAINS/BELONGS_TO edges
- If heads can't specialize by edge type, this will fail in 2 layers

---

## 6. Priority Issue List

Ordered by likely impact on fixing the poor performance (acc ~48-49%):

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| P0 | **Relational bias not head-specific** (Bug 2.1) | High — heads can't specialize by edge type | Medium (rearchitect K projection) |
| P0 | **Item deviations enabled without dropout** (Bug 2.2) | High — model memorizes items, fails on new | Low (change VariationConfig or add dropout) |
| P1 | **No mask augmentations** (Gap 3.1) | High — 5x less training signal | Low (expand dummy dataset size) |
| P1 | **No observed loss** (Gap 2.3) | Medium — LLM signal wasted | Low (add observed weight param) |
| P1 | **No cosine schedule** (Gap 3.2) | Medium — training instability | Low (copy schedule from Marformer trainer) |
| P2 | **No transductive learning** (Gap 3.3) | Medium — test items cold-start | Low (add flag to run script) |
| P2 | **Annotator dropout missing** (related to Bug 2.2) | Medium — can't generalize to new annotators | Medium (add per-entity dropout in forward) |
| P3 | **Shallow depth** (2-4 layers for multi-hop paths) | Medium — entity info needs more hops | Low (increase num_layers) |
| P3 | **Attention head specialization monitoring** | Diagnostic | Low (log per-edge-type attention weights) |

---

## 7. Evaluation Methodology Comparison

| Aspect | Marformer | Entity Marformer |
|--------|-----------|-----------------|
| Primary eval metric | Accuracy on `test_missing` rating tokens | Accuracy on `test_missing` rating tokens |
| Secondary metrics | RMSE, entropy, per-attribute breakdown | xent by status/type (`observed/masked/missing`) |
| Eval mode | Separate eval engine runs | Inline `evaluate_entity_marformer_split` |
| Predictives | Saves `train_predictives.json` + `test_predictives.json` | Not implemented |
| TensorBoard | Full: grad norms, param norms, LR, attention weights | Basic: train/loss, reg_loss |
| Attention diagnostics | `collect_attention_stats` per block | None |
| Training history | JSON serialized per epoch | JSON serialized per epoch (good) |

The Marformer has significantly more diagnostic tooling (grad/param norms, per-head attention stats). The attention diagnostics are specifically what Jason asked for: "We should investigate after training what the heads are actually doing." Adding per-edge-type average attention weights to Entity Marformer logging would help diagnose whether the relational attention fix is working.

---

## 8. Summary and Next Steps

Entity Marformer currently achieves ~48-49% accuracy (near chance for 5-class) vs. Marformer's substantially better results. The gap is likely a combination of:

1. **Broken relational mechanism** (Bug 2.1) — the core innovation is not working as intended
2. **Item deviation memorization** (Bug 2.2) — model cheats instead of generalizing
3. **5x less training signal** (Gap 3.1) — mask augmentations missing
4. **LLM signal wasted** (Gap 2.3) — observed loss zero

Fixes in order:
1. Fix `RelationalAttentionBlock`: implement head-specific edge bits in K (not broadcast)
2. Set `ItemEntityType.variation.enabled = False` (or add 100% item dropout)
3. Add annotator deviation dropout (Jason's Option 3)
4. Add 5 mask augmentations per epoch
5. Add observed loss weight (even at weight=0.1)
6. Add cosine schedule + warmup
7. Enable transductive learning in run script
8. Run sanity check on synthetic relational graph task

The architecture itself (entity tokens + typed graph + relational attention) is sound and well-motivated. The implementation has specific fixable bugs, not fundamental design flaws.
