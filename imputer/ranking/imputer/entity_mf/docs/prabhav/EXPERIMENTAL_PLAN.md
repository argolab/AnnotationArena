# Entity Marformer — Experimental Plan

**Date:** 2026-03-20
**Context:** 33-run final ablation study revealed that the Pointer Network is the primary driver of performance (loss ~0.95). No other single ablation broke the ~1.39 random-guess baseline. DevNorm and Item Reg showed minor benefit. This plan refines the architecture and validates each decision systematically.

---

## Background: LayerNorm Design Choice

The current model concatenates the feature stream (75-dim) and param stream (5-dim) into an 80-dim vector and applies a single LayerNorm over the combined representation. This is problematic: the param stream is a structured 5-dim vector (mask bit + class logits) with different semantics and scale than the feature stream. The joint LayerNorm statistics are dominated by the 75-dim feature side, meaning the param stream is effectively normalized relative to features it has no relation to.

Three options considered:
1. **Option 1** — LayerNorm on features only everywhere; concat with params after. *(Most principled.)*
2. **Option 2** — LayerNorm on features only before first sublayer, joint afterwards. *(No clear motivation; dropped.)*
3. **Option 3** — Joint LayerNorm on concat(features, params). *(Current design; all 33 ablation runs used this.)*

Option 1 is adopted going forward as the principled fix.

---

## Relation Value Augmentation Fix

The current `use_rel_value` implementation uses per-head value embeddings `[H, R, head_dim]` aggregated per-head. We align this to the simpler shared formulation from a colleague's implementation:

```
val_rel[i, j, d] = sum_r  edge_mask[i, j, r] * value_rel[r, d]       # [L, L, D]
attn_mean[b, i, j] = mean over heads of attn[b, h, i, j]              # [B, L, L]
bias[b, i, d] = sum_j  attn_mean[b, i, j] * val_rel[i, j, d]         # [B, L, D]
bias_h = bias reshaped to [B, H, L, head_dim]
out = out + bias_h
```

Single shared `value_rel [R, D]` embedding; aggregation uses mean attention across heads. This is the version used in all Section 1 experiments.

---

## Section 1 — LayerNorm Ablation

**Goal:** Determine whether Option 1 LayerNorm changes performance characteristics for the base model, Pointer Network, and Relation Value Augmentation.

**Base model:** Same config as `run_exp00_base.sh` (shared-bias, scale-shared-rel, item_dropout=0.0, no features) but with **Option 1 LayerNorm**.

| Experiment | LayerNorm | Features | Runs |
|---|---|---|---|
| Base | Option 1 | None | 2 |
| + Pointer | Option 1 | Pointer Network | 2 |
| + Rel Value | Option 1 | Rel Value Aug (shared, colleague version) | 2 |

**Total: 6 runs.**

We already have Option 3 results for all three from the 33-run study. No need to re-run Option 3.

**Expected outcome:** Best architecture from Section 1 becomes **EF1**. Hypothesis: Pointer + Option 1 LN.

---

## Section 2 — Minor Ablations on EF1

**Goal:** Identify which minor architectural additions improve EF1.

**Base model:** EF1 (best from Section 1).

| Experiment | Addition | Runs |
|---|---|---|
| Base + Dev Norm | LayerNorm applied to deviation before adding to type centroid | 3 |
| Base + Item Reg | L2 regularization on item deviation tables (weight 1e-3) | 3 |
| Base + Item Reg + Dev Norm | Both combined | 3 |
| Base + Add-One Attention | exp(s) / (1 + sum exp(s)), no forced attention | 3 |

**Total: 12 runs.**

Note: Attribute Reg is deferred — can be added later if Item Reg shows strong benefit.

**Expected outcome:** Best architecture becomes **EF2**. Hypothesis: Item Reg + Dev Norm likely to help. Add-One Attention theoretically well-motivated for missing data setting (tokens with no relevant neighbors should be allowed to attend to nothing).

---

## Section 3 — Relational Edge Ablation (Sanity Check)

**Goal:** Confirm that relational edges carry information beyond the pointer network.

**Base model:** EF2.

**Setup:** Remove relational edge mask from attention scoring (zero out edge_mask / disable Q_rel contribution) while **keeping the Pointer Network** (K_aug). This means:
- Entity tokens do NOT connect to their rating observations via relational scoring.
- Rating observations still share pointer edges when they share entities (same annotator, same attribute, same item).

If relational edges carry genuine information, performance should degrade significantly compared to EF2.

| Experiment | Runs |
|---|---|
| EF2 without relational edges (pointer only) | 3 |

**Total: 3 runs.** (EF2 itself serves as the comparison baseline — already run in Section 2.)

---

## Section 4 — Dimensionality and Stability Analysis

**Goal:** Understand whether training instability comes from model capacity or data/initialization variance.

### 4a — Seed Control (Tom Wang's Suggestion)

Isolate the two sources of training variance:
- **Masking seed**: controls which ratings are masked per epoch/batch.
- **Initialization seed**: controls model weight initialization.

Run EF2 with:
1. Fixed init seed, varying masking seed
2. Fixed masking seed, varying init seed

This identifies which source drives run-to-run variance before trying to fix it with more capacity.

### 4b — Dimensionality Sweep

Base model: EF2 config. Sweep feature/model dimensionality:

| model_dim (feature + param) | Runs |
|---|---|
| 80 (75 + 5) — current | already have from EF2 |
| 160 (155 + 5) | 3 |
| 320 (315 + 5) | 3 |

**Total: 6 runs** (plus seed control runs from 4a, count TBD).

If higher dimensionality stabilizes training or improves performance, the resulting model becomes **EF3**. Otherwise EF2 stands.

---

## Next Steps (To Be Discussed)

- **Real data evaluation:** Test EF2/EF3 on HANNA and SummEval without LLM observations.
- **Artificial data:** Run with STAN-generated data as a controlled generalization test.
- Additional architectural experiments TBD based on Section 4 findings.

---

## Naming Convention

| Name | Definition |
|---|---|
| EF1 | Best model from Section 1 (LayerNorm ablation) |
| EF2 | Best model from Section 2 (minor ablations on EF1) |
| EF3 | EF2 with increased dimensionality if Section 4b shows benefit |
