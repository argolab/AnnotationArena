## Entity Marformer (`entity_mf`) Overview

This package is a minimal, research-oriented implementation of an **entity-centric Marformer** that generalizes beyond the rating/ranking domain while staying easy to hack on.

It is intentionally small and modular: you define **types**, we build an **entity graph**, run a **relational transformer**, and compute **type-specific losses** with clear breakdowns.

---

## Big picture

- **Goal**: Learn representations for heterogeneous entities and variables (attributes, annotators, items, ratings, rankings) in a unified transformer, with flexible per-type adapters and losses.
- **Key ideas**:
  - Tokens represent both **entities** and **random variables**.
  - A small **type system** defines how to encode inputs and compute losses.
  - A typed **entity graph** provides edges (relationships) for relational attention.
  - The **EntityMarformer** runs attention over this graph and returns a parameter stream for variables.
  - A separate **evaluation layer** turns parameter predictions into masked-only training loss plus breakdown metrics.

---

## Directory map

- `types.py` – type system and domain-3 concrete types.
- `data.py` – `Token` / `EntityGraph` and bundle → graph conversion.
- `model.py` – `EntityMarformer` and relational attention block.
- `config.py` – small config dataclass for model hyperparameters.
- `eval.py` – loss aggregation and status-wise breakdown (`LossStat`).
- `masking.py` – pluggable masking strategies (currently MCAR).
- `train.py` – Lightning module and CLI entry point.

You can mostly treat these as **independent layers**: Types → Data → Model → Eval/Masking → Training.

---

## Type system (`types.py`)

### Core abstractions

- **`VariationConfig`**: config for per-entity deviation embeddings.
  - `enabled`, `num_entities`, `reg_weight` (L2 penalty in the model).

- **`EntityType` (ABC)** – one per logical type (e.g. attribute, rating).
  - `name`: string identifier.
  - `is_variable`: `True` for variables (have loss), `False` for pure entities.
  - `param_dim`: local parameter dimension (0 for entities).
  - `variation`: `VariationConfig`.
  - `build_param(raw_data, device, global_param_dim) -> Tensor`:
    - Input adapter: raw observation → parameter vector in the **global** param space.
  - `compute_loss(predicted_params, tokens, type_mask, global_param_dim) -> Tensor`:
    - Legacy scalar loss (still implemented, but new code uses breakdown).
  - `compute_loss_breakdown(...) -> LossBreakdown`:
    - Returns per-type masked/observed/missing means and counts (see below).

- **`LossBreakdown`** (per-type):
  - `trainable_loss`: scalar Tensor (mean over **masked** tokens of this type).
  - `loss_observed`, `loss_masked`, `loss_missing`: floats (metrics only).
  - `n_observed`, `n_masked`, `n_missing`: counts for aggregation.

- **`NullEntityType(EntityType)`**:
  - For types that **never** incur direct loss and have **no param content**.
  - `build_param` → zeros; `compute_loss` → zero.
  - Domain-3 entity types just inherit from this and specialize `variation`.

### Domain-3 types

- **Entities (no direct loss)** – all inherit from `NullEntityType`:
  - `AttributeEntityType`: closed class, variation enabled.
  - `AnnotatorEntityType`: variation enabled with regularization.
  - `ItemEntityType`: variation disabled (items always new).

- **Variables (carry loss)**:
  - `RatingVariableType`:
    - `param_dim = 1 + num_classes`.
    - First dimension is a **status/mask bit**; the rest are class logits.
    - Observed ratings encoded in **logit space** with a large spike (`logit_high`).
    - `compute_loss_breakdown`:
      - CE over logits for all tokens with labels.
      - Buckets by `token.status` (0=missing,1=masked,2=observed).
      - `trainable_loss` = mean CE over **masked only**.
  - `PairwiseRankingVariableType`:
    - `param_dim = 1 + max_rank_size`.
    - Uses `PlackettLuceLoss` over small ranking sets.
    - `compute_loss_breakdown` parallels the rating case but with PL loss.

- **`build_default_domain3_types(...)`**:
  - Convenience factory building the canonical domain-3 registry:
    - `"attribute"`, `"annotator"`, `"item"`, `"rating"`, `"ranking_pairwise"`.

---

## Data and graph (`data.py`)

### Tokens and relationships

- **`Token`**:
  - `type_name`: which `EntityType` this token belongs to.
  - `entity_id`: index within that type (e.g. annotator id).
  - `status`: `0=missing`, `1=masked`, `2=observed`.
  - `raw_data`: type-specific payload (e.g. `rating_value`, `ranking_order`, flags).

- **`Relationship`**:
  - `name`, `source_type`, `target_type`, optional `inverse`.
  - Used to define edge labels like `ANNOTATOR`, `ANNOTATOR_INV`, `ITEM`, etc.

### `EntityGraph`

- Holds:
  - `types`: mapping `type_name -> EntityType`.
  - `tokens`: flat list of `Token`.
  - `edges`: list of `(src_index, tgt_index, rel_name)`.
  - Relationship indexing (`rel_name -> rel_id`).
- Provides:
  - `build_edge_masks(device) -> Tensor[L, L, num_relationships]`:
    - Dense 0/1 indicators used by relational attention.

### Bundle conversion

- **`bundle_to_entity_graph(bundle, ranking_vars, types)`**:
  - Input:
    - `GroundTruthBundle` (existing data format).
    - `ranking_vars`: list of `RankingData` (observed, masked, missing).
    - `types`: type registry (usually from `build_default_domain3_types`).
  - Output:
    - A single `EntityGraph` with:
      - A token per variable (rating / ranking), with proper `status` and `raw_data`.
      - A token per entity (attribute / annotator / item).
      - Edge structure wiring variables to their entities (and inverses).

This lets the model operate directly on existing bundles with no changes to the data pipeline.

---

## Model (`model.py`) and config (`config.py`)

### Config

- **`EntityMarformerConfig`**:
  - `embedding_dim`, `num_layers`, `attention_heads`, `dropout`,
  - `d_ff` (FFN size), `logit_high`, `temperature`, `normalize_parameter`.

### EntityMarformer

At a high level:

1. **Embeddings and deviations**
   - Per-type base embeddings (`[num_types, D]`).
   - Optional per-entity deviation tables per type (if `variation.enabled`).
2. **Streams**
   - **Feature stream**: type embedding + deviation for entity tokens; appropriate type embedding for variables.
   - **Param stream**: built per-token via `type.build_param(raw_data, ...)`.
   - Streams are concatenated and processed similarly to the legacy imputer (FFN on combined, then split).
3. **Relational attention**
   - `RelationalAttentionBlock` uses:
     - Standard multi-head Q/K/V projections.
     - Edge masks `edge_mask[L, L, R]` from the graph.
     - Relationship parameters to modulate attention scores by edge type (ROPE-like idea).
4. **Output**
   - The model returns a tensor `params` of shape `[1, L, global_param_dim]`.
   - Variable types read their slice of this param stream in `compute_loss_breakdown`.

The model is deliberately thin: **it does not know about masking or evaluation**; it just maps an `EntityGraph` to a parameter stream.

---

## Evaluation (`eval.py`)

- **`LossStat`** (global aggregate):
  - `trainable_loss`: mean loss over **all masked variables** (backprop objective).
  - `loss_observed`, `loss_masked`, `loss_missing`: global means by status (metrics).
  - `n_observed`, `n_masked`, `n_missing`: global counts.

- **`compute_loss_stat(params, graph, types, global_param_dim, device)`**:
  - For each `EntityType`:
    - Builds `type_mask` for tokens of that type.
    - Calls `t.compute_loss_breakdown(...)` to get a `LossBreakdown`.
  - Aggregates per-type breakdowns into a single `LossStat`:
    - `trainable_loss`: **count-weighted** mean across all masked tokens.
    - Status-wise metrics: count-weighted means across types.

This keeps all “how do we evaluate and aggregate across types?” logic outside the model and the types.

---

## Masking (`masking.py`)

- **Goal**: pluggable training masking strategies (currently **MCAR** only).

- **`MCARConfig(masking_rate)`**:
  - Probability/rate of masking observed variables per step.

- **`MaskingStrategy` interface**:
  - `mask(observed_vars: List[RankingData]) -> List[RankingData]`.

- **`MCARMasking`**:
  - Current default:
    - Randomly selects a subset of observed variables and sets `status=1` (masked); others remain `status=2` (observed).
    - Returns a new list of `RankingData` with updated statuses.

- **`build_default_masking_strategy(masking_rate)`**:
  - Returns an `MCARMasking` instance today; can branch in the future.

---

## Training (`train.py`)

### Lightning module

- **`EntityMarformerLightningModule`**:
  - Holds:
    - `model: EntityMarformer`
    - `train_observed`, `train_missing`: `List[RankingData]`.
    - `bundle`: `GroundTruthBundle`.
    - `types`: type registry.
    - `masking_strategy`: a `MaskingStrategy` (currently MCAR).
  - `train_dataloader`: dummy one-batch loader (graph is rebuilt each step from `train_*`).
  - `training_step`:
    1. Apply masking: `masked_or_observed = masking_strategy.mask(train_observed)`.
    2. Build full-instance vars: `train_vars = masked_or_observed + train_missing`.
    3. Build graph: `graph = bundle_to_entity_graph(bundle, train_vars, types)`.
    4. Forward: `params = model(graph, device=device)`.
    5. Compute global losses: `loss_stat = compute_loss_stat(params, graph, model.types, global_param_dim, device)`.
    6. Add deviation regularization and backprop on:
       - `loss = loss_stat.trainable_loss + reg_loss`.
    7. Log:
       - `train/loss` (trainable + reg),
       - `train/trainable_loss`,
       - `train/loss_masked`, `train/loss_observed`, `train/loss_missing`,
       - `train/reg_loss` when non-zero.

### CLI

- `main()` in `train.py`:
  - Reads `--data-dir`, `--epochs`, `--lr`, `--weight-decay`, `--masking-rate`, `--device`.
  - Loads bundle + configs via `load_bundle_and_converter`.
  - Builds domain-3 types, initial graph, and `EntityMarformer`.
  - Wraps model in `EntityMarformerLightningModule`.
  - Runs `pl.Trainer(...).fit(...)`.

Minimal usage:

```bash
python -m imputer.entity_mf.train \
  --data-dir PATH/TO/BUNDLE_DIR \
  --epochs 50 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --masking-rate 0.15 \
  --device cuda
```

---

## How to extend / hack

- **Add a new variable type**:
  - Subclass `EntityType`.
  - Implement `build_param` and `compute_loss_breakdown`.
  - Register it in your type registry (e.g. extend `build_default_domain3_types` or build your own).

- **Change masking**:
  - Implement a new `MaskingStrategy` in `masking.py`.
  - Swap `build_default_masking_strategy` or pass a different strategy into the Lightning module.

- **Change evaluation**:
  - Adjust `LossBreakdown` for your types and `compute_loss_stat` in `eval.py`.

The intent is that each layer is small and swappable so you can iterate quickly on research ideas without digging through a monolithic trainer. 

