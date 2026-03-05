## Entity Marformer Transductive Behavior

```mermaid
flowchart TD
  cfg["Config(flags)"] --> transFlag["transductive=true/false"]

  subgraph dataPrep ["__init__() persistent splits"]
    bundle["GroundTruthBundle"] -->|converter.create_variables_from_bundle| trainObs["train_observed"]
    bundle --> trainMiss["train_missing"]
    bundle --> testObs["test_observed"]
    bundle --> testMiss["test_missing"]

    trainObs --> trainAll["train_all = obs+miss"]
    trainMiss --> trainAll
    testObs --> testAll["test_all = obs+miss"]
    testMiss --> testAll
  end

  subgraph trainingStep ["training_step() (per item-chunk)"]
    direction LR
    transFlag -->|false| obsSrc["observed_sources = train_observed"]
    transFlag -->|false| missSrc["missing_sources = train_missing"]
    transFlag -->|true| obsSrcT["observed_sources = train_observed + test_observed"]
    transFlag -->|true| missSrcT["missing_sources = train_missing + test_missing"]

    obsSrc --> items["all_items = union(item_ids)"]
    missSrc --> items
    obsSrcT --> items
    missSrcT --> items

    items --> chunking["item_chunks (optional max_item)"]
    chunking --> chunkSel["filter vars whose items in chunk"]

    chunkSel --> chunkMask["masked_or_observed = masking_strategy.mask(chunk_observed)"]
    chunkSel --> chunkMiss["chunk_missing"]
    chunkMask --> varsChunk["train_vars = masked_or_observed + chunk_missing"]
    chunkMiss --> varsChunk
    varsChunk --> graphBuild["variable_list_to_entity_graph(train_vars, types)"]
    graphBuild --> forward["params = model(entity_graph)"]
    forward --> trainLoss["trainable_loss = compute_trainable_loss (masked-only)"]
  end

  subgraph epochEnd ["on_train_epoch_end()"]
    direction LR
    trainAll --> nonTransTrain["evaluate_entity_marformer_split(split='train', variables=train_all)"]
    testAll  --> anyTestEval["evaluate_entity_marformer_split(split='test', variables=test_all)"]

    transFlag -->|true| combBranch["transductive path"]
    transFlag -->|false| nonTransBranch["non-transductive path"]

    nonTransBranch --> nonTransTrain
    nonTransBranch --> anyTestEval

    combBranch --> combVars["combined = train_all + test_all"]
    combVars --> combEval["evaluate_entity_marformer_split(split='combined', variables=combined)"]
    combBranch --> anyTestEval
  end
```

### Splits and persistent state

- In `EntityMarformerLightningModule.__init__`, we construct four splits directly from the bundle via the shared `DataConverter`:
  - `self.train_observed`, `self.train_missing`
  - `self.test_observed`, `self.test_missing`
- For evaluation we also define:
  - `self.train_all = train_observed + train_missing`
  - `self.test_all  = test_observed + test_missing`
- These four lists are **never mutated** by training; they act as a canonical view of the bundle partition.

### Training-time behavior

- The `transductive` flag only affects **which variables are visible to the training step**, not how losses are computed:
  - Start each step with:
    - `observed_sources = self.train_observed`
    - `missing_sources  = self.train_missing`
  - If `transductive` is enabled:
    - `observed_sources += self.test_observed`
    - `missing_sources  += self.test_missing`
- For each item-chunk:
  - Filter `observed_sources` / `missing_sources` by the items in the chunk.
  - If `max_item` is set, `training_step` splits items into multiple chunks and combines
    chunk losses as a weighted average by the number of masked tokens (fallback weight = 1).
  - Apply `masking_strategy.mask(chunk_observed)` to get `masked_or_observed`.
  - Build `train_vars = masked_or_observed + chunk_missing`.
  - Convert to an `EntityGraph` using `variable_list_to_entity_graph(train_vars, types)`.
  - Run the model once: `params = model(graph, device=device)`.
  - Compute the scalar **trainable loss** via `compute_trainable_loss`, which:
    - Aggregates per-type `LossBreakdown` objects,
    - Uses only the **masked** subset to define the objective.
  - Add deviation regularization and backprop on this scalar.

### Epoch-end evaluation (non-transductive)

- When `transductive=False`, `on_train_epoch_end` evaluates splits separately:
  - Train split:
    - `train_eval = evaluate_entity_marformer_split(model, "train", self.train_all, ...)`
  - Test split:
    - `test_eval  = evaluate_entity_marformer_split(model, "test",  self.test_all,  ...)`
- Both calls:
  - Build a full graph from **observed + missing** variables,
  - Let the `status` field (0/1/2) drive missing/masked/observed decomposition,
  - Return `EntityEvalResults(split, metrics)` where `metrics` encodes per-status, per-type metrics.

### Epoch-end evaluation (transductive)

- When `transductive=True` we want:
  - One **combined** view of how the model behaves on the merged graph,
  - A clean **test-only** view for reporting.
- `on_train_epoch_end` therefore runs:
  - Combined evaluation:
    - `combined = self.train_all + self.test_all`
    - `combined_eval = evaluate_entity_marformer_split(model, "combined", combined, ...)`
    - Printed as `[combined_missing] acc=... xent=...` by reading `combined_eval.metrics["missing"]["rating"]`.
    - Stored under `epoch_metrics["combined_eval"]`.
  - Test-only evaluation:
    - `test_eval = evaluate_entity_marformer_split(model, "test", self.test_all, ...)`
    - Printed as `[test_missing] acc=... xent=...`.
    - Stored under `epoch_metrics["test_eval"]`.

### Invariants and interpretation

- **Training objective**:
  - Always masked-only loss, aggregated across all variable types and all masked tokens in the training graph.
  - Unaffected by whether we are in transductive or non-transductive mode.
- **Transductive vs non-transductive**:
  - Non-transductive:
    - Training graphs see **train** variables only.
    - Evaluation splits are train-only and test-only graphs.
  - Transductive:
    - Training graphs see **train + test** variables together.
    - Evaluation still exposes:
      - Combined stats (`combined_eval`) for debugging overall behavior,
      - Test-only stats (`test_eval`) for primary reporting.
