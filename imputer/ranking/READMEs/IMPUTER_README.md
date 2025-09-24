## Imputer (RVFormer) — Structure, APIs, Training, and Evaluation

This document consolidates the latest understanding of the imputer system for Domain 3 and is intended to stay in sync with the updated paper and Jason’s comments (see `imputer/ranking/READMEs/domain_3_updates.tex`).

### Scope

- Architecture overview and key components
- Public APIs used during training and evaluation
- Data flow and masking strategies
- Evaluation metrics and how they are computed

### Architecture Overview

<!-- Should there be a new module named Atom. Atom type is used in embedings of different Atom types (item, criterion, annotator) -->
- Core modules live under `imputer/ranking/imputer/`:
  - `embedding.py`: compositional embeddings for criteria (attributes), annotators, and items; supports mixed variable types (ratings C-way; pairwise rankings 2-way) concatenate atoms with their atom var type (100 prefix for critieron,, 010 prefix for annotator, 001 for item)
    - notice that criteria embeddings are always learned, annotator embeddings are half learnable and half fresh random, with the learnable part subject to dropouts at training time. while the item embeddigns are fresh at each forward pass, and can not be learned.
  - `ranking_imputer.py`: high-level imputation model wrapper
  - `trainer.py` / `multi_instance_trainer.py`: training loops over instances, batching, masking, logging. Since now we only have two instance.  A training and a testing instance. 
  - `losses.py`: cross-entropy and ranking losses
  - `eval.py`: `EvaluationResults` schema and helpers

Variable encoding follows the unified parameter vector scheme described in the paper: mask bit + rating logits (C) + ranking logits (2), with type-appropriate heads.

### Public APIs (Typical Usage)

- Dataset instances provide lists of variables with fields: `{attribute, annotator, item, value}` for ratings and `{attribute, annotator, items[2], order[2]}` for pairwise rankings.
- Training loop APIs:
  - Provide observed subset; apply random masking for robust training
  - Forward pass consumes observed context and predicts both masked and evaluation targets
  - Loss computed only on designated supervision positions

### Training

- Progressive instance-level batching supported (see runners)
- Masking rate is a first-class hyperparameter for both training and evaluation
- Optimizer: AdamW with standard scheduler; gradient clipping recommended

### Evaluation

- Metrics:
  - Rating accuracy, RMSE
  - Ranking accuracy (binary)
  - Log-loss on masked positions
- Protocols:
  - Within-instance validation (mask M% of observed during training)
  - Test-time evaluation: treat provided observed variables as context; predict masked variables

### Ablations / Observation Removal (Planned)

We will study sensitivity to observation types by selectively removing observation subsets during training/evaluation while keeping the variables present in the model/data:
- Remove the third annotator’s ratings (for cases where the first two annotators differ by >1)
- Remove all rankings (keep ratings only)
- Remove all ratings (keep rankings only)

These switches allow comparison of MCMC vs. RVFormer under different supervision mixes using the same generated instances.

Where do ablations live?
- Data-generation ablation: change the observation protocol v_r (e.g., disable third annotator, disable pairwise generation) — affects which variables are observed vs. missing while preserving full universe in GT.
- Imputer-level ablation: keep data fixed but mask out subsets before training/eval to simulate limited supervision; Stan baseline can use the same observed sets.

We will support both, but default to data-generation protocol toggles to align with the paper’s design.

### Interop with Stan Baseline

- The imputer is evaluated side-by-side with the Stan baseline:
  - Stan uses posterior predictive marginalization via `generated quantities` (see `STAN_README.md`)
  - Imputer uses neural predictions directly; uncertainty handled via softmax distributions

### Paper Sync

This README tracks the updated paper structure and Jason’s guidance:
- Clear separation of baseline (Bayesian) vs. neural (RVFormer) approaches
- Unified variable representation and compositional embedding story
- Evaluation protocols consistent across systems for fair comparison

TODOs to reflect in main text:
- Document the observation protocol (two annotators per item + conditional third; capped tie-break rankings)
- Clarify how masking is used to emulate missing-at-random vs. protocol-driven missingness
- Include ablation toggles and report their effects across systems

### Logging and Traces (Save Everything)

- Persist all training/evaluation traces: configs, seeds, model checkpoints, per-epoch metrics, prediction dumps for masked variables, and any auxiliary artifacts used in plots/tables.
- Directory structure mirrors the Stan runs (`runs/<timestamp>/...`) so cross-system analysis is straightforward.


