# Marformer

Marformer is a transformer-based imputation model for
incomplete data: attributes x annotators x items, represented as
typed random-variable tokens with no positional embeddings. See
`imputer/README.md` for the model itself and its type system.

## Experimental design

Marformer is evaluated against Stan on two kinds of data:

**Artificial data (`scripts/DOMAIN3-SIGMOID/`)** — synthetic ratings drawn
from a known generative model (CPM),
`STAN/stan_models/tensor_generation_sigmoid.stan`, so ground truth is known
exactly. Two held-out expansion axes:

- **Item expansion** — grow the training set over items, holding out a fixed
  block of test items/annotators.
- **Annotator expansion** — grow the training set over annotators, same fixed
  test block.

Each axis has a transductive and a non-transductive variant. Marformer is
compared against:

- **Stan Oracle** — the correctly-specified Stan inference model (matches the
  generator exactly) — the performance ceiling.
- **Stan Misspecified** — three structurally wrong variants (shared
  thresholds, shared annotator projection, or both) — how much accuracy
  costs when the baseline's assumptions are wrong, for comparison against
  Marformer, which makes no such parametric assumptions.

**Real data (`scripts/LLMRubric/`)** — human and LLM annotation ratings
(no known ground-truth generative model). Marformer is compared against a
Stan CPM-Sigmoid fit (the same tensor model, now misspecified by construction
since it isn't the true generator) and imputation baselines (MIWAE,
ReMasker).

## Layout

- `imputer/` — the Marformer model. See `imputer/README.md`.
- `STAN/` — the Stan data-generation and inference pipeline. See `STAN/README.md`.
- `scripts/` — experiment runners and plotting. See `scripts/README.md`.
- `DATA/`, `RESULTS/`, `PLOTS/` — generated data, experiment outputs, figures
  (gitignored except where noted).
