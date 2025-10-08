## STAN Pipeline: Data Generation, Sampling, Evaluation, and GT-Init

This document captures the current design for our Stan-based experiments in Domain 3 (ratings + pairwise rankings). It consolidates the latest changes discussed in the updated paper draft and addresses key points raised in Jason's comments (see `imputer/ranking/READMEs/domain_3_updates.tex`).

### Models

- Data generation: `imputer/ranking/models/iclr_data_generation.stan`
  - Emits complete ground-truth parameters and synthetic annotations, including:
    - `embeddings[K,D]`, `mean_preferences[I,D]`, `annotator_preferences[I*J,D]`
    - `rating_probs[IJ] simplex[C]`, `rating_thresholds[IJ] vector[C]`
    - `base_scores[IJ,K]`
    - All ratings `all_rating_values[IJ,K]` and observed flags
    - Pairwise rankings from tied groups (configurable)

- Domain model (inference): `imputer/ranking/models/domain_model.stan`
  - Parameters sampled: `embeddings_raw[K,D]`, `mean_preferences[I,D]`, `annotator_preferences[I*J,D]`, and `rating_thresholds_increments[IJ,C-2]`
  - Transformed parameters: unit-normalized `embeddings[K,D]`, `base_scores[IJ,K]`, and full `rating_thresholds[IJ] vector[C+1]` with identification constraints (`[-inf, 0, ... , +inf]`)
  - Likelihoods: ordered probit for ratings; Bradley–Terry style pairwise ranking with temperature
  - Generated quantities:
    - `log_lik_ratings`, `log_lik_rankings`, `total_log_lik`
    - Posterior predictive samples for missing variables:
      - `missing_rating_predictions[N_missing_ratings]`
      - `missing_ranking_predictions[N_missing_rankings]`

### Observation Protocol (v_r) and Pairwise Generation (Updated)

- Observation mask generation v_r (protocol to emulate real annotation):
  - Assign two random annotators to each item; both rate all criteria (observed).
  - Add a third annotator for those (i,k) where the first two annotators’ ratings differ by more than 1 on the Likert scale: |x_{ijk} - x_{ij'k}| > 1.
  - For each observed triplet (i,j,k), ask annotator j to also annotate up to 10 pairwise rankings x_{ijkk'} where r_{ijk'} = 1 and x_{ijk'} = x_{ijk} (same rating bin). This yields Bradley–Terry tie-breaking within tied groups. By chance, both x_{ijkk'} and x_{ijk'k} may be annotated.
  - Cap per-item pairwise load to at most 10 comparisons (configurable). This prevents quadratic explosion while preserving informative local orderings.

Implication: Although the full variable universe is size I·J·(K + K*(K-1)/2), we only observe a sparse subset. Ratings can be dense; pairwise rankings are generated only from the above procedure, capped per (i,j,k).

### Stan Data Interface (Domain Model)

Required fields (subset):
- Dimensions: `K, I, J, D, C, ranking_size`
- Observed ratings: `N_ratings`, `rating_attributes`, `rating_annotators`, `rating_items`, `rating_values`
- Observed rankings: `N_rankings`, `ranking_attributes`, `ranking_annotators`, `ranking_items`, `ranking_orders`
- Hyperparameters: `sigma_annotator`, `sigma_measurement`, `alpha_dirichlet`, `temperature`, `sigma_embedding_prior`, `sigma_preference_prior`
- Missing variables to predict (new):
  - `N_missing_ratings`, `missing_rating_attributes`, `missing_rating_annotators`, `missing_rating_items`
  - `N_missing_rankings`, `missing_ranking_attributes`, `missing_ranking_annotators`, `missing_ranking_items`

Storage note: Because pairwise rankings are sparse and capped, we pass explicit index lists for both observed and missing variables, rather than dense tensors. This keeps the Stan data small and aligns with the protocol.

### Parameter Taxonomy: Data Constants vs. Model Parameters vs. Hyperparameters

- Data constants (provided from Python, fixed for a run): `K, I, J, D, C, ranking_size`.
- Hyperparameters (fixed, control priors/likelihoods): `sigma_annotator, sigma_measurement, alpha_dirichlet, temperature, sigma_embedding_prior, sigma_preference_prior`.
- Model parameters (sampled with priors): `embeddings_raw / embeddings`, `mean_preferences`, `annotator_preferences`, `rating_thresholds_increments` (with transformed `rating_thresholds`).

Naming: we will consistently call the first group "data constants", the second "hyperparameters", and the third "model parameters".

These align with Stan program blocks per the reference manual (see “data”, “parameters”, “transformed parameters”, “model”, “generated quantities”) [Stan program blocks](https://mc-stan.org/docs/reference-manual/blocks.html).

### Ground Truth Loading and Oracle Initialization for MCMC

Use `imputer/ranking/utils/ground_truth_extractor.py` (property-first API):

```python
from pathlib import Path
from imputer.ranking.utils.ground_truth_extractor import GroundTruthExtractor

gtx = GroundTruthExtractor(Path(".../iclr_complete_ground_truth.json"))
gt = gtx.complete
dims = gtx.dimensions  # {'K','I','J','D','C'}
```

Mapping GT → Stan `inits`:
- `embeddings_raw[K,D]`: start from `gt['embeddings']` and add small noise (Stan normalizes in transformed parameters)
- `mean_preferences[I,D]`: `gt['mean_preferences']`
- `annotator_preferences[I*J,D]`: `gt['annotator_preferences']`
- `rating_thresholds_increments[IJ,C-2]`: differences between adjacent internal thresholds from `gt['rating_thresholds']` (skip `-inf`, `0`, `+inf`); use absolute increments and a small positive default if needed

This yields an oracle init that places chains near the true posterior mode while preserving chain diversity via small noise.

### Posterior Predictive and Marginalization

- We rely on Stan to perform marginalization by sampling the parameters and generating posterior predictive draws in `generated quantities`.
- For missing ratings and pairwise rankings, each draw produces a prediction; aggregating across draws provides the marginal predictive distribution.
- This is the correct Bayesian evaluation vs. using posterior means (a lossy point estimate).

Additionally, we will emit separate log-likelihood components in `generated quantities` to distinguish observed vs. missing contributions for diagnostics:
- `log_lik_ratings_obs`, `log_lik_ratings_missing`
- `log_lik_rankings_obs`, `log_lik_rankings_missing`

These are calculated using the same likelihood formulas but partitioned by whether each variable is in the observed or missing lists.

Client-side (Python) evaluation should:
- Extract `missing_rating_predictions` (and rankings if used) across draws
- Compute per-missing-variable empirical probabilities (frequency over draws)
- Score with log-loss/accuracy using these marginal probabilities

### Reference Scripts

- Posterior predictive + GT-init: `imputer/ranking/experiments/ground_truth_mcmc_marginalization.py`
  - Note: the previous "optimization check" path is deprecated and will be removed; we will keep a single posterior/evaluation check script for clarity.

Each script demonstrates: model compile, Stan data assembly, optional oracle initialization, sampling, extraction of generated quantities, and evaluation.

### Practical Notes

- Identification: first threshold fixed at 0; increments are positive. Embedding normalization is currently used for identification, but given embeddings are generated standard normal in data gen, we will re-evaluate whether normalization is necessary (TODO below).
- Temperature scales ranking logits; measurement noise used for ordered probit.
- Keep `iter_warmup`, `iter_sampling`, `adapt_delta`, and `max_treedepth` configurable; use multiple chains.
- Storage: enabling posterior predictive in `generated quantities` increases CSV size (expected/desired for proper marginalization).

### Python–Stan Integration (Minimal, Self-Contained API)

A small set of functions is sufficient and reusable across scripts:

- `compile_model(stan_file: str) -> CmdStanModel`
- `generate_data(config: DataGenConfig) -> GroundTruthBundle`  
  - Inputs: data constants (K,I,J,D,C, ranking caps), hyperparams (sigma_annotator, sigma_measurement, alpha_dirichlet, temperature), seed
  - Output: complete GT package with parameters, all ratings, sparse capped pairwise, observed masks, per-variable indices, and log-likelihood partitions from data-gen Stan
- `prepare_stan_data(observed: ObservedSet, missing: MissingSet, config: DomainConfig) -> Dict`
  - ObservedSet/MissingSet schemas: lists of (i,j,k,c) for ratings and (i,j,k1,k2,order) for pairwise; packed into Stan index arrays
- `make_oracle_inits(ground_truth, dims, noise=0.1) -> List[Dict]`
- `sample(model, data, inits, mcmc_cfg) -> fit`
- `extract_predictives(fit) -> Dict`  
  - Returns: posterior predictive samples for missing lists, plus (planned) observed log-likelihood aggregates from `generated quantities`
- `evaluate(predictives, ground_truth_missing) -> metrics`  
  - Computes accuracy/log-loss by marginalizing over posterior predictive samples

These cover compile → data assembly → oracle init → sampling → predictive extraction → evaluation, and are sufficient for end-to-end experiments.

### Paper Sync — TODOs to Reflect in Main Text

- Use Bradley–Terry for pairwise; ordered probit for ratings (done in code; ensure text states this)
- No manual marginalization in Python; use `generated quantities` (implemented; ensure text states this)
- Generate rankings only from tied groups and cap per-item comparisons at 10 (ensure text specifies protocol and rationale)
- Distinguish data constants, hyperparameters, and model parameters (add explicit terminology)
- Add observed/missing log-likelihood partitions in Stan (code change pending; reflect once merged)
- Revisit embedding normalization vs. generative assumptions (decide and update model + text)
- Deprecate/remove old "optimization check"; keep a single posterior evaluation script and name it clearly

### Logging and Traces (Save Everything)

- Save all inputs (configs, seeds), Stan CSVs (all draws), and derived artifacts (summaries, predictive samples, log-likelihood partitions) to a run directory.
- Favor high-fidelity logs so that figures/tables can be regenerated without re-running MCMC.
- Structure: `runs/<timestamp>/configs.json`, `stan_csv/`, `predictives/`, `metrics.json`, `artifacts/`.

Note: Larger storage footprint is acceptable to avoid re-computation.


