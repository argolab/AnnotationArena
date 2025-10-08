## STAN Implementation Plan (Modules, Classes, Files)

This plan instantiates the simplified architecture from STAN_README.md with concrete files, classes, and functions. It preserves full functionality while minimizing moving parts.

### File Layout (current)

```
imputer/ranking/
  models/
    iclr_data_generation.stan          # data-gen model (GT + full annotations + partitions)
    domain_model.stan                   # inference model (observed-only ll; predictives for missing)

  stan/
    pipeline/
      __init__.py
      configs.py                        # dataclasses: DataGenConfig, DomainConfig, McmcConfig
      bundle.py                         # dataclasses: GroundTruthBundle, ObservedSet, MissingSet
      io.py                             # run dirs, logging/traces, CSV management
      data_gen.py                       # Python wrapper for Stan data generation
      prepare.py                        # prepare_stan_data() for domain model
      inits.py                          # make_inits() from GT or random
      sampling.py                       # compile_model(), sample()
      extract.py                        # extract_predictives(), extract_logliks()
      evaluate.py                       # evaluate() on missing via predictives

  scripts/
    generate_data.py                    # CLI wrapper around Stan data generation
    infer.py                            # CLI wrapper around sampling + evaluation
```

### Core Dataclasses

`stan/pipeline/configs.py`
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class DataGenConfig:
    K: int; I: int; J: int; D: int; C: int
    # observation protocol (v_r) switches
    enable_third_annotator: bool = True
    enable_pairwise: bool = True
    pairwise_cap_per_item: int = 10
    # hyperparameters for generation
    sigma_annotator: float = 0.3
    sigma_measurement: float = 0.1
    alpha_dirichlet: float = 2.0
    temperature: float = 0.5
    # split + seed
    train_fraction: float = 0.8
    seed: Optional[int] = None

@dataclass
class DomainConfig:
    K: int; I: int; J: int; D: int; C: int
    ranking_size: int = 2
    sigma_annotator: float = 0.3
    sigma_measurement: float = 0.1
    alpha_dirichlet: float = 2.0
    temperature: float = 0.5
    sigma_embedding_prior: float = 1.0
    sigma_preference_prior: float = 1.0

@dataclass
class McmcConfig:
    chains: int = 4
    iter_warmup: int = 1000
    iter_sampling: int = 2000
    adapt_delta: float = 0.8
    max_treedepth: int = 15
    seed: Optional[int] = 42
```

`stan/pipeline/bundle.py`
```python
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

@dataclass
class GroundTruthBundle:
    # parameters
    embeddings: np.ndarray
    mean_preferences: np.ndarray
    annotator_preferences: np.ndarray
    rating_probs: np.ndarray
    rating_thresholds: np.ndarray
    base_scores: np.ndarray
    # full annotations
    all_ratings: List[Dict]
    all_pairwise: List[Dict]
    # observed/missing partitions
    observed_ratings: List[Dict]
    missing_ratings: List[Dict]
    observed_pairwise: List[Dict]
    missing_pairwise: List[Dict]
    # diagnostics
    stats: Dict[str, Any]
    # log-likelihood partitions from data-gen (optional but desirable)
    log_lik_ratings_obs: float | None
    log_lik_ratings_missing: float | None
    log_lik_rankings_obs: float | None
    log_lik_rankings_missing: float | None

@dataclass
class ObservedSet:
    ratings: List[Dict]          # {'attribute','annotator','item','value'}
    pairwise: List[Dict]         # {'attribute','annotator','items':[k1,k2],'order':[1,2]}

@dataclass
class MissingSet:
    ratings: List[Dict]          # same schema but unknown at inference time
    pairwise: List[Dict]
```

### Module Responsibilities & Function Signatures

`stan/pipeline/io.py`
- `new_run_dir(root: Path) -> Path`  // create runs/<timestamp>
- `save_configs(run_dir, data_cfg, domain_cfg, mcmc_cfg)`
- `save_bundle(run_dir, bundle: GroundTruthBundle)`
- `save_predictives(run_dir, predictives: Dict)`
- `save_metrics(run_dir, metrics: Dict)`
- `save_fit_csvs(run_dir, fit)`

`stan/pipeline/data_gen.py`
- `generate_data(cfg: DataGenConfig) -> GroundTruthBundle`
  - Compiles `iclr_data_generation.stan` via cmdstanpy
  - Samples with `fixed_param=True` for data generation
  - Extracts parameters, annotations, and partitions from Stan output
  - Converts Stan arrays to `GroundTruthBundle` format
- `extract_bundle_from_stan_output(fit, config) -> GroundTruthBundle`
  - Converts Stan generated quantities to Python data structures

`stan/pipeline/prepare.py`
- `prepare_stan_data(observed: ObservedSet, missing: MissingSet, domain_cfg: DomainConfig) -> Dict`
  - Packs observed lists into Stan arrays for ratings/rankings
  - Adds missing lists for posterior predictive sampling indices

`stan/pipeline/inits.py`
- `make_inits_from_gt(bundle: GroundTruthBundle, noise: float, mcmc: McmcConfig) -> List[Dict]`
- `make_inits_random(domain_cfg: DomainConfig, mcmc: McmcConfig) -> List[Dict]`

`stan/pipeline/sampling.py`
- `compile_model(stan_file: str) -> CmdStanModel`
- `sample(model, data: Dict, inits: List[Dict], mcmc: McmcConfig) -> CmdStanMCMC`

`stan/pipeline/extract.py`
- `extract_predictives(fit) -> Dict`  
  - returns arrays for `missing_rating_predictions` and `missing_ranking_predictions`
- `extract_loglik_obs(fit) -> Dict`  
  - returns aggregates for observed log-likelihood (ratings/rankings) if emitted

`stan/pipeline/evaluate.py`
- `evaluate_marginal(predictives: Dict, missing: MissingSet, C: int) -> Dict`  
  - accuracy/log-loss over missing via empirical frequencies across draws

### Scripts (CLIs)

`scripts/generate_data.py`
- Args: DataGenConfig parameters (K, I, J, D, C, hyperparameters, seed)
- Calls `data_gen.generate_data()` to interface with Stan
- Saves bundle + configs under `runs/<timestamp>`

`scripts/infer.py`
- Args: path to run dir (bundle), DomainConfig JSON, McmcConfig JSON, `--init {gt,random}`
- Prepares Stan data from bundle (observed/missing), runs sampling, extracts predictives, evaluates, and saves all artifacts in the same run dir (subfolder `inference/`).

### Stan Model Interfaces (delta vs current)

`iclr_data_generation.stan` (data-gen)
- Keep: parameters, full annotations, tie-break pairwise (capped at `pairwise_cap_per_item` per (i,j,k)), counts.
- Add (recommended):
  - Observed/missing indicator construction per v_r protocol into arrays/lists for ratings and pairwise (or emit sufficient info to reconstruct in Python deterministically).
  - Generated quantities: `log_lik_ratings_obs`, `log_lik_ratings_missing`, `log_lik_rankings_obs`, `log_lik_rankings_missing` based on the full annotations and chosen v_r.

`domain_model.stan` (inference)
- Data: observed ratings/rankings; missing indices for predictives.
- Generated quantities: posterior predictives for missing; `log_lik_ratings_obs`, `log_lik_rankings_obs` for observed only.
- Identification: keep minimal (remove embedding normalization if not required); retain threshold increment construction.

### Migration Plan (from current code)

1) ✅ Introduce `imputer/ranking/stan/pipeline/` with configs, bundle, io modules; keep `domain_model_trainer.py` and `iclr_data_generator.py` intact initially.
2) ✅ Update `iclr_data_generation.stan` with simplified pairwise protocol (per-item cap, no group size params).
3) ✅ Add Python wrapper `data_gen.py` for Stan data generation with cmdstanpy integration.
4) Implement `prepare_stan_data()` to build Stan arrays for both observed and missing indices from the bundle.
5) Implement `inits` (GT/random) with noise control; wire into sampling.
6) Implement `extract` and `evaluate` with posterior predictive marginalization (ratings first; rankings optional).
7) Add `scripts/generate_data.py` and `scripts/infer.py` as the only CLIs; deprecate older experiment scripts.
8) Stan deltas:
   - domain_model.stan: ensure missing indices and predictives are emitted; add observed log-lik aggregates.
   - iclr_data_generation.stan: (optional) emit observed/missing log-lik partitions directly; otherwise compute in Python from emitted full annotations.
9) Logging: adopt unified run dir structure across both scripts; ensure CSVs, configs, predictives, and metrics are saved.

### Milestones

- ✅ Phase 0: Pipeline skeleton + configs + bundle + IO; no predictives yet.
- ✅ Phase 1: Data generation via Stan with Python wrapper (cmdstanpy integration).
- Phase 2: Prepare Stan data for inference.
- Phase 3: Inits and sampling utilities.
- Phase 4: Predictives + evaluation (ratings) + CLI.
- Phase 5: Unified logging and run directories.
- Phase 6: Rankings support end-to-end.
- Phase 7: Observed/missing log-likelihood partitions.
- Phase 8: Cleanup/deprecate old scripts; finalize docs.

### Data Generation Protocol

**Simultaneous Generation:**
- **Single Stan Model**: Generates both training and test instances simultaneously
- **Shared Components**: Criteria embeddings (mean_preferences) are shared across instances
- **Separate Items**: Training and test items have different embeddings (K_train vs K_test)
- **Annotator Sets**: Training uses first 2/3 annotators (J=9 → annotators 1-6), test uses last 2/3 (annotators 4-9)
- **Overlap**: Middle 1/3 annotators (4-6) appear in both training and test

**Observation Protocol:**
1. **Two Random Annotators**: For each item, assign two random annotators from the active set
2. **All Criteria Rating**: Both annotators rate the item on all I criteria
3. **Third Annotator**: If ratings differ by >1, add third annotator (ablation: `enable_third_annotator`)
4. **Pairwise Rankings**: For each (i,j,k), find items k' with same rating, generate up to 10 pairwise comparisons (ablation: `enable_pairwise_rankings`)

**Bradley-Terry Model:**
- Uses Gumbel noise for tie-breaking in pairwise rankings
- Temperature parameter controls noise level
- No categorical (Likert) or listwise (Plackett-Luce) comparisons

**Ablation Studies:**
- `enable_third_annotator`: Control third annotator addition
- `enable_pairwise_rankings`: Control pairwise ranking generation
- `pairwise_cap_per_item`: Control number of comparisons per item

### Notes on Simplicity Choices

- Two CLIs (generate_data, infer) and one pipeline module are sufficient for all experiments.
- All ablations live in DataGenConfig (v_r switches) to keep inference/evaluation identical.
- Always save full traces; avoid recomputation; keep API small and composable.
- **Data generation**: Stan handles synthesis via `iclr_data_generation.stan`; Python wrapper (`data_gen.py`) provides cmdstanpy interface and data conversion.
- **Clean separation**: Stan owns generative logic; Python handles I/O, data conversion, and pipeline scaffolding.


