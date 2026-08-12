# scripts

Two script families are essential and actively maintained:

- **`DOMAIN3-SIGMOID/`** — synthetic data (Item/Annotator expansion) experiments:
  Marformer vs. Stan Oracle/Misspecified baselines.
- **`LLMRubric/`** — the same Stan/Marformer comparison on real LLMRubric
  annotation data.

Everything else in `scripts/` supports one of those two:

- `DOMAIN3/` — generic runner scripts (`common.sh`, `run_stan_group.sh`,
  `run_marformer_group.sh`, `run_marformer_eval.sh`) that `DOMAIN3-SIGMOID/`'s
  per-size scripts delegate to.
- `BASELINES/` — MIWAE / ReMasker baseline runners, plotted alongside
  Stan/Marformer in the LLMRubric figures.
- `utils/plot_domain3_generalization.py` — shared plotting helpers imported by
  both `DOMAIN3-SIGMOID/plot_domain3_sigmoid.py` and
  `LLMRubric/plot_llmrubric_cpm_sigmoid.py`. 

## Data

### Synthetic (DOMAIN3-SIGMOID)

```bash
bash scripts/DOMAIN3-SIGMOID/generate_data.sh
```

Generates one base tensor bundle via `STAN/stan_models/tensor_generation_sigmoid.stan`
(`STAN/stan_code/scripts/generate_data.py`), then splits it into Item- and
Annotator-expansion sets, each Transductive and Non-Transductive
(`STAN/stan_code/scripts/domain3subsplit.py`), under `DATA/STAN/DOMAIN3-SIGMOID/`.
Dimensions, generative hyperparameters, and the size lists for each split are
all constants at the top of `generate_data.sh` — see "Other sizes or params"
below.

### Real data (LLMRubric)

Already-collected LLM Rubric annotation bundles live under
`DATA/LLM_RUBRIC/LLMRubric_225_25_9_<size>/` for a fixed set of sizes
(10, 20, 30, 40, 50, 75, 100, 125, 150, 175). There is no generation script —
if a size you need isn't present, ask the repo owner for it rather than
trying to synthesize it.

## Running experiments

### DOMAIN3-SIGMOID

Stan (Oracle + 3 misspecified models, **transductive splits only**) — one
script per model family per size group:

```bash
bash scripts/DOMAIN3-SIGMOID/StanOracle/ItemExpansion/run_size500.sh
bash scripts/DOMAIN3-SIGMOID/StanMisspecs/Bin/ItemExpansion/run_sizes_100_200_300_400_500.sh
# ...and the Proj/, Both/ siblings, plus each family's AnnotatorExpansion/ scripts
```

Each delegates to `scripts/DOMAIN3-SIGMOID/run_stan_group.sh` ->
`scripts/DOMAIN3/run_stan_group.sh`, which stages the given `--stan-file`,
runs `STAN/stan_code/scripts/run_inference.py` (MCMC fit) then
`evaluate_predictions.py`, writing to `RESULTS/STAN/DOMAIN3-SIGMOID/{ORACLE,MISSPEC}/{ITEM,ANNOT}/`.

Marformer (**transductive** via `Marformer/`, **non-transductive** via
`MarformerNT/`):

```bash
bash scripts/DOMAIN3-SIGMOID/Marformer/ItemExpansion/run_sizes_100_200_300.sh
bash scripts/DOMAIN3-SIGMOID/MarformerNT/ItemExpansion/run_sizes_100_200_300.sh
```

Delegates to `scripts/DOMAIN3-SIGMOID/run_marformer_group.sh` ->
`scripts/DOMAIN3/run_marformer_group.sh`, which calls
`python -m imputer.entity_mf.train`, writing to
`RESULTS/MARFORMER/DOMAIN3-SIGMOID/` (transductive) or
`RESULTS/MARFORMER-NT/DOMAIN3-SIGMOID/` (non-transductive).

### LLMRubric

Stan CPM-Sigmoid, one script per size (or size group):

```bash
bash scripts/LLMRubric/CPMSigmoid/run_size125.sh
```

Runs `run_inference.py` + `evaluate_predictions.py` directly (no generic
runner here) against `DATA/LLM_RUBRIC/LLMRubric_225_25_9_<size>/`, writing to
`RESULTS/STAN/LLM_RUBRIC/CPMSigmoid/`.

Marformer on LLMRubric has no wrapper script — run `train.py` directly:

```bash
python -m imputer.entity_mf.train \
    --data-dir DATA/LLM_RUBRIC/LLMRubric_225_25_9_125 \
    --run-name LLMRubric_225_25_9_125 \
    [architecture/training flags — see imputer/entity_mf/train.py --help]
```

pointing `--data-dir` at the size you want; results are expected under
`RESULTS/MARFORMER/LLM_RUBRIC/LLMRubric_225_25_9_<size>/` (see
`plot_llmrubric_cpm_sigmoid.py`'s `marformer_root`).

Baselines (MIWAE, ReMasker):

```bash
bash scripts/BASELINES/LLMRubric/run_miwae_llmrubric.sh
bash scripts/BASELINES/LLMRubric/run_remasker_llmrubric.sh
```

## Plotting

```bash
python scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py [--part all|item|annot] [--cache] [--cache-baselines] [--calibration] [--device ...]
python scripts/LLMRubric/plot_llmrubric_cpm_sigmoid.py [--cache] [--best] [--device ...]
```

`--cache` reuses cached probability payloads instead of reloading/re-running
Marformer eval. Output figures go under `PLOTS/DOMAIN3-SIGMOID/` and
`PLOTS/LLM_RUBRIC/` respectively.

## Other sizes or params

- **A new size for an existing model/axis, on already-generated data**: copy
  the closest `run_size*.sh`, change `--sizes` (DOMAIN3-SIGMOID Stan/Marformer
  scripts) or the `SIZES=(...)` array (LLMRubric scripts).
- **A new synthetic data size**: add it to the relevant `*_SIZES` variable
  (`ITEM_TRANS_SIZES`, `ITEM_NT_SIZES`, `ANNOT_TRANS_SIZES`, `ANNOT_NT_SIZES`)
  near the top of `generate_data.sh` and rerun it — safe to rerun, it
  regenerates the base bundle and splits under `DATA/`.
- **Generative hyperparameters** (`D`, `T`, `sigma_u`, `sigma_v`, `sigma_uit`,
  `sigma_measurement`, `kappa`, ...): the same constants block in
  `generate_data.sh` (values are currently set to the inference model's prior
  means — see the comment above them).
- **Marformer architecture/training hyperparameters** (`embedding_dim`,
  `num_layers`, `lr`, masking rate, etc.): **not** exposed through
  `run_marformer_group.sh`'s CLI (only `--axis`/`--protocol`/`--sizes`/
  `--use-embed-unembed`) — either edit the hardcoded
  `python -m imputer.entity_mf.train` call in `scripts/DOMAIN3/run_marformer_group.sh`,
  or call `train.py` directly with your own flags for one-off runs.
