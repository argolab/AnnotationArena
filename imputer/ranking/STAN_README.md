# Stan Pipeline for Annotation Arena Ranking

This directory contains the Stan-based pipeline for generating synthetic annotation data and running MCMC inference for the Annotation Arena ranking project.

## Directory Structure

```
imputer/ranking/
├── stan/                          # Stan pipeline modules
│   ├── pipeline/                  # Core pipeline modules
│   │   ├── bundle.py             # Data structures
│   │   ├── configs.py            # Configuration classes
│   │   ├── data_gen.py           # Data generation
│   │   ├── inference.py           # MCMC inference
│   │   ├── predictives.py        # Prediction evaluation
│   │   └── io.py                 # I/O utilities
│   ├── scripts/                   # CLI scripts
│   │   ├── generate_data.py       # Data generation
│   │   ├── run_inference.py       # MCMC inference
│   │   ├── evaluate_predictions.py # Evaluation
│   │   └── run_full_experiment.py # End-to-end pipeline
│   └── tests/                     # Unit tests
├── models/                        # Stan models
│   ├── iclr_data_generation.stan  # Data generation model
│   └── domain_model.stan          # Inference model
└── OUTPUT/                        # Default output directory
    ├── generated_data/            # Generated datasets
    ├── domain_model/
    │   ├── runs/                  # MCMC inference results
    │   └── eval/                  # Evaluation results
    └── imputer/                   # Imputer results (future)
```

## Setup

```bash
# Activate virtual environment
source ../../venv/bin/activate

# Set Python path
export PYTHONPATH=.

# Run from imputer/ranking directory
cd imputer/ranking
```

## Phase-by-Phase Commands

### Phase 1: Data Generation

Generate synthetic annotation data with train/test instances:

```bash
# Basic data generation
python stan/scripts/generate_data.py

# Custom parameters
python stan/scripts/generate_data.py --K-train 5 --K-test 3 --I 3 --J 6 --D 8 --C 5 --seed 42

# Ablation studies
python stan/scripts/generate_data.py --disable-third-annotator --disable-pairwise-rankings

# Custom output directory
python stan/scripts/generate_data.py --output-dir custom_data --run-name my_experiment
```

**Output**: `OUTPUT/generated_data/run_YYYYMMDD_HHMMSS/`
- `data_bundle.json`: Complete dataset with ground truth
- `configs.json`: Generation configuration

### Phase 2: MCMC Inference

Run MCMC inference on generated data:

```bash
# Basic inference (requires data bundle)
python stan/scripts/run_inference.py --data-bundle OUTPUT/generated_data/run_YYYYMMDD_HHMMSS/data_bundle.json

# Custom MCMC parameters
python stan/scripts/run_inference.py --data-bundle <bundle> --chains 4 --iter-warmup 1000 --iter-sampling 1000

# Different initialization strategies
python stan/scripts/run_inference.py --data-bundle <bundle> --init-strategy ground_truth
python stan/scripts/run_inference.py --data-bundle <bundle> --init-strategy random

# Learning modes
python stan/scripts/run_inference.py --data-bundle <bundle> --use-train-only
python stan/scripts/run_inference.py --data-bundle <bundle> --use-test-only
# Default: transductive (both train and test)
```

**Output**: `OUTPUT/domain_model/runs/run_YYYYMMDD_HHMMSS/`
- `domain_model-*.csv`: MCMC samples
- `configs.json`: Inference configuration

### Phase 3: Evaluation

Evaluate MCMC predictions against ground truth:

```bash
# Basic evaluation
python stan/scripts/evaluate_predictions.py --mcmc-dir OUTPUT/domain_model/runs/run_YYYYMMDD_HHMMSS --data-bundle OUTPUT/generated_data/run_YYYYMMDD_HHMMSS/data_bundle.json

# Verbose output
python stan/scripts/evaluate_predictions.py --mcmc-dir <mcmc_dir> --data-bundle <bundle> --verbose
```

**Output**: `OUTPUT/domain_model/eval/run_YYYYMMDD_HHMMSS/`
- `predictive_metrics.json`: Summary metrics
- `rating_predictions.csv`: Individual rating predictions
- `rating_probabilities.csv`: Probability distributions
- `pairwise_predictions.csv`: Pairwise ranking predictions

### Phase 4: End-to-End Pipeline

Run complete experiment pipeline:

```bash
# Basic full experiment
python stan/scripts/run_full_experiment.py

# Custom parameters
python stan/scripts/run_full_experiment.py --K-train 5 --K-test 3 --chains 2 --iter-warmup 100 --iter-sampling 100

# Skip stages
python stan/scripts/run_full_experiment.py --skip-data-gen --skip-inference
python stan/scripts/run_full_experiment.py --skip-evaluation

# Different learning modes
python stan/scripts/run_full_experiment.py --use-train-only
python stan/scripts/run_full_experiment.py --use-test-only
```

**Output**: `OUTPUT/domain_model/run_YYYYMMDD_HHMMSS/`
- `experiment_config.json`: Complete experiment configuration
- `data/`: Data generation results
- `inference/`: MCMC inference results  
- `evaluation/`: Evaluation results

## Quick Start Examples

### Minimal Example
```bash
# Generate small dataset
python stan/scripts/generate_data.py --K-train 3 --K-test 2 --I 2 --J 3 --D 4 --C 3

# Run inference
python stan/scripts/run_inference.py --data-bundle OUTPUT/generated_data/run_*/data_bundle.json --chains 2 --iter-warmup 50 --iter-sampling 50

# Evaluate results
python stan/scripts/evaluate_predictions.py --mcmc-dir OUTPUT/domain_model/runs/run_* --data-bundle OUTPUT/generated_data/run_*/data_bundle.json
```

### Full Experiment
```bash
# Run everything in one command
python stan/scripts/run_full_experiment.py --K-train 5 --K-test 3 --chains 2 --iter-warmup 100 --iter-sampling 100
```

## Testing

Run unit tests:

```bash
# All tests
python -m pytest stan/tests/ -v

# Specific test modules
python -m pytest stan/tests/test_data_gen_stan.py -v
python -m pytest stan/tests/test_inference.py -v
python -m pytest stan/tests/test_predictives.py -v
```

## Configuration

All scripts automatically extract configuration from the data generation `configs.json` file, including:
- Dimensions: `K_train`, `K_test`, `I`, `J`, `D`, `C`
- Hyperparameters: `sigma_annotator`, `sigma_measurement`, `alpha_dirichlet`, `temperature`
- Data generation settings: `enable_third_annotator`, `enable_pairwise_rankings`, `pairwise_cap_per_item`

## Logging

The inference script includes comprehensive logging to help debug issues:

- **Console Output**: Real-time progress and results
- **Log File**: Detailed logging saved to `inference.log` in the current directory
- **Log Levels**: INFO for normal operation, WARNING for issues, ERROR for failures

Key logged information:
- Data bundle loading and configuration extraction
- Stan model compilation status
- Initialization strategy details (ground truth vs random)
- MCMC sampling parameters and progress
- Diagnostic information (divergent transitions, mixing quality)

### Initialization Strategies

- **`ground_truth`**: Uses actual parameter values from data generation (recommended for testing)
- **`random`**: Uses random initialization with range [-2, 2] (default for production)
- **`file`**: Uses custom initialization file (advanced users)

## Output Organization

- **Generated Data**: `OUTPUT/generated_data/` - Synthetic datasets with ground truth
- **Domain Model Runs**: `OUTPUT/domain_model/runs/` - MCMC inference results
- **Domain Model Eval**: `OUTPUT/domain_model/eval/` - Prediction evaluations
- **Imputer Results**: `OUTPUT/imputer/` - Future imputer model results

All outputs include timestamps and organized directory structures for easy tracking and comparison of experiments.
