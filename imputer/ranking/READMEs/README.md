# Ranking System for Annotation Arena

This directory contains the implementation of a sophisticated synthetic ranking and rating system for machine learning experiments. The system models human annotation behaviors using hierarchical Bayesian methods with **cmdstanpy** for inference and neural transformers for progressive imputation.

## Overview

The ranking system extends the progressive imputation codebase by adding:

1. **Synthetic Human Annotations**: Generate realistic rating and pairwise ranking data using Stan
2. **Mixed Annotation Types**: Unary ratings and pairwise rankings 
3. **Hierarchical Modeling**: Model individual annotator differences and preferences
4. **Bayesian Inference**: Use cmdstanpy for MCMC sampling and uncertainty quantification
5. **Neural Imputation**: Transformer-based progressive imputation with masking
6. **Pure Imputation Evaluation**: Train on ALL data, test on ALL data (no artificial masking)

## Core Concept

**Embedding-Based Scoring**: Items have latent embeddings `e ∈ ℝᴰ`, annotators have preference vectors `v ∈ ℝᴰ`, and scores are computed as `score = v · e`. Different noise models and binning strategies generate different types of observational data.

## Documentation

- **[RANKING_NOTES.md](RANKING_NOTES.md)**: Complete mathematical framework and model specifications
- **[PYSTAN_INTEGRATION.md](PYSTAN_INTEGRATION.md)**: PyStan implementation guide and code examples
- **[imputer/README.md](../imputer/README.md)**: Neural imputer architecture and training details

## Key Features

### Annotation Types Supported

1. **Unary Categorical Ratings** 
   - "Rate this item 1-5 stars"
   - Model: Base score + Gaussian noise → thresholded into categories
   - Implementation: `base_scores[ij, k] + normal_rng(0, σ_measurement)` binned by rating thresholds

2. **Pairwise Rankings** 
   - "Which of these 2 items is better?"  
   - Model: Sigmoid preference with temperature scaling
   - Implementation: `P(item1 > item2) = sigmoid((score1 - score2) / temperature)`

### Model Hierarchy

```
Items: eₖ ~ N(0,I)                    # Item embeddings
Attributes: vᵢ ~ N(0,I)               # Mean preference vectors  
Annotators: vᵢⱼ ~ N(vᵢ, σ²I)          # Individual preferences
Scores: zᵢⱼₖ = vᵢⱼ · eₖ               # Base scores
Observations: f(zᵢⱼₖ + noise)         # Various noise/binning models
```

## Implementation Status

### Phase 1: Data Generation ✅ **COMPLETED**
- [x] Hierarchical Bayesian data generation using Stan
- [x] Mixed annotation types (ratings + pairwise rankings)
- [x] Complete annotation space generation with deterministic train/test splits
- [x] Configurable hyperparameters (annotator variance, measurement noise, etc.)
- [x] ICLR pairwise ranking dataset generation

### Phase 2: Domain Model ✅ **COMPLETED**
- [x] Stan MCMC model for inference from observed annotations
- [x] Rating likelihood: Gaussian noise + ordered thresholding  
- [x] Pairwise ranking likelihood: Sigmoid preference model
- [x] Pure imputation training (train on ALL data, test on ALL data)
- [x] Simple accuracy metrics (rating accuracy, ranking accuracy)
- [x] Training log-likelihood and test log-loss evaluation

### Phase 3: Neural Imputer ✅ **COMPLETED**
- [x] Transformer architecture with additive compositional embeddings
- [x] Mixed annotation heads (rating + ranking)
- [x] Progressive masking training protocol
- [x] Conditional masking for training evaluation
- [x] Pure imputation testing capability
- [x] Modular embedding providers and loss strategies

### Phase 4: Comparison Framework ✅ **COMPLETED**
- [x] Domain vs Neural model comparison on same datasets
- [x] Pure imputation evaluation metrics
- [x] Training time and accuracy benchmarking
- [x] Centralized configuration system

## Current Architecture

### Neural Imputer Architecture

**Transformer-Based Imputation**:
- **Input**: Mixed rating and ranking variables with conditional masking
- **Embedding**: Additive compositional (attribute + annotator + item(s))
- **Encoder**: 4-layer transformer with 8 attention heads
- **Heads**: Rating classifier (5-way) + Ranking utilities (pairwise)
- **Training**: Progressive masking with configurable masking rates

**Key Components**:
- `MultiVariableImputer`: Main model class with transformer backbone
- `DataConverter`: Handles data loading and batch creation with masking
- `ImputerTrainer`: Training loop with mixed losses and evaluation

### Domain Model Architecture

**Bayesian Hierarchical Model**:
- **Parameters**: Item embeddings, preference vectors, rating thresholds
- **Likelihood**: Rating (ordered thresholds) + Ranking (sigmoid pairwise)
- **Inference**: MCMC sampling with cmdstanpy
- **Evaluation**: Direct prediction from posterior means

**Key Features**:
- Ordered threshold parameterization for numerical stability
- Temperature-scaled pairwise preferences
- Pure imputation evaluation (no artificial masking)

## Configuration System

**Centralized Configuration** (`config.py`):
```python
@dataclass
class ExperimentConfig:
    K: int = 30   # number of items
    I: int = 10   # number of attributes  
    J: int = 5    # number of annotators
    D: int = 64   # embedding dimension
    C: int = 5    # number of rating categories
    
    sigma_annotator: float = 0.3    # annotator preference variance
    sigma_measurement: float = 0.1  # measurement noise variance
    temperature: float = 0.5        # ranking temperature
```

## Getting Started

### 1. Install Dependencies
```bash
conda install -c conda-forge cmdstanpy
pip install numpy scipy pandas matplotlib torch tqdm
```

### 2. Generate ICLR Pairwise Dataset
```bash
python iclr_data_generator.py
```
This creates:
- `generated_data/iclr_complete_train.json` - Training annotations
- `generated_data/iclr_complete_test.json` - Test annotations  
- `generated_data/iclr_complete_ground_truth.json` - True embeddings/preferences
- `generated_data/iclr_complete_stats.json` - Dataset statistics

### 3. Train Domain Model (Bayesian Baseline)
```bash
python domain_model_trainer.py
```
This runs MCMC inference and reports:
- Training log-likelihood
- Test rating accuracy
- Test ranking accuracy  
- Training time

### 4. Train Neural Imputer
```bash
python run_experiment_imputer.py --epochs 50 --masking_rate 0.5
```
This trains the transformer imputer with:
- Conditional masking during training
- Pure imputation evaluation on test set
- Training loss plots and model saving

## Current Results

### Domain Model Performance
- **Training**: Learns from ALL training annotations (no masking)
- **Testing**: Predicts ALL test annotations (pure imputation)
- **Metrics**: Rating accuracy, ranking accuracy, log-likelihoods

### Neural Imputer Performance  
- **Training**: Progressive masking with configurable rates
- **Testing**: Pure imputation evaluation
- **Metrics**: Rating loss, ranking loss, combined accuracy

## Research Applications

This system enables investigation of:

- **Pure vs Conditional Imputation**: Training with masking vs testing without
- **Domain vs Neural Models**: Bayesian MCMC vs transformer comparison
- **Annotation Efficiency**: Which annotation types are most informative?
- **Human Modeling**: Individual differences in preference modeling
- **Scalability**: Performance on varying dataset sizes

## Key Files

### Data Generation
- `iclr_data_generator.py` - ICLR pairwise dataset generation
- `models/complete_data_generator.stan` - Hierarchical data generation model

### Domain Model
- `domain_model_trainer.py` - Bayesian inference training
- `models/domain_model.stan` - Stan model for MCMC sampling

### Neural Imputer
- `run_experiment_imputer.py` - Neural imputer training script
- `imputer/` - Modular imputer package
  - `models.py` - Transformer architecture
  - `trainer.py` - Training loop and evaluation
  - `data.py` - Data loading and batching

### Configuration
- `config.py` - Centralized experiment configuration

## Integration with Existing Codebase

The ranking system is designed to complement the existing progressive imputation framework:

- **Shared Infrastructure**: Logging, visualization, experiment management
- **Modular Design**: Independent but compatible with existing experiments
- **Common Interfaces**: Similar data structures and evaluation metrics
- **Extensible Architecture**: Easy to add new annotation types and inference methods

This creates a comprehensive annotation research platform combining progressive learning with sophisticated human behavior modeling.

---

*For detailed technical specifications, see the documentation files in this directory.*