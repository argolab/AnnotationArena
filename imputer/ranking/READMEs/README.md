# Ranking System for Annotation Arena

This directory contains the implementation of a sophisticated synthetic ranking and rating system for machine learning experiments. The system models human annotation behaviors using hierarchical Bayesian methods with **cmdstanpy** for inference.

## Overview

The ranking system extends the progressive imputation codebase by adding:

1. **Synthetic Human Annotations**: Generate realistic rating and ranking data using Stan
2. **Mixed Annotation Types**: Unary ratings and listwise rankings (pairwise comparisons removed) 
3. **Hierarchical Modeling**: Model individual annotator differences and preferences
4. **Bayesian Inference**: Use cmdstanpy for MCMC sampling and uncertainty quantification
5. **Progressive Training**: Compare domain model (MCMC) vs neural imputer on increasing data budgets

## Core Concept

**Embedding-Based Scoring**: Items have latent embeddings `e ∈ ℝᴰ`, annotators have preference vectors `v ∈ ℝᴰ`, and scores are computed as `score = v · e`. Different noise models and binning strategies generate different types of observational data.

## Documentation

- **[RANKING_NOTES.md](RANKING_NOTES.md)**: Complete mathematical framework and model specifications
- **[PYSTAN_INTEGRATION.md](PYSTAN_INTEGRATION.md)**: PyStan implementation guide and code examples

## Key Features

### Annotation Types Supported

1. **Unary Categorical Ratings** 
   - "Rate this item 1-5 stars"
   - Model: Base score + Gaussian noise → thresholded into categories
   - Implementation: `base_scores[ij, k] + normal_rng(0, σ_measurement)` binned by rating thresholds

2. **Listwise Rankings** 
   - "Rank these K items from best to worst"  
   - Model: Plackett-Luce with Gumbel noise for ranking generation
   - Implementation: `base_scores[ij, k] / temperature + gumbel_noise` sorted for ranking order

**Note**: Pairwise comparisons were removed as they are a special case of listwise rankings.

### Model Hierarchy

```
Items: eₖ ~ N(0,I)                    # Item embeddings
Attributes: vᵢ ~ N(0,I)               # Mean preference vectors  
Annotators: vᵢⱼ ~ N(vᵢ, σ²I)          # Individual preferences
Scores: zᵢⱼₖ = vᵢⱼ · eₖ               # Base scores
Observations: f(zᵢⱼₖ + noise)         # Various noise/binning models
```

## Implementation Status

**Phase 1: Data Generation** ✅ **COMPLETED**
- [x] Hierarchical Bayesian data generation using Stan
- [x] Mixed annotation types (ratings + rankings)
- [x] Complete annotation space generation with deterministic train/test splits
- [x] Configurable hyperparameters (annotator variance, measurement noise, etc.)

**Phase 2: Domain Model** ✅ **COMPLETED**
- [x] Stan MCMC model for inference from observed annotations
- [x] Rating likelihood: Gaussian noise + thresholding  
- [x] Ranking likelihood: Plackett-Luce model
- [x] Progressive training with increasing data budgets
- [x] KL divergence and log-likelihood evaluation

**Phase 3: Neural Imputer** 🔄 **IN PROGRESS**
- [x] Transformer architecture with additive compositional embeddings
- [x] Mixed annotation heads (rating + ranking)
- [x] Progressive masking training protocol
- [ ] Ranking evaluation fixes and Plackett-Luce loss integration

**Phase 4: Comparison Framework** ⏳ **PENDING**
- [ ] Domain vs Neural model comparison on same datasets
- [ ] Visualization and analysis of results
- [ ] Performance benchmarking

## Research Applications

This system enables investigation of:

- **Annotation Efficiency**: Which query types extract information most efficiently?
- **Active Learning**: How to select informative annotation requests?
- **Human Modeling**: How much individual difference modeling is necessary?
- **Inference Methods**: MCMC sampling vs MAP estimation trade-offs
- **Scalability**: Performance on large item sets and annotator pools

## Getting Started

1. **Install Dependencies**:
   ```bash
   conda install -c conda-forge cmdstanpy
   pip install numpy scipy pandas matplotlib torch
   ```

2. **Generate Data**:
   ```bash
   python complete_data_generator.py
   ```

3. **Run Domain Model**:
   ```bash
   python domain_model_trainer.py
   ```

4. **Run Neural Imputer**:
   ```bash
   python neural_imputer_trainer.py
   ```

## Current Implementation

### Data Generation (`complete_data_generator.py`)
- **Stan Model**: `models/complete_data_generator.stan` 
- **Hierarchical Generation**: Items → Attributes → Annotators → Scores → Annotations
- **Output**: Train/test splits with ground truth embeddings and preferences

### Domain Model (`domain_model_trainer.py`)  
- **Stan Model**: `models/domain_model.stan`
- **MCMC Inference**: Learn embeddings and preferences from observed annotations
- **Progressive Training**: 20%, 50%, 100% data budgets with time tracking

### Neural Imputer (`neural_imputer_trainer.py`)
- **Architecture**: Transformer with additive compositional embeddings (e_i + e_j + e_k)
- **Mixed Heads**: Rating head (categorical) + Ranking head (Plackett-Luce utilities)
- **Training**: Progressive masking with increasing data budgets

## Integration with Existing Codebase

The ranking system is designed to complement the existing progressive imputation framework:

- **Shared Infrastructure**: Logging, visualization, experiment management
- **Modular Design**: Independent but compatible with existing experiments
- **Common Interfaces**: Similar data structures and evaluation metrics
- **Extensible Architecture**: Easy to add new annotation types and inference methods

### Code Reuse from Gaussian Experiments

**Available Components from `../gaussian/`:**
- **Genz Algorithm**: Advanced Monte Carlo integration for binned multivariate normals (directly applicable to our binning model)
- **MAP Estimation Framework**: Domain baseline pattern with training time tracking  
- **Neural Architecture**: Transformer-based imputer with similarity smoothing
- **Evaluation Pipeline**: KL divergence computation and model comparison framework
- **Experiment Runner**: Three-way comparison (Ground Truth vs Domain vs Neural)

These components can be adapted for ranking with minimal modifications since both problems involve binning continuous latent variables and comparing neural vs domain-specific models.

This creates a comprehensive annotation research platform combining progressive learning with sophisticated human behavior modeling.

---

*For detailed technical specifications, see the documentation files in this directory.*