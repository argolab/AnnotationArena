# Ranking System for Annotation Arena

This directory contains the implementation of a sophisticated synthetic ranking and rating system for machine learning experiments. The system models human annotation behaviors using hierarchical Bayesian methods with Stan/PyStan for inference.

## Overview

The ranking system extends the progressive imputation codebase by adding:

1. **Synthetic Human Annotations**: Generate realistic rating and ranking data
2. **Multiple Annotation Types**: Unary ratings, pairwise comparisons, listwise rankings  
3. **Hierarchical Modeling**: Model individual annotator differences and preferences
4. **Bayesian Inference**: Use Stan for principled uncertainty quantification
5. **Active Learning**: Intelligent query selection for annotation efficiency

## Core Concept

**Embedding-Based Scoring**: Items have latent embeddings `e ∈ ℝᴰ`, annotators have preference vectors `v ∈ ℝᴰ`, and scores are computed as `score = v · e`. Different noise models and binning strategies generate different types of observational data.

## Documentation

- **[RANKING_NOTES.md](RANKING_NOTES.md)**: Complete mathematical framework and model specifications
- **[PYSTAN_INTEGRATION.md](PYSTAN_INTEGRATION.md)**: PyStan implementation guide and code examples

## Key Features

### Annotation Types Supported

1. **Unary Categorical Ratings** 
   - "Rate this item 1-5 stars"
   - Gaussian noise + binning model

2. **Pairwise Comparisons**
   - "Is item A better than item B?"
   - Categorical responses: "much better", "slightly better", "same", etc.

3. **Listwise Rankings** 
   - "Rank these 5 items from best to worst"
   - Plackett-Luce model with Gumbel noise

4. **Best/Worst Selection**
   - "Pick the best and worst from this set"
   - Simplified ranking with lower cognitive load

### Model Hierarchy

```
Items: eₖ ~ N(0,I)                    # Item embeddings
Attributes: vᵢ ~ N(0,I)               # Mean preference vectors  
Annotators: vᵢⱼ ~ N(vᵢ, σ²I)          # Individual preferences
Scores: zᵢⱼₖ = vᵢⱼ · eₖ               # Base scores
Observations: f(zᵢⱼₖ + noise)         # Various noise/binning models
```

## Implementation Status

**Phase 1: Foundation** *(Planned)*
- [ ] Basic synthetic data generation
- [ ] Simple Stan model for unary ratings
- [ ] PyStan integration layer
- [ ] Model validation framework

**Phase 2: Extensions** *(Future)*  
- [ ] Multi-annotator hierarchical models
- [ ] Pairwise comparison models
- [ ] Ranking model integration
- [ ] Active learning strategies

**Phase 3: Integration** *(Future)*
- [ ] Connection to progressive imputation experiments
- [ ] Large-scale evaluation framework
- [ ] Performance optimization

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
   pip install pystan numpy scipy pandas
   ```

2. **Review Documentation**: Start with `RANKING_NOTES.md` for the mathematical framework

3. **Explore Examples**: Implementation examples in `PYSTAN_INTEGRATION.md`

4. **Run Tests**: Validation scripts (to be implemented)

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