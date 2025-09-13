# Domain Model — Bayesian Annotation Inference

This document describes the Bayesian hierarchical model used for annotation inference in the ranking system. The domain model serves as a probabilistic baseline for comparison with neural imputation methods.

## Overview

The domain model uses **cmdstanpy** for MCMC inference to learn latent embeddings and preferences from observed annotations. It implements a hierarchical Bayesian model that captures individual annotator differences and measurement noise.

## Model Architecture

### Hierarchical Structure

```
Items: eₖ ~ N(0, σ_e²I)                    # Item embeddings [K × D]
Attributes: vᵢ ~ N(0, σ_v²I)               # Mean preference vectors [I × D]  
Annotators: vᵢⱼ ~ N(vᵢ, σ_a²I)             # Individual preferences [I×J × D]
Scores: zᵢⱼₖ = vᵢⱼ · eₖ                     # Base utility scores
Observations: Various likelihood models
```

### Parameters

**Latent Variables**:
- `embeddings[K, D]`: Item embeddings in D-dimensional space
- `mean_preferences[I, D]`: Mean preference vectors per attribute
- `annotator_preferences[I*J, D]`: Individual annotator preferences
- `rating_thresholds_raw[I*J, C-1]`: Rating threshold parameters

**Hyperparameters**:
- `sigma_annotator`: Annotator preference variance (default: 0.3)
- `sigma_measurement`: Measurement noise variance (default: 0.1)  
- `temperature`: Ranking temperature scaling (default: 0.5)
- `sigma_embedding_prior`: Embedding prior scale (default: 1.0)
- `sigma_preference_prior`: Preference prior scale (default: 1.0)

## Likelihood Models

### Rating Likelihood

**Model**: Ordered threshold model with Gaussian noise
```
z_ijk = v_ij · e_k                           # Base score
y_ijk = z_ijk + ε, ε ~ N(0, σ_m²)            # Noisy observation
rating = c if Q_{c-1} < y_ijk ≤ Q_c          # Threshold binning
```

**Implementation**:
```stan
// Rating thresholds: [-∞, Q₁, Q₂, ..., Q_{C-1}, +∞]
real upper_prob = Phi((Q_c - z_ijk) / sigma_measurement);
real lower_prob = Phi((Q_{c-1} - z_ijk) / sigma_measurement);
real prob = upper_prob - lower_prob;
target += log(prob);
```

**Threshold Parameterization**:
- Uses free parameters instead of ordered constraints for numerical stability
- First threshold: `Q₁ = raw₁`
- Subsequent: `Q_c = Q_{c-1} + exp(raw_c)` (ensures ordering)

### Pairwise Ranking Likelihood

**Model**: Temperature-scaled sigmoid preference
```
z_ij1 = v_ij · e_1                          # Score for item 1
z_ij2 = v_ij · e_2                          # Score for item 2
P(item1 > item2) = sigmoid((z_ij1 - z_ij2) / τ)
```

**Implementation**:
```stan
real score1 = base_scores[ij_idx, item1] / temperature;
real score2 = base_scores[ij_idx, item2] / temperature;

if (item1_ranks_first) {
    target += log_inv_logit(score1 - score2);
} else {
    target += log_inv_logit(score2 - score1);
}
```

## Prior Specifications

### Embedding Priors
```stan
// Item embeddings: weakly informative
for (k in 1:K) {
    embeddings[k] ~ normal(0, sigma_embedding_prior);  // σ = 1.0
}

// Mean preferences: weakly informative  
for (i in 1:I) {
    mean_preferences[i] ~ normal(0, sigma_preference_prior);  // σ = 1.0
}
```

### Annotator Preference Hierarchy
```stan
// Individual preferences around attribute means
for (i in 1:I) {
    for (j in 1:J) {
        int idx = (i-1)*J + j;
        annotator_preferences[idx] ~ normal(mean_preferences[i], sigma_annotator);
    }
}
```

### Threshold Priors
```stan
// Rating thresholds with moderate informativeness
for (ij in 1:(I*J)) {
    rating_thresholds_raw[ij, 1] ~ normal(0, 1.0);      // First threshold
    for (c in 2:(C-1)) {
        rating_thresholds_raw[ij, c] ~ normal(0, 0.5);  // Log-spacings
    }
}
```

## Training Protocol

### Pure Imputation Paradigm

**Training Phase**:
1. Observe ALL training annotations (ratings + pairwise rankings)
2. Use MCMC to sample from posterior distribution
3. No artificial masking during training

**Testing Phase**:
1. Predict ALL test annotations (pure imputation)
2. Use posterior means for point predictions
3. Evaluate accuracy and log-likelihood

### MCMC Configuration

**Default Settings**:
```python
config = DomainModelConfig(
    chains=2,                    # Number of MCMC chains
    iter_warmup=500,            # Warmup iterations per chain
    iter_sampling=2000,         # Sampling iterations per chain
    adapt_delta=0.8,            # Target acceptance rate
    max_treedepth=15,           # Maximum tree depth
)
```

**Initialization**:
- Small random embeddings: `N(0, 0.5²)`
- Ordered random thresholds for each annotator-attribute pair
- Annotator preferences close to mean preferences

## Prediction and Evaluation

### Rating Prediction

**Posterior Predictive**:
```python
# Extract posterior means
embeddings = np.mean(fit.stan_variable('embeddings'), axis=0)
preferences = np.mean(fit.stan_variable('annotator_preferences'), axis=0)
thresholds = np.mean(fit.stan_variable('rating_thresholds_raw'), axis=0)

# Compute base score
base_score = preferences[ij_idx] @ embeddings[k]

# Create threshold boundaries
full_thresholds = [-∞, thresholds[ij_idx], +∞]

# Compute category probabilities
for c in range(C):
    upper_prob = norm.cdf((Q_c - base_score) / sigma_measurement)
    lower_prob = norm.cdf((Q_{c-1} - base_score) / sigma_measurement)
    prob[c] = upper_prob - lower_prob

# Predict most likely category
predicted_rating = argmax(prob) + 1
```

### Ranking Prediction

**Pairwise Comparison**:
```python
# Compute item scores
score1 = preferences[ij_idx] @ embeddings[item1]
score2 = preferences[ij_idx] @ embeddings[item2]

# Predict preference
predicted_preference = score1 > score2
```

### Accuracy Metrics

**Rating Accuracy**:
```python
correct_ratings = sum(predicted_rating == true_rating for all test ratings)
rating_accuracy = correct_ratings / num_test_ratings
```

**Ranking Accuracy**:
```python
correct_rankings = sum(predicted_preference == true_preference for all test pairs)
ranking_accuracy = correct_rankings / num_test_pairs
```

## Implementation Details

### Stan Model File

**Location**: `models/domain_model.stan`

**Key Sections**:
- `data`: Input dimensions and observed annotations
- `parameters`: Latent embeddings, preferences, thresholds
- `transformed parameters`: Base scores and ordered thresholds
- `model`: Prior specifications and likelihoods
- `generated quantities`: Log-likelihood computation for evaluation

### Python Interface

**Trainer Class**: `DomainModelTrainer`

**Key Methods**:
```python
def load_data(data_path: Path) -> Dict[str, Any]
def prepare_stan_data(observed_data: Dict, config: DomainModelConfig) -> Dict
def train_and_evaluate(data_path: Path, config: DomainModelConfig) -> DomainModelResults
def compute_test_accuracy(fit, test_data: Dict, stan_data: Dict) -> Dict[str, float]
```

### Configuration Integration

**Centralized Config**:
```python
from config import ExperimentConfig
exp_config = ExperimentConfig()

config = DomainModelConfig(
    sigma_annotator=exp_config.sigma_annotator,
    sigma_measurement=exp_config.sigma_measurement,
    temperature=exp_config.temperature
)
```

## Usage

### Basic Training

```bash
python domain_model_trainer.py
```

### Programmatic Usage

```python
from domain_model_trainer import DomainModelTrainer, DomainModelConfig
from pathlib import Path

# Initialize trainer
trainer = DomainModelTrainer()

# Configure model
config = DomainModelConfig(
    chains=2,
    iter_warmup=500,
    iter_sampling=2000
)

# Train and evaluate
data_path = Path("generated_data")
results = trainer.train_and_evaluate(data_path, config, seed=12345)

print(f"Rating accuracy: {results.test_rating_accuracy:.3f}")
print(f"Ranking accuracy: {results.test_ranking_accuracy:.3f}")
```

## Performance Characteristics

### Computational Complexity

**Training Time**: O(chains × iterations × N_annotations)
- Typical: 2 chains × 2500 iterations × 2000 annotations ≈ 10-60 minutes
- Scales linearly with number of observations
- Parallel chains reduce wall-clock time

**Memory Usage**: O(I×J×D + K×D + samples×parameters)
- Dominated by annotator preferences: I×J×D
- Sample storage: chains×iterations×parameter_count

### Convergence Diagnostics

**Rhat Statistic**: Should be < 1.1 for all parameters
**Effective Sample Size**: Should be > 400 per chain
**Divergent Transitions**: Should be < 1% of total samples

### Typical Performance

**ICLR Dataset** (I=10, J=5, K=30, D=64):
- Training time: ~30-60 minutes (2 chains, 2500 iterations each)
- Rating accuracy: 60-80% (baseline random: 20%)
- Ranking accuracy: 65-85% (baseline random: 50%)

## Advantages and Limitations

### Advantages

**Theoretical Foundation**:
- Principled uncertainty quantification
- Hierarchical modeling of annotator differences
- Interpretable parameters and structure

**Robustness**:
- Handles missing data naturally
- No overfitting concerns with proper priors
- Uncertainty estimates for predictions

### Limitations

**Computational Cost**:
- Slow inference compared to neural methods
- Doesn't scale to very large datasets
- Requires MCMC convergence monitoring

**Model Flexibility**:
- Fixed hierarchical structure
- Limited capacity compared to neural networks
- Requires manual model specification

## Comparison with Neural Imputer

### Complementary Strengths

**Domain Model**:
- Interpretable structure and parameters
- Uncertainty quantification
- Principled handling of hierarchy
- Strong theoretical foundation

**Neural Imputer**:
- Flexible representation learning
- Fast inference after training
- Scalable to large datasets
- End-to-end optimization

### Evaluation Framework

Both models are evaluated on the same datasets using:
- Pure imputation protocol (train on ALL, test on ALL)
- Identical accuracy metrics
- Training time comparisons
- Same hyperparameter configurations

This enables direct comparison of Bayesian vs neural approaches to annotation imputation.

---

*For Stan model implementation details, see `models/domain_model.stan`. For usage examples, see `domain_model_trainer.py`.*