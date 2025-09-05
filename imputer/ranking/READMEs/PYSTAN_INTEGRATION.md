# PyStan Integration Guide for Ranking System

## PyStan Overview

**PyStan** is the Python interface to Stan, a platform for statistical modeling and high-performance statistical computation. It's ideal for our Bayesian inference needs in the ranking system.

## Installation and Setup

```bash
# Install PyStan (latest version)
pip install pystan

# Alternative: CmdStanPy (often more stable)
pip install cmdstanpy
```

## Key PyStan Concepts for Our System

### 1. Stan Model Structure

Stan models have distinct blocks:
```stan
data {
    // Input data (observations, dimensions, etc.)
}

parameters {
    // Variables to be sampled (embeddings, preference vectors)
}

model {
    // Priors and likelihood computations
}

generated quantities {
    // Derived quantities, predictions
}
```

### 2. Integration with Our Ranking System

**Data Block:**
```stan
data {
    int<lower=1> K;           // number of items
    int<lower=1> I;           // number of attributes  
    int<lower=1> J;           // number of annotators
    int<lower=1> D;           // embedding dimension
    int<lower=1> C;           // number of rating categories
    
    // Observations
    int<lower=1> N_ratings;   // number of rating observations
    int<lower=1,upper=K> item_ids[N_ratings];
    int<lower=1,upper=I> attr_ids[N_ratings]; 
    int<lower=1,upper=J> annotator_ids[N_ratings];
    int<lower=1,upper=C> ratings[N_ratings];
    
    // Hyperparameters
    real<lower=0> sigma_annotator;  // annotator noise
    real<lower=0> alpha_dirichlet;  // rating threshold concentration
}
```

**Parameters Block:**
```stan
parameters {
    matrix[K, D] embeddings;          // item embeddings eₖ
    matrix[I, D] mean_preferences;    // mean preference vectors vᵢ
    matrix[I*J, D] annotator_preferences; // individual preferences vᵢⱼ
    
    // Rating thresholds per annotator-attribute pair
    simplex[C] rating_probs[I*J];     // Dirichlet-distributed probabilities
}
```

**Model Block:**
```stan
model {
    // Priors
    for (k in 1:K) {
        embeddings[k] ~ normal(0, 1);
    }
    
    for (i in 1:I) {
        mean_preferences[i] ~ normal(0, 1);
    }
    
    // Likelihood computation for ratings
    for (n in 1:N_ratings) {
        int k = item_ids[n];
        int i = attr_ids[n];
        int j = annotator_ids[n];
        int c = ratings[n];
        
        real z_score = dot_product(annotator_preferences[i*J + j], embeddings[k]);
        
        // Custom likelihood computation as described in notes
        target += rating_likelihood(c, z_score, rating_probs[i*J + j], sigma_annotator);
    }
}
```

### 3. Python Integration Workflow

**Basic PyStan Usage:**
```python
import pystan
import numpy as np

# Compile Stan model
model_code = """
// Stan model code here
"""

model = pystan.StanModel(model_code=model_code)

# Prepare data
stan_data = {
    'K': num_items,
    'I': num_attributes,
    'J': num_annotators,
    'D': embedding_dim,
    'C': num_categories,
    'N_ratings': len(observations),
    'item_ids': item_indices,
    'attr_ids': attribute_indices,
    'annotator_ids': annotator_indices,
    'ratings': rating_values,
    'sigma_annotator': 0.1,
    'alpha_dirichlet': 1.0
}

# Run MCMC sampling
fit = model.sampling(data=stan_data, 
                    iter=2000, 
                    chains=4,
                    warmup=1000)

# Extract results
embeddings_samples = fit['embeddings']
preferences_samples = fit['mean_preferences']
```

### 4. Advanced Features for Our System

**Vectorized Operations:**
Stan is highly optimized for vectorized operations:
```stan
// Efficient dot products
vector[K] scores = annotator_preferences[ij] * embeddings';

// Vectorized normal computations  
embeddings ~ normal(0, 1);  // applies to all elements
```

**Custom Functions:**
Define likelihood computations as functions:
```stan
functions {
    real rating_likelihood(int rating, real z_score, 
                          vector thresholds, real sigma) {
        // Custom likelihood computation
        real upper_thresh = thresholds[rating];
        real lower_thresh = (rating > 1) ? thresholds[rating-1] : negative_infinity();
        
        return log(Phi((upper_thresh - z_score) / sigma) - 
                  Phi((lower_thresh - z_score) / sigma));
    }
}
```

**Generated Quantities for Predictions:**
```stan
generated quantities {
    // Predict ratings for new items
    vector[K] predicted_ratings[I, J];
    
    for (i in 1:I) {
        for (j in 1:J) {
            for (k in 1:K) {
                real z_score = dot_product(annotator_preferences[i*J + j], embeddings[k]);
                // Generate predicted rating from z_score
                predicted_ratings[i,j,k] = predicted_rating(z_score, rating_probs[i*J + j]);
            }
        }
    }
}
```

## Implementation Strategy

### Phase 1: Basic Unary Ratings
1. **Simple Stan Model:** Single attribute, single annotator
2. **Synthetic Data Generation:** Known embeddings and preferences
3. **Recovery Validation:** Compare inferred vs ground truth
4. **Python Wrapper Classes:** Clean interface to Stan model

### Phase 2: Multi-Annotator Extension  
1. **Hierarchical Preferences:** Add annotator-specific variations
2. **Rating Thresholds:** Individual binning parameters
3. **Cross-Validation:** Hold-out testing framework

### Phase 3: Ranking Extensions
1. **Plackett-Luce Integration:** Add ranking likelihood  
2. **Mixed Observation Types:** Ratings + rankings together
3. **Active Learning:** Query selection strategies

### Phase 4: Optimization
1. **MAP Estimation:** Gradient-based alternative to MCMC
2. **Variational Inference:** Faster approximate posterior
3. **GPU Acceleration:** Stan GPU support for large-scale inference

## Error Handling and Debugging

**Common PyStan Issues:**
```python
# Check for sampling issues
print(fit.summary())  # Look for Rhat > 1.1, low n_eff

# Trace plots for convergence diagnosis
fit.plot()

# Extract divergent transitions
divergences = fit.get_sampler_params(inc_warmup=False)
```

**Model Validation:**
```python
# Prior predictive checks
prior_samples = model.sampling(data=stan_data, 
                              algorithm='Fixed_param',
                              iter=1000)

# Posterior predictive checks  
posterior_predictions = fit['predicted_ratings']
```

## Integration with Existing Codebase

The ranking system should integrate cleanly with the existing progressive imputation codebase:

```
ranking/
├── models/
│   ├── stan_models/           # .stan files
│   ├── rating_model.py        # PyStan wrapper classes
│   └── ranking_model.py       # Ranking-specific models
├── data/
│   ├── synthetic_generator.py # Generate synthetic annotations  
│   └── data_loader.py         # Interface to Stan data format
├── inference/
│   ├── stan_interface.py      # PyStan integration layer
│   └── evaluation.py          # Model validation and metrics
└── experiments/
    ├── rating_experiments.py  # Experimental framework
    └── active_learning.py     # Query selection strategies
```

This structure mirrors the existing codebase organization while providing clean separation between the ranking system and the progressive imputation components.

## Performance Considerations

**MCMC Tuning:**
- **Warmup:** Usually need 1000+ warmup samples for complex models
- **Thinning:** May need to thin chains for memory efficiency  
- **Parallel Chains:** Use multiple chains for convergence diagnostics

**Large-Scale Data:**
- **Minibatching:** Process subsets of observations
- **Approximate Methods:** Consider variational inference for speed
- **Caching:** Compile Stan models once, reuse across experiments

**Memory Management:**
- Stan models can use significant memory for large parameter spaces
- Consider hierarchical centering vs non-centering parameterizations
- Use sparse representations where appropriate