# Synthetic Ranking and Rating System - Technical Notes

## Overview

This document outlines a comprehensive synthetic data generation system for ranking and rating experiments. The system models multiple types of human annotations (unary ratings, pairwise comparisons, listwise rankings) using a hierarchical Bayesian framework with underlying embeddings projected through preference vectors.

## Core Mathematical Framework

### Basic Model Structure

**Items and Embeddings:**
- `K` items indexed by `k ∈ [1,K]` 
- Each item has embedding `eₖ ∈ ℝᴰ ~ N(0,I)`
- Embeddings can be fixed at training or generated for harder test instances

**Annotation Dimensions:**
- `I` attributes that can be rated or compared
- `J` annotators 
- Score generation: `score = vᵢⱼ · eₖ` (dot product of preference vector and embedding)

**Hierarchical Preference Structure:**
```
vᵢ ∈ ℝᴰ ~ N(0,I)           # Mean preference vector for attribute i
vᵢⱼ ∈ ℝᴰ ~ N(vᵢ, σ²I)      # Annotator j's personal preference for attribute i
zᵢⱼₖ = vᵢⱼ · eₖ ∈ ℝ        # Base score for item k on attribute i by annotator j
```

**Measurement Error:**
- `σ²ⱼ` = measurement error variance for annotator j
- `ϵⱼₜ ~ N(0, σ²ⱼ)` = measurement error at time t
- Allows different consistency levels across annotators

## Annotation Types and Data Generation

### 1. Unary Categorical Ratings

**Process:**
1. Add Gaussian noise to base score: `zᵢⱼₖ + ϵⱼₜ`
2. Bin the noised score into `C` categories
3. Result: Binned multivariate Gaussian distribution

**Binning Mechanism:**
```
C = number of rating categories
α = concentration parameter for symmetric Dirichlet
pᵢⱼ ∈ Δ^(C-1) ~ Dir(α/C, ..., α/C)    # Probability mass for each bin
qᵢⱼ = cumsum(pᵢⱼ)                      # Cumulative thresholds [0,1]^C
```

**Rating Assignment:**
```
xᵢⱼₖₜ ∈ [1,C] = argmin_c qᵢⱼc > CDF_ij(zᵢⱼₖ + ϵⱼₜ)
```
Where `CDF_ij` is the CDF of `N(0, ||vᵢⱼ||² + σ²ⱼ)`

**Python Implementation:**
```python
# Binning accomplished with:
rating = np.digitize(z_ijk + epsilon_jt, quantiles, right=True)
```

### 2. Pairwise/Listwise Rankings

**Process:**
1. Add independent Gumbel noise to each item's score
2. Sort by noised scores to get ranking
3. Use Plackett-Luce distribution for sampling

**Mathematical Formulation:**
```
Temp ≥ 0 = temperature parameter
π_ijSt ∈ Perm(S) ~ Plackett-Luce(zᵢⱼₖ/Temp : k ∈ S)
```
Where `S ⊆ [1,K]` is the set of items to rank, and `zᵢⱼₖ/Temp` are treated as logits.

**Properties:**
- Equivalent to adding Gumbel/negative-Gumbel noise independently to each score
- Different samples at different times `t` provide ranking variation
- Temperature controls ranking randomness

### 3. Categorical Pairwise Comparisons

**Process:**
1. Compute score difference: `zᵢⱼₖ - zᵢⱼₖ'`
2. Add Gaussian noise to difference
3. Bin into categories (e.g., "clearly better", "slightly better", "same", etc.)

**Mathematical Formulation:**
```
x_ijkk't ∈ [1,C] = argmin_c qᵢⱼc > CDF_ij(zᵢⱼₖ - zᵢⱼₖ' + ϵⱼₜ)
```
Where the CDF is for `N(0, 2||vᵢⱼ||² + σ²ⱼ)`

**Important Consideration - Active Learning Adaptation:**
The variance `2||vᵢⱼ||² + σ²ⱼ` assumes random comparisons, but active learners will compare items with similar scores. Two approaches:

1. **Fudge Factor:** Multiply the `2` by factor `< 1`
2. **Principled Heuristic:** Assume comparisons are between items with same unary rating, compute actual variance

**Variance for Same-Rating Comparisons:**
Based on truncated normal variance, the effective variance is much smaller (fudge factors ~0.11 for σⱼ=0.1, ~0.28 for σⱼ=0.5).

## Inference with Stan/PyStan

### Stan Model Structure

**Key Insight:** Unlike binned multivariate Gaussian, all underlying `z` values are consistent with observations because `ϵ` noise can always explain discrepancies.

**Integration Strategy:**
- Don't sample `ϵⱼₜ` variables (would require many rejections)
- Integrate over `ϵⱼₜ` analytically since it's independent across observations
- Sample underlying embeddings and preference vectors

**Likelihood Computation:**
For rating observation `xᵢⱼₖₜ = c`:
```
P(xᵢⱼₖₜ = c | zᵢⱼₖ) = Φ((Qᵢⱼ(qᵢⱼc) - zᵢⱼₖ)/σⱼ) - Φ((Qᵢⱼ(qᵢⱼ(c-1)) - zᵢⱼₖ)/σⱼ)
```
Where `Φ` is standard normal CDF and `Qᵢⱼ` is the quantile function.

**Stan Implementation Notes:**
- No `ϵ` variables in parameters block
- Explicit likelihood computation as above
- Joint sampling of embeddings that explain all observations

### PyStan Integration

**Key Libraries:**
- `pystan` or `cmdstanpy` for Stan interface
- `numpy` for numerical operations
- `scipy.stats` for distribution functions

**Workflow:**
1. **Data Preparation:** Convert observations to Stan-compatible format
2. **Model Compilation:** Compile Stan model with likelihood functions  
3. **Sampling:** MCMC sampling of embeddings and preference vectors
4. **Prediction:** Sample from posterior predictive for unobserved ratings/rankings

**Alternative Inference - MAP Estimation:**
- Use gradient ascent to find maximum likelihood embeddings
- Deterministic but faster than full Bayesian inference
- Works well when many observations concentrate probability mass
- May be less effective for sparse observations where marginalization helps

## Additional Query Types

### 1. Best/Worst Selection

**Process:**
- Present set `S ⊆ [1,K]` to annotator
- Ask for single best or worst item
- Less information than full ranking but lower cognitive load

**Mathematical Model:**
```
P(k = best) ∝ exp(zᵢⱼₖ/Temp)  for k ∈ S
P(k = worst) ∝ exp(-zᵢⱼₖ/Temp) for k ∈ S
```

### 2. Best-Worst Scaling (BWS) - Optional Extension

**Process:**
- Present set `S ⊆ [1,K]` with `|S| ≥ 4`
- Ask for both best AND worst items (must be distinct)
- Efficient information extraction method
- `|S|(|S|-1)` possible outcomes

**Model Options:**
1. **Argmax/Argmin with IID noise** (expensive for large `|S|`)
2. **Simple Maxdiff:** `P(k,k') ∝ exp((zᵢⱼₖ - zᵢⱼₖ')/Temp)`
3. **Sequential Selection:** Pick best first, then worst from remaining

**Note:** BWS may be useful for human annotation but not prioritized for synthetic experiments.

## Implementation Considerations

### 1. Model Variants to Implement

**Priority Order:**
1. **Unary categorical ratings** - Foundation for all other methods
2. **Pairwise rankings** - Core comparison mechanism  
3. **Categorical pairwise comparisons** - Rich comparison information
4. **Best/worst selection** - Simplified ranking queries

### 2. Parameter Ranges

**Suggested Defaults:**
- `D = 10-50` (embedding dimension)
- `C = 3-7` (number of rating categories)
- `σ² = 0.01-1.0` (annotator noise variance)
- `α = 1-10` (Dirichlet concentration)
- `Temp = 0.1-2.0` (ranking temperature)

### 3. Validation Approaches

**Synthetic Data Validation:**
1. Generate known embeddings and preferences
2. Simulate annotation process
3. Recover embeddings via inference
4. Compare recovered vs ground truth

**Cross-Validation:**
1. Hold out subset of annotations
2. Train on remaining data
3. Predict held-out annotations
4. Measure prediction accuracy

## Research Questions

### 1. Inference Method Comparison
- Full Bayesian (Stan sampling) vs MAP estimation (gradient ascent)
- When does marginalization beat point estimation?
- Computational trade-offs

### 2. Active Learning Strategies
- How to select informative item pairs for comparison?
- Best/worst vs full ranking vs pairwise comparison efficiency
- Adaptive threshold selection for categorical comparisons

### 3. Annotator Modeling
- How much individual difference modeling is needed?
- Impact of hierarchical vs flat preference structures
- Robustness to model misspecification

## Next Steps

1. **Implement basic unary rating model in Stan**
2. **Create synthetic data generation pipeline**  
3. **Test inference recovery on known ground truth**
4. **Extend to pairwise comparisons**
5. **Develop active learning query selection**
6. **Integration with existing imputation codebase**

## References

- **Stan Documentation:** https://mc-stan.org/docs/
- **PyStan Documentation:** https://pystan.readthedocs.io/en/latest/
- **Plackett-Luce Models:** For ranking distributions
- **Truncated Normal Distributions:** For categorical comparison variance
- **Best-Worst Scaling Literature:** For BWS implementation if needed

---

*This system provides a comprehensive framework for synthetic annotation experiments that can model realistic human rating and ranking behaviors while maintaining mathematical tractability for inference.*