# STAN Model Experiments Implementation Plan

This document outlines the implementation plan for three key STAN model experiments to understand model behavior and performance under different conditions.

## Overview

The STAN model (`iclr_data_generation.stan`) generates synthetic data with known ground truth parameters, allowing us to test model recovery under various conditions. The three experiments focus on:

1. **Overfitting Analysis**: Test STAN on tiny instances to observe overfitting behavior
2. **Oracle Knowledge**: Provide STAN with correct item embeddings (K embeddings)
3. **Correct Initialization**: Initialize STAN with ground truth parameters

## Experiment 1: Tiny Instance Overfitting Analysis

### Objective
Demonstrate overfitting behavior by running STAN on very small instances where the model has more parameters than observations.

### Implementation Plan

#### 1.1 Create Tiny Instance Configuration
**File**: `configs/tiny_instance_overfit.json`

```json
{
  "data_config": {
    "num_instances": 1,
    "K": 5,           // Very few items
    "I": 2,           // Very few attributes  
    "J": 2,           // Very few annotators
    "C": 3,           // Few rating categories
    "D": 16,          // Moderate embedding dimension
    "sigma_annotator": 0.1,
    "sigma_measurement": 0.05,
    "alpha_dirichlet": 1.0,
    "temperature": 0.3,
    "max_pairs_per_tied_group": 5,
    "min_group_size": 2,
    "max_group_size": 3
  },
  "domain_config": {
    "chains": 2,
    "iter_warmup": 200,
    "iter_sampling": 1000,
    "adapt_delta": 0.8,
    "max_treedepth": 10,
    "sample_counts": [100, 200, 500, 1000]
  },
  "experiment_name": "tiny_instance_overfit",
  "enabled_strategies": ["Domain_Model"]
}
```

#### 1.2 Analysis Metrics
- **Parameter Count**: Count total STAN parameters vs observations
- **Training Loss**: Monitor training log-likelihood convergence
- **Test Performance**: Compare training vs test performance
- **Posterior Variance**: Analyze posterior parameter uncertainty
- **Convergence Diagnostics**: Check Rhat, effective sample size

#### 1.3 Expected Results
- Very low training loss (potential overfitting)
- Poor generalization to test data
- High posterior variance due to identifiability issues
- Potential convergence problems

### Implementation Steps

1. **Create Configuration**: Generate `tiny_instance_overfit.json`
2. **Generate Data**: Use partial runner to create tiny instance data
3. **Run Domain Model**: Evaluate with different sample counts to observe overfitting
4. **Analysis Script**: Create `analyze_overfitting.py` to compute parameter/observation ratios
5. **Visualization**: Plot training vs test performance curves

---

## Experiment 2: Oracle K (Correct Item Embeddings)

### Objective
Provide STAN with the correct item embeddings (K embeddings) from ground truth, testing how oracle knowledge of the true embeddings affects model performance and convergence.

### Implementation Plan

#### 2.1 Ground Truth Extraction
**File**: `utils/ground_truth_extractor.py`

```python
class GroundTruthExtractor:
    """Extract ground truth parameters from generated data."""
    
    def extract_true_embeddings(self, data_path: Path) -> np.ndarray:
        """Extract true item embeddings (K embeddings) from ground truth."""
        with open(data_path / "iclr_complete_ground_truth.json", 'r') as f:
            ground_truth = json.load(f)
        return np.array(ground_truth['embeddings'])  # Shape: [K, D]
    
    def extract_true_parameters(self, data_path: Path) -> Dict[str, Any]:
        """Extract all ground truth parameters."""
        with open(data_path / "iclr_complete_ground_truth.json", 'r') as f:
            ground_truth = json.load(f)
        
        return {
            'embeddings': ground_truth['embeddings'],  # K item embeddings
            'mean_preferences': ground_truth['mean_preferences'],
            'annotator_preferences': ground_truth['annotator_preferences'],
            'rating_thresholds': ground_truth['rating_thresholds'],
            'K': len(ground_truth['embeddings']),  # Number of items
            'D': len(ground_truth['embeddings'][0])  # Embedding dimension
        }
```

#### 2.2 Model Checkpoint Approach
**Enhancement**: `domain_model_trainer.py`

```python
def save_model_checkpoint(self, fit, checkpoint_path: Path):
    """Save STAN model parameters as checkpoint."""
    # Extract parameters from fitted model
    extracted_params = fit.extract()
    
    # Convert numpy arrays to lists for JSON compatibility
    params_to_save = {key: value.tolist() for key, value in extracted_params.items()}
    
    with open(checkpoint_path, 'w') as f:
        json.dump(params_to_save, f)
    
    logger.info(f"Model checkpoint saved to {checkpoint_path}")

def load_model_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
    """Load STAN model parameters from checkpoint."""
    with open(checkpoint_path, 'r') as f:
        loaded_params = json.load(f)
    
    # Convert lists back to numpy arrays
    params = {key: np.array(value) for key, value in loaded_params.items()}
    
    logger.info(f"Model checkpoint loaded from {checkpoint_path}")
    return params

def create_oracle_initialization(self, checkpoint_path: Path, 
                               ground_truth_path: Path,
                               noise_scale: float = 0.01) -> List[Dict[str, Any]]:
    """Create initialization using checkpoint + ground truth with noise."""
    
    # Load checkpoint parameters
    checkpoint_params = self.load_model_checkpoint(checkpoint_path)
    
    # Load ground truth
    extractor = GroundTruthExtractor()
    true_params = extractor.extract_true_parameters(ground_truth_path)
    
    def create_oracle_init():
        # Use checkpoint parameters as base
        init_values = {}
        for key, value in checkpoint_params.items():
            if key == 'embeddings':
                # Replace embeddings with ground truth + noise
                true_embeddings = np.array(true_params['embeddings'])
                noise = np.random.normal(0, noise_scale, true_embeddings.shape)
                init_values[key] = (true_embeddings + noise).tolist()
            else:
                init_values[key] = value.tolist() if hasattr(value, 'tolist') else value
        
        return init_values
    
    return [create_oracle_init() for _ in range(self.config.chains)]
```

#### 2.3 Oracle Configuration
**File**: `configs/oracle_k_experiment.json`

```json
{
  "data_config": {
    "num_instances": 3,
    "K": 20,
    "I": 5,
    "J": 4,
    "C": 5,
    "D": 32,
    "sigma_annotator": 0.3,
    "sigma_measurement": 0.1,
    "alpha_dirichlet": 2.0,
    "temperature": 0.5
  },
  "domain_config": {
    "chains": 4,
    "iter_warmup": 500,    // Reduced warmup since starting near optimum
    "iter_sampling": 2000,
    "use_oracle_checkpoint": true,
    "oracle_noise_scale": 0.01,
    "test_masking_rate": 0.5
  },
  "experiment_name": "oracle_k_experiment"
}
```

#### 2.4 Analysis Metrics
- **Embedding Recovery**: Compare learned vs true item embeddings (cosine similarity, MSE)
- **Parameter Recovery**: Measure correlation between learned and true preferences/thresholds
- **Convergence Speed**: Compare convergence with oracle vs learned embeddings
- **Posterior Quality**: Analyze posterior concentration around true values
- **Imputation Accuracy**: Compare rating/ranking accuracy with oracle embeddings

### Implementation Steps

1. **Create Ground Truth Extractor**: Implement parameter extraction utilities
2. **Add Checkpoint Support**: Implement save/load model checkpoints in domain trainer
3. **Generate Data**: Use partial runner to create oracle experiment data
4. **Run Baseline**: Run domain model and save checkpoint
5. **Run Oracle**: Load checkpoint + ground truth embeddings for initialization
6. **Analysis**: Compare embedding recovery and convergence

---

## Experiment 3: Correct Parameter Initialization

### Objective
Initialize STAN with ground truth parameters to test convergence and posterior quality when starting from optimal conditions.

### Implementation Plan

#### 3.1 Full Oracle Initialization
**Enhancement**: `domain_model_trainer.py`

```python
def create_full_oracle_initialization(self, checkpoint_path: Path, 
                                    ground_truth_path: Path,
                                    noise_scale: float = 0.01) -> List[Dict[str, Any]]:
    """Create initialization using checkpoint + all ground truth parameters with noise."""
    
    # Load checkpoint parameters
    checkpoint_params = self.load_model_checkpoint(checkpoint_path)
    
    # Load ground truth
    extractor = GroundTruthExtractor()
    true_params = extractor.extract_true_parameters(ground_truth_path)
    
    def create_oracle_init():
        # Use checkpoint parameters as base
        init_values = {}
        for key, value in checkpoint_params.items():
            if key == 'embeddings':
                # Replace embeddings with ground truth + noise
                true_embeddings = np.array(true_params['embeddings'])
                noise = np.random.normal(0, noise_scale, true_embeddings.shape)
                init_values[key] = (true_embeddings + noise).tolist()
            elif key == 'mean_preferences':
                # Replace mean preferences with ground truth + noise
                true_prefs = np.array(true_params['mean_preferences'])
                noise = np.random.normal(0, noise_scale, true_prefs.shape)
                init_values[key] = (true_prefs + noise).tolist()
            elif key == 'annotator_preferences':
                # Replace annotator preferences with ground truth + noise
                true_prefs = np.array(true_params['annotator_preferences'])
                noise = np.random.normal(0, noise_scale, true_prefs.shape)
                init_values[key] = (true_prefs + noise).tolist()
            elif key == 'rating_thresholds_increments':
                # Replace thresholds with ground truth + noise
                true_thresholds = np.array(true_params['rating_thresholds'])
                # Convert to increments
                increments = np.diff(true_thresholds, axis=1)
                noise = np.random.normal(0, noise_scale, increments.shape)
                init_values[key] = (increments + noise).tolist()
            else:
                init_values[key] = value.tolist() if hasattr(value, 'tolist') else value
        
        return init_values
    
    return [create_oracle_init() for _ in range(self.config.chains)]
```

#### 3.2 Oracle Initialization Configuration
**File**: `configs/oracle_init_experiment.json`

```json
{
  "data_config": {
    "num_instances": 3,
    "K": 25,
    "I": 6,
    "J": 5,
    "C": 5,
    "D": 32,
    "sigma_annotator": 0.3,
    "sigma_measurement": 0.1,
    "alpha_dirichlet": 2.0,
    "temperature": 0.5
  },
  "domain_config": {
    "chains": 4,
    "iter_warmup": 200,    // Much reduced warmup since starting near optimum
    "iter_sampling": 2000,
    "use_oracle_checkpoint": true,
    "oracle_noise_scale": 0.01,  // Small noise for initialization
    "test_masking_rate": 0.5
  },
  "experiment_name": "oracle_init_experiment"
}
```

#### 3.3 Analysis Metrics
- **Convergence Speed**: Compare warmup iterations needed
- **Posterior Quality**: Measure posterior concentration
- **Parameter Recovery**: Accuracy of parameter estimation
- **Chain Mixing**: Analyze chain convergence and mixing

### Implementation Steps

1. **Add Full Oracle Initialization**: Implement checkpoint + all ground truth parameters
2. **Generate Data**: Use partial runner to create oracle init experiment data
3. **Run Baseline**: Run domain model and save checkpoint
4. **Run Oracle**: Load checkpoint + all ground truth parameters for initialization
5. **Analysis**: Measure convergence and recovery improvements

---

## Implementation Timeline

### Phase 1: Infrastructure (Week 1)
- [ ] Create `GroundTruthExtractor` utility class
- [ ] Add oracle configuration support to `ExperimentConfig`
- [ ] Enhance `DomainModelTrainer` with oracle capabilities
- [ ] Create analysis utilities for parameter recovery

### Phase 2: Experiment 1 - Overfitting (Week 1-2)
- [ ] Create tiny instance configuration
- [ ] Generate tiny instance data using partial runner
- [ ] Run overfitting experiments with different sample counts
- [ ] Implement parameter counting analysis
- [ ] Create overfitting analysis script

### Phase 3: Experiment 2 - Oracle K (Week 2)
- [ ] Implement ground truth embedding extraction
- [ ] Modify data preparation for oracle embeddings
- [ ] Generate oracle K data using partial runner
- [ ] Run oracle vs non-oracle comparisons using partial runner
- [ ] Analyze embedding recovery performance

### Phase 4: Experiment 3 - Oracle Init (Week 2-3)
- [ ] Implement oracle parameter initialization
- [ ] Add oracle initialization to training pipeline
- [ ] Generate oracle init data using partial runner
- [ ] Run oracle initialization experiments using partial runner
- [ ] Analyze convergence and recovery improvements

### Phase 5: Analysis & Documentation (Week 3)
- [ ] Create comprehensive analysis scripts
- [ ] Generate comparison plots and tables
- [ ] Document findings and insights
- [ ] Create summary report

---

## Expected Insights

### Experiment 1 (Overfitting)
- **Parameter-to-Observation Ratio**: Quantify when STAN overfits
- **Identifiability Issues**: Understand parameter identifiability problems
- **Convergence Problems**: Identify when MCMC fails to converge

### Experiment 2 (Oracle K)
- **Embedding Recovery Impact**: Measure effect of knowing true item embeddings
- **Parameter Recovery**: Quantify improvement with oracle embeddings
- **Convergence Speed**: Compare convergence with oracle vs learned embeddings

### Experiment 3 (Oracle Init)
- **Initialization Importance**: Measure impact of good initialization
- **Convergence Speed**: Quantify warmup reduction with oracle init
- **Posterior Quality**: Analyze posterior concentration improvements

---

## Running the Three STAN Experiments

The experiments can be run using the `partial_experiment_runner.py` script, which allows for isolated testing of individual components and rapid iteration.

### Experiment 1: Tiny Instance Overfitting Analysis

#### Step 1: Create Tiny Instance Configuration
First, create the configuration file `configs/tiny_instance_overfit.json`:

```json
{
  "data_config": {
    "num_instances": 1,
    "K": 5,
    "I": 2,
    "J": 2,
    "C": 3,
    "D": 16,
    "sigma_annotator": 0.1,
    "sigma_measurement": 0.05,
    "alpha_dirichlet": 1.0,
    "temperature": 0.3,
    "max_pairs_per_tied_group": 5,
    "min_group_size": 2,
    "max_group_size": 3
  },
  "domain_config": {
    "chains": 2,
    "iter_warmup": 200,
    "iter_sampling": 1000,
    "adapt_delta": 0.8,
    "max_treedepth": 10,
    "sample_counts": [100, 200, 500, 1000],
    "test_masking_rate": 0.5
  },
  "experiment_name": "tiny_instance_overfit",
  "enabled_strategies": ["Domain_Model"]
}
```

#### Step 2: Generate Tiny Instance Data
```bash
python partial_experiment_runner.py --config configs/tiny_instance_overfit.json --operation generate_data
```

#### Step 3: Run Domain Model with Different Sample Counts
```bash
# Low samples (potential overfitting)
python partial_experiment_runner.py \
  --config configs/tiny_instance_overfit.json \
  --operation evaluate_domain \
  --data_file generated_data/tiny_instance_overfit_*/train_instances.json \
  --chains 2 \
  --iter_warmup 200 \
  --iter_sampling 100 \
  --instance_idx 0 \
  --output_file overfit_100_samples.json

# Medium samples
python partial_experiment_runner.py \
  --config configs/tiny_instance_overfit.json \
  --operation evaluate_domain \
  --data_file generated_data/tiny_instance_overfit_*/train_instances.json \
  --chains 2 \
  --iter_warmup 200 \
  --iter_sampling 500 \
  --instance_idx 0 \
  --output_file overfit_500_samples.json

# High samples (better convergence)
python partial_experiment_runner.py \
  --config configs/tiny_instance_overfit.json \
  --operation evaluate_domain \
  --data_file generated_data/tiny_instance_overfit_*/train_instances.json \
  --chains 2 \
  --iter_warmup 200 \
  --iter_sampling 2000 \
  --instance_idx 0 \
  --output_file overfit_2000_samples.json
```

#### Step 4: Analyze Overfitting
```bash
python analyze_overfitting.py --results_dir experiment_results/tiny_instance_overfit
```

### Experiment 2: Oracle K (Correct Item Embeddings)

#### Step 1: Create Oracle K Configuration
Create `configs/oracle_k_experiment.json`:

```json
{
  "data_config": {
    "num_instances": 3,
    "K": 20,
    "I": 5,
    "J": 4,
    "C": 5,
    "D": 32,
    "sigma_annotator": 0.3,
    "sigma_measurement": 0.1,
    "alpha_dirichlet": 2.0,
    "temperature": 0.5
  },
  "domain_config": {
    "chains": 4,
    "iter_warmup": 1000,
    "iter_sampling": 2000,
    "use_oracle_embeddings": true,
    "test_masking_rate": 0.5
  },
  "experiment_name": "oracle_k_experiment"
}
```

#### Step 2: Generate Oracle K Data
```bash
python partial_experiment_runner.py --config configs/oracle_k_experiment.json --operation generate_data
```

#### Step 3: Run Oracle vs Non-Oracle Comparison
```bash
# Non-oracle baseline (save checkpoint)
python partial_experiment_runner.py \
  --config configs/oracle_k_experiment.json \
  --operation evaluate_domain \
  --data_file generated_data/oracle_k_experiment_*/train_instances.json \
  --chains 4 \
  --iter_warmup 1000 \
  --iter_sampling 2000 \
  --instance_idx 0 \
  --output_file oracle_k_baseline.json

# Oracle embeddings (load checkpoint + ground truth embeddings)
python partial_experiment_runner.py \
  --config configs/oracle_k_experiment.json \
  --operation evaluate_domain \
  --data_file generated_data/oracle_k_experiment_*/train_instances.json \
  --chains 4 \
  --iter_warmup 500 \
  --iter_sampling 2000 \
  --instance_idx 0 \
  --output_file oracle_k_oracle.json \
  --checkpoint_file oracle_k_baseline_checkpoint.json
```

#### Step 4: Analyze Oracle K Results
```bash
python analyze_oracle_k.py --results_dir experiment_results/oracle_k_experiment
```

### Experiment 3: Oracle Initialization

#### Step 1: Create Oracle Init Configuration
Create `configs/oracle_init_experiment.json`:

```json
{
  "data_config": {
    "num_instances": 3,
    "K": 25,
    "I": 6,
    "J": 5,
    "C": 5,
    "D": 32,
    "sigma_annotator": 0.3,
    "sigma_measurement": 0.1,
    "alpha_dirichlet": 2.0,
    "temperature": 0.5
  },
  "domain_config": {
    "chains": 4,
    "iter_warmup": 500,
    "iter_sampling": 2000,
    "use_oracle_initialization": true,
    "oracle_noise_scale": 0.01,
    "test_masking_rate": 0.5
  },
  "experiment_name": "oracle_init_experiment"
}
```

#### Step 2: Generate Oracle Init Data
```bash
python partial_experiment_runner.py --config configs/oracle_init_experiment.json --operation generate_data
```

#### Step 3: Run Oracle vs Random Initialization
```bash
# Random initialization baseline (save checkpoint)
python partial_experiment_runner.py \
  --config configs/oracle_init_experiment.json \
  --operation evaluate_domain \
  --data_file generated_data/oracle_init_experiment_*/train_instances.json \
  --chains 4 \
  --iter_warmup 1000 \
  --iter_sampling 2000 \
  --instance_idx 0 \
  --output_file oracle_init_random.json

# Oracle initialization (load checkpoint + all ground truth parameters)
python partial_experiment_runner.py \
  --config configs/oracle_init_experiment.json \
  --operation evaluate_domain \
  --data_file generated_data/oracle_init_experiment_*/train_instances.json \
  --chains 4 \
  --iter_warmup 200 \
  --iter_sampling 2000 \
  --instance_idx 0 \
  --output_file oracle_init_oracle.json \
  --checkpoint_file oracle_init_random_checkpoint.json
```

#### Step 4: Analyze Oracle Init Results
```bash
python analyze_oracle_init.py --results_dir experiment_results/oracle_init_experiment
```

### Quick Test Commands

#### Test Data Generation Only
```bash
python partial_experiment_runner.py --config configs/tiny_instance_overfit.json --operation generate_data
```

#### Test Single Strategy Evaluation
```bash
python partial_experiment_runner.py \
  --config configs/tiny_instance_overfit.json \
  --operation evaluate_strategy \
  --strategy Domain_Model \
  --data_file generated_data/tiny_instance_overfit_*/train_instances.json \
  --instance_idx 0
```

#### Test Domain Model with Custom Parameters
```bash
python partial_experiment_runner.py \
  --config configs/tiny_instance_overfit.json \
  --operation evaluate_domain \
  --data_file generated_data/tiny_instance_overfit_*/train_instances.json \
  --chains 1 \
  --iter_warmup 50 \
  --iter_sampling 100 \
  --instance_idx 0
```

### Analysis Scripts
```bash
python analyze_overfitting.py --results_dir experiment_results/tiny_instance_overfit
python analyze_oracle_k.py --results_dir experiment_results/oracle_k_experiment
python analyze_oracle_init.py --results_dir experiment_results/oracle_init_experiment
```

### Benefits of Partial Runner Approach

1. **Rapid Iteration**: Test individual components without full pipeline
2. **Debugging**: Isolate issues to specific operations
3. **Resource Efficiency**: Run only what's needed
4. **Flexibility**: Customize parameters per experiment
5. **Reusability**: Reuse generated data across multiple experiments
6. **Comparison**: Easy A/B testing of different configurations

---

## Technical Notes

### Ground Truth Access
The STAN model generates data with known parameters stored in:
- `iclr_complete_ground_truth.json`: Contains true embeddings, preferences, thresholds
- `iclr_complete_stats.json`: Contains data generation statistics

### Configuration Extensions
New configuration fields needed in `DomainModelConfig`:
- `use_oracle_checkpoint`: Boolean flag for oracle checkpoint initialization
- `oracle_noise_scale`: Noise level for oracle initialization
- `test_masking_rate`: Masking rate for test data evaluation

### Checkpoint Approach Benefits
1. **Minimal Code Changes**: No STAN model modifications needed
2. **Elegant Implementation**: Uses existing cmdstanpy checkpoint functionality
3. **Flexible Oracle Control**: Can selectively replace parameters (embeddings only vs all parameters)
4. **Noise Control**: Add controlled noise to ground truth for MCMC initialization
5. **Reusability**: Same checkpoint can be used for different oracle experiments

### Analysis Tools
Required analysis utilities:
- Parameter counting and ratio computation
- Ground truth parameter extraction
- Convergence diagnostics
- Parameter recovery metrics
- Visualization tools for comparisons

This implementation plan provides a comprehensive framework for understanding STAN model behavior under different conditions and testing the impact of oracle knowledge on model performance.
