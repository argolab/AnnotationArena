# Imputer Experiment System

This system supports both **single instance** and **multi-instance** imputation experiments with a unified configuration approach.

## Quick Start

### Single Instance Experiment (Setup 1)
Train and test on the same instance with masking:

```bash
# Using the new system
python experiment_runner.py --config configs/single_instance.json

# Using legacy system (backwards compatible)
python run_experiment_imputer.py --config_path configs/single_instance.json
```

### Multi-Instance Generalization (Setup 2) 
Train on multiple instances, test generalization on held-out instances:

```bash
python experiment_runner.py --config configs/multi_instance_demo.json
```

## Configuration System

All experiments are controlled by JSON configuration files with three main components:

### 1. Instance Configuration
```json
{
  "K": 30,           // number of items
  "I": 10,           // number of attributes  
  "J": 5,            // number of annotators
  "D": 64,           // embedding dimension
  "C": 5,            // rating categories
  "sigma_annotator": 0.3,        // annotator variance
  "sigma_measurement": 0.1,      // measurement noise
  "alpha_dirichlet": 2.0,        // rating threshold concentration
  "temperature": 0.5,            // ranking temperature
  // ... other data generation parameters
}
```

### 2. Model Configuration
```json
{
  "encoder_layers": 4,
  "attention_heads": 8,
  "embedding_dim": 64,
  "dropout": 0.1,
  "embedding_type": "pairwise"
}
```

### 3. Training Configuration
```json
{
  "epochs": 50,
  "learning_rate": 0.0001,
  "embedding_anchor_reg": 0.0,
  "masking_rate": 0.3,           // fraction to mask for imputation
  "evaluation_frequency": 2       // evaluate every N epochs
}
```

## Experiment Types

### Single Instance (`experiment_type: "single_instance"`)
- **Goal**: Show imputer can recover missing annotations within a single instance
- **Setup**: Train on masked train split, evaluate on masked test split
- **Plots**: 
  - Training loss curves (Total, Rating, Ranking)
  - Test loss over epochs
- **Table**: Test accuracy breakdown (masked/observed × rating/ranking)

### Multi-Instance (`experiment_type: "multi_instance"`)
- **Goal**: Demonstrate generalization to new instances (new items, annotators, attributes)
- **Setup**: Sequential training across multiple instances, evaluate on held-out instances
- **Plots**:
  - Segmented training curves across instances with boundaries
  - Test performance on held-out instances throughout training
- **Table**: Average metrics over test instances

## Directory Structure

The system automatically organizes outputs by experiment type:

```
OUTPUT/IMPUTER/
├── single_instance/           # Single instance experiments
│   ├── plots/
│   ├── models/
│   ├── results/
│   └── config.json
├── multi_instance_3train_2test/  # Multi-instance experiments
│   ├── plots/
│   ├── models/
│   ├── results/
│   └── config.json
└── legacy/                    # Legacy experiments

generated_data/
├── single_instance/
│   └── instance_0/
│       ├── iclr_complete_train.json
│       └── iclr_complete_test.json
└── multi_instance_5instances/
    ├── instance_0/
    ├── instance_1/
    └── ...
```

## Key Features

### 1. Configuration-Driven Data Generation
- All data generation parameters come from config files
- No hardcoded values in any script
- Automatic data directory organization by experiment type

### 2. Fixed Plotting Issues
- Removed redundant training plots (left plot contains all information)
- Single comprehensive training plot with Total/Rating/Ranking losses
- Clean test loss plots with proper styling

### 3. Multi-Instance Training Pipeline
- Sequential training across instances with shared model weights
- Instance boundary visualization in segmented plots
- Evaluation on held-out test instances throughout training

### 4. Backwards Compatibility
- Legacy `run_experiment_imputer.py` still works
- Automatic conversion from command-line args to new config system
- Existing workflows preserved

## Sample Configurations

### Different Instance Parameters
You can vary parameters across instances in multi-instance experiments:

```json
{
  "experiment_type": "multi_instance",
  "instances": [
    {
      "K": 30, "I": 10, "J": 5, 
      "sigma_annotator": 0.3,
      // ... other params
    },
    {
      "K": 30, "I": 10, "J": 5,
      "sigma_annotator": 0.4,  // Different variance
      // ... other params  
    }
  ],
  "train_instance_indices": [0],
  "test_instance_indices": [1]
}
```

### Large Multi-Instance Setup
```json
{
  "experiment_type": "multi_instance",
  "instances": [/* 10 instances with same structure, different random data */],
  "train_instance_indices": [0, 1, 2, 3, 4, 5, 6, 7],  // 8 for training
  "test_instance_indices": [8, 9]                       // 2 for testing
}
```

## Usage Examples

### 1. Generate Data Only
```bash
python experiment_runner.py --config configs/multi_instance_demo.json --generate_only
```

### 2. Run with Custom Config
Create your own config file:

```bash
# Create config programmatically
from config import ExperimentConfig, InstanceConfig

# Single instance
config = ExperimentConfig.create_single_instance()
config.save_to_file("my_single_config.json")

# Multi-instance  
instances = [InstanceConfig() for _ in range(10)]
config = ExperimentConfig.create_multi_instance(
    instances=instances,
    train_instance_indices=list(range(8)),
    test_instance_indices=[8, 9]
)
config.save_to_file("my_multi_config.json")

# Run experiment
python experiment_runner.py --config my_multi_config.json
```

### 3. Legacy Mode (Backwards Compatible)
```bash
python run_experiment_imputer.py \
  --epochs 50 \
  --learning_rate 1e-4 \
  --masking_rate 0.3 \
  --save_plots
```

## Results and Analysis

### Training Plots
- **Single Instance**: Simple training curves showing convergence
- **Multi-Instance**: Segmented plots showing learning across instances with boundaries

### Test Performance
- **Single Instance**: Performance on masked test set 
- **Multi-Instance**: Performance on completely held-out instances (demonstrates generalization)

### Tables
Results tables automatically generated with breakdown by:
- Variable type (rating vs ranking)
- Observation status (masked vs observed vs all)
- Instance averages (for multi-instance)

## Notes

- **Instance Independence**: Even with identical (I,J,K,C), different instances generate different data due to random sampling
- **Memory Efficiency**: Data generated on-demand and cached in structured directories  
- **Generalization Test**: Multi-instance setup tests true compositional generalization to new annotators/items/attributes
- **Reproducible**: All random seeds and configurations saved for reproducibility

This system provides a comprehensive framework for systematic evaluation of the imputer's capabilities across both within-instance imputation and cross-instance generalization.