# Active Learner Framework

A comprehensive framework for active learning in multi-dimensional annotation tasks, featuring the AnnotationArena system for intelligent annotation selection and model training.

## Overview

This framework implements active learning strategies for annotation tasks where human and LLM evaluations are needed across multiple dimensions. The core AnnotationArena system manages variables, observations, and predictions while minimizing annotation costs through intelligent selection strategies.

## Key Features

- **AnnotationArena System**: Manages annotation variables and tracks observations with proper training queue management
- **Multiple Selection Strategies**: Random, gradient-based, entropy-based, and combined variable-level selection
- **Flexible Training Options**: Basic, random masking, and dynamic masking training modes
- **Configuration Management**: Centralized config system with automatic path management
- **Comprehensive Logging**: File and console logging with optional Wandb integration
- **Evaluation Framework**: Structured evaluation with metrics tracking and model comparison

## Installation

```bash
# Clone repository
git clone <repository-url>
cd active-learner

# Install dependencies
pip install torch numpy scipy pandas matplotlib tqdm
pip install sentence-transformers sklearn
pip install wandb  # Optional for experiment tracking
```

## Quick Start

### Basic Usage

```bash
# Run gradient-based active learning with VOI feature selection
python src/activeLearner.py \
    --experiment gradient_voi \
    --dataset hanna \
    --cycles 10 \
    --examples_per_cycle 50 \
    --features_per_example 5

# Run with Wandb logging
python src/activeLearner.py \
    --experiment gradient_voi \
    --dataset hanna \
    --use_wandb \
    --wandb_project my-active-learning \
    --experiment_name test_run
```

### Available Experiments

- `random_all`: Random example selection with all features
- `gradient_all`: Gradient-based example selection with all features  
- `entropy_all`: Entropy-based example selection with all features
- `random_5`: Random example and feature selection
- `gradient_voi`: Gradient examples with VOI feature selection
- `gradient_voi_q0_human`: Gradient + VOI targeting question 0
- `gradient_voi_all_questions`: Gradient + VOI across all questions
- `variable_gradient_comparison`: Combined variable-level gradient selection
- `comparison`: Runs multiple strategies for comparison

## Configuration

The framework uses a centralized configuration system supporting different environments:

```python
# Local development
python src/activeLearner.py --runner local

# Prabhav's cluster environment  
python src/activeLearner.py --runner prabhav
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset to use (hanna, llm_rubric) | hanna |
| `--cycles` | Number of active learning cycles | 10 |
| `--examples_per_cycle` | Examples to select per cycle | 50 |
| `--features_per_example` | Features to select per example | 5 |
| `--train_option` | Training mode (basic, random_masking, dynamic_masking) | dynamic_masking |
| `--use_embedding` | Use text embeddings | True |
| `--active_set_size` | K-centers subset size | 100 |
| `--validation_set_size` | Fixed validation set size | 50 |

## Project Structure

```
src/
├── activeLearner.py      # Main experiment runner
├── annotationArena.py    # Core AnnotationArena class
├── config.py             # Configuration management
├── imputer.py           # Imputer model with training queue system
├── selection.py         # Selection strategy implementations
├── utils.py             # Data management and utilities
└── eval.py              # Evaluation framework

input/
├── data/                # Generated data splits
└── fixed/               # Fixed reference data

output/
├── results/             # Experiment results
├── models/              # Saved model checkpoints
├── plots/               # Generated visualizations
└── logs/                # Experiment logs
```

## Core Components

### AnnotationArena

The central class managing annotation variables and model training:

```python
from annotationArena import AnnotationArena

# Initialize arena
arena = AnnotationArena(model, device)
arena.set_dataset(dataset)

# Register variables for an example
arena.register_example(example_idx, add_all_positions=False)

# Observe a position and update training
arena.observe_position(example_idx, position)
arena.predict(f"example_{example_idx}_position_{position}", train=True)

# Train the model
arena.train(training_type='dynamic_masking', epochs=5)
```

### Selection Strategies

Multiple strategies for intelligent annotation selection:

```python
from selection import SelectionFactory

# Example selection strategies
example_selector = SelectionFactory.create_example_strategy("gradient", model, device)
selected_examples, scores = example_selector.select_examples(dataset, num_to_select=50)

# Feature selection strategies  
feature_selector = SelectionFactory.create_feature_strategy("voi", model, device)
selected_features = feature_selector.select_features(example_idx, dataset, num_to_select=5)
```

### Configuration System

Centralized configuration with automatic path management:

```python
from config import Config, ModelConfig

# Initialize configuration
config = Config("local")  # or "prabhav"

# Get model configuration for dataset
model_config = ModelConfig.get_config("hanna")

# Access paths
data_paths = config.get_data_paths()
exp_paths = config.get_experiment_paths("my_experiment")
```

## Training Modes

The framework supports three training modes:

1. **Basic Training**: Standard supervised training on observed annotations
2. **Random Masking**: Random masking patterns for robust learning
3. **Dynamic Masking**: Adaptive masking based on observation patterns

```bash
# Use dynamic masking (recommended)
python src/activeLearner.py --train_option dynamic_masking

# Configure masking parameters
python src/activeLearner.py \
    --train_option dynamic_masking \
    --num_patterns_per_example 5 \
    --visible_ratio 0.6
```

## Evaluation and Metrics

The framework includes comprehensive evaluation capabilities:

```python
from eval import ModelEvaluator, evaluate_training_progress

# Initialize evaluator
evaluator = ModelEvaluator(config, use_wandb=True)

# Evaluate model on datasets
results = evaluator.evaluate_model(model, test_dataset, "test")

# Track training progress across cycles
datasets = {'train': train_data, 'val': val_data, 'test': test_data}
cycle_results = evaluator.evaluate_active_learning_cycle(model, datasets, cycle_num)
```

## Logging and Monitoring

Comprehensive logging with optional Wandb integration:

```bash
# Configure logging level
python src/activeLearner.py --log_level INFO

# Enable Wandb tracking
python src/activeLearner.py \
    --use_wandb \
    --wandb_project active-learning-project \
    --wandb_entity your-username
```

Logs are automatically saved to timestamped files in the `logs/` directory with full experiment traceability.

## Datasets

The framework supports two main datasets:

- **HANNA**: Human-annotated narratives with 6 evaluation dimensions
- **LLM-Rubric**: Multi-dimensional dialogue evaluation with 9 dimensions

Data preparation is handled automatically with proper train/validation/test splits and active pool management.

## Advanced Usage

### Custom Selection Strategies

Implement custom selection strategies by extending the base classes:

```python
from selection import ExampleSelectionStrategy

class CustomSelectionStrategy(ExampleSelectionStrategy):
    def select_examples(self, dataset, num_to_select, **kwargs):
        # Implement custom selection logic
        return selected_indices, scores
```

### Experiment Customization

The framework allows extensive customization of experimental parameters through the configuration system and command-line arguments. All hyperparameters can be adjusted without code modification.

## Performance Optimization

- **K-centers Algorithm**: Efficient diverse subset selection
- **Training Queue System**: Optimized training with current dataset state
