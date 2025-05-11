# AnnotationArena

A framework for managing and optimizing the annotation process through active learning, designed to reduce annotation costs while maintaining model performance.

## Overview

AnnotationArena dynamically selects which examples and features to annotate using various active learning strategies. The framework supports both example selection (which examples to annotate) and feature selection (which features within an example to annotate).

## Code Structure

- `annotationArena.py`: Core class for managing variables, observations, and predictions
- `imputer.py`: Model for predicting missing annotations
- `selection.py`: Selection strategies for active learning
- `activeLearner.py`: Experiment runner for different active learning strategies
- `utils.py`: Utility functions for data handling and metrics
- `visualizations.py`: Functions for generating plots and visualizations

## Selection Strategies

The framework includes several selection strategies:

**Example Selection:**
- `random`: Randomly selects examples
- `gradient`: Selects examples based on gradient alignment with validation set
- `entropy`: Selects examples with highest prediction uncertainty

**Feature Selection:**
- `random`: Randomly selects features within an example
- `sequential`: Selects features in sequential order
- `voi`: Value of Information based selection
- `fast_voi`: Efficient approximation of VOI
- `entropy`: Selects features with highest prediction uncertainty

## Running Experiments

To run experiments with the default settings:

```bash
python src/activeLearner.py
```

### Command-line Arguments

```bash
python src/activeLearner.py --experiment [experiment_name] --cycles 5 --examples_per_cycle 20 --features_per_example 5
```

- `--experiment`: Experiment to run (`all`, `random_all`, `gradient_all`, `entropy_all`, `random_5`, `gradient_voi`, etc.)
- `--cycles`: Number of active learning cycles
- `--examples_per_cycle`: Number of examples to select per cycle
- `--features_per_example`: Number of features to select per example
- `--loss_type`: Type of loss to use (`cross_entropy`, `l2`)
- `--resample_validation`: Flag to resample validation set during training
- `--run_until_exhausted`: Flag to run until annotation pool is exhausted

## Visualization

After running experiments, visualizations are automatically generated in the `outputs/results/plots` directory, including:

- Comparison of "observe all" strategies (random, gradient, entropy)
- Comparison of feature selection strategies
- VOI comparison plots
- Feature count plots

To manually create visualization plots:

```python
from visualizations import create_plots
create_plots()
```

## Example

Run all strategies and compare:

```bash
python src/activeLearner.py --experiment all --cycles 7 --examples_per_cycle 20 --features_per_example 5
```

Run only entropy-based selection:

```bash
python src/activeLearner.py --experiment entropy_all --cycles 5
```
