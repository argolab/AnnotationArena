"""
Comprehensive data saving utilities for complete experiment reproducibility.

Saves all raw predictions, models, ground truth data, and individual sample
results to enable plot recreation without re-running experiments.
"""

import pickle
import json
import torch
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def save_comprehensive_experiment_data(results: Dict[str, Any],
                                     raw_experiment_data: Dict[str, Any],
                                     config: Dict[str, Any],
                                     output_dir: Path) -> None:
    """
    Save everything from experiments for complete reproducibility.

    Args:
        results: Aggregated experimental results
        raw_experiment_data: Raw data from each experiment (graphs, models, predictions)
        config: Experiment configuration
        output_dir: Base output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / "comprehensive_data" / f"experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving comprehensive experiment data to {experiment_dir}")

    # 1. Save aggregated results (for quick access)
    save_aggregated_results(results, experiment_dir / "aggregated")

    # 2. Save raw experimental data (graphs, datasets, predictions)
    save_raw_experimental_data(raw_experiment_data, experiment_dir / "raw_data")

    # 3. Save trained models
    save_trained_models(raw_experiment_data, experiment_dir / "models")

    # 4. Save configuration and metadata
    save_configuration_and_metadata(config, experiment_dir, timestamp)

    # 5. Create summary manifest
    create_experiment_manifest(experiment_dir, results, config)

    logger.info(f"Comprehensive data saved successfully to {experiment_dir}")


def save_aggregated_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save aggregated results with bootstrap confidence intervals."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as pickle (full data)
    with open(output_dir / "aggregated_results.pkl", 'wb') as f:
        pickle.dump(results, f)

    # Save as JSON (summary only)
    json_results = {}
    for key, experiment_data in results.items():
        key_str = str(key)
        json_results[key_str] = {
            'total_time': experiment_data.get('total_time', 0.0),
            'n_graphs': experiment_data.get('n_graphs', 0),
            'policy_name': experiment_data.get('policy_name', ''),
            'imputer_size': experiment_data.get('imputer_size', ''),
            'n_steps': len(experiment_data.get('results', [])),
            'config': experiment_data.get('config', {})
        }

        # Add final performance metrics
        if experiment_data.get('results'):
            final_result = experiment_data['results'][-1]
            json_results[key_str]['final_metrics'] = {
                'neural_kl': final_result.get('neural_kl', float('inf')),
                'domain_kl': final_result.get('domain_kl', float('inf')),
                'neural_log_loss': final_result.get('neural_log_loss', float('inf')),
                'domain_log_loss': final_result.get('domain_log_loss', float('inf'))
            }

    with open(output_dir / "aggregated_summary.json", 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    logger.info(f"Aggregated results saved to {output_dir}")


def save_raw_experimental_data(raw_data: Dict[str, Any], output_dir: Path) -> None:
    """Save all raw experimental data for complete reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save graph structures and CPTs
    graphs_dir = output_dir / "graph_structures"
    graphs_dir.mkdir(exist_ok=True)

    for graph_id, graph_data in raw_data.get('graphs', {}).items():
        graph_file = graphs_dir / f"graph_{graph_id}.pkl"
        with open(graph_file, 'wb') as f:
            pickle.dump(graph_data, f)

    # Save datasets (training and test)
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    for dataset_id, dataset_data in raw_data.get('datasets', {}).items():
        dataset_file = datasets_dir / f"dataset_{dataset_id}.pkl"
        with open(dataset_file, 'wb') as f:
            pickle.dump(dataset_data, f)

    # Save raw predictions from all models at all budget steps
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    for pred_id, pred_data in raw_data.get('predictions', {}).items():
        pred_file = predictions_dir / f"predictions_{pred_id}.pkl"
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_data, f)

    # Save ground truth posteriors
    truth_dir = output_dir / "ground_truth"
    truth_dir.mkdir(exist_ok=True)

    for truth_id, truth_data in raw_data.get('ground_truth', {}).items():
        truth_file = truth_dir / f"ground_truth_{truth_id}.pkl"
        with open(truth_file, 'wb') as f:
            pickle.dump(truth_data, f)

    logger.info(f"Raw experimental data saved to {output_dir}")


def save_trained_models(raw_data: Dict[str, Any], output_dir: Path) -> None:
    """Save all trained models for potential reuse."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save neural models
    neural_models_dir = output_dir / "neural_models"
    neural_models_dir.mkdir(exist_ok=True)

    for model_id, model_data in raw_data.get('neural_models', {}).items():
        if 'state_dict' in model_data:
            model_file = neural_models_dir / f"neural_model_{model_id}.pt"
            torch.save(model_data['state_dict'], model_file)

        if 'metadata' in model_data:
            metadata_file = neural_models_dir / f"neural_model_{model_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(model_data['metadata'], f, indent=2, default=str)

    # Save EM models
    em_models_dir = output_dir / "em_models"
    em_models_dir.mkdir(exist_ok=True)

    for model_id, model_data in raw_data.get('em_models', {}).items():
        model_file = em_models_dir / f"em_model_{model_id}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)

    logger.info(f"Trained models saved to {output_dir}")


def save_configuration_and_metadata(config: Dict[str, Any], experiment_dir: Path,
                                   timestamp: str) -> None:
    """Save experiment configuration and runtime metadata."""

    # Save full configuration
    config_file = experiment_dir / "experiment_config.json"
    with open(config_file, 'w') as f:
        json_config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
        json.dump(json_config, f, indent=2, default=str)

    # Save runtime metadata
    metadata = {
        'timestamp': timestamp,
        'experiment_id': f"experiment_{timestamp}",
        'creation_date': datetime.now().isoformat(),
        'python_version': str(sys.version),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'data_structure_version': '1.0',
        'description': 'Comprehensive progressive imputation experiment data'
    }

    import sys
    metadata_file = experiment_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Configuration and metadata saved to {experiment_dir}")


def create_experiment_manifest(experiment_dir: Path, results: Dict[str, Any],
                              config: Dict[str, Any]) -> None:
    """Create a manifest describing all saved data."""

    manifest = {
        'experiment_summary': {
            'total_configurations': len(results),
            'node_sizes': config.get('node_sizes', []),
            'missing_rates': config.get('missing_rates', []),
            'imputer_sizes': config.get('imputer_sizes', []),
            'n_graphs': config.get('n_graphs', 1),
            'total_training_samples': config.get('max_samples', 0),
            'test_samples': config.get('test_samples', 0)
        },
        'data_structure': {
            'aggregated/': 'Bootstrap aggregated results with confidence intervals',
            'raw_data/graph_structures/': 'Bayesian network topologies and CPT parameters',
            'raw_data/datasets/': 'Training and test datasets with missing patterns',
            'raw_data/predictions/': 'Raw model predictions for every test sample',
            'raw_data/ground_truth/': 'True posterior distributions for evaluation',
            'models/neural_models/': 'Trained neural imputer model checkpoints',
            'models/em_models/': 'Learned EM model parameters',
            'experiment_config.json': 'Complete experiment configuration',
            'metadata.json': 'Runtime and versioning information'
        },
        'reproducibility_notes': {
            'plot_recreation': 'All plots can be recreated from aggregated/ data',
            'experiment_rerun': 'Complete experiments can be rerun from raw_data/',
            'model_analysis': 'Trained models available for post-hoc analysis',
            'bootstrap_data': 'Individual sample results stored for bootstrap CI computation'
        }
    }

    manifest_file = experiment_dir / "MANIFEST.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Also create a human-readable README
    readme_content = f"""# Progressive Imputation Experiment Data

Generated: {datetime.now().isoformat()}
Experiment ID: {experiment_dir.name}

## Summary
- **Configurations**: {len(results)} total experimental configurations
- **Node Sizes**: {config.get('node_sizes', [])}
- **Missing Rates**: {config.get('missing_rates', [])}
- **Imputer Sizes**: {config.get('imputer_sizes', [])}
- **Graphs per Configuration**: {config.get('n_graphs', 1)}

## Data Structure
```
{experiment_dir.name}/
├── aggregated/                 # Bootstrap aggregated results
│   ├── aggregated_results.pkl  # Full aggregated data with confidence intervals
│   └── aggregated_summary.json # Summary statistics
├── raw_data/                   # Complete raw experimental data
│   ├── graph_structures/       # BN topologies and CPT parameters
│   ├── datasets/              # Training/test data with missing patterns
│   ├── predictions/           # Raw model predictions for every sample
│   └── ground_truth/          # True posterior distributions
├── models/                     # Trained model checkpoints
│   ├── neural_models/         # Neural imputer state dicts
│   └── em_models/             # EM model parameters
├── experiment_config.json      # Complete configuration
├── metadata.json              # Runtime information
└── MANIFEST.json              # Data structure description
```

## Usage
- **Plot Recreation**: Load `aggregated/aggregated_results.pkl` and use `utils/paper_plots.py`
- **Model Analysis**: Load models from `models/` for post-hoc analysis
- **Bootstrap Analysis**: Individual sample results available in aggregated data
- **Complete Rerun**: Use raw_data/ to recreate any aspect of the experiment

## Notes
- All data uses pickle format for Python objects, JSON for metadata
- Bootstrap confidence intervals computed from individual sample results
- Both EM variants (1 restart and 5 restarts) included in all data
- Neural models saved at final budget level for each configuration
"""

    readme_file = experiment_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)

    logger.info(f"Experiment manifest and README created in {experiment_dir}")


class ExperimentDataCollector:
    """Collects raw experimental data during experiment execution."""

    def __init__(self):
        self.graphs = {}
        self.datasets = {}
        self.predictions = {}
        self.ground_truth = {}
        self.neural_models = {}
        self.em_models = {}

    def add_graph(self, graph_id: str, bn, adj_matrix, config: Dict[str, Any]) -> None:
        """Add graph structure data."""
        self.graphs[graph_id] = {
            'bayesian_network': bn,
            'adjacency_matrix': adj_matrix,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }

    def add_dataset(self, dataset_id: str, training_data: List, test_data: List,
                   config: Dict[str, Any]) -> None:
        """Add dataset information."""
        self.datasets[dataset_id] = {
            'training_data': training_data,
            'test_data': test_data,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }

    def add_predictions(self, pred_id: str, neural_preds: Dict[str, Any],
                       em_preds: Dict[str, Any], budget: int) -> None:
        """Add raw prediction data."""
        self.predictions[pred_id] = {
            'neural_predictions': neural_preds,
            'em_predictions': em_preds,
            'budget': budget,
            'timestamp': datetime.now().isoformat()
        }

    def add_ground_truth(self, truth_id: str, true_posteriors: Dict[str, Any],
                        config: Dict[str, Any]) -> None:
        """Add ground truth posterior data."""
        self.ground_truth[truth_id] = {
            'true_posteriors': true_posteriors,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }

    def add_neural_model(self, model_id: str, model, metadata: Dict[str, Any]) -> None:
        """Add trained neural model."""
        self.neural_models[model_id] = {
            'state_dict': model.state_dict() if model else None,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }

    def add_em_model(self, model_id: str, em_model, metadata: Dict[str, Any]) -> None:
        """Add trained EM model."""
        self.em_models[model_id] = {
            'em_model': em_model,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }

    def get_all_data(self) -> Dict[str, Any]:
        """Get all collected data."""
        return {
            'graphs': self.graphs,
            'datasets': self.datasets,
            'predictions': self.predictions,
            'ground_truth': self.ground_truth,
            'neural_models': self.neural_models,
            'em_models': self.em_models
        }