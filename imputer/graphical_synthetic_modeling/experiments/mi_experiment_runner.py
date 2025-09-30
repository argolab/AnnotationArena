"""
Mechanistic Interpretability Experiment Runner for MARFORMER.

Simplified experiment runner focused on layer-wise analysis without EM baseline.
Saves models and detailed layer-wise KL divergence data for analysis.
"""

import logging
import time
import pickle
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from data.graph_generator import generate_experiment_graph
from data.sample_generator import generate_sample_pool, generate_test_dataset
from imputer.architecture import create_model, DEVICE, SampleTuple, compute_max_cpt_size
from imputer.training_eval import (
    train_model, evaluate_model_with_layer_analysis,
    ImputationDataset, collate_batch
)
from experiments.policies import BaseObservationPolicy

logger = logging.getLogger(__name__)


class MIProgressiveExperiment:
    """
    Mechanistic interpretability experiment for analyzing layer-wise representations.

    Focuses on neural model training and layer-wise analysis without EM baseline.
    """

    def __init__(self, config: Dict[str, Any], imputer_sizes: Optional[List[str]] = None):
        """
        Initialize MI experiment.

        Args:
            config: Experiment configuration
            imputer_sizes: List of imputer sizes to test (default: ["Large"])
        """
        self.config = config
        self.n_nodes = config['n_nodes']
        self.target_parents = config['target_parents']
        self.missing_rate = config['missing_rate']
        self.max_samples = config['max_samples']
        self.test_samples = config['test_samples']
        self.seed = config['seed']
        self.cpt_generation = config.get('cpt_generation', 'logistic')
        self.logistic_std = config.get('logistic_std', 1.5)

        if imputer_sizes is None:
            imputer_sizes = ["Large"]
        self.imputer_sizes = imputer_sizes

        # Data containers
        self.bn = None
        self.adj_matrix = None
        self.sample_pool = None
        self.test_dataset = None
        self.neural_models = {}  # Store trained models

        logger.info(f"MI Experiment initialized: {self.n_nodes} nodes, imputer sizes={imputer_sizes}")

    def setup(self) -> None:
        """Generate graph structure and datasets."""
        logger.info("Setting up experiment data...")
        start_time = time.time()

        # Generate graph
        self.bn, self.adj_matrix = generate_experiment_graph(
            self.n_nodes, self.target_parents, self.seed, None,
            self.cpt_generation, self.logistic_std
        )

        # Generate sample pool
        self.sample_pool = generate_sample_pool(
            self.bn, self.adj_matrix, self.n_nodes, self.max_samples,
            self.missing_rate, self.seed
        )

        # Generate test dataset
        self.test_dataset = generate_test_dataset(
            self.bn, self.adj_matrix, self.n_nodes, self.test_samples,
            self.missing_rate, self.seed + 1000
        )

        setup_time = time.time() - start_time
        logger.info(f"Setup completed in {setup_time:.2f}s")
        logger.info(f"Sample pool: {len(self.sample_pool)} samples")
        logger.info(f"Test dataset: {len(self.test_dataset)} samples")

    def _create_neural_models(self) -> Dict[str, Any]:
        """Create neural imputer models."""
        neural_models = {}

        if not self.sample_pool:
            raise ValueError("Sample pool must be generated first")

        sample_inputs, sample_structure, _, _, _, _ = self.sample_pool[0]
        input_dim = sample_inputs.shape[1]
        structure_dim = sample_structure.shape[1]
        max_cpt_size = compute_max_cpt_size(self.bn) if self.bn else 8

        for imputer_size in self.imputer_sizes:
            model = create_model(
                n_nodes=self.n_nodes,
                input_dim=input_dim,
                structure_dim=structure_dim,
                cpt_dim=max_cpt_size,
                model_size=imputer_size
            )
            neural_models[imputer_size] = model

        logger.info(f"Created {len(neural_models)} neural models with cpt_dim={max_cpt_size}")
        return neural_models

    def run_policy_experiment(self, policy: BaseObservationPolicy) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run MI experiment with given policy and all imputer variants.

        Args:
            policy: Observation policy (typically MI_Analysis policy)

        Returns:
            Dict mapping imputer sizes to results lists
        """
        logger.info(f"Running MI experiment with policy: {policy}")

        if self.sample_pool is None or self.test_dataset is None:
            raise ValueError("Must call setup() first")

        # Create neural models
        neural_models = self._create_neural_models()
        results_by_size = {size: [] for size in self.imputer_sizes}

        # Progressive observation loop (for MI: specific budgets)
        for budget, training_data in policy.observe_progressively(self.sample_pool):
            logger.info(f"=" * 80)
            logger.info(f"BUDGET: {budget} samples")
            logger.info(f"=" * 80)

            # Convert to PyTorch datasets
            train_dataset = ImputationDataset(training_data, self.bn)
            val_dataset = ImputationDataset(self.test_dataset[:50], self.bn)

            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

            # Train and evaluate each imputer size
            for imputer_size in self.imputer_sizes:
                logger.info(f"\n--- Training {imputer_size} imputer with {len(training_data)} samples ---")
                neural_start = time.time()

                # Create fresh model for this budget
                neural_model = self._create_neural_models()[imputer_size]

                # Train
                trained_model = train_model(
                    neural_model, train_loader, val_loader,
                    epochs=100, lr=1e-4, patience=45
                )

                # Evaluate with layer-wise analysis
                logger.info(f"Evaluating {imputer_size} with layer-wise analysis...")
                neural_results = evaluate_model_with_layer_analysis(
                    trained_model, self.test_dataset, self.bn,
                    self.n_nodes, n_states=2
                )

                neural_time = time.time() - neural_start

                # Store results
                step_result = {
                    'budget': budget,
                    'n_training_samples': len(training_data),
                    'imputer_size': imputer_size,
                    'neural_time': neural_time,

                    # Standard metrics
                    'neural_kl': neural_results.get('mean_kl', float('inf')),
                    'neural_kl_std': neural_results.get('std_kl', 0.0),
                    'neural_n_evaluations': neural_results.get('n_evaluations', 0),
                    'neural_failed_rate': neural_results.get('failed_rate', 0.0),

                    # Layer-wise analysis (KEY MI DATA)
                    'n_layers': neural_results.get('n_layers', 0),
                    'layer_kl_means': neural_results.get('layer_kl_means'),  # np.ndarray
                    'layer_kl_stds': neural_results.get('layer_kl_stds'),
                    'layer_kl_counts': neural_results.get('layer_kl_counts'),
                    'layer_descriptions': neural_results.get('layer_descriptions', []),
                    'layer_kl_raw_data': neural_results.get('layer_kl_raw_data', {}),  # Sparse dict
                    'sample_metadata': neural_results.get('sample_metadata', [])
                }

                results_by_size[imputer_size].append(step_result)

                logger.info(f"  {imputer_size} Final KL: {step_result['neural_kl']:.4f}")
                logger.info(f"  {imputer_size} Layer KL means: {step_result['layer_kl_means']}")
                logger.info(f"  {imputer_size} Training time: {neural_time:.1f}s")

                # Store model for potential saving
                self.neural_models[f"{imputer_size}_budget_{budget}"] = trained_model

        return results_by_size

    def run_multi_policy_experiment(self, policies: List[BaseObservationPolicy]) -> Dict[str, Dict[str, Any]]:
        """
        Run experiment with multiple policies.

        Args:
            policies: List of policies (typically just one MI policy)

        Returns:
            Dict mapping "{policy_name}_{imputer_size}" to experiment results
        """
        logger.info(f"Running multi-policy MI experiment with {len(policies)} policies")

        all_results = {}

        for policy in policies:
            policy_start = time.time()

            results_by_size = self.run_policy_experiment(policy)
            policy_time = time.time() - policy_start

            # Create entries for each (policy, imputer_size) combination
            for imputer_size, results in results_by_size.items():
                combined_key = f"{policy.name}_{imputer_size}"
                all_results[combined_key] = {
                    'results': results,
                    'total_time': policy_time,
                    'config': {**self.config.copy(), 'imputer_size': imputer_size},
                    'policy_name': policy.name,
                    'imputer_size': imputer_size,
                    'policy_info': policy.get_selection_info(self.sample_pool) if hasattr(policy, 'get_selection_info') else {}
                }

            logger.info(f"Policy {policy.name} completed in {policy_time:.1f}s")

        return all_results


def run_mi_experiment_suite(
    node_sizes: List[int],
    target_parents: float,
    missing_rates: List[float],
    test_samples: int,
    policies: List[BaseObservationPolicy],
    imputer_sizes: List[str],
    n_graphs: int,
    cpt_generation: str,
    logistic_std: float,
    output_dir: Path,
    seed: int = 42
) -> Dict[Any, Dict[str, Any]]:
    """
    Run MI experiment suite across graph sizes.

    Args:
        node_sizes: List of graph sizes
        target_parents: Target parents per node
        missing_rates: List of missing rates
        test_samples: Number of test samples
        policies: List of observation policies (typically one MI policy)
        imputer_sizes: List of imputer sizes
        n_graphs: Number of graph instances
        cpt_generation: CPT generation method
        logistic_std: Std for logistic CPTs
        output_dir: Output directory
        seed: Random seed

    Returns:
        Dict mapping keys to experiment results
    """
    logger.info("=" * 80)
    logger.info("MECHANISTIC INTERPRETABILITY EXPERIMENT SUITE")
    logger.info("=" * 80)
    logger.info(f"Node sizes: {node_sizes}")
    logger.info(f"Imputer sizes: {imputer_sizes}")
    logger.info(f"Missing rates: {missing_rates}")
    logger.info(f"N graphs: {n_graphs}")
    logger.info(f"CPT generation: {cpt_generation} (std={logistic_std})")

    all_results = {}

    # Get max budget from policy
    max_budget = max(policies[0].get_budget_sequence()) if policies else 1000

    for n_nodes in node_sizes:
        for missing_rate in missing_rates:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"GRAPH: {n_nodes} nodes, missing_rate={missing_rate}")
            logger.info(f"{'=' * 60}")

            # Run single graph instance (n_graphs typically 1 for MI)
            for graph_idx in range(n_graphs):
                logger.info(f"\nGraph instance {graph_idx + 1}/{n_graphs}")

                config = {
                    'n_nodes': n_nodes,
                    'target_parents': target_parents,
                    'missing_rate': missing_rate,
                    'max_samples': max_budget,
                    'test_samples': test_samples,
                    'seed': seed + n_nodes * 100 + graph_idx * 10,
                    'cpt_generation': cpt_generation,
                    'logistic_std': logistic_std
                }

                # Create and run experiment
                experiment = MIProgressiveExperiment(config, imputer_sizes)
                experiment.setup()
                results = experiment.run_multi_policy_experiment(policies)

                # Save experiment artifacts
                _save_mi_experiment_artifacts(
                    experiment=experiment,
                    results=results,
                    output_dir=output_dir,
                    n_nodes=n_nodes,
                    missing_rate=missing_rate,
                    graph_idx=graph_idx
                )

                # Store results
                for policy_imputer_key, result_data in results.items():
                    composite_key = (n_nodes, policy_imputer_key, missing_rate, graph_idx)
                    all_results[composite_key] = result_data

    logger.info(f"\nMI experiment suite completed: {len(all_results)} configurations")
    return all_results


def _save_mi_experiment_artifacts(
    experiment: MIProgressiveExperiment,
    results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    n_nodes: int,
    missing_rate: float,
    graph_idx: int
) -> None:
    """Save all MI experiment artifacts."""
    # Create structured output directory
    exp_dir = output_dir / f"nodes_{n_nodes}_missing_{missing_rate:.1f}_graph_{graph_idx}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving MI experiment artifacts to {exp_dir}")

    # Save BN structure
    bn_dir = exp_dir / "bn_structure"
    bn_dir.mkdir(exist_ok=True)

    with open(bn_dir / "bn_structure.pkl", 'wb') as f:
        pickle.dump(experiment.bn, f)
    np.save(bn_dir / "adjacency_matrix.npy", experiment.adj_matrix)
    with open(bn_dir / "graph_metadata.json", 'w') as f:
        json.dump({
            'n_nodes': experiment.n_nodes,
            'target_parents': experiment.target_parents,
            'cpt_generation': experiment.cpt_generation,
            'logistic_std': experiment.logistic_std
        }, f, indent=2)

    # Save test dataset
    test_dir = exp_dir / "test_dataset"
    test_dir.mkdir(exist_ok=True)
    with open(test_dir / "test_samples.pkl", 'wb') as f:
        pickle.dump(experiment.test_dataset, f)
    with open(test_dir / "test_metadata.json", 'w') as f:
        json.dump({
            'n_samples': len(experiment.test_dataset),
            'missing_rate': experiment.missing_rate
        }, f, indent=2)

    # Save results for each budget and imputer size
    for policy_imputer_key, result_data in results.items():
        for step_result in result_data['results']:
            budget = step_result['budget']
            imputer_size = step_result['imputer_size']

            # Create budget directory
            budget_dir = exp_dir / f"budget_{budget:04d}" / imputer_size.lower()
            budget_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            model_key = f"{imputer_size}_budget_{budget}"
            if model_key in experiment.neural_models:
                import torch
                model = experiment.neural_models[model_key]
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': {
                        'n_nodes': model.n_nodes,
                        'n_states': model.n_states,
                        'cpt_dim': model.cpt_dim,
                        'imputer_size': imputer_size
                    }
                }, budget_dir / "model.pt")

            # Save layer-wise analysis results
            analysis_data = {
                'budget': budget,
                'imputer_size': imputer_size,
                'n_training_samples': step_result['n_training_samples'],
                'training_time': step_result['neural_time'],

                # Standard metrics
                'final_kl': step_result['neural_kl'],
                'final_kl_std': step_result['neural_kl_std'],

                # Layer-wise metrics
                'n_layers': step_result['n_layers'],
                'layer_descriptions': step_result['layer_descriptions'],
                'layer_kl_means': step_result['layer_kl_means'].tolist() if step_result['layer_kl_means'] is not None else [],
                'layer_kl_stds': step_result['layer_kl_stds'].tolist() if step_result['layer_kl_stds'] is not None else [],
                'layer_kl_counts': step_result['layer_kl_counts'].tolist() if step_result['layer_kl_counts'] is not None else [],
            }

            with open(budget_dir / "layer_analysis.json", 'w') as f:
                json.dump(analysis_data, f, indent=2)

            # Save raw layer KL data (sparse dict)
            with open(budget_dir / "layer_kl_raw_data.pkl", 'wb') as f:
                pickle.dump(step_result['layer_kl_raw_data'], f)

            # Save sample metadata
            with open(budget_dir / "sample_metadata.pkl", 'wb') as f:
                pickle.dump(step_result['sample_metadata'], f)

    # Save experiment config
    with open(exp_dir / "experiment_config.json", 'w') as f:
        json.dump(experiment.config, f, indent=2, default=str)

    logger.info(f"Saved MI experiment artifacts: {exp_dir}")
