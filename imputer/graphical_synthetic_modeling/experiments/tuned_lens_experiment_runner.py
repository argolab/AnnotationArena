"""
Tuned Lens Experiment Runner for MARFORMER.

Trains tuned lens probes on calibration data and evaluates layer-wise
representations with learned transformations. Extends logit lens approach.
"""

import logging
import time
import pickle
import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path

from data.graph_generator import generate_experiment_graph
from data.sample_generator import generate_sample_pool, generate_test_dataset
from imputer.architecture import create_model, DEVICE, SampleTuple, compute_max_cpt_size
from imputer.training_eval import (
    train_model, ImputationDataset, collate_batch
)
from imputer.tuned_lens_probes import TunedLensGraphImputer
from imputer.tuned_lens_training import train_all_probes, evaluate_tuned_lens_model
from experiments.policies import BaseObservationPolicy

logger = logging.getLogger(__name__)


class TunedLensProgressiveExperiment:
    """
    Tuned lens experiment for analyzing layer-wise representations with learned probes.

    Workflow:
    1. Train base MARFORMER model (same as MI experiment)
    2. Create calibration dataset (separate from training/test)
    3. Train tuned lens probes on calibration data
    4. Evaluate with layer-wise analysis using tuned probes
    """

    def __init__(self, config: Dict[str, Any], imputer_sizes: Optional[List[str]] = None):
        """
        Initialize tuned lens experiment.

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
        self.calibration_samples = config.get('calibration_samples', 500)  # NEW: separate calibration set
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
        self.calibration_dataset = None  # NEW: for probe training
        self.test_dataset = None
        self.base_models = {}  # Store trained base models
        self.tuned_lens_models = {}  # Store tuned lens models

        logger.info(f"Tuned Lens Experiment initialized: {self.n_nodes} nodes, "
                   f"imputer sizes={imputer_sizes}, calibration_samples={self.calibration_samples}")

    def setup(self) -> None:
        """Generate graph structure and datasets."""
        logger.info("Setting up tuned lens experiment data...")
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

        # Generate calibration dataset (separate seed)
        self.calibration_dataset = generate_test_dataset(
            self.bn, self.adj_matrix, self.n_nodes, self.calibration_samples,
            self.missing_rate, self.seed + 500
        )

        # Generate test dataset (different seed from training and calibration)
        self.test_dataset = generate_test_dataset(
            self.bn, self.adj_matrix, self.n_nodes, self.test_samples,
            self.missing_rate, self.seed + 1000
        )

        setup_time = time.time() - start_time
        logger.info(f"Setup completed in {setup_time:.2f}s")
        logger.info(f"Sample pool: {len(self.sample_pool)} samples")
        logger.info(f"Calibration dataset: {len(self.calibration_dataset)} samples")
        logger.info(f"Test dataset: {len(self.test_dataset)} samples")

    def _create_base_models(self) -> Dict[str, Any]:
        """Create base neural imputer models (same as MI experiment)."""
        base_models = {}

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
            base_models[imputer_size] = model

        logger.info(f"Created {len(base_models)} base models with cpt_dim={max_cpt_size}")
        return base_models

    def run_policy_experiment(self, policy: BaseObservationPolicy) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run tuned lens experiment with given policy.

        Workflow per budget:
        1. Train base MARFORMER model
        2. Create TunedLensGraphImputer wrapper (frozen base + probes)
        3. Train probes on calibration data
        4. Evaluate with tuned lens on test data

        Args:
            policy: Observation policy (typically MI_Analysis policy)

        Returns:
            Dict mapping imputer sizes to results lists
        """
        logger.info(f"Running tuned lens experiment with policy: {policy}")

        if self.sample_pool is None or self.calibration_dataset is None or self.test_dataset is None:
            raise ValueError("Must call setup() first")

        results_by_size = {size: [] for size in self.imputer_sizes}

        # Progressive observation loop
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
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Processing {imputer_size} imputer with {len(training_data)} samples")
                logger.info(f"{'=' * 60}")

                # STEP 1: Train base MARFORMER model
                logger.info(f"Step 1/3: Training base {imputer_size} model...")
                base_start = time.time()

                base_model = self._create_base_models()[imputer_size]
                trained_base_model = train_model(
                    base_model, train_loader, val_loader,
                    epochs=100, lr=1e-4, patience=45
                )

                base_time = time.time() - base_start
                logger.info(f"Base model training completed in {base_time:.1f}s")

                # STEP 2: Create tuned lens wrapper with probes
                logger.info(f"Step 2/3: Creating tuned lens probes...")
                n_layers = len(trained_base_model.transformer.layers)
                tuned_lens_model = TunedLensGraphImputer(
                    base_model=trained_base_model,
                    n_layers=n_layers
                ).to(DEVICE)

                # STEP 3: Train probes on calibration data
                logger.info(f"Step 3/3: Training {n_layers + 2} probes on calibration data...")
                probe_start = time.time()

                # Create calibration data loaders
                calibration_train_dataset = ImputationDataset(self.calibration_dataset, self.bn)
                calibration_val_dataset = ImputationDataset(self.test_dataset[:50], self.bn)

                calibration_train_loader = DataLoader(
                    calibration_train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch
                )
                calibration_val_loader = DataLoader(
                    calibration_val_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch
                )

                # Train all probes independently
                trained_tuned_lens_model = train_all_probes(
                    tuned_lens_model=tuned_lens_model,
                    calibration_loader=calibration_train_loader,
                    val_loader=calibration_val_loader,
                    epochs_per_probe=50,
                    lr=1e-3,
                    patience=15
                )

                probe_time = time.time() - probe_start
                logger.info(f"Probe training completed in {probe_time:.1f}s")

                # STEP 4: Evaluate with tuned lens
                logger.info(f"Evaluating {imputer_size} with tuned lens...")
                eval_start = time.time()

                tuned_lens_results = evaluate_tuned_lens_model(
                    trained_tuned_lens_model, self.test_dataset, self.bn,
                    self.n_nodes, n_states=2
                )

                eval_time = time.time() - eval_start
                total_time = base_time + probe_time + eval_time

                # Store results
                step_result = {
                    'budget': budget,
                    'n_training_samples': len(training_data),
                    'n_calibration_samples': self.calibration_samples,
                    'imputer_size': imputer_size,

                    # Timing breakdown
                    'base_training_time': base_time,
                    'probe_training_time': probe_time,
                    'evaluation_time': eval_time,
                    'total_time': total_time,

                    # Standard metrics
                    'tuned_lens_kl': tuned_lens_results.get('mean_kl', float('inf')),
                    'tuned_lens_kl_std': tuned_lens_results.get('std_kl', 0.0),
                    'tuned_lens_n_evaluations': tuned_lens_results.get('n_evaluations', 0),
                    'tuned_lens_failed_rate': tuned_lens_results.get('failed_rate', 0.0),

                    # Layer-wise analysis (KEY TUNED LENS DATA)
                    'n_layers': tuned_lens_results.get('n_layers', 0),
                    'layer_kl_means': tuned_lens_results.get('layer_kl_means'),
                    'layer_kl_stds': tuned_lens_results.get('layer_kl_stds'),
                    'layer_kl_counts': tuned_lens_results.get('layer_kl_counts'),
                    'layer_descriptions': tuned_lens_results.get('layer_descriptions', []),
                    'layer_kl_raw_data': tuned_lens_results.get('layer_kl_raw_data', {}),
                    'sample_metadata': tuned_lens_results.get('sample_metadata', [])
                }

                results_by_size[imputer_size].append(step_result)

                logger.info(f"  {imputer_size} Final KL (tuned lens): {step_result['tuned_lens_kl']:.4f}")
                logger.info(f"  {imputer_size} Layer KL means: {step_result['layer_kl_means']}")
                logger.info(f"  {imputer_size} Total time: {total_time:.1f}s "
                           f"(base: {base_time:.1f}s, probes: {probe_time:.1f}s)")

                # Store models for potential saving
                self.base_models[f"{imputer_size}_budget_{budget}"] = trained_base_model
                self.tuned_lens_models[f"{imputer_size}_budget_{budget}"] = trained_tuned_lens_model

        return results_by_size

    def run_multi_policy_experiment(self, policies: List[BaseObservationPolicy]) -> Dict[str, Dict[str, Any]]:
        """
        Run tuned lens experiment with multiple policies.

        Args:
            policies: List of policies (typically just one MI policy)

        Returns:
            Dict mapping "{policy_name}_{imputer_size}" to experiment results
        """
        logger.info(f"Running multi-policy tuned lens experiment with {len(policies)} policies")

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


def run_tuned_lens_experiment_suite(
    node_sizes: List[int],
    target_parents: float,
    missing_rates: List[float],
    test_samples: int,
    calibration_samples: int,
    policies: List[BaseObservationPolicy],
    imputer_sizes: List[str],
    n_graphs: int,
    cpt_generation: str,
    logistic_std: float,
    output_dir: Path,
    seed: int = 42
) -> Dict[Any, Dict[str, Any]]:
    """
    Run tuned lens experiment suite across graph sizes.

    Args:
        node_sizes: List of graph sizes
        target_parents: Target parents per node
        missing_rates: List of missing rates
        test_samples: Number of test samples
        calibration_samples: Number of calibration samples for probe training
        policies: List of observation policies
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
    logger.info("TUNED LENS EXPERIMENT SUITE")
    logger.info("=" * 80)
    logger.info(f"Node sizes: {node_sizes}")
    logger.info(f"Imputer sizes: {imputer_sizes}")
    logger.info(f"Missing rates: {missing_rates}")
    logger.info(f"Calibration samples: {calibration_samples}")
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

            # Run graph instances
            for graph_idx in range(n_graphs):
                logger.info(f"\nGraph instance {graph_idx + 1}/{n_graphs}")

                config = {
                    'n_nodes': n_nodes,
                    'target_parents': target_parents,
                    'missing_rate': missing_rate,
                    'max_samples': max_budget,
                    'test_samples': test_samples,
                    'calibration_samples': calibration_samples,
                    'seed': seed + n_nodes * 100 + graph_idx * 10,
                    'cpt_generation': cpt_generation,
                    'logistic_std': logistic_std
                }

                # Create and run experiment
                experiment = TunedLensProgressiveExperiment(config, imputer_sizes)
                experiment.setup()
                results = experiment.run_multi_policy_experiment(policies)

                # Save experiment artifacts
                _save_tuned_lens_experiment_artifacts(
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

    logger.info(f"\nTuned lens experiment suite completed: {len(all_results)} configurations")
    return all_results


def _save_tuned_lens_experiment_artifacts(
    experiment: TunedLensProgressiveExperiment,
    results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    n_nodes: int,
    missing_rate: float,
    graph_idx: int
) -> None:
    """Save all tuned lens experiment artifacts."""
    # Create structured output directory
    exp_dir = output_dir / f"nodes_{n_nodes}_missing_{missing_rate:.1f}_graph_{graph_idx}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving tuned lens experiment artifacts to {exp_dir}")

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

    # Save calibration dataset
    calibration_dir = exp_dir / "calibration_dataset"
    calibration_dir.mkdir(exist_ok=True)
    with open(calibration_dir / "calibration_samples.pkl", 'wb') as f:
        pickle.dump(experiment.calibration_dataset, f)
    with open(calibration_dir / "calibration_metadata.json", 'w') as f:
        json.dump({
            'n_samples': len(experiment.calibration_dataset),
            'missing_rate': experiment.missing_rate
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
            budget_dir = exp_dir / f"budget_{budget:04d}" / imputer_size.lower() / "tuned_lens"
            budget_dir.mkdir(parents=True, exist_ok=True)

            # Save base model
            base_model_key = f"{imputer_size}_budget_{budget}"
            if base_model_key in experiment.base_models:
                model = experiment.base_models[base_model_key]
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': {
                        'n_nodes': model.n_nodes,
                        'n_states': model.n_states,
                        'cpt_dim': model.cpt_dim,
                        'imputer_size': imputer_size
                    }
                }, budget_dir / "base_model.pt")

            # Save tuned lens model (probes)
            if base_model_key in experiment.tuned_lens_models:
                tuned_lens_model = experiment.tuned_lens_models[base_model_key]
                torch.save({
                    'probes_state_dict': tuned_lens_model.probes.state_dict(),
                    'n_layers': tuned_lens_model.n_layers,
                    'cpt_dim': tuned_lens_model.cpt_dim
                }, budget_dir / "probes.pt")

            # Save layer-wise analysis results
            analysis_data = {
                'budget': budget,
                'imputer_size': imputer_size,
                'n_training_samples': step_result['n_training_samples'],
                'n_calibration_samples': step_result['n_calibration_samples'],
                'base_training_time': step_result['base_training_time'],
                'probe_training_time': step_result['probe_training_time'],
                'total_time': step_result['total_time'],

                # Standard metrics
                'final_kl': step_result['tuned_lens_kl'],
                'final_kl_std': step_result['tuned_lens_kl_std'],

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

    logger.info(f"Saved tuned lens experiment artifacts: {exp_dir}")

    # Create tuned lens plots
    try:
        from utils.tuned_lens_plotting import create_tuned_lens_plots_for_experiment

        # Get imputer sizes from results
        imputer_sizes = set()
        for result_data in results.values():
            for step_result in result_data['results']:
                imputer_sizes.add(step_result['imputer_size'])

        for imputer_size in imputer_sizes:
            logger.info(f"Creating tuned lens plots for {imputer_size}")
            create_tuned_lens_plots_for_experiment(exp_dir, imputer_size)
    except Exception as e:
        logger.error(f"Failed to create tuned lens plots: {e}", exc_info=True)
