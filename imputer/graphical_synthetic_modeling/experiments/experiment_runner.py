"""
Progressive experiment runner for policy-based imputation.

Coordinates progressive experiments between neural imputers and EM baselines,
providing comprehensive evaluation with KL divergence and log-loss metrics.
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pathlib import Path

from data.graph_generator import generate_experiment_graph
from data.sample_generator import generate_sample_pool, generate_test_dataset
from imputer.architecture import create_model, DEVICE, SampleTuple
from imputer.training_eval import train_model, evaluate_model, evaluate_log_loss, evaluate_cross_entropy, ImputationDataset, collate_batch
from domain.em_model import DomainEMModel
from experiments.policies import BaseObservationPolicy, SampleTuple

logger = logging.getLogger(__name__)


def aggregate_multi_graph_results(graph_results: List[Dict[str, Dict[str, Any]]], 
                                 n_graphs: int) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate experimental results across multiple graph instances.
    
    Computes mean and standard deviation of performance metrics across different
    graph structures to provide proper statistical analysis.
    
    Args:
        graph_results: List of result dicts from each graph instance
        n_graphs: Number of graph instances
        
    Returns:
        Dict with aggregated results including means and standard deviations
    """
    logger.debug(f"Aggregating results from {n_graphs} graph instances")
    
    if not graph_results:
        return {}
    
    # Get all policy-imputer combinations from first graph
    all_keys = list(graph_results[0].keys())
    aggregated = {}
    
    for policy_imputer_key in all_keys:
        logger.debug(f"Aggregating results for {policy_imputer_key}")
        
        # Get progressive results from all graphs for this policy-imputer combination
        all_progressive_results = []
        for graph_result in graph_results:
            if policy_imputer_key in graph_result:
                all_progressive_results.append(graph_result[policy_imputer_key]['results'])
        
        if not all_progressive_results:
            continue
            
        # Get number of budget steps
        n_steps = len(all_progressive_results[0])
        
        # Aggregate step by step
        aggregated_steps = []
        for step_idx in range(n_steps):
            # Collect metrics from all graphs for this budget step
            neural_kls = []
            domain_kls = []
            neural_log_losses = []
            domain_log_losses = []
            true_log_losses = []
            neural_cross_entropies = []
            domain_cross_entropies = []
            true_entropies = []
            neural_times = []
            domain_times = []
            budgets = []
            n_training_samples = []
            
            # Collect individual sample arrays for proper aggregation
            neural_log_loss_arrays = []
            domain_log_loss_arrays = []
            true_log_loss_arrays = []
            neural_cross_entropy_arrays = []
            domain_cross_entropy_arrays = []
            true_entropy_arrays = []
            
            for graph_progressive_results in all_progressive_results:
                if step_idx < len(graph_progressive_results):
                    step_result = graph_progressive_results[step_idx]
                    neural_kls.append(step_result.get('neural_kl', float('inf')))
                    domain_kls.append(step_result.get('domain_kl', float('inf')))
                    neural_log_losses.append(step_result.get('neural_log_loss', float('inf')))
                    domain_log_losses.append(step_result.get('domain_log_loss', float('inf')))
                    true_log_losses.append(step_result.get('true_model_log_loss', float('inf')))
                    neural_cross_entropies.append(step_result.get('mean_cross_entropy', float('inf')))
                    domain_cross_entropies.append(step_result.get('mean_cross_entropy', float('inf')))
                    true_entropies.append(step_result.get('mean_true_entropy', float('inf')))
                    neural_times.append(step_result.get('neural_time', 0.0))
                    domain_times.append(step_result.get('domain_time', 0.0))
                    budgets.append(step_result.get('budget', 0))
                    n_training_samples.append(step_result.get('n_training_samples', 0))
                    
                    # Collect individual sample arrays from each graph
                    neural_log_loss_arrays.extend(step_result.get('neural_log_loss_values', []))
                    domain_log_loss_arrays.extend(step_result.get('domain_log_loss_values', []))
                    true_log_loss_arrays.extend(step_result.get('true_model_log_loss_values', []))
                    neural_cross_entropy_arrays.extend(step_result.get('neural_cross_entropy_values', []))
                    domain_cross_entropy_arrays.extend(step_result.get('domain_cross_entropy_values', []))
                    true_entropy_arrays.extend(step_result.get('true_entropy_values', []))
            
            # Compute aggregated statistics
            aggregated_step = {
                'budget': budgets[0] if budgets else 0,  # Same across graphs
                'n_training_samples': n_training_samples[0] if n_training_samples else 0,
                
                # Neural metrics with statistics
                'neural_kl': np.mean(neural_kls) if neural_kls else float('inf'),
                'neural_kl_std': np.std(neural_kls) if len(neural_kls) > 1 else 0.0,
                'neural_log_loss': np.mean(neural_log_losses) if neural_log_losses else float('inf'),
                'neural_log_loss_std': np.std(neural_log_losses) if len(neural_log_losses) > 1 else 0.0,
                'neural_time': np.mean(neural_times) if neural_times else 0.0,
                'neural_time_std': np.std(neural_times) if len(neural_times) > 1 else 0.0,
                
                # Domain metrics with statistics  
                'domain_kl': np.mean(domain_kls) if domain_kls else float('inf'),
                'domain_kl_std': np.std(domain_kls) if len(domain_kls) > 1 else 0.0,
                'domain_log_loss': np.mean(domain_log_losses) if domain_log_losses else float('inf'),
                'domain_log_loss_std': np.std(domain_log_losses) if len(domain_log_losses) > 1 else 0.0,
                'domain_time': np.mean(domain_times) if domain_times else 0.0,
                'domain_time_std': np.std(domain_times) if len(domain_times) > 1 else 0.0,
                
                # True model metrics
                'true_model_log_loss': np.mean(true_log_losses) if true_log_losses else float('inf'),
                'true_model_log_loss_std': np.std(true_log_losses) if len(true_log_losses) > 1 else 0.0,
                
                # Cross-entropy metrics
                'neural_cross_entropy': np.mean(neural_cross_entropies) if neural_cross_entropies else float('inf'),
                'neural_cross_entropy_std': np.std(neural_cross_entropies) if len(neural_cross_entropies) > 1 else 0.0,
                'domain_cross_entropy': np.mean(domain_cross_entropies) if domain_cross_entropies else float('inf'),
                'domain_cross_entropy_std': np.std(domain_cross_entropies) if len(domain_cross_entropies) > 1 else 0.0,
                'true_entropy': np.mean(true_entropies) if true_entropies else float('inf'),
                'true_entropy_std': np.std(true_entropies) if len(true_entropies) > 1 else 0.0,
                
                # Raw values for detailed analysis (flattened across graphs)
                'neural_kl_values': neural_kls,
                'domain_kl_values': domain_kls,
                'neural_log_loss_values': neural_log_loss_arrays,  # Individual sample arrays
                'domain_log_loss_values': domain_log_loss_arrays,  # Individual sample arrays
                'true_model_log_loss_values': true_log_loss_arrays,  # Individual sample arrays
                'neural_cross_entropy_values': neural_cross_entropy_arrays,  # Individual sample arrays
                'domain_cross_entropy_values': domain_cross_entropy_arrays,  # Individual sample arrays
                'true_entropy_values': true_entropy_arrays,  # Individual sample arrays
                
                # Evaluation counts
                'neural_n_evaluations': len([x for x in neural_kls if not np.isinf(x)]),
                'domain_n_evaluations': len([x for x in domain_kls if not np.isinf(x)]),
                'neural_failed_rate': sum(1 for x in neural_kls if np.isinf(x)) / len(neural_kls) if neural_kls else 1.0,
                'domain_failed_rate': sum(1 for x in domain_kls if np.isinf(x)) / len(domain_kls) if domain_kls else 1.0
            }
            
            aggregated_steps.append(aggregated_step)
        
        # Get metadata from first graph result
        first_graph_data = graph_results[0][policy_imputer_key]
        
        # Compute total time statistics
        all_total_times = [graph_results[i][policy_imputer_key].get('total_time', 0.0) 
                          for i in range(len(graph_results)) if policy_imputer_key in graph_results[i]]
        
        # Create aggregated result structure
        aggregated[policy_imputer_key] = {
            'results': aggregated_steps,
            'total_time': np.mean(all_total_times) if all_total_times else 0.0,
            'total_time_std': np.std(all_total_times) if len(all_total_times) > 1 else 0.0,
            'n_graphs': n_graphs,
            'config': first_graph_data.get('config', {}),
            'policy_name': first_graph_data.get('policy_name', ''),
            'imputer_size': first_graph_data.get('imputer_size', ''),
            'policy_info': first_graph_data.get('policy_info', {})
        }
        
    logger.info(f"Successfully aggregated {len(aggregated)} policy-imputer combinations across {n_graphs} graphs")
    return aggregated


class ProgressiveExperiment:
    """
    Main experiment runner for progressive imputation with clean error handling.
    
    Coordinates training and evaluation of multiple imputer variants against
    domain-specific EM baselines using progressive observation policies.
    """
    
    def __init__(self, config: Dict[str, Any], imputer_sizes: Optional[List[str]] = None):
        """
        Initialize experiment with configuration.
        
        Args:
            config: Experiment parameters dict containing:
                - n_nodes: Number of nodes in graph
                - target_parents: Target parents per node  
                - missing_rate: Fraction of missing nodes in test data
                - max_samples: Maximum training samples available
                - test_samples: Number of test samples
                - seed: Random seed
                - alpha: Optional Dirichlet concentration parameter for CPT generation
            imputer_sizes: List of imputer model sizes ["Tiny", "Small", "Large"]
                          If None, defaults to ["Large"]
        """
        self.config = config
        self.n_nodes = config['n_nodes']
        self.target_parents = config['target_parents']
        self.missing_rate = config['missing_rate']
        self.max_samples = config['max_samples']
        self.test_samples = config['test_samples']
        self.seed = config['seed']
        self.alpha = config.get('alpha', None)  # Dirichlet concentration parameter (None = default CPTs)
        
        # Default to single Large imputer if not specified
        if imputer_sizes is None:
            imputer_sizes = ["Large"]
        self.imputer_sizes = imputer_sizes
        
        # Data containers - populated in setup()
        self.bn = None
        self.adj_matrix = None  
        self.sample_pool: Optional[List[SampleTuple]] = None
        self.test_dataset: Optional[List[SampleTuple]] = None
        
        # Domain model for baseline comparison
        use_likelihood = config.get('use_likelihood_selection', False)
        self.domain_model = DomainEMModel(use_likelihood_selection=use_likelihood)
        
        logger.info(f"Initialized experiment: {self.n_nodes} nodes, {self.max_samples} max samples")
        logger.info(f"Imputer sizes: {imputer_sizes}")
        
    def setup(self) -> None:
        """
        Generate graph structure and datasets for progressive experiments.
        
        Creates the Bayesian network, sample pool with missing data, and test dataset.
        All exceptions bubble up for debugging - no error masking.
        """
        logger.info(f"Setting up experiment data...")
        start_time = time.time()
        
        # Generate graph structure
        logger.debug(f"Generating graph: {self.n_nodes} nodes, {self.target_parents} parents")
        self.bn, self.adj_matrix = generate_experiment_graph(
            self.n_nodes, self.target_parents, self.seed, self.alpha
        )
        
        # Generate sample pool with missing data for progressive experiments
        logger.debug(f"Generating sample pool: {self.max_samples} samples, {self.missing_rate} missing rate")
        self.sample_pool = generate_sample_pool(
            self.bn, self.adj_matrix, self.n_nodes, self.max_samples, 
            self.missing_rate, self.seed
        )
        
        # Generate test dataset with missing values
        logger.debug(f"Generating test dataset: {self.test_samples} samples")
        self.test_dataset = generate_test_dataset(
            self.bn, self.adj_matrix, self.n_nodes, self.test_samples, 
            self.missing_rate, self.seed + 1000
        )
        
        setup_time = time.time() - start_time
        logger.info(f"Data setup completed in {setup_time:.2f}s")
        logger.info(f"Sample pool: {len(self.sample_pool)} samples with missing data")
        logger.info(f"Test dataset: {len(self.test_dataset)} samples")
        
    def _create_neural_models(self) -> Dict[str, Any]:
        """
        Create neural imputer models for all requested sizes.
        
        Returns:
            Dict mapping imputer size to created model
        """
        neural_models = {}
        
        # Determine dimensions from sample data
        if not self.sample_pool:
            raise ValueError("Sample pool must be generated before creating models")
            
        sample_inputs, sample_structure, _, _, _, _ = self.sample_pool[0]
        input_dim = sample_inputs.shape[1]
        structure_dim = sample_structure.shape[1]
        
        # CRITICAL: Compute max CPT size from the actual BN to avoid tensor size mismatches
        from imputer.architecture import compute_max_cpt_size
        max_cpt_size = compute_max_cpt_size(self.bn) if self.bn else 8
        logger.debug(f"Using max_cpt_size={max_cpt_size} for model creation")
        
        # Create models for each requested size
        for imputer_size in self.imputer_sizes:
            logger.debug(f"Creating {imputer_size} imputer model")
            model = create_model(
                n_nodes=self.n_nodes,
                input_dim=input_dim,
                structure_dim=structure_dim,
                cpt_dim=max_cpt_size,  # Pass actual max CPT size
                model_size=imputer_size
            )
            neural_models[imputer_size] = model
            
        logger.info(f"Created {len(neural_models)} neural imputer models with cpt_dim={max_cpt_size}")
        return neural_models
    
    def run_policy_experiment(self, policy: BaseObservationPolicy) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run progressive experiment with given policy and all imputer variants.
        
        Args:
            policy: Observation policy for progressive training data selection
            
        Returns:
            Dict mapping imputer sizes to progressive results lists
            
        Raises:
            Exception: Any failures during training/evaluation bubble up for debugging
        """
        logger.info(f"Running experiment with policy: {policy}")
        
        if self.sample_pool is None or self.test_dataset is None:
            raise ValueError("Must call setup() before running experiments")
            
        # Create neural models
        neural_models = self._create_neural_models()
        
        # Results storage for each imputer size
        results_by_size = {size: [] for size in self.imputer_sizes}
        
        # Progressive observation loop
        for budget, training_data in policy.observe_progressively(self.sample_pool):
            logger.info(f"Budget {budget}: Training on {len(training_data)} samples")
            
            # Convert training data to PyTorch datasets
            train_dataset = ImputationDataset(training_data, self.bn)
            val_dataset = ImputationDataset(self.test_dataset[:50], self.bn)  # Small validation set
            
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)
            
            # Train and evaluate all neural model variants
            neural_results_by_size = {}
            neural_times_by_size = {}
            
            for imputer_size, neural_model in neural_models.items():
                neural_start = time.time()
                
                # Reset model parameters (create fresh model)
                neural_model = self._create_neural_models()[imputer_size]
                
                # Train neural model
                logger.debug(f"Training {imputer_size} neural imputer")
                trained_model = train_model(
                    neural_model, train_loader, val_loader,
                    epochs=100, lr=1e-4, patience=30
                )
                
                # Evaluate neural model (KL divergence)
                logger.debug(f"Evaluating {imputer_size} neural imputer KL")
                neural_results = evaluate_model(
                    trained_model, self.test_dataset, self.bn, 
                    self.n_nodes, n_states=2
                )
                
                # Evaluate neural model (log-loss)
                logger.debug(f"Evaluating {imputer_size} neural imputer log-loss")
                neural_log_loss_results = evaluate_log_loss(
                    trained_model, self.test_dataset, self.bn, self.n_nodes
                )
                
                # Evaluate neural model (cross-entropy)
                logger.debug(f"Evaluating {imputer_size} neural imputer cross-entropy")
                neural_cross_entropy_results = evaluate_cross_entropy(
                    trained_model, self.test_dataset, self.bn, self.n_nodes
                )
                
                neural_time = time.time() - neural_start
                
                # Combine results
                combined_neural_results = {**neural_results, **neural_log_loss_results, **neural_cross_entropy_results}
                neural_results_by_size[imputer_size] = combined_neural_results
                neural_times_by_size[imputer_size] = neural_time
                
                logger.info(f"  Imputer ({imputer_size}): KL={neural_results.get('mean_kl', float('inf')):.4f}, "
                           f"LogLoss={neural_log_loss_results.get('mean_log_loss', float('inf')):.4f}, time={neural_time:.1f}s")
            
            # Train and evaluate domain model (once per budget - same for all imputer variants)
            domain_start = time.time()
            
            logger.debug("Training domain EM model")
            self.domain_model.reset()
            self.domain_model.train(training_data, self.bn, self.adj_matrix, self.n_nodes)
            
            logger.debug("Evaluating domain EM model KL")
            domain_results = self.domain_model.evaluate(self.test_dataset, self.bn, self.n_nodes)
            
            logger.debug("Evaluating domain EM model log-loss") 
            domain_log_loss_results = self.domain_model.evaluate_log_loss(self.test_dataset, self.bn, self.n_nodes)
            
            logger.debug("Evaluating domain EM model cross-entropy")
            domain_cross_entropy_results = self.domain_model.evaluate_cross_entropy(self.test_dataset, self.bn, self.n_nodes)
            
            domain_time = time.time() - domain_start
            
            # Combine domain results
            combined_domain_results = {**domain_results, **domain_log_loss_results, **domain_cross_entropy_results}
            
            logger.info(f"  Domain EM: KL={domain_results.get('mean_kl', float('inf')):.4f}, "
                       f"LogLoss={domain_log_loss_results.get('mean_log_loss', float('inf')):.4f}, time={domain_time:.1f}s")
            
            # Force garbage collection after EM training to prevent memory accumulation
            import gc
            gc.collect()
            
            # Evaluate true model log-loss (ground truth baseline)
            logger.debug("Evaluating true model log-loss")
            true_model_wrapper = DomainEMModel()
            true_model_wrapper.learned_bn = self.bn  # Use ground truth BN
            true_model_wrapper.is_trained = True
            true_model_log_loss_results = true_model_wrapper.evaluate_log_loss(self.test_dataset, self.bn, self.n_nodes)
            
            logger.info(f"  True Model: LogLoss={true_model_log_loss_results.get('mean_log_loss', float('inf')):.4f}")
            
            # Store results for each imputer size
            for imputer_size in self.imputer_sizes:
                neural_results = neural_results_by_size[imputer_size]
                neural_time = neural_times_by_size[imputer_size]
                
                step_result = {
                    'budget': budget,
                    'n_training_samples': len(training_data),
                    # KL divergence metrics
                    'neural_kl': neural_results.get('mean_kl', float('inf')),
                    'neural_kl_std': neural_results.get('std_kl', 0.0),
                    'neural_failed_rate': neural_results.get('failed_rate', 1.0),
                    'neural_n_evaluations': neural_results.get('n_evaluations', 0),
                    'neural_time': neural_time,
                    'domain_kl': combined_domain_results.get('mean_kl', float('inf')),
                    'domain_kl_std': combined_domain_results.get('std_kl', 0.0),
                    'domain_failed_rate': combined_domain_results.get('failed_rate', 1.0),
                    'domain_n_evaluations': combined_domain_results.get('n_evaluations', 0),
                    'domain_time': domain_time,
                    # Log-loss metrics
                    'neural_log_loss': neural_results.get('mean_log_loss', float('inf')),
                    'neural_log_loss_std': neural_results.get('std_log_loss', 0.0),
                    'domain_log_loss': combined_domain_results.get('mean_log_loss', float('inf')),
                    'domain_log_loss_std': combined_domain_results.get('std_log_loss', 0.0),
                    'true_model_log_loss': true_model_log_loss_results.get('mean_log_loss', float('inf')),
                    'true_model_log_loss_std': true_model_log_loss_results.get('std_log_loss', 0.0),
                    # Raw values for detailed analysis
                    'neural_kl_distribution': neural_results.get('kl_distribution', []),
                    'neural_log_loss_values': neural_results.get('log_loss_values', []),
                    'domain_log_loss_values': combined_domain_results.get('log_loss_values', []),
                    'true_model_log_loss_values': true_model_log_loss_results.get('log_loss_values', []),
                    # Cross-entropy values
                    'neural_cross_entropy_values': combined_neural_results.get('cross_entropy_values', []),
                    'domain_cross_entropy_values': combined_domain_results.get('cross_entropy_values', []),
                    'true_entropy_values': combined_neural_results.get('true_entropy_values', [])
                }
                
                results_by_size[imputer_size].append(step_result)
                
        return results_by_size
    
    def run_multi_policy_experiment(self, policies: List[BaseObservationPolicy]) -> Dict[str, Dict[str, Any]]:
        """
        Run experiment with multiple policies and all imputer variants.
        
        Args:
            policies: List of observation policies to compare
            
        Returns:
            Dict mapping "{policy_name}_{imputer_size}" to experiment results
        """
        logger.info(f"Running multi-policy experiment with {len(policies)} policies")
        
        all_results = {}
        
        for policy in policies:
            policy_start = time.time()
            logger.debug(f"Starting policy: {policy.name}")
            
            results_by_size = self.run_policy_experiment(policy)
            policy_time = time.time() - policy_start
            
            # Create separate entries for each (policy, imputer_size) combination
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


def run_experiment_suite(node_sizes: List[int], target_parents: float = 1.0, 
                        missing_rate: float = 0.4, max_samples: int = 3000, 
                        test_samples: int = 250, 
                        policies: Optional[List[BaseObservationPolicy]] = None,
                        imputer_sizes: Optional[List[str]] = None,
                        n_graphs: int = 1) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """
    Run experiments across multiple graph sizes with comprehensive evaluation.
    
    Args:
        node_sizes: List of graph sizes to test (e.g., [5, 10, 15])
        target_parents: Target parents per node for graph generation
        missing_rate: Missing rate for test data
        max_samples: Maximum training samples available
        test_samples: Number of test samples
        policies: List of observation policies to test (default: RandomExamplePolicy)
        imputer_sizes: List of imputer sizes to test (default: ["Large"])
        n_graphs: Number of random graph instances per configuration for statistical analysis
        
    Returns:
        Dict mapping (n_nodes, "{policy_name}_{imputer_size}") to aggregated experiment results
        
    Raises:
        Exception: Any experimental failures bubble up for debugging
    """
    # Default policies and imputer sizes
    if policies is None:
        from experiments.policies import RandomExamplePolicy
        policies = [RandomExamplePolicy()]
        
    if imputer_sizes is None:
        imputer_sizes = ["Large"]
        
    logger.info(f"Running experiment suite: {len(node_sizes)} graph sizes, "
               f"{len(policies)} policies, {len(imputer_sizes)} imputer sizes, {n_graphs} graphs each")
    logger.info(f"Graph sizes: {node_sizes}")
    logger.info(f"Policies: {[p.name for p in policies]}")
    logger.info(f"Imputer sizes: {imputer_sizes}")
    
    total_experiments = len(node_sizes) * len(policies) * len(imputer_sizes) * n_graphs
    logger.info(f"Total experiment runs: {total_experiments}")
    
    all_results = {}
    
    for n_nodes in node_sizes:
        logger.info(f"\\n{'='*60}")
        logger.info(f"GRAPH SIZE: {n_nodes} nodes")
        logger.info(f"{'='*60}")
        
        # Define alpha candidates for Dirichlet CPT sampling (emphasizing challenging sparse cases)
        alpha_candidates = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
        
        # Run multiple graph instances for statistical analysis
        graph_results = []
        
        for graph_idx in range(n_graphs):
            # Alpha selection: None for single graph (backward compatibility), random for multi-graph
            if n_graphs == 1:
                alpha = None  # Use current generateCPT() behavior
                logger.info(f"\\nGraph instance {graph_idx + 1}/{n_graphs} for {n_nodes} nodes (using default CPT generation)")
            else:
                alpha = np.random.choice(alpha_candidates)
                logger.info(f"\\nGraph instance {graph_idx + 1}/{n_graphs} for {n_nodes} nodes (using Dirichlet α={alpha:.1f})")
            
            # Configuration for this graph instance
            config = {
                'n_nodes': n_nodes,
                'target_parents': target_parents,
                'missing_rate': missing_rate,
                'max_samples': max_samples,
                'test_samples': test_samples,
                'seed': 42 + n_nodes * 1000 + graph_idx * 10,  # Unique seed per graph instance
                'alpha': alpha  # Add alpha parameter for Dirichlet CPT sampling
            }
            
            # Create and setup experiment
            experiment = ProgressiveExperiment(config, imputer_sizes)
            experiment.setup()
            
            # Run with all policies
            results = experiment.run_multi_policy_experiment(policies)
            graph_results.append(results)
            
            logger.debug(f"Graph instance {graph_idx + 1} completed: {len(results)} configurations")
        
        # Aggregate results across multiple graph instances
        logger.info(f"\\nAggregating results across {n_graphs} graph instances for {n_nodes} nodes...")
        aggregated_results = aggregate_multi_graph_results(graph_results, n_graphs)
        
        # Store aggregated results with (n_nodes, policy_imputer) key
        for policy_imputer_key, aggregated_data in aggregated_results.items():
            composite_key = (n_nodes, policy_imputer_key)
            all_results[composite_key] = aggregated_data
            
        logger.debug(f"Graph size {n_nodes} aggregation completed: {len(aggregated_results)} configurations")
            
    logger.info(f"\\nExperiment suite completed: {len(all_results)} total configurations")
    
    return all_results


def save_experiment_results(results: Dict[Any, Dict[str, Any]], output_dir: Path) -> None:
    """
    Save experiment results to structured output files.
    
    Args:
        results: Experiment results from run_experiment_suite
        output_dir: Directory to save results
    """
    import pickle
    import json
    from datetime import datetime
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results as pickle
    pickle_path = output_dir / f"results_{timestamp}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
        
    logger.info(f"Saved full results to {pickle_path}")
    
    # Save summary as JSON (excluding large arrays)
    summary_results = {}
    for key, experiment_data in results.items():
        summary_key = str(key)  # Convert tuple key to string
        summary_results[summary_key] = {
            'total_time': experiment_data['total_time'],
            'config': experiment_data['config'],
            'policy_name': experiment_data['policy_name'],
            'imputer_size': experiment_data['imputer_size'],
            'n_steps': len(experiment_data['results']),
            'final_performance': experiment_data['results'][-1] if experiment_data['results'] else {}
        }
    
    json_path = output_dir / f"summary_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(summary_results, f, indent=2, default=str)
        
    logger.info(f"Saved summary to {json_path}")