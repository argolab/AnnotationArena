"""
Progressive experiment runner for policy-based imputation.

Main experiment runner that coordinates policies, models, and evaluation.
"""

import logging
import time
from typing import List, Dict, Any
import numpy as np

from data.graph_generator import generate_experiment_graph
from data.sample_generator import generate_sample_pool, generate_test_dataset
from models.neural_imputer import NeuralParameterEmbeddingImputer
from models.domain_model import DomainEMModel

logger = logging.getLogger(__name__)

class ProgressiveExperiment:
    """
    Main experiment runner for progressive imputation.
    """
    
    def __init__(self, config, imputer_sizes=None):
        """
        Initialize experiment with configuration.
        
        Args:
            config: Dict with experiment parameters
                - n_nodes: Number of nodes in graph
                - target_parents: Target parents per node
                - missing_rate: Fraction of missing nodes in test data
                - max_samples: Maximum training samples available
                - test_samples: Number of test samples
                - seed: Random seed
            imputer_sizes: List of imputer model sizes ["Tiny", "Small", "Large"]
        """
        self.config = config
        self.n_nodes = config['n_nodes']
        self.target_parents = config['target_parents']
        self.missing_rate = config['missing_rate']
        self.max_samples = config['max_samples']
        self.test_samples = config['test_samples']
        self.seed = config['seed']
        
        # Default to single Large imputer if not specified
        if imputer_sizes is None:
            imputer_sizes = ["Large"]
        self.imputer_sizes = imputer_sizes
        
        # Data will be generated in setup
        self.bn = None
        self.adj_matrix = None
        self.sample_pool = None
        self.test_dataset = None
        
        # Models - create multiple imputer variants
        self.neural_models = {}
        for imputer_size in imputer_sizes:
            self.neural_models[imputer_size] = NeuralParameterEmbeddingImputer(model_size=imputer_size)
        
        self.domain_model = DomainEMModel()
        
        logger.info(f"Initialized experiment: {self.n_nodes} nodes, {self.max_samples} max samples")
        logger.info(f"Imputer sizes: {imputer_sizes}")
        
    def setup(self):
        """Generate graph structure and datasets."""
        logger.info(f"Setting up experiment data...")
        start_time = time.time()
        
        # Generate graph structure
        self.bn, self.adj_matrix = generate_experiment_graph(
            self.n_nodes, self.target_parents, self.seed
        )
        
        # Generate sample pool with missing data for progressive experiments
        self.sample_pool = generate_sample_pool(
            self.bn, self.adj_matrix, self.n_nodes, self.max_samples, 
            self.missing_rate, self.seed
        )
        
        # Generate test dataset with missing values
        self.test_dataset = generate_test_dataset(
            self.bn, self.adj_matrix, self.n_nodes, self.test_samples, 
            self.missing_rate, self.seed + 1000
        )
        
        setup_time = time.time() - start_time
        logger.info(f"Data setup completed in {setup_time:.2f}s")
        logger.info(f"Sample pool: {len(self.sample_pool)} samples with missing data")
        logger.info(f"Test dataset: {len(self.test_dataset)} samples")
        
    def run_policy_experiment(self, policy):
        """
        Run progressive experiment with given policy and all imputer variants.
        
        Args:
            policy: Observation policy to use
            
        Returns:
            Dict mapping imputer sizes to results lists
        """
        logger.info(f"Running experiment with policy: {policy}")
        
        if self.sample_pool is None:
            raise ValueError("Must call setup() before running experiments")
            
        # Results for each imputer size
        results_by_size = {size: [] for size in self.imputer_sizes}
        
        for budget, training_data in policy.observe_progressively(self.sample_pool):
            logger.info(f"Budget {budget}: Training on {len(training_data)} samples")
            
            # Train and evaluate all neural model variants
            neural_results_by_size = {}
            neural_times_by_size = {}
            
            for imputer_size, neural_model in self.neural_models.items():
                neural_start = time.time()
                neural_model.reset()
                neural_model.train(training_data, self.bn, self.adj_matrix, self.n_nodes)
                neural_results = neural_model.evaluate(self.test_dataset, self.bn, self.adj_matrix, self.n_nodes)
                neural_time = time.time() - neural_start
                
                neural_results_by_size[imputer_size] = neural_results
                neural_times_by_size[imputer_size] = neural_time
                
                logger.info(f"  Imputer ({imputer_size}): KL={neural_results.get('mean_kl', float('inf')):.4f}, time={neural_time:.1f}s")
            
            # Train and evaluate domain model once (same for all imputer variants)
            domain_start = time.time()
            self.domain_model.reset()
            self.domain_model.train(training_data, self.bn, self.adj_matrix, self.n_nodes)
            domain_results = self.domain_model.evaluate(self.test_dataset, self.bn, self.n_nodes)
            domain_time = time.time() - domain_start
            
            logger.info(f"  Domain EM: KL={domain_results.get('mean_kl', float('inf')):.4f}, time={domain_time:.1f}s")
            
            # Store results for each imputer size
            for imputer_size in self.imputer_sizes:
                neural_results = neural_results_by_size[imputer_size]
                neural_time = neural_times_by_size[imputer_size]
                
                step_result = {
                    'budget': budget,
                    'n_training_samples': len(training_data),
                    'neural_kl': neural_results.get('mean_kl', float('inf')),
                    'neural_failed_rate': neural_results.get('failed_rate', 1.0),
                    'neural_time': neural_time,
                    'domain_kl': domain_results.get('mean_kl', float('inf')),
                    'domain_failed_rate': domain_results.get('failed_rate', 1.0),
                    'domain_time': domain_time
                }
                
                results_by_size[imputer_size].append(step_result)
            
        return results_by_size
    
    def run_multi_policy_experiment(self, policies):
        """
        Run experiment with multiple policies.
        
        Args:
            policies: List of policies to compare
            
        Returns:
            Dict mapping (policy_name, imputer_size) to results
        """
        logger.info(f"Running multi-policy experiment with {len(policies)} policies")
        
        all_results = {}
        
        for policy in policies:
            policy_start = time.time()
            results_by_size = self.run_policy_experiment(policy)
            policy_time = time.time() - policy_start
            
            # Create separate entries for each imputer size
            for imputer_size, results in results_by_size.items():
                combined_key = f"{policy.name}_{imputer_size}"
                all_results[combined_key] = {
                    'results': results,
                    'total_time': policy_time,  # Same training time for all sizes (trained together)
                    'config': {**self.config.copy(), 'imputer_size': imputer_size},
                    'policy_name': policy.name,
                    'imputer_size': imputer_size
                }
            
            logger.info(f"Policy {policy.name} completed in {policy_time:.1f}s")
            
        return all_results

def run_experiment_suite(node_sizes, target_parents=1.0, missing_rate=0.4, 
                        max_samples=3000, test_samples=250, policies=None):
    """
    Run experiments across multiple graph sizes.
    
    Args:
        node_sizes: List of graph sizes to test
        target_parents: Target parents per node
        missing_rate: Missing rate for test data
        max_samples: Maximum training samples
        test_samples: Number of test samples
        policies: List of policies to test
        
    Returns:
        Dict mapping (n_nodes, policy_name) to results
    """
    if policies is None:
        from policies.random_example_policy import RandomExamplePolicy
        policies = [RandomExamplePolicy()]
        
    logger.info(f"Running experiment suite: {len(node_sizes)} graph sizes, {len(policies)} policies")
    
    all_results = {}
    
    for n_nodes in node_sizes:
        logger.info(f"\\n{'='*60}")
        logger.info(f"GRAPH SIZE: {n_nodes} nodes")
        logger.info(f"{'='*60}")
        
        config = {
            'n_nodes': n_nodes,
            'target_parents': target_parents,
            'missing_rate': missing_rate,
            'max_samples': max_samples,
            'test_samples': test_samples,
            'seed': 42 + n_nodes * 1000  # Different seed per graph size
        }
        
        experiment = ProgressiveExperiment(config)
        experiment.setup()
        
        results = experiment.run_multi_policy_experiment(policies)
        
        # Store results with node size key
        for policy_name, policy_results in results.items():
            key = (n_nodes, policy_name)
            all_results[key] = policy_results
            
    logger.info(f"\\nExperiment suite completed: {len(all_results)} configurations")
    
    return all_results