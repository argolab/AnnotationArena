"""
Progressive experiment runner with VOI-based Active Feature Acquisition.

This module handles experiments where policies can select individual node 
observations rather than complete samples, using Value of Information to 
guide the selection process.

Author: Prabhav Singh
"""

import logging
import time
from typing import List, Dict, Any
import numpy as np
import torch

from data.graph_generator import generate_experiment_graph
from data.sample_generator import generate_sample_pool, generate_test_dataset
from data.sample_generator_afa import generate_afa_sample_pool, generate_afa_test_dataset, get_samples_with_observations
from models.neural_imputer import NeuralParameterEmbeddingImputer
from models.domain_model import DomainEMModel
from policies.afa_policy import AFAPolicy
from policies.random_fa_policy import RandomFAPolicy

logger = logging.getLogger(__name__)

class ProgressiveVOIExperiment:
    """
    Experiment runner for VOI-based progressive imputation.
    
    Handles both traditional example-level policies (RandomExample) and
    node-level policies (AFA) that can select individual observations.
    """
    
    def __init__(self, config):
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
        """
        self.config = config
        self.n_nodes = config['n_nodes']
        self.target_parents = config['target_parents']
        self.missing_rate = config['missing_rate']
        self.max_samples = config['max_samples']
        self.test_samples = config['test_samples']
        self.seed = config['seed']
        
        # Data will be generated in setup
        self.bn = None
        self.adj_matrix = None
        self.sample_pool = None
        self.test_dataset = None
        
        # Models
        self.neural_model = NeuralParameterEmbeddingImputer()
        self.domain_model = DomainEMModel()
        
        logger.info(f"Initialized VOI experiment: {self.n_nodes} nodes, {self.max_samples} max samples")
        
    def setup(self):
        """Generate graph structure and datasets."""
        logger.info(f"Setting up VOI experiment data...")
        start_time = time.time()
        
        # Generate graph structure
        self.bn, self.adj_matrix = generate_experiment_graph(
            self.n_nodes, self.target_parents, self.seed
        )
        
        # Generate sample pool - completely empty for Pure AFA
        self.sample_pool = generate_afa_sample_pool(
            self.bn, self.adj_matrix, self.n_nodes, self.max_samples, self.seed
        )
        
        # Generate test dataset with missing values (traditional approach for evaluation)
        self.test_dataset = generate_afa_test_dataset(
            self.bn, self.adj_matrix, self.n_nodes, self.test_samples, 
            self.missing_rate, self.seed + 1000
        )
        
        setup_time = time.time() - start_time
        logger.info(f"VOI data setup completed in {setup_time:.2f}s")
        logger.info(f"Sample pool: {len(self.sample_pool)} completely empty samples (Pure AFA)")
        logger.info(f"Test dataset: {len(self.test_dataset)} samples")
        
    def run_node_level_experiment(self, node_policy):
        """
        Run progressive experiment with node-level policy (AFA or Random FA).
        
        Args:
            node_policy: Policy that selects individual node observations (AFA or Random FA)
            
        Returns:
            List of results at each budget step
        """
        logger.info(f"Running node-level experiment with policy: {node_policy}")
        
        if self.sample_pool is None:
            raise ValueError("Must call setup() before running experiments")
            
        results = []
        
        # Get budget sequence (cumulative node observations)
        budget_sequence = node_policy.get_budget_sequence()
        current_budget = 0
        
        for target_budget in budget_sequence:
            nodes_to_select = target_budget - current_budget
            
            logger.info(f"Budget step: {target_budget} total nodes ({nodes_to_select} new selections)")
            
            # Select nodes based on policy type
            if nodes_to_select > 0:
                if isinstance(node_policy, AFAPolicy):
                    # VOI-based selection (requires trained model)
                    if hasattr(self.neural_model, 'model') and self.neural_model.model is not None:
                        selected_observations = node_policy.select_nodes_with_voi(
                            self.sample_pool, nodes_to_select, 
                            self.neural_model.model, self.bn, self.adj_matrix, self.n_nodes
                        )
                    else:
                        # No trained model yet - select randomly for first step
                        logger.warning("No trained model for VOI - using random selection for this step")
                        selected_observations = [(0, 0, 0.0)]  # Placeholder
                        
                elif isinstance(node_policy, RandomFAPolicy):
                    # Random selection
                    selected_observations = node_policy.select_nodes_randomly(
                        self.sample_pool, nodes_to_select, self.n_nodes
                    )
                else:
                    raise ValueError(f"Unknown node-level policy type: {type(node_policy)}")
                
                # Apply observations to sample pool
                node_policy.apply_observations(self.sample_pool, selected_observations)
                
                logger.info(f"Applied {len(selected_observations)} node observations")
            
            # Get samples with observations (Pure AFA approach)
            training_samples = get_samples_with_observations(self.sample_pool)
            n_training_samples = len(training_samples)
            logger.info(f"Training on {n_training_samples} samples with observations")
            
            if n_training_samples == 0:
                logger.warning("No training samples available - skipping this budget step")
                continue
            
            # Train and evaluate neural model
            neural_start = time.time()
            self.neural_model.reset()
            # Use self-supervised training for AFA experiments to avoid cheating
            use_afa = isinstance(node_policy, (AFAPolicy, RandomFAPolicy))
            self.neural_model.train(training_samples, self.bn, self.adj_matrix, self.n_nodes, use_afa=use_afa)
            neural_results = self.neural_model.evaluate(self.test_dataset, self.bn, self.adj_matrix, self.n_nodes)
            neural_time = time.time() - neural_start
            
            # Train and evaluate domain model
            domain_start = time.time()
            self.domain_model.reset()
            self.domain_model.train(training_samples, self.bn, self.adj_matrix, self.n_nodes)
            domain_results = self.domain_model.evaluate(self.test_dataset, self.bn, self.n_nodes)
            domain_time = time.time() - domain_start
            
            # Store results
            step_result = {
                'budget': target_budget,  # Total node observations
                'n_training_samples': n_training_samples,
                'n_node_observations': target_budget,
                'neural_kl': neural_results.get('mean_kl', float('inf')),
                'neural_failed_rate': neural_results.get('failed_rate', 1.0),
                'neural_time': neural_time,
                'domain_kl': domain_results.get('mean_kl', float('inf')),
                'domain_failed_rate': domain_results.get('failed_rate', 1.0),
                'domain_time': domain_time
            }
            
            results.append(step_result)
            current_budget = target_budget
            
            logger.info(f"Budget {target_budget} results:")
            logger.info(f"  Neural: KL={step_result['neural_kl']:.4f}, time={neural_time:.1f}s")
            logger.info(f"  Domain: KL={step_result['domain_kl']:.4f}, time={domain_time:.1f}s")
            
        return results
    
    def run_traditional_experiment(self, policy):
        """
        Run experiment with traditional example-level policy (e.g., RandomExample).
        
        Args:
            policy: Traditional policy that selects complete samples
            
        Returns:
            List of results at each budget step
        """
        logger.info(f"Running traditional experiment with policy: {policy}")
        
        if self.sample_pool is None:
            raise ValueError("Must call setup() before running experiments")
            
        results = []
        
        for budget, training_data in policy.observe_progressively(self.sample_pool):
            logger.info(f"Budget {budget}: Training on {len(training_data)} samples")
            
            # Train and evaluate neural model
            neural_start = time.time()
            self.neural_model.reset()
            self.neural_model.train(training_data, self.bn, self.adj_matrix, self.n_nodes)
            neural_results = self.neural_model.evaluate(self.test_dataset, self.bn, self.adj_matrix, self.n_nodes)
            neural_time = time.time() - neural_start
            
            # Train and evaluate domain model
            domain_start = time.time()
            self.domain_model.reset()
            self.domain_model.train(training_data, self.bn, self.adj_matrix, self.n_nodes)
            domain_results = self.domain_model.evaluate(self.test_dataset, self.bn, self.n_nodes)
            domain_time = time.time() - domain_start
            
            # Store results
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
            
            results.append(step_result)
            
            logger.info(f"Budget {budget} results:")
            logger.info(f"  Neural: KL={step_result['neural_kl']:.4f}, time={neural_time:.1f}s")
            logger.info(f"  Domain: KL={step_result['domain_kl']:.4f}, time={domain_time:.1f}s")
            
        return results
    
    def run_multi_policy_experiment(self, policies):
        """
        Run experiment with multiple policies (mix of traditional and AFA).
        
        Args:
            policies: List of policies to compare
            
        Returns:
            Dict mapping policy names to results
        """
        logger.info(f"Running multi-policy VOI experiment with {len(policies)} policies")
        
        all_results = {}
        
        for policy in policies:
            policy_start = time.time()
            
            # Check if this is a node-level policy (AFA or Random FA)
            if isinstance(policy, (AFAPolicy, RandomFAPolicy)):
                results = self.run_node_level_experiment(policy)
            else:
                results = self.run_traditional_experiment(policy)
                
            policy_time = time.time() - policy_start
            
            all_results[policy.name] = {
                'results': results,
                'total_time': policy_time,
                'config': self.config.copy()
            }
            
            logger.info(f"Policy {policy.name} completed in {policy_time:.1f}s")
            
            # Reset for next policy (fresh sample pool)
            if hasattr(policy, 'observed_nodes'):
                policy.observed_nodes.clear()
            self.setup()  # Regenerate fresh sample pool
            
        return all_results


def run_voi_experiment_suite(node_sizes, target_parents=1.0, missing_rate=0.4, 
                           max_samples=3000, test_samples=250, policies=None):
    """
    Run VOI experiments across multiple graph sizes.
    
    Args:
        node_sizes: List of graph sizes to test
        target_parents: Target parents per node
        missing_rate: Missing rate for test data
        max_samples: Maximum training samples
        test_samples: Number of test samples
        policies: List of policies to test (mix of traditional and AFA)
        
    Returns:
        Dict mapping (n_nodes, policy_name) to results
    """
    if policies is None:
        policies = [
            RandomFAPolicy(start_budget=10, nodes_per_cycle=150, max_budget=1000),
            AFAPolicy(start_budget=10, nodes_per_cycle=150, max_budget=1000)
        ]
        
    logger.info(f"Running VOI experiment suite: {len(node_sizes)} graph sizes, {len(policies)} policies")
    
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
        
        experiment = ProgressiveVOIExperiment(config)
        experiment.setup()
        
        results = experiment.run_multi_policy_experiment(policies)
        
        # Store results with node size key
        for policy_name, policy_results in results.items():
            key = (n_nodes, policy_name)
            all_results[key] = policy_results
            
    logger.info(f"\\nVOI experiment suite completed: {len(all_results)} configurations")
    
    return all_results