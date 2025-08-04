"""
Multi-graph experiment runner for progressive imputation.

Runs experiments across multiple graph instances for statistical averaging.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict

from .progressive_experiment import ProgressiveExperiment

logger = logging.getLogger(__name__)

class MultiGraphExperiment:
    """
    Experiment runner that averages results across multiple graph instances.
    """
    
    def __init__(self, n_graphs=10):
        """
        Initialize multi-graph experiment.
        
        Args:
            n_graphs: Number of graph instances to average over
        """
        self.n_graphs = n_graphs
        logger.info(f"Initialized multi-graph experiment with {n_graphs} graphs per configuration")
    
    def run_multi_graph_experiment(self, base_config, policies):
        """
        Run experiment across multiple graph instances.
        
        Args:
            base_config: Base configuration dict
            policies: List of policies to test
            
        Returns:
            Dict with averaged results and statistics
        """
        logger.info(f"Running multi-graph experiment: {self.n_graphs} graphs")
        
        # Collect results from all graph instances
        all_results = []
        
        for graph_idx in range(self.n_graphs):
            logger.info(f"\n{'='*50}")
            logger.info(f"GRAPH INSTANCE {graph_idx + 1}/{self.n_graphs}")
            logger.info(f"{'='*50}")
            
            # Create config with unique seed for this graph
            config = base_config.copy()
            config['seed'] = base_config['seed'] + graph_idx * 10000
            
            # Run experiment on this graph instance
            experiment = ProgressiveExperiment(config)
            experiment.setup()
            
            graph_results = experiment.run_multi_policy_experiment(policies)
            all_results.append(graph_results)
            
            logger.info(f"Graph {graph_idx + 1} completed")
        
        # Aggregate results across all graphs
        aggregated_results = self._aggregate_results(all_results, base_config)
        
        logger.info(f"Multi-graph experiment completed: {self.n_graphs} graphs averaged")
        
        return aggregated_results
    
    def _aggregate_results(self, all_results, base_config):
        """
        Aggregate results across multiple graph instances.
        
        Args:
            all_results: List of result dicts from each graph
            base_config: Base configuration
            
        Returns:
            Dict with mean, std, and individual results
        """
        logger.debug("Aggregating results across graph instances...")
        
        # Organize results by policy
        policy_results = defaultdict(list)
        
        for graph_results in all_results:
            for policy_name, policy_data in graph_results.items():
                policy_results[policy_name].append(policy_data)
        
        # Compute statistics for each policy
        aggregated = {}
        
        for policy_name, policy_data_list in policy_results.items():
            logger.debug(f"Aggregating results for policy: {policy_name}")
            
            # Extract budget steps (should be same across all graphs)
            budget_steps = [len(data['results']) for data in policy_data_list]
            if len(set(budget_steps)) > 1:
                logger.warning(f"Inconsistent budget steps for {policy_name}: {budget_steps}")
            
            n_steps = min(budget_steps)
            
            # Aggregate metrics at each budget step
            aggregated_steps = []
            
            for step_idx in range(n_steps):
                step_metrics = {
                    'neural_kl': [],
                    'domain_kl': [],
                    'neural_failed_rate': [],
                    'domain_failed_rate': [],
                    'neural_time': [],
                    'domain_time': [],
                    'budget': None,
                    'n_training_samples': None
                }
                
                for graph_data in policy_data_list:
                    if step_idx < len(graph_data['results']):
                        step_result = graph_data['results'][step_idx]
                        
                        step_metrics['neural_kl'].append(step_result['neural_kl'])
                        step_metrics['domain_kl'].append(step_result['domain_kl'])
                        step_metrics['neural_failed_rate'].append(step_result['neural_failed_rate'])
                        step_metrics['domain_failed_rate'].append(step_result['domain_failed_rate'])
                        step_metrics['neural_time'].append(step_result['neural_time'])
                        step_metrics['domain_time'].append(step_result['domain_time'])
                        
                        # These should be the same across graphs
                        if step_metrics['budget'] is None:
                            step_metrics['budget'] = step_result['budget']
                            step_metrics['n_training_samples'] = step_result['n_training_samples']
                
                # Compute mean and std for each metric
                aggregated_step = {
                    'budget': step_metrics['budget'],
                    'n_training_samples': step_metrics['n_training_samples'],
                    'neural_kl_mean': np.mean(step_metrics['neural_kl']),
                    'neural_kl_std': np.std(step_metrics['neural_kl']),
                    'domain_kl_mean': np.mean(step_metrics['domain_kl']),
                    'domain_kl_std': np.std(step_metrics['domain_kl']),
                    'neural_failed_rate_mean': np.mean(step_metrics['neural_failed_rate']),
                    'neural_failed_rate_std': np.std(step_metrics['neural_failed_rate']),
                    'domain_failed_rate_mean': np.mean(step_metrics['domain_failed_rate']),
                    'domain_failed_rate_std': np.std(step_metrics['domain_failed_rate']),
                    'neural_time_mean': np.mean(step_metrics['neural_time']),
                    'neural_time_std': np.std(step_metrics['neural_time']),
                    'domain_time_mean': np.mean(step_metrics['domain_time']),
                    'domain_time_std': np.std(step_metrics['domain_time']),
                    # Keep individual values for further analysis
                    'neural_kl_values': step_metrics['neural_kl'],
                    'domain_kl_values': step_metrics['domain_kl']
                }
                
                aggregated_steps.append(aggregated_step)
            
            # Compute total time statistics
            total_times = [data['total_time'] for data in policy_data_list]
            
            aggregated[policy_name] = {
                'results': aggregated_steps,
                'total_time_mean': np.mean(total_times),
                'total_time_std': np.std(total_times),
                'total_time_values': total_times,
                'config': base_config.copy(),
                'n_graphs': self.n_graphs,
                'individual_results': policy_data_list  # Keep for detailed analysis
            }
            
            logger.debug(f"Policy {policy_name}: {len(aggregated_steps)} budget steps aggregated")
        
        return aggregated

def run_multi_graph_experiment_suite(node_sizes, target_parents=1.0, missing_rate=0.4,
                                    max_samples=3000, test_samples=250, policies=None,
                                    n_graphs=10):
    """
    Run multi-graph experiments across multiple node sizes.
    
    Args:
        node_sizes: List of graph sizes to test
        target_parents: Target parents per node
        missing_rate: Missing rate for test data
        max_samples: Maximum training samples
        test_samples: Number of test samples
        policies: List of policies to test
        n_graphs: Number of graph instances per configuration
        
    Returns:
        Dict mapping (n_nodes, policy_name) to aggregated results
    """
    if policies is None:
        from policies.random_example_policy import RandomExamplePolicy
        policies = [RandomExamplePolicy()]
    
    logger.info(f"Running multi-graph experiment suite:")
    logger.info(f"  Node sizes: {node_sizes}")
    logger.info(f"  Graphs per size: {n_graphs}")
    logger.info(f"  Total experiments: {len(node_sizes) * n_graphs}")
    
    all_results = {}
    
    for n_nodes in node_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"NODE SIZE: {n_nodes} nodes ({n_graphs} graph instances)")
        logger.info(f"{'='*60}")
        
        base_config = {
            'n_nodes': n_nodes,
            'target_parents': target_parents,
            'missing_rate': missing_rate,
            'max_samples': max_samples,
            'test_samples': test_samples,
            'seed': 42 + n_nodes * 1000  # Base seed for this node size
        }
        
        # Run multi-graph experiment
        multi_graph_runner = MultiGraphExperiment(n_graphs)
        node_results = multi_graph_runner.run_multi_graph_experiment(base_config, policies)
        
        # Store results with node size key
        for policy_name, policy_results in node_results.items():
            key = (n_nodes, policy_name)
            all_results[key] = policy_results
            
            # Log summary
            final_result = policy_results['results'][-1]
            logger.info(f"Node size {n_nodes}, Policy {policy_name}:")
            logger.info(f"  Final Neural KL: {final_result['neural_kl_mean']:.4f} ± {final_result['neural_kl_std']:.4f}")
            logger.info(f"  Final Domain KL: {final_result['domain_kl_mean']:.4f} ± {final_result['domain_kl_std']:.4f}")
            logger.info(f"  Total time: {policy_results['total_time_mean']:.1f} ± {policy_results['total_time_std']:.1f}s")
    
    logger.info(f"\nMulti-graph experiment suite completed: {len(all_results)} configurations")
    
    return all_results