"""
Main entry point for progressive imputation experiments with VOI.

This script runs experiments comparing traditional policies (RandomExample)
with Active Feature Acquisition (AFA) using Value of Information.
"""

import logging
import os
import torch
from policies.afa_policy import AFAPolicy
from policies.random_fa_policy import RandomFAPolicy
from experiments.progressive_experiment_voi import run_voi_experiment_suite
from visualization.plots import create_experiment_report

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/progressive_imputation_voi.log')
        ]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger('data').setLevel(logging.INFO)
    logging.getLogger('models').setLevel(logging.INFO) 
    logging.getLogger('policies').setLevel(logging.INFO)
    logging.getLogger('experiments').setLevel(logging.INFO)
    
    # Reduce verbosity of external libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)

def main():
    """Main function to run VOI progressive imputation experiments."""
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("PROGRESSIVE IMPUTATION EXPERIMENTS WITH VOI")
    logger.info("="*60)
    
    # Experiment configuration
    node_sizes = [5]  # Start with small graph for testing
    target_parents = 1.4
    missing_rate = 0.6
    max_samples = 3000
    test_samples = 250
    
    # Create policies to compare
    policies = [
        AFAPolicy(
            start_budget=5,
            nodes_per_cycle=500,  # Smaller for testing  
            max_budget=2500,
            seed=42
        ),
        RandomFAPolicy(
            start_budget=5,
            nodes_per_cycle=500,  # Smaller for testing  
            max_budget=2500,
            seed=42
        )
    ]
    
    logger.info(f"Configuration:")
    logger.info(f"  Node sizes: {node_sizes}")
    logger.info(f"  Target parents: {target_parents}")
    logger.info(f"  Missing rate: {missing_rate}")
    logger.info(f"  Max samples: {max_samples}")
    logger.info(f"  Test samples: {test_samples}")
    logger.info(f"  Policies: {[p.name for p in policies]}")
    
    try:
        # Run experiment suite
        results = run_voi_experiment_suite(
            node_sizes=node_sizes,
            target_parents=target_parents,
            missing_rate=missing_rate,
            max_samples=max_samples,
            test_samples=test_samples,
            policies=policies
        )
        
        # Print summary
        logger.info("\\n" + "="*60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*60)
        
        for (n_nodes, policy_name), policy_results in results.items():
            logger.info(f"\\nGraph {n_nodes} nodes, Policy {policy_name}:")
            
            final_result = policy_results['results'][-1]  # Last budget step
            logger.info(f"  Final budget: {final_result['budget']}")
            logger.info(f"  Final Neural KL: {final_result['neural_kl']:.4f}")
            logger.info(f"  Final Domain KL: {final_result['domain_kl']:.4f}")
            logger.info(f"  Total time: {policy_results['total_time']:.1f}s")
            
            # Print improvement over budget
            first_result = policy_results['results'][0]
            neural_improvement = first_result['neural_kl'] / final_result['neural_kl']
            domain_improvement = first_result['domain_kl'] / final_result['domain_kl']
            
            logger.info(f"  Neural improvement: {neural_improvement:.2f}x")
            logger.info(f"  Domain improvement: {domain_improvement:.2f}x")
            
            # Special info for node-level policies
            if policy_name in ["AFA_VOI", "Random_FA"]:
                n_samples = final_result.get('n_training_samples', 0)
                n_observations = final_result.get('n_node_observations', final_result['budget'])
                logger.info(f"  Training samples used: {n_samples}")
                logger.info(f"  Total node observations: {n_observations}")
        
        logger.info("\\nVOI experiments completed successfully!")
        
        # Generate comprehensive plots
        logger.info("\\n" + "="*60)
        logger.info("GENERATING VISUALIZATION PLOTS")
        logger.info("="*60)
        
        try:
            # Create output directories
            os.makedirs('plots', exist_ok=True)
            
            # Create plots and save to plots/ directory
            create_experiment_report(results, output_dir="plots")
            
            logger.info("\\nVisualization completed! Check plots/ directory for saved figures.")
            
        except Exception as plot_error:
            logger.warning(f"Plotting failed: {plot_error}")
            logger.info("Experiment results are still valid, continuing without plots...")
        
    except Exception as e:
        logger.error(f"VOI experiment failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()