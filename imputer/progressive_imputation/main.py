"""
Main entry point for progressive imputation experiments.
"""

import logging
import os
from policies.random_example_policy import RandomExamplePolicy
from experiments.multi_graph_experiment import run_multi_graph_experiment_suite
from visualization.multi_graph_plots import create_multi_graph_experiment_report

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'logs')
    
    # Create logs directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, 'progressive_imputation.log')
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
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
    """Main function to run progressive imputation experiments."""
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("PROGRESSIVE IMPUTATION EXPERIMENTS")
    logger.info("="*60)
    
    # Experiment configurations
    node_sizes = [5]  # Start with 5 nodes
    target_parents = 1.5
    missing_rates = [0.5, 0.7]  # Train/test on same missing rates
    imputer_sizes = ["Tiny", "Small", "Large"]  # Different model architectures
    max_samples = 2500
    test_samples = 250
    n_graphs = 10  # Number of graphs to average over
    
    # Create policies
    policies = [
        RandomExamplePolicy(
            start_examples=10, 
            increment=250, 
            max_examples=max_samples,
            seed=42
        )
    ]
    
    logger.info(f"Configuration:")
    logger.info(f"  Node sizes: {node_sizes}")
    logger.info(f"  Target parents: {target_parents}")
    logger.info(f"  Missing rates: {missing_rates}")
    logger.info(f"  Imputer sizes: {imputer_sizes}")
    logger.info(f"  Max samples: {max_samples}")
    logger.info(f"  Test samples: {test_samples}")
    logger.info(f"  Graphs per configuration: {n_graphs}")
    logger.info(f"  Total experiments: {len(missing_rates)} × {len(imputer_sizes)} = {len(missing_rates) * len(imputer_sizes)}")
    
    try:
        # Run experiments for each missing rate and imputer size combination
        all_experiments = {}
        
        for missing_rate in missing_rates:
            logger.info(f"\n{'='*80}")
            logger.info(f"MISSING RATE: {missing_rate}")
            logger.info(f"{'='*80}")
            
            for imputer_size in imputer_sizes:
                logger.info(f"\n{'-'*60}")
                logger.info(f"IMPUTER SIZE: {imputer_size}")
                logger.info(f"{'-'*60}")
                
                # Create experiment identifier
                experiment_key = f"missing_{missing_rate}_imputer_{imputer_size}"
                
                # Run multi-graph experiment suite for this configuration
                results = run_multi_graph_experiment_suite(
                    node_sizes=node_sizes,
                    target_parents=target_parents,
                    missing_rate=missing_rate,
                    max_samples=max_samples,
                    test_samples=test_samples,
                    policies=policies,
                    n_graphs=n_graphs,
                    imputer_size=imputer_size  # Pass imputer size
                )
                
                all_experiments[experiment_key] = {
                    'results': results,
                    'missing_rate': missing_rate,
                    'imputer_size': imputer_size
                }
                
                # Generate plots for this configuration
                logger.info(f"\nGenerating plots for {experiment_key}...")
                
                # Get the directory where this script is located  
                script_dir = os.path.dirname(os.path.abspath(__file__))
                plots_dir = os.path.join(script_dir, 'plots', f'missing_{missing_rate}', f'imputer_{imputer_size}')
                
                # Create output directories
                os.makedirs(plots_dir, exist_ok=True)
                
                # Create plots for this configuration
                create_multi_graph_experiment_report(results, output_dir=plots_dir)
                
                logger.info(f"Plots saved to {plots_dir}/")
        
        # Print comprehensive summary
        logger.info(f"\n{'='*80}")
        logger.info(f"COMPREHENSIVE EXPERIMENT SUMMARY")
        logger.info(f"{'='*80}")
        
        for experiment_key, experiment_data in all_experiments.items():
            results = experiment_data['results']
            missing_rate = experiment_data['missing_rate']
            imputer_size = experiment_data['imputer_size']
            
            logger.info(f"\nExperiment: {experiment_key}")
            logger.info(f"Missing Rate: {missing_rate}, Imputer Size: {imputer_size}")
            
            for (n_nodes, policy_name), policy_results in results.items():
                final_result = policy_results['results'][-1]  # Last budget step
                logger.info(f"  Final {imputer_size} KL: {final_result['neural_kl_mean']:.4f} ± {final_result['neural_kl_std']:.4f}")
                logger.info(f"  Final Domain KL: {final_result['domain_kl_mean']:.4f} ± {final_result['domain_kl_std']:.4f}")
                
                # Winner
                winner = imputer_size if final_result['neural_kl_mean'] < final_result['domain_kl_mean'] else 'Domain EM'
                ratio = final_result['neural_kl_mean'] / final_result['domain_kl_mean']
                logger.info(f"  Winner: {winner} (ratio: {ratio:.2f})")
        
        logger.info("\n" + "="*80)
        logger.info("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
        # Print summary
        logger.info("\\n" + "="*60)
        logger.info("MULTI-GRAPH EXPERIMENT SUMMARY")
        logger.info("="*60)
        
        for (n_nodes, policy_name), policy_results in results.items():
            logger.info(f"\\nGraph {n_nodes} nodes, Policy {policy_name} ({policy_results['n_graphs']} graphs):")
            
            final_result = policy_results['results'][-1]  # Last budget step
            logger.info(f"  Final budget: {final_result['budget']}")
            logger.info(f"  Final Neural KL: {final_result['neural_kl_mean']:.4f} ± {final_result['neural_kl_std']:.4f}")
            logger.info(f"  Final Domain KL: {final_result['domain_kl_mean']:.4f} ± {final_result['domain_kl_std']:.4f}")
            logger.info(f"  Total time: {policy_results['total_time_mean']:.1f} ± {policy_results['total_time_std']:.1f}s")
            
            # Print improvement over budget
            first_result = policy_results['results'][0]
            neural_improvement = first_result['neural_kl_mean'] / final_result['neural_kl_mean']
            domain_improvement = first_result['domain_kl_mean'] / final_result['domain_kl_mean']
            
            logger.info(f"  Neural improvement: {neural_improvement:.2f}x")
            logger.info(f"  Domain improvement: {domain_improvement:.2f}x")
            
            # Winner
            winner = 'Neural' if final_result['neural_kl_mean'] < final_result['domain_kl_mean'] else 'Domain'
            ratio = final_result['neural_kl_mean'] / final_result['domain_kl_mean']
            logger.info(f"  Winner: {winner} (ratio: {ratio:.2f})")
        
        logger.info("\\nExperiments completed successfully!")
        
        # Generate comprehensive plots
        logger.info("\\n" + "="*60)
        logger.info("GENERATING MULTI-GRAPH VISUALIZATION PLOTS")
        logger.info("="*60)
        
        try:
            # Get the directory where this script is located  
            script_dir = os.path.dirname(os.path.abspath(__file__))
            plots_dir = os.path.join(script_dir, 'plots')
            
            # Create output directories
            os.makedirs(plots_dir, exist_ok=True)
            
            # Create multi-graph plots with error bars and separate node plots
            create_multi_graph_experiment_report(results, output_dir=plots_dir)
            
            logger.info(f"\\nMulti-graph visualization completed! Check {plots_dir}/ directory for saved figures.")
            
        except Exception as plot_error:
            logger.warning(f"Plotting failed: {plot_error}")
            logger.info("Experiment results are still valid, continuing without plots...")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()