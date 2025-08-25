#!/usr/bin/env python3
"""
Clean entry point for progressive imputation experiments.

Provides command-line interface for running comprehensive experiments comparing
neural imputers against domain-specific EM baselines across multiple graph sizes.
"""

import argparse
import logging
import os
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Import our clean modules
from experiments.experiment_runner import run_experiment_suite, save_experiment_results
from experiments.policies import RandomExamplePolicy
from utils.visualization import create_experiment_report
from utils.runtime_plotting import create_runtime_analysis_report
from utils.step_progression_plots import create_step_progression_report


def setup_logging(output_dir: Path, log_level: str = "INFO", enable_debug_file: bool = False) -> None:
    """
    Setup structured logging to separate files.
    
    Creates log files based on configuration:
    - info.log: INFO level messages (also to stdout)  
    - debug.log: DEBUG level messages (only if enable_debug_file=True)
    - error.log: ERROR level messages with full tracebacks
    
    Args:
        output_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_debug_file: Whether to create debug.log file (saves space)
    """
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Info log file (INFO and above) 
    info_handler = logging.FileHandler(logs_dir / "info.log")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(info_handler)
    
    # Debug log file (DEBUG and above) - only if enabled
    if enable_debug_file:
        debug_handler = logging.FileHandler(logs_dir / "debug.log")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(debug_handler)
    
    # Error log file (ERROR only)
    error_handler = logging.FileHandler(logs_dir / "error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Set specific logger levels to reduce noise
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    if enable_debug_file:
        logger.info(f"Logging initialized: level={log_level}, output_dir={output_dir}, debug_file=enabled")
        logger.debug(f"Log files created in {logs_dir}/")
    else:
        logger.info(f"Logging initialized: level={log_level}, output_dir={output_dir}, debug_file=disabled")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for experiment configuration.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Progressive Imputation Experiments with Neural Transformers vs Domain EM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Graph configuration
    parser.add_argument(
        '--node-sizes', type=int, nargs='+', default=[5, 10, 15],
        help='Graph sizes (number of nodes) to test'
    )
    parser.add_argument(
        '--target-parents', type=float, default=1.0,
        help='Target parents per node for graph generation'
    )
    
    # Missing data configuration
    parser.add_argument(
        '--missing-rates', type=float, nargs='+', default=[0.4],
        help='Missing data rates for test samples'
    )
    
    # Model configuration
    parser.add_argument(
        '--imputer-sizes', choices=['Tiny', 'Small', 'Large'], nargs='+', 
        default=['Large'], help='Neural imputer model sizes to test'
    )
    
    # Training configuration
    parser.add_argument(
        '--max-samples', type=int, default=3000,
        help='Maximum training samples available to policies'
    )
    parser.add_argument(
        '--test-samples', type=int, default=250,
        help='Number of test samples for evaluation'
    )
    parser.add_argument(
        '--start-examples', type=int, default=10,
        help='Starting number of training examples'
    )
    parser.add_argument(
        '--increment', type=int, default=150,
        help='Increment in training examples per progressive step'
    )
    
    # Experiment configuration
    parser.add_argument(
        '--n-graphs', type=int, default=1,
        help='Number of random graph instances per configuration'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir', type=Path, 
        default=Path(__file__).parent / "outputs",
        help='Output directory for results, plots, and logs'
    )
    parser.add_argument(
        '--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO', help='Logging level'
    )
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='Enable debug.log file creation (saves disk space when disabled)'
    )
    parser.add_argument(
        '--save-results', action='store_true', default=True,
        help='Save detailed results to pickle/json files'
    )
    parser.add_argument(
        '--use-likelihood-selection', action='store_true', default=False,
        help='Use log-likelihood for EM model selection (default: iteration-based, safer)'
    )
    
    # New visualization options
    parser.add_argument(
        '--plot-runtimes', action='store_true',
        help='Include runtime performance plots in visualization'
    )
    parser.add_argument(
        '--plot-step-progression', action='store_true', 
        help='Create step-by-step progression plots showing learning dynamics'
    )
    
    return parser.parse_args()


def create_experiment_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Create experiment configuration dictionary from parsed arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Configuration dictionary for experiments
    """
    return {
        'node_sizes': args.node_sizes,
        'target_parents': args.target_parents,
        'missing_rates': args.missing_rates,
        'imputer_sizes': args.imputer_sizes,
        'max_samples': args.max_samples,
        'test_samples': args.test_samples,
        'start_examples': args.start_examples,
        'increment': args.increment,
        'n_graphs': args.n_graphs,
        'seed': args.seed,
        'output_dir': str(args.output_dir),
        'log_level': args.log_level,
        'save_results': args.save_results,
        'use_likelihood_selection': args.use_likelihood_selection,
        'plot_runtimes': args.plot_runtimes,
        'plot_step_progression': args.plot_step_progression
    }


def create_observation_policies(config: Dict[str, Any]) -> List[RandomExamplePolicy]:
    """
    Create observation policies for progressive experiments.
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        List of observation policies
    """
    policies = [
        RandomExamplePolicy(
            start_examples=config['start_examples'],
            increment=config['increment'],
            max_examples=config['max_samples'],
            seed=config['seed']
        )
    ]
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created {len(policies)} observation policies")
    for policy in policies:
        logger.debug(f"Policy: {policy}")
        
    return policies


def run_experiments(config: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    """
    Run comprehensive progressive imputation experiments.
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        Complete experimental results
        
    Raises:
        Exception: Any experimental failures bubble up for debugging
    """
    logger = logging.getLogger(__name__)
    
    # Create observation policies
    policies = create_observation_policies(config)
    
    # Log experiment configuration
    logger.info("="*80)
    logger.info("PROGRESSIVE IMPUTATION EXPERIMENTS")
    logger.info("="*80)
    
    logger.info("Configuration:")
    logger.info(f"  Node sizes: {config['node_sizes']}")
    logger.info(f"  Missing rates: {config['missing_rates']}")
    logger.info(f"  Imputer sizes: {config['imputer_sizes']}")
    logger.info(f"  Training samples: {config['start_examples']} to {config['max_samples']} (increment: {config['increment']})")
    logger.info(f"  Test samples: {config['test_samples']}")
    logger.info(f"  Random graphs: {config['n_graphs']} per configuration")
    logger.info(f"  Random seed: {config['seed']}")
    
    total_experiments = len(config['node_sizes']) * len(config['missing_rates']) * len(config['imputer_sizes'])
    logger.info(f"  Total experiments: {total_experiments}")
    
    # Run experiment suite for each missing rate
    logger.info("\\nStarting experimental runs...")
    
    all_results = {}
    
    for missing_rate in config['missing_rates']:
        logger.info(f"\\nRunning experiments for missing rate: {missing_rate}")
        
        results = run_experiment_suite(
            node_sizes=config['node_sizes'],
            target_parents=config['target_parents'],
            missing_rate=missing_rate,
            max_samples=config['max_samples'],
            test_samples=config['test_samples'],
            policies=policies,
            imputer_sizes=config['imputer_sizes'],
            n_graphs=config['n_graphs']
        )
        
        # Store results with missing rate as part of the key structure
        for key, value in results.items():
            # Create new key: (n_nodes, policy_imputer, missing_rate)
            extended_key = (key[0], key[1], missing_rate)
            value['missing_rate'] = missing_rate
            all_results[extended_key] = value
    
    logger.info(f"\\nExperimental runs completed: {len(all_results)} configurations")
    
    return all_results


def save_results(results: Dict[Any, Dict[str, Any]], config: Dict[str, Any]) -> None:
    """
    Save experimental results to structured output files.
    
    Args:
        results: Experimental results from run_experiments
        config: Experiment configuration dictionary
    """
    if not config['save_results']:
        return
        
    logger = logging.getLogger(__name__)
    output_dir = Path(config['output_dir'])
    
    # Save detailed results
    logger.info("Saving experimental results...")
    save_experiment_results(results, output_dir / "results")
    
    # Save configuration
    config_path = output_dir / "results" / "experiment_config.json"
    with open(config_path, 'w') as f:
        # Convert Path objects to strings for JSON serialization
        json_config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
        json.dump(json_config, f, indent=2)
        
    logger.info(f"Configuration saved to {config_path}")


def create_visualizations(results: Dict[Any, Dict[str, Any]], config: Dict[str, Any]) -> None:
    """
    Create comprehensive visualization report with timestamp organization.
    
    Args:
        results: Experimental results from run_experiments
        config: Experiment configuration dictionary
    """
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    output_dir = Path(config['output_dir'])
    
    # Create timestamp folder for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}"
    
    # Add configuration details to folder name
    node_sizes_str = "_".join(map(str, config['node_sizes']))
    missing_rates_str = "_".join([f"{mr:.1f}" for mr in config['missing_rates']])
    imputer_sizes_str = "_".join(config['imputer_sizes'])
    
    experiment_folder = f"{experiment_name}_nodes_{node_sizes_str}_missing_{missing_rates_str}_imputers_{imputer_sizes_str}_graphs_{config['n_graphs']}"
    plots_dir = output_dir / "plots" / experiment_folder
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating visualization report in: {plots_dir}")
    
    # Group results by missing rate and create separate reports
    for missing_rate in config['missing_rates']:
        logger.info(f"Creating visualizations for missing rate: {missing_rate}")
        
        # Create missing rate subfolder
        missing_rate_dir = plots_dir / f"missing_rate_{missing_rate:.1f}"
        missing_rate_dir.mkdir(exist_ok=True)
        
        # Filter results for this missing rate
        filtered_results = {}
        for key, value in results.items():
            if len(key) == 3 and key[2] == missing_rate:  # (n_nodes, policy_imputer, missing_rate)
                # Convert back to original key format for visualization functions
                original_key = (key[0], key[1])  # (n_nodes, policy_imputer)
                filtered_results[original_key] = value
        
        if filtered_results:
            # Create visualizations for all node sizes together
            create_experiment_report(
                results=filtered_results,
                output_dir=str(missing_rate_dir),
                missing_rate=missing_rate
            )
            
            # Add new visualization modules if requested
            if config.get('plot_runtimes'):
                create_runtime_analysis_report(
                    results=filtered_results,
                    output_dir=str(missing_rate_dir),
                    missing_rate=missing_rate
                )
            
            if config.get('plot_step_progression'):
                create_step_progression_report(
                    results=filtered_results,
                    output_dir=str(missing_rate_dir),
                    missing_rate=missing_rate
                )
            
            # Create individual node size subfolders
            node_sizes_in_results = set(key[0] for key in filtered_results.keys())
            for n_nodes in sorted(node_sizes_in_results):
                node_dir = missing_rate_dir / f"nodes_{n_nodes}"
                node_dir.mkdir(exist_ok=True)
                
                # Filter results for this specific node size
                node_results = {k: v for k, v in filtered_results.items() if k[0] == n_nodes}
                
                if node_results:
                    create_experiment_report(
                        results=node_results,
                        output_dir=str(node_dir),
                        missing_rate=missing_rate
                    )
        else:
            logger.warning(f"No results found for missing rate: {missing_rate}")
    
    logger.info(f"Visualization report completed: {plots_dir}/")


def main() -> None:
    """
    Main entry point for progressive imputation experiments.
    
    Parses command-line arguments, sets up logging, runs experiments,
    saves results, and creates visualizations.
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Create output directories
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "plots").mkdir(exist_ok=True)
        (args.output_dir / "results").mkdir(exist_ok=True)
        
        # Setup logging
        setup_logging(args.output_dir, args.log_level, args.debug)
        
        logger = logging.getLogger(__name__)
        logger.info("Progressive imputation experiments starting...")
        
        # Create experiment configuration
        config = create_experiment_config(args)
        
        # Run experiments
        results = run_experiments(config)
        
        # Save results
        save_results(results, config)
        
        # Create visualizations  
        create_visualizations(results, config)
        
        # Final summary
        logger.info("="*80)
        logger.info("EXPERIMENTS COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Logs available in: {args.output_dir / 'logs'}/")
        logger.info(f"Plots available in: {args.output_dir / 'plots'}/")
        
        if results:
            logger.info(f"Total configurations: {len(results)}")
            
            # Print brief summary
            for (n_nodes, policy_imputer_key, missing_rate), experiment_data in results.items():
                if experiment_data['results']:
                    final_result = experiment_data['results'][-1]
                    neural_kl = final_result.get('neural_kl', float('inf'))
                    domain_kl = final_result.get('domain_kl', float('inf'))
                    winner = 'Neural' if neural_kl < domain_kl else 'Domain'
                    logger.info(f"  {n_nodes} nodes, {policy_imputer_key}, missing_rate={missing_rate}: Final winner = {winner}")
        
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.warning("Experiment interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        # Logger may not be initialized if error occurs early
        try:
            logger = logging.getLogger(__name__)
            logger.error(f"Experiment failed: {e}", exc_info=True)
        except:
            print(f"FATAL ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()