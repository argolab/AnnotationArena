#!/usr/bin/env python3
"""Run ICLR imputer experiments with mixed training and conditional evaluation."""

import argparse
import logging
import json
from pathlib import Path

from config import load_config
from experiment_runner_iclr import ExperimentRunnerICLR

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ICLR imputer experiments")

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to experiment configuration file'
    )

    parser.add_argument(
        '--masking-rate', '-m',
        type=float,
        default=0.5,
        help='Masking rate for conditional imputation (default: 0.5)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Override output directory from config'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        help='Override device from config'
    )

    parser.add_argument(
        '--generate-plots',
        action='store_true',
        help='Generate additional standalone plots (automatic after experiment)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()

def main():
    """Main function to run ICLR experiments."""
    args = parse_args()

    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting ICLR imputer experiments")

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override config parameters if specified
    if args.output_dir:
        config.base_output_dir = args.output_dir
    if args.device:
        config.device = args.device

    # Log experiment parameters
    logger.info(f"Experiment type: {config.experiment_type}")
    logger.info(f"Training instances: {config.train_instance_indices}")
    logger.info(f"Test instances: {config.test_instance_indices}")
    logger.info(f"Masking rate: {args.masking_rate}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Device: {config.device}")

    # Validate configuration
    config.validate()

    if config.experiment_type != "multi_instance":
        raise ValueError("ICLR experiments require multi_instance experiment type")

    if len(config.test_instance_indices) == 0:
        raise ValueError("ICLR experiments require at least one test instance")

    # Create experiment runner
    runner = ExperimentRunnerICLR(config)

    # Run experiment
    try:
        logger.info("Starting experiment execution")
        results = runner.run_experiment(masking_rate=args.masking_rate)

        # Log summary results
        logger.info("Experiment completed successfully!")

        # Summary statistics
        pretraining_time = results.get('pretraining_time', 0.0)
        total_time = results['total_time']

        # logger.info(f"Pretraining time: {pretraining_time:.2f} seconds")
        # logger.info(f"Total experiment time: {total_time:.2f} seconds")

        # Test instance results summary
        for test_idx in config.test_instance_indices:
            test_results = results['test_results'][test_idx]

            # Method 1: Pretrained only
            method1 = test_results['pretrained_only']
            logger.info(f"Test instance {test_idx} - Pretrained only: "
                       f"Total loss={method1['total_log_loss']:.4f}")

            # Method 2: Pretrained + Finetuned
            method2 = test_results['pretrained_finetuned']
            logger.info(f"Test instance {test_idx} - Pretrained + Finetuned: "
                       f"Total loss={method2['total_log_loss']:.4f}")

            # Method 3: No pretrain
            method3 = test_results['no_pretrain_finetuned']
            logger.info(f"Test instance {test_idx} - No pretrain: "
                       f"Total loss={method3['total_log_loss']:.4f}")

            # Method 4: Domain model (best samples)
            domain_results = test_results['domain_model']
            if domain_results:
                best_samples = max(domain_results.keys())
                best_result = domain_results[best_samples]
                logger.info(f"Test instance {test_idx} - Domain model ({best_samples} samples): "
                           f"Total loss={best_result.total_log_loss:.4f}")

        logger.info(f"Results saved to {config.output_dir}")

        # Generate additional plots if requested
        if args.generate_plots:
            logger.info("Generating additional standalone plots...")
            try:
                from iclr_visualization import ICLRResultsAnalyzer
                results_file = config.output_dir / "results" / "iclr_results.json"
                standalone_viz_dir = config.output_dir / "standalone_visualizations"

                analyzer = ICLRResultsAnalyzer(str(results_file))
                analyzer.create_comprehensive_report(standalone_viz_dir)

                logger.info(f"Standalone visualizations saved to {standalone_viz_dir}")

            except Exception as viz_error:
                logger.error(f"Failed to generate additional plots: {viz_error}")

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

    return results

if __name__ == "__main__":
    main()