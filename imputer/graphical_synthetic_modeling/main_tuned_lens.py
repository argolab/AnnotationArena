#!/usr/bin/env python3
"""
Tuned Lens experiments for MARFORMER.

Extends logit lens analysis by training learned affine transformations (probes)
for each transformer layer on a calibration dataset. This allows us to better
align intermediate representations with the output space.

Workflow:
1. Train base MARFORMER at budgets [50, 500, 2000]
2. Train tuned lens probes on separate calibration data (500 samples)
3. Evaluate layer-wise KL divergence with tuned transformations
4. Compare with logit lens baseline to measure improvement
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from experiments.tuned_lens_experiment_runner import run_tuned_lens_experiment_suite
from experiments.mi_policy import MechanisticInterpretabilityPolicy


def setup_logging(output_dir: Path, log_level: str = "INFO") -> None:
    """
    Setup logging for tuned lens experiments.

    Args:
        output_dir: Output directory for logs
        log_level: Logging level
    """
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler (simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # File handler (detailed format)
    file_handler = logging.FileHandler(logs_dir / "tuned_lens_experiment.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    # Error file handler
    error_handler = logging.FileHandler(logs_dir / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tuned Lens Analysis for MARFORMER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Graph configuration
    parser.add_argument('--node-sizes', type=int, nargs='+', default=[5],
                       help='Graph sizes to analyze')
    parser.add_argument('--target-parents', type=float, default=3.0,
                       help='Target average parents per node')

    # Training budgets
    parser.add_argument('--budgets', type=int, nargs='+', default=[50, 500, 2000],
                       help='Training budgets for base model')
    parser.add_argument('--test-samples', type=int, default=500,
                       help='Number of test samples for evaluation')
    parser.add_argument('--calibration-samples', type=int, default=500,
                       help='Number of calibration samples for probe training')

    # Model configuration
    parser.add_argument('--imputer-sizes', choices=['Tiny', 'Small', 'Large'],
                       nargs='+', default=['Large'],
                       help='MARFORMER model sizes to analyze')

    # Data configuration
    parser.add_argument('--missing-rates', type=float, nargs='+', default=[0.5],
                       help='Missing data rates (fraction of nodes unobserved)')
    parser.add_argument('--n-graphs', type=int, default=1,
                       help='Number of graph instances (typically 1 for MI)')

    # CPT generation
    parser.add_argument('--cpt-generation', choices=['default', 'dirichlet', 'logistic'],
                       default='logistic',
                       help='CPT generation method')
    parser.add_argument('--logistic-std', type=float, default=1.5,
                       help='Standard deviation for logistic regression CPT weights')

    # Output and reproducibility
    parser.add_argument('--output-dir', type=Path, default=Path('./OUTPUT_TUNED_LENS'),
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')

    args = parser.parse_args()

    # Validate arguments
    if any(b <= 0 for b in args.budgets):
        parser.error("All budgets must be positive")
    if args.test_samples <= 0:
        parser.error("test-samples must be positive")
    if args.calibration_samples <= 0:
        parser.error("calibration-samples must be positive")
    if any(r < 0 or r > 1 for r in args.missing_rates):
        parser.error("missing-rates must be in [0, 1]")

    return args


def print_experiment_summary(args: argparse.Namespace) -> None:
    """Print experiment configuration summary."""
    print("=" * 80)
    print("TUNED LENS EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Graph sizes:          {args.node_sizes}")
    print(f"Target parents:       {args.target_parents}")
    print(f"Training budgets:     {sorted(args.budgets)}")
    print(f"Test samples:         {args.test_samples}")
    print(f"Calibration samples:  {args.calibration_samples}")
    print(f"Model sizes:          {args.imputer_sizes}")
    print(f"Missing rates:        {args.missing_rates}")
    print(f"N graph instances:    {args.n_graphs}")
    print(f"CPT generation:       {args.cpt_generation} (std={args.logistic_std})")
    print(f"Random seed:          {args.seed}")
    print(f"Output directory:     {args.output_dir}")
    print("=" * 80)
    print()


def main() -> None:
    """Main entry point for tuned lens experiments."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        setup_logging(args.output_dir, args.log_level)
        logger = logging.getLogger(__name__)

        # Print experiment summary
        print_experiment_summary(args)
        logger.info("Starting Tuned Lens experiments...")

        # Create MI policy with specific budgets (same as logit lens)
        policy = MechanisticInterpretabilityPolicy(
            budgets=sorted(args.budgets),
            seed=args.seed
        )

        # Run experiment suite
        start_time = datetime.now()
        results = run_tuned_lens_experiment_suite(
            node_sizes=args.node_sizes,
            target_parents=args.target_parents,
            missing_rates=args.missing_rates,
            test_samples=args.test_samples,
            calibration_samples=args.calibration_samples,
            policies=[policy],
            imputer_sizes=args.imputer_sizes,
            n_graphs=args.n_graphs,
            cpt_generation=args.cpt_generation,
            logistic_std=args.logistic_std,
            output_dir=args.output_dir,
            seed=args.seed
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Print final summary
        print()
        print("=" * 80)
        print("TUNED LENS EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Total duration:       {duration:.1f}s ({duration/60:.1f} min)")
        print(f"Results saved to:     {args.output_dir}")
        print(f"Configurations:       {len(results)}")
        print(f"Budgets analyzed:     {sorted(args.budgets)}")
        print(f"Model sizes:          {args.imputer_sizes}")
        print()

        # Print per-budget summary
        if results:
            print("Per-Budget Summary:")
            print("-" * 80)
            for key, result_data in results.items():
                if result_data['results']:
                    for step_result in result_data['results']:
                        budget = step_result['budget']
                        imputer_size = step_result['imputer_size']
                        final_kl = step_result['tuned_lens_kl']
                        n_layers = step_result['n_layers']
                        layer_kl_means = step_result.get('layer_kl_means', [])

                        print(f"  Budget {budget:4d} | {imputer_size:5s} | Final KL: {final_kl:.4f} | Layers: {n_layers}")
                        if layer_kl_means is not None and len(layer_kl_means) > 0:
                            layer_kl_list = [f"{kl:.3f}" for kl in layer_kl_means]
                            print(f"            Layer KL progression: [{', '.join(layer_kl_list)}]")
            print("-" * 80)

        print()
        print("To analyze results, examine:")
        print(f"  - Layer-wise KL data:  {args.output_dir}/*/budget_*/*/tuned_lens/layer_analysis.json")
        print(f"  - Raw KL data:         {args.output_dir}/*/budget_*/*/tuned_lens/layer_kl_raw_data.pkl")
        print(f"  - Base models:         {args.output_dir}/*/budget_*/*/tuned_lens/base_model.pt")
        print(f"  - Trained probes:      {args.output_dir}/*/budget_*/*/tuned_lens/probes.pt")
        print(f"  - Logs:                {args.output_dir}/logs/")
        print()
        print("Compare with logit lens results in ./OUTPUT_MI/")
        print("=" * 80)

        logger.info("Tuned lens experiments completed successfully")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user", file=sys.stderr)
        logging.getLogger(__name__).warning("Experiment interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}", file=sys.stderr)
        logging.getLogger(__name__).error(f"Experiment failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
