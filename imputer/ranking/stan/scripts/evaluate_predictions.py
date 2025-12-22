#!/usr/bin/env python3
"""
CLI script for evaluating posterior predictions against ground truth.

Usage:
    python stan/scripts/evaluate_predictions.py --mcmc-dir runs/inference_test/run_20250923_234251 --data-bundle generated_data/run_20250923_230222/data_bundle.json --output-dir runs/evaluation_test

It use the bundle to get the missing attributes, then use the mcmc-dir to get the predictions, then evaluate the predictions against the ground truth.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import sys

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stan.pipeline.predictives import evaluate_predictives, save_predictives
from stan.pipeline.bundle import GroundTruthBundle
from stan.pipeline.io import new_run_dir, save_json
import cmdstanpy


def main():
    parser = argparse.ArgumentParser(description="Evaluate posterior predictions against ground truth")
    
    # Required arguments
    parser.add_argument("--mcmc-dir", required=True, help="Directory containing MCMC CSV files")
    parser.add_argument("--data-bundle", required=True, help="Path to data bundle JSON file")
    parser.add_argument("--output-dir", default="OUTPUT/domain_model/eval", help="Output directory for evaluation results")
    
    # Optional arguments
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name (default: auto-generated)")
    parser.add_argument("--csv-pattern", default="domain_model-*.csv", help="Pattern for MCMC CSV files")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    
    args = parser.parse_args()
    
    # Load data bundle
    print(f"Loading data bundle from {args.data_bundle}")
    with open(args.data_bundle, 'r') as f:
        bundle_data = json.load(f)
    
    bundle = GroundTruthBundle.from_dict(bundle_data)
    
    # Create output directory
    output_dir = new_run_dir(args.output_dir, run_name=args.run_name)
    print(f"Output directory: {output_dir}")
    
    # Load MCMC fit
    print(f"Loading MCMC results from {args.mcmc_dir}")
    mcmc_dir = Path(args.mcmc_dir)
    
    # Find CSV files
    csv_files = list(mcmc_dir.glob(args.csv_pattern))
    if not csv_files:
        print(f"Error: No CSV files found in {mcmc_dir} matching pattern {args.csv_pattern}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Load fit object
    try:
        # Convert Path objects to strings for cmdstanpy
        csv_file_paths = [str(f) for f in csv_files]
        fit = cmdstanpy.from_csv(csv_file_paths)
    except Exception as e:
        print(f"Error loading MCMC fit: {e}")
        sys.exit(1)
    
    # Extract configuration from data generation configs
    data_bundle_dir = Path(args.data_bundle).parent
    configs_path = data_bundle_dir / "configs.json"
    
    if configs_path.exists():
        with open(configs_path, 'r') as f:
            configs_data = json.load(f)
        datagen_config = configs_data["datagen"]
        config = {
            "K_train": datagen_config["K_train"],
            "K_test": datagen_config["K_test"],
            "I": datagen_config["I"],
            "J": datagen_config["J"],
            "D": datagen_config["D"],
            "C": datagen_config["C"],
            "temperature": datagen_config.get("temperature", 1.0),
        }
    else:
        # Fallback to bundle stats if configs.json not found
        config = {
            "K_train": bundle.stats["K_train"],
            "K_test": bundle.stats["K_test"],
            "I": 3,  # Fallback defaults
            "J": 6,
            "D": 8,
            "C": 5,
            "temperature": 1.0,
        }
    
    print(f"\nConfiguration:")
    print(f"  K_train: {config['K_train']}")
    print(f"  K_test: {config['K_test']}")
    print(f"  Missing ratings: {len(bundle.missing_ratings)}")
    print(f"  Missing pairwise: {len(bundle.missing_pairwise)}")
    
    # Evaluate predictions
    print(f"\nEvaluating predictions...")
    try:
        results = evaluate_predictives(fit, bundle, config)
        
        print("Evaluation completed successfully!")
        
        # Save results
        save_predictives(output_dir, results)
        print(f"Results saved to {output_dir}")
        
        # Print summary metrics
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"\nRating Predictions (Missing Data):")
        print(f"  Accuracy: {results.metrics['rating_missing_accuracy']:.3f}")
        print(f"  MAE: {results.metrics['rating_missing_mae']:.3f}")
        print(f"  Log-likelihood: {results.metrics['rating_missing_log_likelihood']:.3f}")
        print(f"  Calibration error: {results.metrics['rating_missing_calibration_error']:.3f}")
        print(f"  N missing: {results.metrics['n_missing_ratings']}")
        print(f"  N observed: {results.metrics.get('n_observed_ratings', 0)}")
        
        print(f"\nPairwise Predictions (Missing Data):")
        print(f"  Accuracy: {results.metrics['pairwise_missing_accuracy']:.3f}")
        print(f"  Log-likelihood: {results.metrics['pairwise_missing_log_likelihood']:.3f}")
        print(f"  AUC: {results.metrics['pairwise_missing_auc']:.3f}")
        print(f"  N missing: {results.metrics['n_missing_pairwise']}")
        print(f"  N observed: {results.metrics.get('n_observed_pairwise', 0)}")
        
        print(f"\nLog-likelihood Summary:")
        print(f"  Observed ratings: {results.metrics['log_lik_ratings_obs_mean']:.3f} ± {results.metrics['log_lik_ratings_obs_std']:.3f}")
        print(f"  Observed pairwise: {results.metrics['log_lik_pairwise_obs_mean']:.3f} ± {results.metrics['log_lik_pairwise_obs_std']:.3f}")
        print(f"  Observed pairwise accuracy: {results.metrics.get('pairwise_observed_accuracy', float('nan')):.3f}")
        print(f"  Total: {results.metrics['total_log_lik_mean']:.3f} ± {results.metrics['total_log_lik_std']:.3f}")
        
        if args.verbose:
            print(f"\nDetailed Metrics:")
            for key, value in results.metrics.items():
                print(f"  {key}: {value}")
        
        # Save configuration
        eval_config = {
            "mcmc_dir": str(args.mcmc_dir),
            "data_bundle": str(args.data_bundle),
            "csv_pattern": args.csv_pattern,
            "config": config,
        }
        save_json(eval_config, output_dir / "evaluation_config.json")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
