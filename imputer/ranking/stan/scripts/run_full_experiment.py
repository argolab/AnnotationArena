#!/usr/bin/env python3
"""
CLI script for running complete experiments: data generation, MCMC inference, and evaluation.

Usage:
    python stan/scripts/run_full_experiment.py --output-dir experiments/full_test --K-train 5 --K-test 3 --chains 2 --iter-warmup 100 --iter-sampling 100
"""

import argparse
import json
from pathlib import Path
import sys
import subprocess

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stan.pipeline.io import new_run_dir, save_json


def main():
    parser = argparse.ArgumentParser(description="Run complete experiment: data generation + MCMC + evaluation")
    
    # Data generation arguments
    parser.add_argument("--K-train", type=int, default=5, help="Number of training items")
    parser.add_argument("--K-test", type=int, default=3, help="Number of test items")
    parser.add_argument("--I", type=int, default=3, help="Number of criteria")
    parser.add_argument("--J", type=int, default=6, help="Number of annotators")
    parser.add_argument("--D", type=int, default=8, help="Embedding dimension")
    parser.add_argument("--C", type=int, default=5, help="Number of rating categories")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    # MCMC arguments
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains")
    parser.add_argument("--iter-warmup", type=int, default=1000, help="Number of warmup iterations")
    parser.add_argument("--iter-sampling", type=int, default=1000, help="Number of sampling iterations")
    parser.add_argument("--init-strategy", choices=["random", "ground_truth"], default="ground_truth", help="Initialization strategy")
    parser.add_argument("--use-train-only", action="store_true", help="Use only training instance data")
    parser.add_argument("--use-test-only", action="store_true", help="Use only test instance data")
    
    # Output arguments
    parser.add_argument("--output-dir", default="OUTPUT/domain_model", help="Output directory for experiment")
    parser.add_argument("--skip-data-gen", action="store_true", help="Skip data generation (use existing data)")
    parser.add_argument("--skip-inference", action="store_true", help="Skip MCMC inference")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation")
    
    args = parser.parse_args()
    
    # Create experiment directory
    exp_dir = new_run_dir(args.output_dir)
    print(f"Experiment directory: {exp_dir}")
    
    # Store experiment configuration
    exp_config = {
        "data_generation": {
            "K_train": args.K_train,
            "K_test": args.K_test,
            "I": args.I,
            "J": args.J,
            "D": args.D,
            "C": args.C,
            "seed": args.seed,
        },
        "mcmc_inference": {
            "chains": args.chains,
            "iter_warmup": args.iter_warmup,
            "iter_sampling": args.iter_sampling,
            "init_strategy": args.init_strategy,
            "use_train_only": args.use_train_only,
            "use_test_only": args.use_test_only,
        }
    }
    save_json(exp_config, exp_dir / "experiment_config.json")
    
    # Step 1: Data Generation
    data_dir = exp_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    if not args.skip_data_gen:
        print(f"\n=== STEP 1: DATA GENERATION ===")
        cmd = [
            "python", "stan/scripts/generate_data.py",
            "--output-dir", str(data_dir),
            "--K-train", str(args.K_train),
            "--K-test", str(args.K_test),
            "--I", str(args.I),
            "--J", str(args.J),
            "--D", str(args.D),
            "--C", str(args.C),
        ]
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Data generation failed with exit code {result.returncode}")
            sys.exit(1)
        print("Data generation completed successfully!")
    else:
        print("Skipping data generation")
    
    # Find the generated data bundle
    data_bundles = list(data_dir.glob("*/data_bundle.json"))
    if not data_bundles:
        print("Error: No data bundle found")
        sys.exit(1)
    
    data_bundle_path = data_bundles[0]  # Use the first (most recent) bundle
    print(f"Using data bundle: {data_bundle_path}")
    
    # Step 2: MCMC Inference
    inference_dir = exp_dir / "inference"
    inference_dir.mkdir(exist_ok=True)
    
    if not args.skip_inference:
        print(f"\n=== STEP 2: MCMC INFERENCE ===")
        cmd = [
            "python", "stan/scripts/run_inference.py",
            "--data-bundle", str(data_bundle_path),
            "--output-dir", str(inference_dir),
            "--chains", str(args.chains),
            "--iter-warmup", str(args.iter_warmup),
            "--iter-sampling", str(args.iter_sampling),
            "--init-strategy", args.init_strategy,
        ]
        if args.use_train_only:
            cmd.append("--use-train-only")
        if args.use_test_only:
            cmd.append("--use-test-only")
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"MCMC inference failed with exit code {result.returncode}")
            sys.exit(1)
        print("MCMC inference completed successfully!")
    else:
        print("Skipping MCMC inference")
    
    # Find the MCMC results
    mcmc_dirs = list(inference_dir.glob("*/"))
    if not mcmc_dirs:
        print("Error: No MCMC results found")
        sys.exit(1)
    
    mcmc_dir = mcmc_dirs[0]  # Use the first (most recent) result
    print(f"Using MCMC results: {mcmc_dir}")
    
    # Step 3: Evaluation
    evaluation_dir = exp_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    
    if not args.skip_evaluation:
        print(f"\n=== STEP 3: EVALUATION ===")
        cmd = [
            "python", "stan/scripts/evaluate_predictions.py",
            "--mcmc-dir", str(mcmc_dir),
            "--data-bundle", str(data_bundle_path),
            "--output-dir", str(evaluation_dir),
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Evaluation failed with exit code {result.returncode}")
            sys.exit(1)
        print("Evaluation completed successfully!")
    else:
        print("Skipping evaluation")
    
    # Find the evaluation results
    eval_dirs = list(evaluation_dir.glob("*/"))
    if not eval_dirs:
        print("Error: No evaluation results found")
        sys.exit(1)
    
    eval_dir = eval_dirs[0]  # Use the first (most recent) result
    print(f"Using evaluation results: {eval_dir}")
    
    # Load and display final results
    metrics_path = eval_dir / "predictive_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Rating Predictions (Missing Data):")
        print(f"  Accuracy: {metrics['rating_missing_accuracy']:.3f}")
        print(f"  MAE: {metrics['rating_missing_mae']:.3f}")
        print(f"  Log-likelihood: {metrics['rating_missing_log_likelihood']:.3f}")
        print(f"  Calibration error: {metrics['rating_missing_calibration_error']:.3f}")
        print(f"  N missing: {metrics['n_missing_ratings']}")
        print(f"  N observed: {metrics.get('n_observed_ratings', 0)}")
        
        print(f"\nPairwise Predictions (Missing Data):")
        print(f"  Accuracy: {metrics['pairwise_missing_accuracy']:.3f}")
        print(f"  Log-likelihood: {metrics['pairwise_missing_log_likelihood']:.3f}")
        print(f"  AUC: {metrics['pairwise_missing_auc']:.3f}")
        print(f"  N missing: {metrics['n_missing_pairwise']}")
        print(f"  N observed: {metrics.get('n_observed_pairwise', 0)}")
        
        print(f"\nLog-likelihood Summary:")
        print(f"  Observed ratings: {metrics['log_lik_ratings_obs_mean']:.3f} ± {metrics['log_lik_ratings_obs_std']:.3f}")
        print(f"  Observed pairwise: {metrics['log_lik_pairwise_obs_mean']:.3f} ± {metrics['log_lik_pairwise_obs_std']:.3f}")
        print(f"  Total: {metrics['total_log_lik_mean']:.3f} ± {metrics['total_log_lik_std']:.3f}")
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
