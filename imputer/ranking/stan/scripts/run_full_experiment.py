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

# ANSI colors
CYAN = "\033[96m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Run complete experiment: data generation + MCMC + evaluation")
    
    # Data generation arguments
    parser.add_argument("--K-train", type=int, default=15, help="Number of training items")
    parser.add_argument("--K-test", type=int, default=15, help="Number of test items")
    parser.add_argument("--I", type=int, default=10, help="Number of criteria")
    parser.add_argument("--J", type=int, default=9, help="Number of annotators")
    parser.add_argument("--D", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--C", type=int, default=5, help="Number of rating categories")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    # MCMC arguments
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains")
    parser.add_argument("--iter-warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--iter-sampling", type=int, default=50, help="Number of sampling iterations")
    parser.add_argument("--init-strategy", choices=["random", "ground_truth"], default="random", help="Initialization strategy")
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
    print(f"{CYAN}Experiment directory: {exp_dir}{RESET}")
    
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
        print(f"\n{CYAN}=== STEP 1: DATA GENERATION ==={RESET}")
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
        
        print(f"{CYAN}Running:{RESET} {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Data generation failed with exit code {result.returncode}")
            sys.exit(1)
        print(f"{CYAN}Data generation completed successfully!{RESET}")
    else:
        print("Skipping data generation")
    
    # Find the generated data bundle
    data_bundles = list(data_dir.glob("*/data_bundle.json"))
    if not data_bundles:
        print("Error: No data bundle found")
        sys.exit(1)
    
    data_bundle_path = data_bundles[0]  # Use the first (most recent) bundle
    print(f"{CYAN}Using data bundle: {data_bundle_path}{RESET}")
    
    # Step 2: MCMC Inference (run two variants: ground_truth and random)
    inference_dir = exp_dir / "inference"
    inference_dir.mkdir(exist_ok=True)

    mcmc_variants = [
        ("ground_truth", "ground_truth"),
        ("random", "random"),
    ]
    mcmc_result_dirs = {}

    if not args.skip_inference:
        print(f"\n{CYAN}=== STEP 2: MCMC INFERENCE ==={RESET}")
        for variant_name, init_strategy in mcmc_variants:
            variant_out = inference_dir / variant_name
            variant_out.mkdir(exist_ok=True)
            cmd = [
                "python", "stan/scripts/run_inference.py",
                "--data-bundle", str(data_bundle_path),
                "--output-dir", str(variant_out),
                "--chains", str(args.chains),
                "--iter-warmup", str(args.iter_warmup),
                "--iter-sampling", str(args.iter_sampling),
                "--init-strategy", init_strategy,
            ]
            if args.use_train_only:
                cmd.append("--use-train-only")
            if args.use_test_only:
                cmd.append("--use-test-only")

            print(f"{CYAN}Running ({variant_name}):{RESET} {' '.join(cmd)}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"MCMC inference ({variant_name}) failed with exit code {result.returncode}")
                sys.exit(1)
            print(f"{CYAN}MCMC inference ({variant_name}) completed successfully!{RESET}")

            # capture the created run dir
            variant_runs = list(variant_out.glob("*/"))
            if not variant_runs:
                print(f"{CYAN}Error: No MCMC results found for variant {variant_name}{RESET}")
                sys.exit(1)
            mcmc_result_dirs[variant_name] = variant_runs[0]
    else:
        print(f"{CYAN}Skipping MCMC inference{RESET}")
        # discover most recent results for both variants
        for variant_name, _ in mcmc_variants:
            variant_out = inference_dir / variant_name
            runs = list(variant_out.glob("*/"))
            if not runs:
                print(f"{CYAN}Error: No MCMC results found for variant {variant_name}{RESET}")
                sys.exit(1)
            mcmc_result_dirs[variant_name] = runs[0]

    print(f"{CYAN}Using MCMC results:{RESET}")
    for k, v in mcmc_result_dirs.items():
        print(f"  {k}: {v}")
    
    # Step 3: Evaluation (for both variants)
    evaluation_dir = exp_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)

    eval_result_dirs = {}
    if not args.skip_evaluation:
        print(f"\n{CYAN}=== STEP 3: EVALUATION ==={RESET}")
        for variant_name in mcmc_result_dirs:
            variant_eval_out = evaluation_dir / variant_name
            variant_eval_out.mkdir(exist_ok=True)
            cmd = [
                "python", "stan/scripts/evaluate_predictions.py",
                "--mcmc-dir", str(mcmc_result_dirs[variant_name]),
                "--data-bundle", str(data_bundle_path),
                "--output-dir", str(variant_eval_out),
            ]
            print(f"{CYAN}Running ({variant_name}):{RESET} {' '.join(cmd)}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"Evaluation ({variant_name}) failed with exit code {result.returncode}")
                sys.exit(1)
            print(f"{CYAN}Evaluation ({variant_name}) completed successfully!{RESET}")

            runs = list(variant_eval_out.glob("*/"))
            if not runs:
                print(f"{CYAN}Error: No evaluation results found for variant {variant_name}{RESET}")
                sys.exit(1)
            eval_result_dirs[variant_name] = runs[0]
    else:
        print(f"{CYAN}Skipping evaluation{RESET}")
        for variant_name in mcmc_result_dirs:
            variant_eval_out = evaluation_dir / variant_name
            runs = list(variant_eval_out.glob("*/"))
            if not runs:
                print(f"{CYAN}Error: No evaluation results found for variant {variant_name}{RESET}")
                sys.exit(1)
            eval_result_dirs[variant_name] = runs[0]

    print(f"{CYAN}Using evaluation results:{RESET}")
    for k, v in eval_result_dirs.items():
        print(f"  {k}: {v}")
    
    # Load and display final results for each variant
    print(f"\n{CYAN}=== FINAL RESULTS ==={RESET}")
    for variant_name, variant_eval_dir in eval_result_dirs.items():
        metrics_path = variant_eval_dir / "predictive_metrics.json"
        if not metrics_path.exists():
            print(f"Missing metrics for {variant_name}: {metrics_path}")
            continue
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        print(f"\n{CYAN}Variant: {variant_name}{RESET}")
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
