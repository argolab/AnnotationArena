#!/usr/bin/env python3
"""
CLI script for running MCMC inference with the domain model.

Usage:
    python stan/scripts/run_inference.py --data-bundle generated_data/run_20250923_230222/data_bundle.json --output-dir runs/inference_test
"""

import argparse
import json
from pathlib import Path
import sys

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stan.pipeline.inference import InferenceConfig, run_mcmc_inference
from stan.pipeline.bundle import GroundTruthBundle
from stan.pipeline.io import new_run_dir, save_bundle, save_configs


def main():
    parser = argparse.ArgumentParser(description="Run MCMC inference with domain model")
    
    # Required arguments
    parser.add_argument("--data-bundle", required=True, help="Path to data bundle JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    
    # Data configuration
    parser.add_argument("--use-train-only", action="store_true", help="Use only training instance data")
    parser.add_argument("--use-test-only", action="store_true", help="Use only test instance data")
    
    # MCMC configuration
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains")
    parser.add_argument("--iter-warmup", type=int, default=1000, help="Number of warmup iterations")
    parser.add_argument("--iter-sampling", type=int, default=1000, help="Number of sampling iterations")
    parser.add_argument("--seed", type=int, help="Random seed for MCMC")
    parser.add_argument("--adapt-delta", type=float, default=0.8, help="Adapt delta for NUTS")
    parser.add_argument("--max-treedepth", type=int, default=10, help="Maximum tree depth for NUTS")
    
    # Initialization strategy
    parser.add_argument("--init-strategy", choices=["random", "ground_truth", "file"], 
                       default="random", help="Initialization strategy")
    parser.add_argument("--init-file", help="Path to initialization file (if init-strategy=file)")
    
    # Model configuration
    parser.add_argument("--stan-file", help="Path to domain_model.stan (defaults to models/domain_model.stan)")
    
    # Other options
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_train_only and args.use_test_only:
        print("Error: Cannot use both --use-train-only and --use-test-only")
        sys.exit(1)
    
    if args.init_strategy == "file" and not args.init_file:
        print("Error: --init-file required when --init-strategy=file")
        sys.exit(1)
    
    # Load data bundle
    print(f"Loading data bundle from {args.data_bundle}")
    with open(args.data_bundle, 'r') as f:
        bundle_data = json.load(f)
    
    bundle = GroundTruthBundle.from_dict(bundle_data)
    
    # Create output directory
    output_dir = new_run_dir(args.output_dir)
    print(f"Output directory: {output_dir}")
    
    # Create inference configuration
    inference_config = InferenceConfig(
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        seed=args.seed,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        init_strategy=args.init_strategy,
        init_file=args.init_file,
        show_progress=not args.no_progress
    )
    
    # Save inference configuration
    config_dict = {
        "data_bundle": str(args.data_bundle),
        "use_train_only": args.use_train_only,
        "use_test_only": args.use_test_only,
        "chains": args.chains,
        "iter_warmup": args.iter_warmup,
        "iter_sampling": args.iter_sampling,
        "seed": args.seed,
        "adapt_delta": args.adapt_delta,
        "max_treedepth": args.max_treedepth,
        "init_strategy": args.init_strategy,
        "init_file": args.init_file,
        "stan_file": args.stan_file,
    }
    save_configs(output_dir, inference=config_dict)
    
    # Print configuration
    print("\nInference Configuration:")
    print(f"  Data bundle: {args.data_bundle}")
    print(f"  Use train only: {args.use_train_only}")
    print(f"  Use test only: {args.use_test_only}")
    print(f"  Chains: {args.chains}")
    print(f"  Warmup iterations: {args.iter_warmup}")
    print(f"  Sampling iterations: {args.iter_sampling}")
    print(f"  Seed: {args.seed}")
    print(f"  Init strategy: {args.init_strategy}")
    print(f"  Adapt delta: {args.adapt_delta}")
    print(f"  Max tree depth: {args.max_treedepth}")
    
    # Extract data generation config from bundle stats
    from stan.pipeline.configs import DataGenConfig
    stats = bundle.stats
    data_config = DataGenConfig(
        K_train=stats["K_train"],
        K_test=stats["K_test"],
        I=3,  # Default values - should be extracted from bundle
        J=6,
        D=8,
        C=5,
        sigma_annotator=0.3,
        sigma_measurement=0.1,
        alpha_dirichlet=2.0,
        temperature=0.5
    )
    
    print(f"\nData Configuration:")
    print(f"  K_train: {data_config.K_train}")
    print(f"  K_test: {data_config.K_test}")
    print(f"  I: {data_config.I}")
    print(f"  J: {data_config.J}")
    print(f"  D: {data_config.D}")
    print(f"  C: {data_config.C}")
    
    # Run MCMC inference
    print(f"\nStarting MCMC inference...")
    try:
        fit = run_mcmc_inference(
            bundle=bundle,
            config=data_config,
            inference_config=inference_config,
            stan_file=args.stan_file,
            use_train_only=args.use_train_only,
            use_test_only=args.use_test_only
        )
        
        print("MCMC inference completed successfully!")
        
        # Save the fit object
        fit_path = output_dir / "mcmc_fit.csv"
        fit.save_csvfiles(str(output_dir))
        print(f"MCMC samples saved to {fit_path}")
        
        # Print diagnostics
        print(f"\nDiagnostics:")
        print(f"  Divergent transitions: {fit.divergences}")
        
        if fit.divergences.sum() > 0:
            print(f"  WARNING: {fit.divergences.sum()} divergent transitions detected!")
            print(f"  Consider increasing adapt_delta or max_treedepth")
        else:
            print(f"  No divergent transitions - good mixing!")
        
    except Exception as e:
        print(f"Error during MCMC inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
