#!/usr/bin/env python3
"""
CLI script for running MCMC inference with the domain model.

Usage:
    python stan/scripts/run_inference.py --data-bundle generated_data/run_20250923_230222/data_bundle.json --output-dir runs/inference_test
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
import sys

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stan.pipeline.inference import InferenceConfig, run_mcmc_inference
from stan.pipeline.bundle import GroundTruthBundle
from stan.pipeline.io import new_run_dir, save_bundle, save_configs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inference.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run MCMC inference with domain model")
    
    # Required arguments
    parser.add_argument("--data-bundle", required=True, help="Path to data bundle JSON file")
    parser.add_argument("--output-dir", default="OUTPUT/domain_model/runs", help="Output directory for results")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name (default: auto-generated timestamp)")
    
    # Data configuration
    parser.add_argument("--use-train-only", action="store_true", help="Use only training instance data")
    parser.add_argument("--use-test-only", action="store_true", help="Use only test instance data")
    
    # MCMC configuration
    parser.add_argument("--chains", type=int, default=8, help="Number of MCMC chains")
    parser.add_argument("--iter-warmup", type=int, default=500, help="Number of warmup iterations")
    parser.add_argument("--iter-sampling", type=int, default=2000, help="Number of sampling iterations")
    parser.add_argument("--seed", type=int, help="Random seed for MCMC")
    parser.add_argument("--adapt-delta", type=float, default=0.8, help="Adapt delta for NUTS")
    parser.add_argument("--max-treedepth", type=int, default=10, help="Maximum tree depth for NUTS")
    
    # Initialization strategy
    parser.add_argument("--init-strategy", choices=["random", "ground_truth", "file"], 
                       default="random", help="Initialization strategy")
    parser.add_argument("--init-file", help="Path to initialization file (if init-strategy=file)")
    
    # Model configuration
    parser.add_argument("--stan-file", help="Path to domain_model.stan (defaults to models/domain_model.stan)")
    parser.add_argument("--override-D", type=int, default=None,
                       help="Override embedding dimension D for Stan (e.g. use half of data-gen D)")
    parser.add_argument("--override-sigma-annotator", type=float, default=None,
                       help="Override sigma_annotator for Stan")
    parser.add_argument("--override-sigma-measurement", type=float, default=None,
                       help="Override sigma_measurement for Stan")
    parser.add_argument("--override-alpha-dirichlet", type=float, default=None,
                       help="Override alpha_dirichlet for Stan")
    parser.add_argument("--override-temperature", type=float, default=None,
                       help="Override temperature for Stan")
    parser.add_argument("--use-dist", action="store_true",
                       help="Use distributional labels (rating_dists) instead of hard labels. "
                            "Requires a dist bundle (observed_ratings with rating_dist field). "
                            "Defaults to stan_dist_model.stan.")
    
    # Other options
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument("--overwrite-existing-data", action="store_true", help="Overwrite existing output directory if it exists")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_train_only and args.use_test_only:
        print("Error: Cannot use both --use-train-only and --use-test-only")
        sys.exit(1)
    
    if args.init_strategy == "file" and not args.init_file:
        print("Error: --init-file required when --init-strategy=file")
        sys.exit(1)
    
    # Load data bundle
    logger.info(f"Loading data bundle from {args.data_bundle}")
    print(f"Loading data bundle from {args.data_bundle}")
    with open(args.data_bundle, 'r') as f:
        bundle_data = json.load(f)
    
    bundle = GroundTruthBundle.from_dict(bundle_data)
    logger.info(f"Loaded bundle with {len(bundle.missing_ratings)} missing ratings and {len(bundle.missing_pairwise)} missing pairwise")
    
    # Create output directory
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Handle --overwrite-existing-data flag: remove existing directory if it exists
    if args.run_name:
        potential_run_dir = output_root / args.run_name
        if potential_run_dir.exists() and args.overwrite_existing_data:
            print("\033[91mWARNING: Overwriting existing output directory: {}\033[0m".format(potential_run_dir))
            shutil.rmtree(potential_run_dir)
    
    output_dir = new_run_dir(output_root, run_name=args.run_name)
    logger.info(f"Created output directory: {output_dir}")
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
    
    # Extract data generation config from bundle stats and configs
    from stan.pipeline.configs import DataGenConfig
    stats = bundle.stats
    
    # Try to load from configs.json first
    data_bundle_dir = Path(args.data_bundle).parent
    configs_path = data_bundle_dir / "configs.json"
    
    if configs_path.exists():
        with open(configs_path, 'r') as f:
            configs_data = json.load(f)
        datagen_config = configs_data["datagen"]
        data_config = DataGenConfig(
            K_train=datagen_config["K_train"],
            K_test=datagen_config["K_test"],
            I=datagen_config["I"],
            J=datagen_config["J"],
            D=datagen_config.get("D", 8),
            C=datagen_config["C"],
            sigma_annotator=datagen_config.get("sigma_annotator", 0.5),
            sigma_measurement=datagen_config.get("sigma_measurement", 0.1),
            alpha_dirichlet=datagen_config.get("alpha_dirichlet", 10.0),
            temperature=datagen_config.get("temperature", 1.0),
        )
    else:
        raise ValueError(f"Configs file not found at {configs_path}")

    # Override parameters if requested
    if args.override_D is not None:
        print(f"\n  Overriding D: {data_config.D} -> {args.override_D}")
        data_config.D = args.override_D
    if args.override_sigma_annotator is not None:
        print(f"\n  Overriding sigma_annotator: {data_config.sigma_annotator} -> {args.override_sigma_annotator}")
        data_config.sigma_annotator = args.override_sigma_annotator
    if args.override_sigma_measurement is not None:
        print(f"\n  Overriding sigma_measurement: {data_config.sigma_measurement} -> {args.override_sigma_measurement}")
        data_config.sigma_measurement = args.override_sigma_measurement
    if args.override_alpha_dirichlet is not None:
        print(f"\n  Overriding alpha_dirichlet: {data_config.alpha_dirichlet} -> {args.override_alpha_dirichlet}")
        data_config.alpha_dirichlet = args.override_alpha_dirichlet
    if args.override_temperature is not None:
        print(f"\n  Overriding temperature: {data_config.temperature} -> {args.override_temperature}")
        data_config.temperature = args.override_temperature

    print(f"\nData Configuration:")
    print(f"  K_train: {data_config.K_train}")
    print(f"  K_test: {data_config.K_test}")
    print(f"  I: {data_config.I}")
    print(f"  J: {data_config.J}")
    print(f"  D: {data_config.D}")
    print(f"  C: {data_config.C}")
    
    # Save augmented configs (for evaluate_predictions.py when fields are absent from configs.json)
    augmented_configs = {
        "datagen": {
            "K_train": data_config.K_train,
            "K_test":  data_config.K_test,
            "I":       data_config.I,
            "J":       data_config.J,
            "D":       data_config.D,
            "C":       data_config.C,
            "sigma_annotator":   data_config.sigma_annotator,
            "sigma_measurement": data_config.sigma_measurement,
            "alpha_dirichlet":   data_config.alpha_dirichlet,
            "temperature":       data_config.temperature,
        }
    }
    with open(output_dir / "augmented_configs.json", "w") as f:
        json.dump(augmented_configs, f, indent=2)

    # Run MCMC inference
    logger.info("Starting MCMC inference")
    print(f"\nStarting MCMC inference...")
    try:
        if args.use_dist:
            #########################################################
            # Dist mode: all ratings use soft expected log-likelihood.
            # Requires a dist bundle where each observed rating has a
            # 'rating_dist' field (simplex over C categories).
            # Human ratings have one-hot rating_dist; LLM ratings have
            # actual distributions.  Both are handled uniformly.
            #########################################################
            from stan.pipeline.inference import compile_domain_model

            default_stan_dist = str(
                Path(__file__).resolve().parents[2] / "models" / "stan_dist_model.stan"
            )
            stan_file_dist = args.stan_file or default_stan_dist
            print(f"Dist mode: Stan model = {stan_file_dist}")

            K = data_config.K_train + data_config.K_test
            observed = bundle.observed_ratings
            missing  = bundle.missing_ratings

            # Build rating_dists — convert int value to one-hot if rating_dist absent
            C = data_config.C
            def _to_dist(r):
                if "rating_dist" in r:
                    d = r["rating_dist"]
                    s = sum(d)
                    return [x / s for x in d]   # normalise for safety
                else:
                    oh = [0.0] * C
                    oh[r["value"] - 1] = 1.0
                    return oh

            stan_data = {
                "K": K,
                "I": data_config.I,
                "J": data_config.J,
                "D": data_config.D,
                "C": C,
                "N_ratings":           len(observed),
                "rating_attributes":   [r["attribute"] for r in observed],
                "rating_annotators":   [r["annotator"]  for r in observed],
                "rating_items":        [r["item"]        for r in observed],
                "rating_dists":        [_to_dist(r)      for r in observed],
                "N_missing_ratings":              len(missing),
                "missing_rating_attributes":      [r["attribute"] for r in missing],
                "missing_rating_annotators":      [r["annotator"]  for r in missing],
                "missing_rating_items":           [r["item"]        for r in missing],
                "sigma_annotator":   data_config.sigma_annotator,
                "sigma_measurement": data_config.sigma_measurement,
                "alpha_dirichlet":   data_config.alpha_dirichlet,
                "temperature":       data_config.temperature,
            }
            print(f"  N_ratings={len(observed)}  N_missing={len(missing)}  K={K}")

            model = compile_domain_model(stan_file_dist)
            fit = model.sample(
                data=stan_data,
                chains=inference_config.chains,
                iter_warmup=inference_config.iter_warmup,
                iter_sampling=inference_config.iter_sampling,
                seed=inference_config.seed,
                adapt_delta=inference_config.adapt_delta,
                max_treedepth=inference_config.max_treedepth,
                inits=1.0,
                show_progress=inference_config.show_progress,
                show_console=True,
            )
        else:
            #########################################################
            fit = run_mcmc_inference(
                bundle=bundle,
                config=data_config,
                inference_config=inference_config,
                stan_file=args.stan_file,
                use_train_only=args.use_train_only,
                use_test_only=args.use_test_only
            )
            #########################################################

        logger.info("MCMC inference completed successfully")
        print("MCMC inference completed successfully!")

        # Save the fit object
        fit_path = output_dir / "mcmc_fit.csv"
        fit.save_csvfiles(str(output_dir))
        logger.info(f"MCMC samples saved to {output_dir}")
        print(f"MCMC samples saved to {fit_path}")

        # Print diagnostics
        print(f"\nDiagnostics:")
        print(f"  Divergent transitions: {fit.divergences}")

        if fit.divergences.sum() > 0:
            logger.warning(f"{fit.divergences.sum()} divergent transitions detected!")
            print(f"  WARNING: {fit.divergences.sum()} divergent transitions detected!")
            print(f"  Consider increasing adapt_delta or max_treedepth")
        else:
            logger.info("No divergent transitions - good mixing!")
            print(f"  No divergent transitions - good mixing!")

    except Exception as e:
        logger.error(f"Error during MCMC inference: {e}")
        print(f"Error during MCMC inference: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
