#!/usr/bin/env python3
"""
CLI script for generating synthetic data using Stan.

Usage:
    # Generate both train and test instances simultaneously
    PYTHONPATH=. python stan/scripts/generate_data.py --output-dir runs/both --K-train 10 --K-test 10 --I 3 --J 9
    
    # Ablation: disable third annotator (deprecated in generator)
    PYTHONPATH=. python stan/scripts/generate_data.py --disable-third-annotator --output-dir runs/no_third --K-train 10 --K-test 10 --I 3 --J 9
"""

import argparse
import json
import shutil
from pathlib import Path

from stan.pipeline.configs import DataGenConfig
from stan.pipeline.data_gen import generate_data
from stan.pipeline.io import new_run_dir, save_configs, save_bundle


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data using Stan")
    
    # Data dimensions
    parser.add_argument("--K-train", type=int, default=10, help="Number of items in training instance")
    parser.add_argument("--K-test", type=int, default=10, help="Number of items in test instance")
    parser.add_argument("--I", type=int, default=10, help="Number of attributes")
    parser.add_argument("--J", type=int, default=9, help="Number of annotators")
    parser.add_argument("--D", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--C", type=int, default=5, help="Number of rating categories")
    parser.add_argument("--d-annotator", type=int, default=None,
                       help="Annotator embedding dimension (default: D). Lower values give low-rank factorization.")
    parser.add_argument("--use-factored-annotator", action="store_true", default=True,
                       help="Use new factored annotator model V_ij = v_i + u_j * M_i (default: True)")
    parser.add_argument("--use-spherical-annotator", action="store_true",
                       help="Use old spherical annotator model V_ij ~ N(v_i, sigma^2) (disables factored)")
    parser.add_argument("--derive-thresholds-from-annotator", action="store_true",
                       help="Derive rating thresholds from annotator embedding u_j (consistent annotator style, fewer d.f.)")
    
    # Observation protocol controls
    # Deprecated in generator; keep flags for backward CLI compatibility (ignored)
    parser.add_argument("--enable-third-annotator", action="store_true", default=True,
                       help="[deprecated] Enable third annotator for disagreement > 1")
    parser.add_argument("--disable-third-annotator", action="store_true",
                       help="[deprecated] Disable third annotator (ablation)")
    parser.add_argument("--enable-pairwise-rankings", action="store_true", default=True,
                       help="Enable pairwise rankings (ablation)")
    parser.add_argument("--disable-pairwise-rankings", action="store_true",
                       help="Disable pairwise rankings (ablation)")
    
    # Generation parameters
    parser.add_argument(
        "--pairwise-cap-per-item",
        type=int,
        default=10,
        help="Max pairwise comparisons per item",
    )
    parser.add_argument(
        "--sigma-annotator",
        type=float,
        default=0.5,
        help="Annotator preference noise",
    )
    parser.add_argument(
        "--sigma-measurement",
        type=float,
        default=0.1,
        help="Measurement noise standard deviation",
    )
    # Legacy name was --alpha-dirichlet; keep it as a deprecated alias for backward compatibility.
    parser.add_argument(
        "--kappa",
        "--alpha-dirichlet",
        dest="kappa",
        type=float,
        default=2.0,
        help="Dirichlet concentration parameter (previously --alpha-dirichlet)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for pairwise ranking generation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--factor-decay", type=float, default=None,
                       help="CP tensor decomposition: T_d = factor_decay^(d-1). Only for tensor_data_generation.stan.")
    parser.add_argument("--use-log-scores", action="store_true",
                       help="Apply log() to raw CP scores before binning (misspecifies Stan's linear likelihood)")
    parser.add_argument("--use-logistic-link", action="store_true",
                       help="Use inv_logit instead of Phi for binning (misspecifies Stan's probit link)")
    parser.add_argument("--use-normal-loadings", action="store_true",
                       help="Use N(0,1) loadings instead of Exp(1) for TCP model (allows cancellation)")

    # Observation protocol
    parser.add_argument("--observation-protocol", type=str, default="tie_breaking",
                       choices=["tie_breaking", "mcar", "extended_rankings"],
                       help="Observation protocol: tie_breaking (sequential MAR), mcar (IID missing completely at random), extended_rankings (pairwise for all pairs)")
    parser.add_argument("--mcar-missing-rate", type=float, default=0.5,
                       help="Missing rate for MCAR protocol (default: 0.5 = 50%% missing)")
    parser.add_argument("--pairwise-observation-rate", type=float, default=1.0,
                       help="For tie_breaking: fraction of missing pairwise rankings to observe (default: 1.0 = all missing remain missing, 0.3 = 30%% of missing become observed)")

    # Axis-invariance controls for easy-data sanity experiments
    parser.add_argument(
        "--hold-I-constant",
        action="store_true",
        help="Force ratings to be invariant to criteria/attributes I (no I-dependence)",
    )
    parser.add_argument(
        "--hold-J-constant",
        action="store_true",
        help="Force ratings to be invariant to annotators J (no J-dependence)",
    )
    parser.add_argument(
        "--hold-K-constant",
        action="store_true",
        help="Force ratings to be invariant to items K (no K-dependence)",
    )
    
    # Output
    parser.add_argument("--output-dir", type=str, default="OUTPUT/generated_data",
                       help="Output directory for generated data")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Custom run name (default: auto-generated)")
    parser.add_argument("--stan-file", type=str, default=None,
                       help="Path to iclr_data_generation.stan (default: models/iclr_data_generation.stan)")
    parser.add_argument("--overwrite-existing-data", action="store_true",
                       help="Overwrite existing output directory if it exists")
    
    args = parser.parse_args()
    
    # Handle ablation flags
    enable_pairwise_rankings = args.enable_pairwise_rankings and not args.disable_pairwise_rankings
    use_factored_annotator = args.use_factored_annotator and not args.use_spherical_annotator

    # Create configuration
    config = DataGenConfig(
        K_train=args.K_train,
        K_test=args.K_test,
        I=args.I,
        J=args.J,
        D=args.D,
        C=args.C,
        d_annotator=args.d_annotator,
        use_factored_annotator=use_factored_annotator,
        derive_thresholds_from_annotator=args.derive_thresholds_from_annotator,
        enable_pairwise_rankings=enable_pairwise_rankings,
        pairwise_cap_per_item=args.pairwise_cap_per_item,
        sigma_annotator=args.sigma_annotator,
        sigma_measurement=args.sigma_measurement,
        alpha_dirichlet=args.kappa,
        temperature=args.temperature,
        observation_protocol=args.observation_protocol,
        mcar_missing_rate=args.mcar_missing_rate,
        pairwise_observation_rate=args.pairwise_observation_rate,
        seed=args.seed,
        hold_I_constant=args.hold_I_constant,
        hold_J_constant=args.hold_J_constant,
        hold_K_constant=args.hold_K_constant,
        factor_decay=args.factor_decay,
        use_log_scores=args.use_log_scores,
        use_logistic_link=args.use_logistic_link,
        use_normal_loadings=args.use_normal_loadings,
    )
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Handle --overwrite-existing-data flag: remove existing directory if it exists
    if args.run_name:
        potential_run_dir = output_path / args.run_name
        if potential_run_dir.exists() and args.overwrite_existing_data:
            print("\033[91mWARNING: Overwriting existing data directory: {}\033[0m".format(potential_run_dir))
            shutil.rmtree(potential_run_dir)
    
    run_dir = new_run_dir(output_path, run_name=args.run_name)
    
    print(f"Generating data with config: {config}")
    print(f"Output directory: {run_dir}")
    
    # Generate data using Stan
    bundle = generate_data(config, stan_file=args.stan_file)
    
    # Save configuration and data
    save_configs(run_dir, datagen=config)
    
    # Convert numpy arrays to lists for JSON serialization
    bundle_dict = {
        "embeddings": bundle.embeddings.tolist(),
        "mean_preferences": bundle.mean_preferences.tolist(),
        "annotator_preferences": bundle.annotator_preferences.tolist(),
        "rating_probs": bundle.rating_probs.tolist(),
        "rating_cumprobs": bundle.rating_cumprobs.tolist(),
        "rating_thresholds_z": bundle.rating_thresholds_z.tolist(),
        "base_scores": bundle.base_scores.tolist(),
        "all_ratings": bundle.all_ratings,
        "all_pairwise": bundle.all_pairwise,
        "observed_ratings": bundle.observed_ratings,
        "missing_ratings": bundle.missing_ratings,
        "observed_pairwise": bundle.observed_pairwise,
        "missing_pairwise": bundle.missing_pairwise,
        "missing_ratings_indexes_in_test_instance": bundle.missing_ratings_indexes_in_test_instance,
        "stats": bundle.stats,
        "train_posterior_rating_probs": (bundle.train_posterior_rating_probs.tolist() if bundle.train_posterior_rating_probs is not None else None),
        "test_posterior_rating_probs": (bundle.test_posterior_rating_probs.tolist() if bundle.test_posterior_rating_probs is not None else None),
    }
    
    save_bundle(run_dir, bundle_dict)
    
    # Print summary
    print("\nData generation complete!")
    print(f"Total ratings: {bundle.stats['total_ratings']}")
    print(f"Observed ratings: {bundle.stats['observed_ratings']}")
    print(f"Missing ratings: {bundle.stats['missing_ratings']}")
    print(f"Observation rate: {bundle.stats['observation_rate']:.2%}")
    print(f"Total pairwise: {bundle.stats['total_pairwise']}")
    print(f"Observed pairwise: {bundle.stats['observed_pairwise']}")
    print(f"Missing pairwise: {bundle.stats['missing_pairwise']}")


if __name__ == "__main__":
    main()
