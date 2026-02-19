#!/usr/bin/env python3
"""
CLI script for generating synthetic data using Stan.

Usage:
    # Generate both train and test instances simultaneously
    PYTHONPATH=. python stan/scripts/generate_data.py --output-dir runs/both --K-train 10 --K-test 10 --I 3 --J 9

    # Enable pairwise rankings (default is off)
    PYTHONPATH=. python stan/scripts/generate_data.py --enable-pairwise-rankings --output-dir runs/with_pairwise ...
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from stan.pipeline.configs import DataGenConfig, STAN_TYPE_REQUIRED
from stan.pipeline.data_gen import generate_data
from stan.pipeline.io import new_run_dir, save_configs, save_bundle, save_json


def _parse_stan_arg(s: str) -> tuple:
    """Parse KEY=VALUE into (key, value). Value is int, float, or str."""
    if "=" not in s:
        raise ValueError(f"Invalid --stan-arg: expected KEY=VALUE, got {s!r}")
    key, value = s.split("=", 1)
    key = key.strip()
    value = value.strip()
    if value.lower() in ("true", "1"):
        return (key, True)
    if value.lower() in ("false", "0"):
        return (key, False)
    try:
        return (key, int(value))
    except ValueError:
        pass
    try:
        return (key, float(value))
    except ValueError:
        pass
    return (key, value)


def main():
    # ========== 1. Parse CLI ==========
    parser = argparse.ArgumentParser(description="Generate synthetic data using Stan")

    # ---------- Output & model choice (most important) ----------
    parser.add_argument("--output-dir", type=str, default="OUTPUT/generated_data",
                        help="Output directory for generated data")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom run name (default: auto-generated)")
    parser.add_argument("--stan-type", type=str, default="factored-dot-product",
                        choices=["normal-noise-dot-product", "factored-dot-product", "discrete", "tensor"],
                        help="Stan model: normal-noise-dot-product, factored-dot-product, discrete, tensor.")
    parser.add_argument("--stan-file", type=str, default=None,
                        help="Path to .stan file (overrides --stan-type when set)")
    parser.add_argument("--overwrite-existing-data", action="store_true",
                        help="Overwrite existing output directory if it exists")

    # ---------- Core dimensions ----------
    parser.add_argument("--K-train", type=int, default=10, help="Number of items in training instance")
    parser.add_argument("--K-test", type=int, default=10, help="Number of items in test instance")
    parser.add_argument("--I", type=int, default=10, help="Number of attributes")
    parser.add_argument("--J", type=int, default=9, help="Number of annotators")
    parser.add_argument("--D", type=int, default=64, help="Embedding dimension (embedding types only)")
    parser.add_argument("--C", type=int, default=5, help="Number of rating categories")

    # ---------- Core behavior ----------
    parser.add_argument("--enable-pairwise-rankings", action="store_true",
                        help="Enable pairwise rankings (default: off)")
    parser.add_argument("--pairwise-cap-per-item", type=int, default=10,
                        help="Max pairwise comparisons per item")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # ---------- Observation protocol ----------
    parser.add_argument("--observation-protocol", type=str, default="tie_breaking",
                        choices=["tie_breaking", "mcar", "extended_rankings"],
                        help="Observation protocol: tie_breaking, mcar, extended_rankings")
    parser.add_argument("--mcar-missing-rate", type=float, default=0.5,
                        help="Missing rate for MCAR protocol (default: 0.5)")
    parser.add_argument("--pairwise-observation-rate", type=float, default=1.0,
                        help="For tie_breaking: fraction of missing pairwise to observe (default: 1.0)")

    # ---------- Shared generation (noise/temp; used by multiple stan types) ----------
    parser.add_argument("--sigma-annotator", type=float, default=0.5,
                        help="Annotator preference noise (embedding/tensor types)")
    parser.add_argument("--sigma-measurement", type=float, default=0.1,
                        help="Measurement noise standard deviation")
    parser.add_argument("--kappa", "--alpha-dirichlet", dest="kappa", type=float, default=2.0,
                        help="Dirichlet concentration (rating thresholds)")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature for pairwise ranking generation")

    # ---------- Stan subtype options ----------
    parser.add_argument("--stan-arg", action="append", default=None, metavar="KEY=VALUE",
                        help="Model-specific Stan data (repeatable). E.g. --stan-arg M=6 --stan-arg S=3 (discrete).")
    # Embedding types (normal-noise-dot-product, factored-dot-product); tensor uses D for all dimensions.
    parser.add_argument("--d-annotator", type=int, default=None,
                        help="Annotator embedding dimension (default: D). Normal-noise and factored-dot-product only.")
    parser.add_argument("--derive-thresholds-from-annotator", action="store_true", default=False,
                        help="Derive rating thresholds from annotator embedding (factored-dot-product only).")
    # tensor only:
    parser.add_argument("--factor-decay", type=float, default=None,
                        help="CP factor decay (tensor only).")
    
    args = parser.parse_args()

    # ========== 2. Build DataGenConfig (core + type-specific fields from args and --stan-arg) ==========
    stan_arg: dict = {}
    if args.stan_arg:
        for s in args.stan_arg:
            k, v = _parse_stan_arg(s)
            stan_arg[k] = v

    stan_type = args.stan_type
    required = STAN_TYPE_REQUIRED[stan_type]
    # Defaults from CLI for type-specific fields (overridden by --stan-arg when provided)
    defaults = {
        "D": args.D,
        "d_annotator": args.d_annotator if args.d_annotator is not None else args.D,
        "sigma_annotator": args.sigma_annotator,
        "sigma_measurement": args.sigma_measurement,
        "kappa": args.kappa,
        "temperature": args.temperature,
        "factor_decay": args.factor_decay,
    }
    if stan_type == "normal-noise-dot-product":
        defaults["use_factored_annotator"] = 0
        defaults["derive_thresholds_from_annotator"] = 0
    elif stan_type == "factored-dot-product":
        defaults["use_factored_annotator"] = 1
        defaults["derive_thresholds_from_annotator"] = 1 if args.derive_thresholds_from_annotator else 0
    type_kwargs = {k: stan_arg.get(k, defaults.get(k)) for k in required}
    if "d_annotator" in required and type_kwargs.get("d_annotator") is None:
        type_kwargs["d_annotator"] = type_kwargs.get("D") or args.D
    if "factor_decay" in required and type_kwargs.get("factor_decay") is None and args.factor_decay is not None:
        type_kwargs["factor_decay"] = args.factor_decay
    for k in list(type_kwargs):
        if type_kwargs[k] is None and k in required:
            raise ValueError(f"For stan_type={stan_type!r}, required field {k!r} is missing. Pass via CLI or --stan-arg {k}=...")

    config = DataGenConfig(
        K_train=args.K_train,
        K_test=args.K_test,
        I=args.I,
        J=args.J,
        C=args.C,
        enable_pairwise_rankings=args.enable_pairwise_rankings,
        pairwise_cap_per_item=args.pairwise_cap_per_item,
        observation_protocol=args.observation_protocol,
        mcar_missing_rate=args.mcar_missing_rate,
        pairwise_observation_rate=args.pairwise_observation_rate,
        seed=args.seed,
        stan_type=stan_type,
        **type_kwargs,
    )

    # ========== 3. Output directory ==========
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
    print(f"Stan type: {config.stan_type}")
    print(f"Output directory: {run_dir}")

    # ========== 4. Run data generation ==========
    bundle = generate_data(config, stan_file=args.stan_file)

    # ========== 5. Save outputs (configs, Stan data JSON for inference, bundle) ==========
    from dataclasses import asdict
    datagen_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else config.__dict__.copy()
    save_configs(run_dir, datagen=datagen_dict)

    stan_data = config.to_stan_data()
    save_json(stan_data, run_dir / "stan_data.json")

    # Bundle (ratings, pairwise, ground truth)
    # Dynamically include only fields that are present (not None)
    bundle_dict = {
        "all_ratings": bundle.all_ratings,
        "all_pairwise": bundle.all_pairwise,
        "observed_ratings": bundle.observed_ratings,
        "missing_ratings": bundle.missing_ratings,
        "observed_pairwise": bundle.observed_pairwise,
        "missing_pairwise": bundle.missing_pairwise,
        "missing_ratings_indexes_in_test_instance": bundle.missing_ratings_indexes_in_test_instance,
        "stats": bundle.stats,
    }
    
    # Add optional standard fields if present
    if bundle.embeddings is not None:
        bundle_dict["embeddings"] = bundle.embeddings.tolist()
    if bundle.mean_preferences is not None:
        bundle_dict["mean_preferences"] = bundle.mean_preferences.tolist()
    if bundle.annotator_preferences is not None:
        bundle_dict["annotator_preferences"] = bundle.annotator_preferences.tolist()
    if bundle.rating_probs is not None:
        bundle_dict["rating_probs"] = bundle.rating_probs.tolist()
    if bundle.rating_cumprobs is not None:
        bundle_dict["rating_cumprobs"] = bundle.rating_cumprobs.tolist()
    if bundle.rating_thresholds_z is not None:
        bundle_dict["rating_thresholds_z"] = bundle.rating_thresholds_z.tolist()
    if bundle.base_scores is not None:
        bundle_dict["base_scores"] = bundle.base_scores.tolist()
    if bundle.train_posterior_rating_probs is not None:
        bundle_dict["train_posterior_rating_probs"] = bundle.train_posterior_rating_probs.tolist()
    if bundle.test_posterior_rating_probs is not None:
        bundle_dict["test_posterior_rating_probs"] = bundle.test_posterior_rating_probs.tolist()
    
    # Add extra ground truth if present
    if bundle.extra_ground_truth:
        for key, value in bundle.extra_ground_truth.items():
            if isinstance(value, np.ndarray):
                bundle_dict[key] = value.tolist()
            else:
                bundle_dict[key] = value
    
    save_bundle(run_dir, bundle_dict)

    # ========== 6. Summary ==========
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
