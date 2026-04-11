#!/usr/bin/env python3
"""
CLI script for generating synthetic data using the annotator-split Stan model.

In this setting all annotators share the same K items, but are partitioned into
train / val / test groups. Every annotator in each group rates every item.
Pairwise rankings are always disabled.

Usage:
    PYTHONPATH=. python stan/scripts/generate_data_annotator_split.py \\
        --output-dir runs/annot_split \\
        --K 20 --J-train 6 --J-val 3 --J-test 3 --I 3 --C 5
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from STAN.stan_code.pipeline.configs import AnnotatorSplitConfig
from STAN.stan_code.pipeline.data_gen import generate_data_annotator_split
from STAN.stan_code.pipeline.io import new_run_dir, save_configs, save_bundle, save_json


def main():
    # ========== 1. Parse CLI ==========
    parser = argparse.ArgumentParser(
        description="Generate synthetic data using the annotator-split Stan model."
    )

    # ---------- Output ----------
    parser.add_argument("--output-dir", type=str, default="OUTPUT/generated_data_annotator_split",
                        help="Output directory for generated data")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom run name (default: auto-generated)")
    parser.add_argument("--stan-file", type=str, default=None,
                        help="Path to .stan file (overrides default annotator_split_generation.stan)")
    parser.add_argument("--overwrite-existing-data", action="store_true",
                        help="Overwrite existing output directory if it exists")

    # ---------- Core dimensions ----------
    parser.add_argument("--K", type=int, default=20,
                        help="Number of shared items (same across all instances)")
    parser.add_argument("--J-train", type=int, default=6,
                        help="Number of training annotators")
    parser.add_argument("--J-val", type=int, default=3,
                        help="Number of validation annotators")
    parser.add_argument("--J-test", type=int, default=3,
                        help="Number of test annotators")
    parser.add_argument("--I", type=int, default=3,
                        help="Number of attributes")
    parser.add_argument("--C", type=int, default=5,
                        help="Number of rating categories")
    parser.add_argument("--D", type=int, default=64,
                        help="Item/attribute embedding dimension")
    parser.add_argument("--d-annotator", type=int, default=None,
                        help="Annotator embedding dimension (default: same as D)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # ---------- Generative model ----------
    parser.add_argument("--sigma-annotator", type=float, default=0.5,
                        help="Annotator preference noise")
    parser.add_argument("--sigma-measurement", type=float, default=0.1,
                        help="Measurement noise standard deviation")
    parser.add_argument("--kappa", type=float, default=2.0,
                        help="Dirichlet concentration for rating thresholds")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature (unused without pairwise, kept for config parity)")
    parser.add_argument("--use-factored-annotator", action="store_true", default=True,
                        help="Use factored annotator model (default: on)")
    parser.add_argument("--no-factored-annotator", dest="use_factored_annotator",
                        action="store_false",
                        help="Use spherical annotator model instead")
    parser.add_argument("--derive-thresholds-from-annotator", action="store_true", default=False,
                        help="Derive rating thresholds from annotator embedding")

    # ---------- Observation protocol ----------
    parser.add_argument("--observation-protocol", type=str, default="mcar",
                        choices=["mcar"],
                        help="Observation protocol (only mcar supported for annotator-split)")
    parser.add_argument("--mcar-missing-rate", type=float, default=0.5,
                        help="Fraction of ratings to mark missing under MCAR (default: 0.5)")
    parser.add_argument("--enable-pairwise-rankings", action="store_true",
                        help="(ignored for annotator-split; pairwise always disabled)")

    args = parser.parse_args()

    # ========== 2. Build AnnotatorSplitConfig ==========
    d_annotator = args.d_annotator if args.d_annotator is not None else args.D

    config = AnnotatorSplitConfig(
        K=args.K,
        J_train=args.J_train,
        J_val=args.J_val,
        J_test=args.J_test,
        I=args.I,
        C=args.C,
        D=args.D,
        d_annotator=d_annotator,
        sigma_annotator=args.sigma_annotator,
        sigma_measurement=args.sigma_measurement,
        kappa=args.kappa,
        temperature=args.temperature,
        use_factored_annotator=1 if args.use_factored_annotator else 0,
        derive_thresholds_from_annotator=1 if args.derive_thresholds_from_annotator else 0,
        observation_protocol=args.observation_protocol,
        mcar_missing_rate=args.mcar_missing_rate,
        seed=args.seed,
    )

    # ========== 3. Output directory ==========
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.run_name:
        potential_run_dir = output_path / args.run_name
        if potential_run_dir.exists() and args.overwrite_existing_data:
            print("\033[91mWARNING: Overwriting existing data directory: {}\033[0m".format(potential_run_dir))
            shutil.rmtree(potential_run_dir)

    run_dir = new_run_dir(output_path, run_name=args.run_name)

    print(f"Generating annotator-split data with config: {config}")
    print(f"  Total annotators: {config.J} (train={config.J_train}, val={config.J_val}, test={config.J_test})")
    print(f"  Shared items: {config.K}")
    print(f"Output directory: {run_dir}")

    # ========== 4. Run data generation ==========
    bundle = generate_data_annotator_split(config, stan_file=args.stan_file)

    # ========== 5. Save outputs ==========
    from dataclasses import asdict
    datagen_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else config.__dict__.copy()
    # J is a property, not a field — add it explicitly for reference
    datagen_dict["J"] = config.J
    save_configs(run_dir, datagen=datagen_dict)

    stan_data = config.to_stan_data()
    save_json(stan_data, run_dir / "stan_data.json")

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
    if bundle.val_posterior_rating_probs is not None:
        bundle_dict["val_posterior_rating_probs"] = bundle.val_posterior_rating_probs.tolist()
    if bundle.test_posterior_rating_probs is not None:
        bundle_dict["test_posterior_rating_probs"] = bundle.test_posterior_rating_probs.tolist()

    if bundle.extra_ground_truth:
        for key, value in bundle.extra_ground_truth.items():
            bundle_dict[key] = value.tolist() if isinstance(value, np.ndarray) else value

    save_bundle(run_dir, bundle_dict)

    # ========== 6. Summary ==========
    s = bundle.stats
    print("\nData generation complete!")
    print(f"Total ratings:    {s['total_ratings']}")
    print(f"Observed ratings: {s['observed_ratings']}  ({s['observation_rate']:.2%})")
    print(f"  Train: {s['train_observed']}/{s['train_ratings']} ({s['train_observation_rate']:.2%})")
    print(f"  Val:   {s['val_observed']}/{s['val_ratings']} ({s['val_observation_rate']:.2%})")
    print(f"  Test:  {s['test_observed']}/{s['test_ratings']} ({s['test_observation_rate']:.2%})")


if __name__ == "__main__":
    main()