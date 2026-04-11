#!/usr/bin/env python3
"""
CLI script for evaluating posterior predictions against ground truth.

Usage:
    python stan/scripts/evaluate_predictions.py \\
        --mcmc-dir   OUTPUT/domain_model/runs/my_run \\
        --data-bundle OUTPUT/generated_data/my_run/data_bundle.json \\
        --output-dir OUTPUT/domain_model/eval
"""

import argparse
import json
import pandas as pd
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from STAN.stan_code.pipeline.predictives import evaluate_predictives, save_predictives
from STAN.stan_code.pipeline.bundle import GroundTruthBundle
from STAN.stan_code.pipeline.configs import DataGenConfig, AnnotatorSplitConfig
from STAN.stan_code.pipeline.io import new_run_dir, save_json
from dataclasses import fields
import cmdstanpy


def load_config_dict(configs_path: Path) -> dict:
    """
    Load config from configs.json and return a flat dict with K, J, I, D, C, temperature.
    Works for both item-split (DataGenConfig) and annotator-split (AnnotatorSplitConfig).
    """
    with open(configs_path) as f:
        configs_data = json.load(f)

    dg = configs_data.get("datagen", configs_data)

    if "J_train" in dg:
        # Annotator-split
        valid_keys = {f.name for f in fields(AnnotatorSplitConfig)}
        dg_filtered = {k: v for k, v in dg.items() if k in valid_keys}
        cfg = AnnotatorSplitConfig(**dg_filtered)
        return {
            "K": cfg.K,
            "J": cfg.J,
            "I": cfg.I,
            "D": cfg.D if cfg.D is not None else 8,
            "C": cfg.C,
            "temperature": cfg.temperature if cfg.temperature is not None else 1.0,
        }
    else:
        # Item-split
        valid_keys = {f.name for f in fields(DataGenConfig)}
        dg_filtered = {k: v for k, v in dg.items() if k in valid_keys}
        if "kappa" not in dg_filtered and "alpha_dirichlet" in dg:
            dg_filtered["kappa"] = dg["alpha_dirichlet"]
        cfg = DataGenConfig(**dg_filtered)
        K_val = getattr(cfg, "K_val", 0)
        return {
            "K": cfg.K_train + K_val + cfg.K_test,
            "K_train": cfg.K_train,
            "K_val": K_val,
            "K_test": cfg.K_test,
            "J": cfg.J,
            "I": cfg.I,
            "D": cfg.D if cfg.D is not None else 8,
            "C": cfg.C,
            "temperature": cfg.temperature if cfg.temperature is not None else 1.0,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate posterior predictions against ground truth"
    )

    parser.add_argument("--mcmc-dir", required=True,
                        help="Directory containing MCMC CSV files")
    parser.add_argument("--data-bundle", required=True,
                        help="Path to data_bundle.json")
    parser.add_argument("--output-dir", default="OUTPUT/domain_model/eval",
                        help="Output directory for evaluation results")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom run name (default: auto-generated)")
    parser.add_argument("--csv-pattern", default="domain_model-*.csv",
                        help="Pattern for MCMC CSV files")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed results")
    parser.add_argument("--overwrite-existing-data", action="store_true",
                        help="Overwrite existing output directory if it exists")

    args = parser.parse_args()

    # ── Load bundle ────────────────────────────────────────────────────────────
    print(f"Loading data bundle from {args.data_bundle}")
    with open(args.data_bundle) as f:
        bundle_data = json.load(f)
    bundle = GroundTruthBundle.from_dict(bundle_data)

    # ── Load config ────────────────────────────────────────────────────────────
    # Look for configs.json next to the bundle file
    bundle_dir = Path(args.data_bundle).parent
    configs_path = bundle_dir / "configs.json"
    if not configs_path.exists():
        raise FileNotFoundError(f"configs.json not found at {configs_path}")

    config = load_config_dict(configs_path)
    print(f"Config: K={config['K']}, J={config['J']}, I={config['I']}, "
          f"D={config['D']}, C={config['C']}")

    # ── Output directory ───────────────────────────────────────────────────────
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        potential = output_root / args.run_name
        if potential.exists() and args.overwrite_existing_data:
            print(f"\033[91mWARNING: Overwriting existing output directory: {potential}\033[0m")
            shutil.rmtree(potential)

    output_dir = new_run_dir(args.output_dir, run_name=args.run_name)
    print(f"Output directory: {output_dir}")

    # ── Load MCMC fit ──────────────────────────────────────────────────────────
    print(f"Loading MCMC results from {args.mcmc_dir}")
    mcmc_dir = Path(args.mcmc_dir)
    csv_files = list(mcmc_dir.glob(args.csv_pattern))
    if not csv_files:
        print(f"Error: No CSV files found in {mcmc_dir} matching {args.csv_pattern}")
        sys.exit(1)
    print(f"Found {len(csv_files)} CSV files")

    try:
        fit = cmdstanpy.from_csv([str(f) for f in csv_files])
    except Exception as e:
        print(f"Error loading MCMC fit: {e}")
        sys.exit(1)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print(f"\nMissing ratings (total):      {len(bundle.missing_ratings)}")
    print(f"Missing ratings (test only):  "
          f"{len(bundle.missing_ratings_indexes_in_test_instance)}")
    print(f"Missing pairwise:             {len(bundle.missing_pairwise)}")

    print(f"\nEvaluating predictions...")
    try:
        results = evaluate_predictives(fit, bundle, config)

        print("Evaluation completed successfully!")
        save_predictives(output_dir, results)
        print(f"Results saved to {output_dir}")

        print(f"\n=== EVALUATION RESULTS ===")
        print(f"\nRating Predictions (test missing only):")
        print(f"  Accuracy:          {results.metrics['rating_missing_accuracy']:.3f}")
        print(f"  MAE:               {results.metrics['rating_missing_mae']:.3f}")
        print(f"  Log-likelihood:    {results.metrics['rating_missing_log_likelihood']:.3f}")
        print(f"  Calibration error: {results.metrics['rating_missing_calibration_error']:.3f}")
        print(f"  N missing (test):  {results.metrics['n_missing_ratings']}")
        print(f"  N observed:        {results.metrics.get('n_observed_ratings', 0)}")

        print(f"\nPairwise Predictions (missing):")
        print(f"  Accuracy:       {results.metrics['pairwise_missing_accuracy']:.3f}")
        print(f"  Log-likelihood: {results.metrics['pairwise_missing_log_likelihood']:.3f}")
        print(f"  AUC:            {results.metrics['pairwise_missing_auc']:.3f}")
        print(f"  N missing:      {results.metrics['n_missing_pairwise']}")

        print(f"\nLog-likelihood Summary (observed):")
        print(f"  Ratings:  {results.metrics['log_lik_ratings_obs_mean']:.3f} "
              f"± {results.metrics['log_lik_ratings_obs_std']:.3f}")
        print(f"  Pairwise: {results.metrics['log_lik_pairwise_obs_mean']:.3f} "
              f"± {results.metrics['log_lik_pairwise_obs_std']:.3f}")
        print(f"  Total:    {results.metrics['total_log_lik_mean']:.3f} "
              f"± {results.metrics['total_log_lik_std']:.3f}")

        if args.verbose:
            print(f"\nDetailed Metrics:")
            for key, value in results.metrics.items():
                print(f"  {key}: {value}")

        eval_config = {
            "mcmc_dir":    str(args.mcmc_dir),
            "data_bundle": str(args.data_bundle),
            "csv_pattern": args.csv_pattern,
            "config":      config,
        }
        save_json(eval_config, output_dir / "evaluation_config.json")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
