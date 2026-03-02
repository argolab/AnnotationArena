#!/usr/bin/env python3
"""
MCMC inference using the Dirichlet observation model for LLM distributional labels.

Trains on observed ratings:
  - Human annotators  → ordinal probit hard-label likelihood: log P(c | model)
  - LLM annotators    → Dirichlet likelihood: q_n ~ Dir(alpha_llm * pi_ijk)

Predicts missing ratings (test-set human annotations) via posterior predictive.

Output is compatible with stan/scripts/evaluate_predictions.py — the generated
quantities block has the same variable names as stan_dist_model.stan.

Usage (from repo root):
    python stan/scripts/run_inference_dirichlet.py \\
        --data-bundle OUTPUT/generated_data/llm_rubric_dist/data_bundle.json \\
        --run-name    llmrubric_dir \\
        --alpha-llm   5.0
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cmdstanpy

from stan.pipeline.bundle import GroundTruthBundle
from stan.pipeline.configs import DataGenConfig
from stan.pipeline.inference import compile_domain_model
from stan.pipeline.io import new_run_dir, save_configs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("inference_dirichlet.log")],
)
logger = logging.getLogger(__name__)

# Default Stan model path (relative to repo root)
_STAN_FILE = str(Path(__file__).resolve().parents[2] / "models" / "stan_dirichlet_model.stan")


def _is_llm(r: dict) -> int:
    """Return 1 if rating has a soft LLM distribution, 0 if hard human one-hot."""
    dist = r.get("rating_dist")
    if dist is None:
        return 0
    return 1 if max(dist) < 1.0 - 1e-6 else 0


def _to_dist(r: dict, C: int, eps: float = 1e-6) -> list:
    """Return a C-simplex for this rating with no exact zeros.

    LLM ratings: use rating_dist (normalised for safety).
    Human ratings: construct one-hot from value.

    Epsilon is added to every entry before renormalising.  This is required
    for LLM distributions passed to dirichlet_lpdf — the Dirichlet log-density
    contains log(q_c) terms, so any exact zero in q causes log(0) = -inf and
    Stan fails to find a valid initial value.  Human one-hots are never passed
    to dirichlet_lpdf (routed through ordinal probit instead), but we apply
    the same treatment uniformly for consistency.
    """
    if "rating_dist" in r:
        d = list(r["rating_dist"])
        s = sum(d)
        d = [x / s for x in d]   # normalise first
    else:
        d = [0.0] * C
        d[r["value"] - 1] = 1.0
    # Add epsilon and renormalise to a proper simplex with no exact zeros
    d = [x + eps for x in d]
    s = sum(d)
    return [x / s for x in d]


def main():
    parser = argparse.ArgumentParser(
        description="Run MCMC with Dirichlet observation model for LLM labels"
    )

    # Data
    parser.add_argument("--data-bundle", required=True,
                        help="Path to dist data_bundle.json (must have rating_dist fields)")
    parser.add_argument("--output-dir", default="OUTPUT/domain_model/runs",
                        help="Root output directory")
    parser.add_argument("--run-name", default=None,
                        help="Custom run name (default: auto timestamp)")
    parser.add_argument("--overwrite-existing-data", action="store_true",
                        help="Remove existing run dir before writing")

    # MCMC
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--iter-warmup", type=int, default=500)
    parser.add_argument("--iter-sampling", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--adapt-delta", type=float, default=0.85)
    parser.add_argument("--max-treedepth", type=int, default=12)

    # Model overrides
    parser.add_argument("--stan-file", default=None,
                        help="Override path to stan_dirichlet_model.stan")
    parser.add_argument("--override-D", type=int, default=None)
    parser.add_argument("--override-sigma-annotator", type=float, default=None)
    parser.add_argument("--override-sigma-measurement", type=float, default=None)
    parser.add_argument("--override-alpha-dirichlet", type=float, default=None)
    parser.add_argument("--override-temperature", type=float, default=None)

    # Dirichlet-specific
    parser.add_argument("--alpha-llm", type=float, default=5.0,
                        help="Dirichlet concentration for LLM observations "
                             "(higher → tighter around model's predicted pi). "
                             "Default: 5.0")

    args = parser.parse_args()

    # ── Load bundle ────────────────────────────────────────────────────────────
    bundle_path = Path(args.data_bundle)
    print(f"Loading data bundle from {bundle_path}")
    with open(bundle_path) as f:
        bundle_data = json.load(f)
    bundle = GroundTruthBundle.from_dict(bundle_data)

    # ── Load configs ───────────────────────────────────────────────────────────
    configs_path = bundle_path.parent / "configs.json"
    if not configs_path.exists():
        raise FileNotFoundError(f"configs.json not found at {configs_path}")
    with open(configs_path) as f:
        configs_data = json.load(f)
    dg = configs_data["datagen"]

    data_config = DataGenConfig(
        K_train=dg["K_train"],
        K_test=dg["K_test"],
        I=dg["I"],
        J=dg["J"],
        D=dg.get("D", 8),
        C=dg["C"],
        sigma_annotator=dg.get("sigma_annotator", 0.5),
        sigma_measurement=dg.get("sigma_measurement", 0.1),
        alpha_dirichlet=dg.get("alpha_dirichlet", 10.0),
        temperature=dg.get("temperature", 1.0),
    )

    # Apply overrides
    if args.override_D is not None:
        print(f"  Overriding D: {data_config.D} -> {args.override_D}")
        data_config.D = args.override_D
    if args.override_sigma_annotator is not None:
        data_config.sigma_annotator = args.override_sigma_annotator
    if args.override_sigma_measurement is not None:
        data_config.sigma_measurement = args.override_sigma_measurement
    if args.override_alpha_dirichlet is not None:
        data_config.alpha_dirichlet = args.override_alpha_dirichlet
    if args.override_temperature is not None:
        data_config.temperature = args.override_temperature

    K   = data_config.K_train + data_config.K_test
    C   = data_config.C

    print(f"\nData config: K={K} (train={data_config.K_train}, test={data_config.K_test}), "
          f"I={data_config.I}, J={data_config.J}, D={data_config.D}, C={C}")

    # ── Create output directory ────────────────────────────────────────────────
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.run_name:
        potential = output_root / args.run_name
        if potential.exists() and args.overwrite_existing_data:
            print(f"\033[91mWARNING: Removing existing dir: {potential}\033[0m")
            shutil.rmtree(potential)

    output_dir = new_run_dir(output_root, run_name=args.run_name)
    print(f"Output directory: {output_dir}")

    # ── Build Stan data dict ───────────────────────────────────────────────────
    observed = bundle.observed_ratings
    missing  = bundle.missing_ratings

    is_llm_flags   = [_is_llm(r)        for r in observed]
    rating_values  = [r["value"]         for r in observed]  # int 1..C for all
    rating_dists   = [_to_dist(r, C)     for r in observed]  # simplex[C] for all

    n_llm   = sum(is_llm_flags)
    n_human = len(is_llm_flags) - n_llm
    print(f"Observed ratings: {len(observed)} total  ({n_human} human, {n_llm} LLM)")
    print(f"Missing ratings:  {len(missing)}")
    print(f"alpha_llm = {args.alpha_llm}")

    stan_data = {
        "K": K,
        "I": data_config.I,
        "J": data_config.J,
        "D": data_config.D,
        "C": C,
        "N_ratings":         len(observed),
        "rating_attributes": [r["attribute"] for r in observed],
        "rating_annotators": [r["annotator"]  for r in observed],
        "rating_items":      [r["item"]        for r in observed],
        "rating_values":     rating_values,
        "rating_dists":      rating_dists,
        "is_llm_rating":     is_llm_flags,
        "N_missing_ratings":              len(missing),
        "missing_rating_attributes":      [r["attribute"] for r in missing],
        "missing_rating_annotators":      [r["annotator"]  for r in missing],
        "missing_rating_items":           [r["item"]        for r in missing],
        "sigma_annotator":   data_config.sigma_annotator,
        "sigma_measurement": data_config.sigma_measurement,
        "alpha_dirichlet":   data_config.alpha_dirichlet,
        "temperature":       data_config.temperature,
        "alpha_llm":         args.alpha_llm,
    }

    # ── Save augmented configs (needed by evaluate_predictions.py) ─────────────
    augmented_configs = {
        "datagen": {
            "K_train": data_config.K_train,
            "K_test":  data_config.K_test,
            "I":       data_config.I,
            "J":       data_config.J,
            "D":       data_config.D,
            "C":       C,
            "sigma_annotator":   data_config.sigma_annotator,
            "sigma_measurement": data_config.sigma_measurement,
            "alpha_dirichlet":   data_config.alpha_dirichlet,
            "temperature":       data_config.temperature,
        },
        "dirichlet": {
            "alpha_llm": args.alpha_llm,
            "n_human_ratings": n_human,
            "n_llm_ratings":   n_llm,
        },
    }
    with open(output_dir / "augmented_configs.json", "w") as f:
        json.dump(augmented_configs, f, indent=2)

    # ── Compile and run ────────────────────────────────────────────────────────
    stan_file = args.stan_file or _STAN_FILE
    print(f"\nCompiling Stan model: {stan_file}")
    model = compile_domain_model(stan_file)

    print("Starting MCMC inference...")
    try:
        fit = model.sample(
            data=stan_data,
            chains=args.chains,
            iter_warmup=args.iter_warmup,
            iter_sampling=args.iter_sampling,
            seed=args.seed,
            adapt_delta=args.adapt_delta,
            max_treedepth=args.max_treedepth,
            inits=1.0,
            show_progress=True,
            show_console=True,
        )
    except Exception as e:
        logger.error(f"MCMC failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("MCMC completed!")
    fit.save_csvfiles(str(output_dir))
    print(f"Samples saved to {output_dir}")

    divergences = fit.divergences
    if divergences.sum() > 0:
        print(f"\033[93mWARNING: {divergences.sum()} divergent transitions. "
              f"Consider raising --adapt-delta or --max-treedepth.\033[0m")
    else:
        print("No divergent transitions — good mixing!")


if __name__ == "__main__":
    main()
