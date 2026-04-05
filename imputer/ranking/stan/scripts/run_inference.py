#!/usr/bin/env python3
"""
CLI script for running MCMC inference with the domain model.

Supports all Stan model types (normal-noise-dot-product, factored-dot-product,
discrete, tensor) and an optional two-round inference mode (--use-train-only):

  Round 1: Train on training-instance observed ratings only.
           Posterior means of type-specific annotator parameters are extracted
           and frozen as fixed data for Round 2.
  Round 2: Train on test-instance ratings with annotator parameters frozen
           from Round 1. Only item embeddings are free.

Data format: every observed rating must have a 'rating_dist' field.
  - Human ratings  → one-hot distribution
  - LLM ratings    → soft probability vector (max < 1)
The Dirichlet observation model is always used for LLM ratings.

Usage:
    # Single-round inference
    python run_inference.py \\
        --data-bundle OUTPUT/generated_data/my_run/data_bundle.json \\
        --output-dir  OUTPUT/domain_model/runs \\
        --stan-type   factored-dot-product \\
        --alpha-llm   5.0

    # Two-round inference
    python run_inference.py \\
        --data-bundle OUTPUT/generated_data/my_run/data_bundle.json \\
        --output-dir  OUTPUT/domain_model/runs \\
        --stan-type   factored-dot-product \\
        --alpha-llm   5.0 \\
        --use-train-only
"""

import argparse
import json
import logging
import shutil
from dataclasses import fields
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cmdstanpy

from stan.pipeline.bundle import GroundTruthBundle
from stan.pipeline.configs import DataGenConfig
from stan.pipeline.inference import InferenceConfig, compile_domain_model
from stan.pipeline.io import new_run_dir, save_configs
from stan.scripts.generate_data import _parse_stan_arg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("inference.log"),
    ],
)
logger = logging.getLogger(__name__)

# Default Stan model paths
_MODELS_DIR = Path(__file__).resolve().parents[2] / "stan_models"

_DEFAULT_STAN_FILES = {
    "discrete":                  str(_MODELS_DIR / "discrete_model.stan"),
    "normal-noise-dot-product":  str(_MODELS_DIR / "normal_noise_dot_product_model.stan"),
    "factored-dot-product":      str(_MODELS_DIR / "normal_noise_dot_product_model.stan"),
    "tensor":                    str(_MODELS_DIR / "tensor_model.stan"),
}

# Parameters frozen in Round 2 per stan type.
# Each entry is a list of (stan_param_name, shape_fn) where shape_fn takes
# (I, J, D, C) and returns the expected numpy shape.
# fmt: off
_ROUND2_FROZEN_PARAMS = {
    "normal-noise-dot-product": [
        ("mean_preferences",      lambda I, J, D, C: (I, D)),
        ("annotator_preferences", lambda I, J, D, C: (I * J, D)),
        ("rating_probs",          lambda I, J, D, C: (I * J, C)),
    ],
    "factored-dot-product": [
        ("mean_preferences",      lambda I, J, D, C: (I, D)),
        ("annotator_preferences", lambda I, J, D, C: (I * J, D)),
        ("rating_probs",          lambda I, J, D, C: (I * J, C)),
    ],
    # TODO: fill in frozen params for discrete and tensor when two-round
    # support is needed for those types.
    "discrete": [],
    "tensor": [
        ("v_loadings",   lambda I, J, D, C: (I, D)),
        ("u_loadings",   lambda I, J, D, C: (J, D)),
        ("rating_probs", lambda I, J, D, C: (J, C)),
    ],
}
# fmt: on


# ── Rating helpers ─────────────────────────────────────────────────────────────

def _is_llm(r: dict) -> int:
    """Return 1 if rating carries a soft LLM distribution, 0 for human one-hot."""
    dist = r.get("rating_dist")
    if dist is None:
        return 0
    return 1 if max(dist) < 1.0 - 1e-6 else 0


def _to_dist(r: dict, C: int, eps: float = 1e-6) -> list:
    """Return a C-simplex for this rating with no exact zeros.

    LLM ratings:   use rating_dist (renormalised for safety).
    Human ratings: construct one-hot from value.

    eps is added to every entry before renormalising so that distributions
    passed to dirichlet_lpdf never contain exact zeros.
    """
    if "rating_dist" in r:
        d = list(r["rating_dist"])
        s = sum(d)
        d = [x / s for x in d]
    else:
        d = [0.0] * C
        d[r["value"] - 1] = 1.0
    d = [x + eps for x in d]
    s = sum(d)
    return [x / s for x in d]


# ── Stan data builders ─────────────────────────────────────────────────────────

def build_stan_data(
    observed: list,
    missing: list,
    data_config: DataGenConfig,
    K: int,
    alpha_llm: float,
) -> dict:
    """Build the Stan data dict for a single sampling round.

    Includes both the ordinal-probit fields (rating_values) and the
    Dirichlet fields (rating_dists, is_llm_rating, alpha_llm) so the
    Stan model can branch on is_llm_rating per observation.
    """
    C = data_config.C
    return {
        "K":                          K,
        "I":                          data_config.I,
        "J":                          data_config.J,
        "D":                          data_config.D,
        "C":                          C,
        "kappa":                      data_config.kappa,
        "factor_decay":               data_config.factor_decay,
        "N_ratings":                  len(observed),
        "rating_attributes":          [r["attribute"] for r in observed],
        "rating_annotators":          [r["annotator"]  for r in observed],
        "rating_items":               [r["item"]        for r in observed],
        "rating_values":              [r["value"]        for r in observed],
        "rating_dists":               [_to_dist(r, C)    for r in observed],
        "is_llm_rating":              [_is_llm(r)        for r in observed],
        "N_missing_ratings":          len(missing),
        "missing_rating_attributes":  [r["attribute"] for r in missing],
        "missing_rating_annotators":  [r["annotator"]  for r in missing],
        "missing_rating_items":       [r["item"]        for r in missing],
        "sigma_annotator":            data_config.sigma_annotator,
        "sigma_measurement":          data_config.sigma_measurement,
        "alpha_dirichlet":            data_config.kappa,
        "temperature":                data_config.temperature,
        "alpha_llm":                  alpha_llm,
        "d_annotator":                data_config.d_annotator,
        "use_factored_annotator":     data_config.use_factored_annotator,
    }


def build_round2_stan_data(
    fit_r1: cmdstanpy.CmdStanMCMC,
    observed_test: list,
    missing_test: list,
    data_config: DataGenConfig,
    K: int,
    alpha_llm: float,
    stan_type: str,
) -> dict:
    """Build Stan data for Round 2 by extracting posterior means of
    annotator-level parameters from the Round 1 fit and freezing them.

    The set of frozen parameters is determined by _ROUND2_FROZEN_PARAMS[stan_type].
    For types with no frozen params defined (discrete, tensor), this falls back
    to a standard single-round data dict (no freezing).

    Args:
        fit_r1:        Completed Round 1 CmdStanMCMC fit object.
        observed_test: Observed ratings for the test instance.
        missing_test:  Missing ratings for the test instance.
        data_config:   DataGenConfig with model dimensions.
        K:             Total number of items (K_train + K_test).
        alpha_llm:     Dirichlet concentration for LLM observations.
        stan_type:     Stan model type string.

    Returns:
        Stan data dict with frozen annotator parameters added as fixed data.
    """
    I, J, D, C = data_config.I, data_config.J, data_config.D, data_config.C

    # Base data (same structure as Round 1, but on test ratings only)
    stan_data_r2 = build_stan_data(observed_test, missing_test, data_config, K, alpha_llm)

    frozen_specs = _ROUND2_FROZEN_PARAMS.get(stan_type, [])
    if not frozen_specs:
        logger.warning(
            f"No frozen parameters defined for stan_type={stan_type!r}. "
            "Running Round 2 without any frozen annotator parameters."
        )
        return stan_data_r2

    # Extract posterior means for each frozen parameter
    draws = fit_r1.draws_pd()

    for param_name, shape_fn in frozen_specs:
        shape = shape_fn(I, J, D, C)
        arr = np.zeros(shape)

        if len(shape) == 2:
            rows, cols = shape
            for i in range(1, rows + 1):
                for j in range(1, cols + 1):
                    col = f"{param_name}[{i},{j}]"
                    arr[i - 1, j - 1] = draws[col].mean()
        elif len(shape) == 1:
            (size,) = shape
            for i in range(1, size + 1):
                col = f"{param_name}[{i}]"
                arr[i - 1] = draws[col].mean()
        else:
            raise ValueError(
                f"Unsupported shape {shape} for frozen parameter {param_name!r}."
            )

        # Renormalise simplex parameters (posterior mean may not sum to exactly 1)
        if param_name == "rating_probs":
            for i in range(arr.shape[0]):
                arr[i] /= arr[i].sum()

        stan_data_r2[param_name] = arr.tolist()
        print(f"  Frozen {param_name:30s} shape={list(arr.shape)}")

    return stan_data_r2


# ── MCMC runner ────────────────────────────────────────────────────────────────

def _run_sample(
    model: cmdstanpy.CmdStanModel,
    stan_data: dict,
    args: argparse.Namespace,
    label: str,
) -> cmdstanpy.CmdStanMCMC:
    """Run model.sample with shared MCMC settings from parsed args."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    fit = model.sample(
        data=stan_data,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        seed=args.seed,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        inits=1.0,
        show_progress=not args.no_progress,
        show_console=False,
    )
    divergences = fit.divergences.sum()
    if divergences > 0:
        print(
            f"\033[93mWARNING ({label}): {divergences} divergent transitions. "
            f"Consider raising --adapt-delta or --max-treedepth.\033[0m"
        )
    else:
        print(f"No divergent transitions in {label} — good mixing!")
    return fit


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MCMC inference with domain model (all Stan types)"
    )

    # ── Input / output ────────────────────────────────────────────────────────
    parser.add_argument("--data-bundle", required=True,
                        help="Path to data_bundle.json (must have rating_dist fields)")
    parser.add_argument("--output-dir", default="OUTPUT/domain_model/runs",
                        help="Root output directory for results")
    parser.add_argument("--run-name", default=None,
                        help="Custom run name (default: auto timestamp)")
    parser.add_argument("--overwrite-existing-data", action="store_true",
                        help="Remove existing run dir before writing")

    # ── Stan model ────────────────────────────────────────────────────────────
    parser.add_argument("--stan-type", default=None,
                        choices=list(_DEFAULT_STAN_FILES),
                        help="Stan model type. Default: read from configs.json.")
    parser.add_argument("--stan-file", default=None,
                        help="Override path to Round 1 Stan model file.")
    parser.add_argument("--stan-file-round2", default=None,
                        help="Override path to Round 2 Stan model file "
                             "(only used with --use-train-only).")
    parser.add_argument("--stan-arg", action="append", metavar="KEY=VALUE",
                        help="Extra Stan data fields (repeatable). "
                             "E.g. --stan-arg DEBUG_INIT=1 for tensor.")

    # ── Two-round mode ────────────────────────────────────────────────────────
    parser.add_argument("--use-train-only", action="store_true",
                        help="Two-round inference: Round 1 on train ratings, "
                             "Round 2 on test ratings with annotator params "
                             "frozen from Round 1 posterior means.")

    # ── MCMC ──────────────────────────────────────────────────────────────────
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--iter-warmup", type=int, default=500)
    parser.add_argument("--iter-sampling", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--adapt-delta", type=float, default=0.85)
    parser.add_argument("--max-treedepth", type=int, default=12)
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bar")

    # ── Model parameter overrides ─────────────────────────────────────────────
    parser.add_argument("--override-D", type=int, default=None,
                        help="Override embedding dimension D")
    parser.add_argument("--override-sigma-annotator", type=float, default=None)
    parser.add_argument("--override-sigma-measurement", type=float, default=None)
    parser.add_argument("--override-alpha-dirichlet", type=float, default=None,
                        help="Override kappa / alpha_dirichlet")
    parser.add_argument("--override-temperature", type=float, default=None)

    # ── Dirichlet-specific ────────────────────────────────────────────────────
    parser.add_argument("--alpha-llm", type=float, default=5.0,
                        help="Dirichlet concentration for LLM observations (default: 5.0)")

    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load bundle ────────────────────────────────────────────────────────────
    bundle_path = Path(args.data_bundle)
    print(f"Loading data bundle from {bundle_path}")
    with open(bundle_path) as f:
        bundle_data = json.load(f)
    bundle = GroundTruthBundle.from_dict(bundle_data)
    logger.info(
        f"Loaded bundle: {len(bundle.observed_ratings)} observed, "
        f"{len(bundle.missing_ratings)} missing ratings"
    )

    # ── Load configs ───────────────────────────────────────────────────────────
    configs_path = Path(f"/home/hshi33/scratchjeisner1/hshi33/AnnotationArena/imputer/ranking/stan/configs/{args.stan_type}.json")
    if not configs_path.exists():
        raise FileNotFoundError(f"configs.json not found at {configs_path}")
    with open(configs_path) as f:
        configs_data = json.load(f)

    dg = configs_data["datagen"]
    valid_keys = {f.name for f in fields(DataGenConfig)}
    dg_filtered = {k: v for k, v in dg.items() if k in valid_keys}
    # Backwards compat: alpha_dirichlet was renamed to kappa
    if "kappa" not in dg_filtered and "alpha_dirichlet" in dg:
        dg_filtered["kappa"] = dg["alpha_dirichlet"]
    data_config = DataGenConfig(**dg_filtered)

    # Stan type: CLI override > configs.json
    stan_type = args.stan_type if args.stan_type is not None else data_config.stan_type
    print(f"Stan type: {stan_type}  (data generated as: {data_config.stan_type})")

    # Apply parameter overrides
    if args.override_D is not None:
        print(f"  Overriding D: {data_config.D} → {args.override_D}")
        data_config.D = args.override_D
    if args.override_sigma_annotator is not None:
        data_config.sigma_annotator = args.override_sigma_annotator
    if args.override_sigma_measurement is not None:
        data_config.sigma_measurement = args.override_sigma_measurement
    if args.override_alpha_dirichlet is not None:
        data_config.kappa = args.override_alpha_dirichlet
    if args.override_temperature is not None:
        data_config.temperature = args.override_temperature

    K = data_config.K_train + data_config.K_test
    print(
        f"Config: K={K} (train={data_config.K_train}, test={data_config.K_test}), "
        f"I={data_config.I}, J={data_config.J}, D={data_config.D}, C={data_config.C}"
    )

    # Parse --stan-arg overrides
    stan_arg: dict = {}
    for s in args.stan_arg or []:
        k, v = _parse_stan_arg(s)
        stan_arg[k] = v
    if stan_type == "tensor":
        stan_arg.setdefault("DEBUG_INIT", 0)

    # ── Resolve Stan files ─────────────────────────────────────────────────────
    stan_file_r1 = args.stan_file or _DEFAULT_STAN_FILES[stan_type]

    # Round 2 file: explicit override, or derive from Round 1 file by convention
    # (append _freeze before .stan, e.g. domain_model_freeze.stan)
    if args.stan_file_round2:
        stan_file_r2 = args.stan_file_round2
    else:
        stan_file_r2 = stan_file_r1.replace(".stan", "_freeze.stan")

    # ── Output directory ───────────────────────────────────────────────────────
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.run_name:
        potential = output_root / args.run_name
        if potential.exists() and args.overwrite_existing_data:
            print(f"\033[91mWARNING: Removing existing dir: {potential}\033[0m")
            shutil.rmtree(potential)

    output_dir = new_run_dir(output_root, run_name=args.run_name)
    print(f"Output directory: {output_dir}")

    # ── Save configs ───────────────────────────────────────────────────────────
    inference_cfg_dict = {
        "data_bundle":      str(args.data_bundle),
        "stan_type":        stan_type,
        "stan_file_r1":     stan_file_r1,
        "stan_file_r2":     stan_file_r2 if args.use_train_only else None,
        "stan_arg":         stan_arg,
        "use_train_only":   args.use_train_only,
        "chains":           args.chains,
        "iter_warmup":      args.iter_warmup,
        "iter_sampling":    args.iter_sampling,
        "seed":             args.seed,
        "adapt_delta":      args.adapt_delta,
        "max_treedepth":    args.max_treedepth,
        "alpha_llm":        args.alpha_llm,
    }
    save_configs(output_dir, inference=inference_cfg_dict)

    augmented_configs = {
        "datagen": {
            "K_train":           data_config.K_train,
            "K_test":            data_config.K_test,
            "I":                 data_config.I,
            "J":                 data_config.J,
            "D":                 data_config.D,
            "C":                 data_config.C,
            "sigma_annotator":   data_config.sigma_annotator,
            "sigma_measurement": data_config.sigma_measurement,
            "alpha_dirichlet":   data_config.kappa,
            "temperature":       data_config.temperature,
        },
        "dirichlet": {
            "alpha_llm":      args.alpha_llm,
            "two_round_mode": args.use_train_only,
        },
    }

    # ── Run inference ──────────────────────────────────────────────────────────
    try:
        if not args.use_train_only:
            # ── Single round ───────────────────────────────────────────────────
            observed = bundle.observed_ratings
            missing  = bundle.missing_ratings

            n_llm   = sum(_is_llm(r) for r in observed)
            n_human = len(observed) - n_llm
            print(f"\nObserved: {len(observed)} ({n_human} human, {n_llm} LLM) | "
                  f"Missing: {len(missing)} | alpha_llm={args.alpha_llm}")

            stan_data = build_stan_data(observed, missing, data_config, K, args.alpha_llm)
            stan_data.update(stan_arg)

            print(f"\nCompiling Stan model: {stan_file_r1}")
            model = compile_domain_model(stan_file_r1)
            fit = _run_sample(model, stan_data, args, label="Single-round MCMC")

        else:
            # ── Round 1: train ratings ─────────────────────────────────────────
            observed_train = [r for r in bundle.observed_ratings if r["instance"] == "train"]
            missing_train  = [r for r in bundle.missing_ratings  if r["instance"] == "train"]

            n_llm_r1   = sum(_is_llm(r) for r in observed_train)
            n_human_r1 = len(observed_train) - n_llm_r1
            print(f"\n[Round 1] Observed train: {len(observed_train)} "
                  f"({n_human_r1} human, {n_llm_r1} LLM) | "
                  f"Missing train: {len(missing_train)}")

            stan_data_r1 = build_stan_data(
                observed_train, missing_train, data_config, K, args.alpha_llm
            )
            stan_data_r1.update(stan_arg)

            print(f"\nCompiling Round 1 Stan model: {stan_file_r1}")
            model_r1 = compile_domain_model(stan_file_r1)
            fit_r1 = _run_sample(model_r1, stan_data_r1, args, label="Round 1 (train)")

            # ── Round 2: test ratings, frozen annotator params ─────────────────
            observed_test = [r for r in bundle.observed_ratings if r["instance"] == "test"]
            missing_test  = [r for r in bundle.missing_ratings  if r["instance"] == "test"]

            n_llm_r2   = sum(_is_llm(r) for r in observed_test)
            n_human_r2 = len(observed_test) - n_llm_r2
            print(f"\n[Round 2] Observed test: {len(observed_test)} "
                  f"({n_human_r2} human, {n_llm_r2} LLM) | "
                  f"Missing test: {len(missing_test)}")
            print(f"Freezing annotator parameters from Round 1 posterior means "
                  f"(stan_type={stan_type!r})...")

            stan_data_r2 = build_round2_stan_data(
                fit_r1=fit_r1,
                observed_test=observed_test,
                missing_test=missing_test,
                data_config=data_config,
                K=K,
                alpha_llm=args.alpha_llm,
                stan_type=stan_type,
            )
            stan_data_r2.update(stan_arg)

            print(f"\nCompiling Round 2 Stan model: {stan_file_r2}")
            model_r2 = compile_domain_model(stan_file_r2)
            fit = _run_sample(model_r2, stan_data_r2, args, label="Round 2 (test, frozen params)")

            augmented_configs["dirichlet"].update({
                "round1_n_human_ratings": n_human_r1,
                "round1_n_llm_ratings":   n_llm_r1,
                "round2_n_human_ratings": n_human_r2,
                "round2_n_llm_ratings":   n_llm_r2,
            })

    except Exception as e:
        logger.error(f"MCMC failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ── Save outputs ───────────────────────────────────────────────────────────
    with open(output_dir / "augmented_configs.json", "w") as f:
        json.dump(augmented_configs, f, indent=2)

    fit.save_csvfiles(str(output_dir))
    print(f"\nMCMC complete. Samples saved to {output_dir}")

    divergences = fit.divergences.sum()
    if divergences > 0:
        print(f"\033[93mWARNING: {divergences} divergent transitions in final fit.\033[0m")
    else:
        print("No divergent transitions in final fit — good mixing!")


if __name__ == "__main__":
    main()
