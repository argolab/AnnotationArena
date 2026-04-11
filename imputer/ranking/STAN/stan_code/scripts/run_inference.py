#!/usr/bin/env python3
"""
CLI script for running MCMC inference with the domain model.

Supports all Stan model types (normal-noise-dot-product, factored-dot-product,
discrete, tensor) and both item-split and annotator-split data bundles.

Instance handling:
  - Observed ratings passed to Stan: instance in ("train", "val")
  - Missing ratings passed to Stan (for prediction): instance == "test" only

Data format: every observed rating must have a 'rating_dist' field.
  - Human ratings  → one-hot distribution
  - LLM ratings    → soft probability vector (max < 1)
The Dirichlet observation model is always used for LLM ratings.

Usage:
    # Item-split bundle
    python run_inference.py \\
        --data-bundle OUTPUT/generated_data/my_run/data_bundle.json \\
        --configs     OUTPUT/generated_data/my_run/configs.json \\
        --output-dir  OUTPUT/domain_model/runs \\
        --stan-type   factored-dot-product \\
        --alpha-llm   5.0

    # Annotator-split bundle
    python run_inference.py \\
        --data-bundle OUTPUT/generated_data/annot_run/data_bundle.json \\
        --configs     OUTPUT/generated_data/annot_run/configs.json \\
        --output-dir  OUTPUT/domain_model/runs \\
        --stan-type   factored-dot-product \\
        --alpha-llm   5.0
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

from STAN.stan_code.pipeline.bundle import GroundTruthBundle
from STAN.stan_code.pipeline.configs import DataGenConfig, AnnotatorSplitConfig
from STAN.stan_code.pipeline.inference import InferenceConfig, compile_domain_model
from STAN.stan_code.pipeline.io import new_run_dir, save_configs
from STAN.stan_code.scripts.generate_data import _parse_stan_arg

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


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(configs_path: Path):
    """
    Load the data generation config from configs.json.

    Returns an AnnotatorSplitConfig if the config has J_train/J_val/J_test fields
    (annotator-split bundle), otherwise returns a DataGenConfig (item-split bundle).
    Also returns a string "annotator_split" or "item_split" indicating the mode.
    """
    with open(configs_path) as f:
        configs_data = json.load(f)

    dg = configs_data.get("datagen", configs_data)

    # Detect annotator-split by presence of J_train
    if "J_train" in dg:
        valid_keys = {f.name for f in fields(AnnotatorSplitConfig)}
        dg_filtered = {k: v for k, v in dg.items() if k in valid_keys}
        return AnnotatorSplitConfig(**dg_filtered), "annotator_split"
    else:
        valid_keys = {f.name for f in fields(DataGenConfig)}
        dg_filtered = {k: v for k, v in dg.items() if k in valid_keys}
        if "kappa" not in dg_filtered and "alpha_dirichlet" in dg:
            dg_filtered["kappa"] = dg["alpha_dirichlet"]
        return DataGenConfig(**dg_filtered), "item_split"


def get_K_J(config, mode: str):
    """Extract total K and J from whichever config type was loaded."""
    if mode == "annotator_split":
        return config.K, config.J
    else:
        # item-split: K is train + val + test, J is total annotators
        K = config.K_train + config.K_val + config.K_test if hasattr(config, "K_val") \
            else config.K_train + config.K_test
        return K, config.J


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


# ── Stan data builder ──────────────────────────────────────────────────────────

def build_stan_data(
    observed: list,
    missing: list,
    config,
    mode: str,
    K: int,
    J: int,
    alpha_llm: float,
) -> dict:
    """
    Build the Stan data dict for inference.

    Works for both item-split (DataGenConfig) and annotator-split
    (AnnotatorSplitConfig). Item and annotator IDs in the bundle are already
    in the correct 1-indexed space for Stan — no remapping needed since:
      - item-split:      train=1..K_train, val=K_train+1..K_train+K_val,
                         test=K_train+K_val+1..K  (all contiguous)
      - annotator-split: all items share IDs 1..K;
                         annotator IDs remapped at subset time to be contiguous

    Args:
        observed: Observed ratings (train + val instances).
        missing:  Missing ratings (test instance only).
        config:   DataGenConfig or AnnotatorSplitConfig.
        mode:     "item_split" or "annotator_split".
        K:        Total number of items passed to Stan.
        J:        Total number of annotators passed to Stan.
        alpha_llm: Dirichlet concentration for LLM observations.
    """
    C = config.C

    return {
        "K":                         K,
        "I":                         config.I,
        "J":                         J,
        "D":                         config.D,
        "C":                         C,
        "d_annotator":               config.d_annotator,
        "use_factored_annotator":    int(config.use_factored_annotator)
                                     if config.use_factored_annotator is not None else 1,
        "kappa":                     config.kappa if hasattr(config, "kappa")
                                     else config.kappa,
        "sigma_annotator":           config.sigma_annotator,
        "sigma_measurement":         config.sigma_measurement,
        "alpha_dirichlet":           config.kappa,
        "temperature":               config.temperature,
        "alpha_llm":                 alpha_llm,
        # Observed ratings
        "N_ratings":                 len(observed),
        "rating_attributes":         [r["attribute"] for r in observed],
        "rating_annotators":         [r["annotator"]  for r in observed],
        "rating_items":              [r["item"]        for r in observed],
        "rating_values":             [r["value"]       for r in observed],
        "rating_dists":              [_to_dist(r, C)   for r in observed],
        "is_llm_rating":             [_is_llm(r)       for r in observed],
        # Missing ratings (test only — these are what we predict)
        "N_missing_ratings":         len(missing),
        "missing_rating_attributes": [r["attribute"] for r in missing],
        "missing_rating_annotators": [r["annotator"]  for r in missing],
        "missing_rating_items":      [r["item"]        for r in missing],
    }


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
        description="Run MCMC inference with domain model (item-split and annotator-split)"
    )

    # ── Input / output ────────────────────────────────────────────────────────
    parser.add_argument("--data-bundle", required=True,
                        help="Path to data_bundle.json")
    parser.add_argument("--configs", required=True,
                        help="Path to configs.json for the data bundle")
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
                        help="Override path to Stan model file.")
    parser.add_argument("--stan-arg", action="append", metavar="KEY=VALUE",
                        help="Extra Stan data fields (repeatable). "
                             "E.g. --stan-arg DEBUG_INIT=1 for tensor.")
    parser.add_argument("--real-data", type=bool, default=False)

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
    configs_path = Path(args.configs)
    if not configs_path.exists():
        raise FileNotFoundError(f"configs.json not found at {configs_path}")

    if not args.real_data:
        data_config, mode = load_config(configs_path)
    else:
        data_config, mode = load_config(f"STAN/stan_code/configs/{args.stan_type}.json")
        with open(configs_path, "r") as file:
            data = json.load(file)
        data_config.K_train = data["datagen"]["K_train"]
        data_config.K_val = data["datagen"]["K_val"]
        data_config.K_test = data["datagen"]["K_test"]
        data_config.I = data["datagen"]["I"]
        data_config.J = data["datagen"]["J"]
        data_config.C = data["datagen"]["C"]
    print(f"Bundle mode: {mode}")

    # Stan type: CLI override > configs.json
    stan_type = args.stan_type if args.stan_type is not None else data_config.stan_type
    print(f"Stan type: {stan_type}")

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

    K, J = get_K_J(data_config, mode)
    print(f"Config: K={K}, J={J}, I={data_config.I}, D={data_config.D}, C={data_config.C}")
    if mode == "item_split":
        k_train = data_config.K_train
        k_val   = getattr(data_config, "K_val", 0)
        k_test  = data_config.K_test
        print(f"  Item-split: K_train={k_train}, K_val={k_val}, K_test={k_test}")
    else:
        print(f"  Annotator-split: J_train={data_config.J_train}, "
              f"J_val={data_config.J_val}, J_test={data_config.J_test}")

    # Parse --stan-arg overrides
    stan_arg: dict = {}
    for s in args.stan_arg or []:
        k, v = _parse_stan_arg(s)
        stan_arg[k] = v
    if stan_type == "tensor":
        stan_arg.setdefault("DEBUG_INIT", 0)

    # ── Instance filtering ─────────────────────────────────────────────────────
    # Observed: train + val treated as training signal
    # Missing to predict: test only
    observed = bundle.observed_ratings        # train + val + test observed
    missing  = bundle.missing_ratings         # train + val + test missing

    n_llm   = sum(_is_llm(r) for r in observed)
    n_human = len(observed) - n_llm
    print(f"\nObserved (train+val): {len(observed)} ({n_human} human, {n_llm} LLM)")
    print(f"Missing  (test only): {len(missing)}")
    print(f"alpha_llm={args.alpha_llm}")

    # ── Resolve Stan file ──────────────────────────────────────────────────────
    stan_file = args.stan_file or _DEFAULT_STAN_FILES[stan_type]

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
        "data_bundle":   str(args.data_bundle),
        "configs":       str(args.configs),
        "bundle_mode":   mode,
        "stan_type":     stan_type,
        "stan_file":     stan_file,
        "stan_arg":      stan_arg,
        "chains":        args.chains,
        "iter_warmup":   args.iter_warmup,
        "iter_sampling": args.iter_sampling,
        "seed":          args.seed,
        "adapt_delta":   args.adapt_delta,
        "max_treedepth": args.max_treedepth,
        "alpha_llm":     args.alpha_llm,
        "n_observed":    len(observed),
        "n_missing":     len(missing),
        "n_llm":         n_llm,
        "n_human":       n_human,
    }
    save_configs(output_dir, inference=inference_cfg_dict)

    # ── Build Stan data ────────────────────────────────────────────────────────
    stan_data = build_stan_data(
        observed=observed,
        missing=missing,
        config=data_config,
        mode=mode,
        K=K,
        J=J,
        alpha_llm=args.alpha_llm,
    )
    stan_data.update(stan_arg)

    # ── Run inference ──────────────────────────────────────────────────────────
    try:
        print(f"\nCompiling Stan model: {stan_file}")
        model = compile_domain_model(stan_file)
        fit = _run_sample(model, stan_data, args, label="MCMC inference")

    except Exception as e:
        logger.error(f"MCMC failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ── Save outputs ───────────────────────────────────────────────────────────
    fit.save_csvfiles(str(output_dir))
    print(f"\nMCMC complete. Samples saved to {output_dir}")

    divergences = fit.divergences.sum()
    if divergences > 0:
        print(f"\033[93mWARNING: {divergences} divergent transitions in final fit.\033[0m")
    else:
        print("No divergent transitions — good mixing!")


if __name__ == "__main__":
    main()
