#!/usr/bin/env python3
"""
CLI for MCMC inference with the tensor-prototype model, for item-split and
annotator-split data bundles. Called by scripts/DOMAIN3/run_stan_group.sh and
scripts/LLMRubric/CPMSigmoid/*.sh.

Use --model-variant (oracle / misspec-bin / misspec-proj / misspec-both) to
pick one of the 4 tensor inference models in stan_models/ by name, or pass
--stan-file directly.

    python run_inference.py --data-bundle <bundle.json> --configs <configs.json> \\
        --output-dir OUTPUT/domain_model/runs --model-variant oracle
"""

import argparse
import csv
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
from STAN.stan_code.pipeline.inference import InferenceConfig, compile_domain_model, create_init_from_ground_truth_tensor
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
    "tensor": str(_MODELS_DIR / "tensor_model_sigmoid.stan"),
}

# Named tensor model variants, selected via --model-variant (--stan-file overrides).
_TENSOR_MODEL_VARIANTS = {
    "oracle":        str(_MODELS_DIR / "tensor_model_sigmoid.stan"),
    "misspec-bin":   str(_MODELS_DIR / "tensor_shared_annotator_binning_raw_z_sigmoid.stan"),
    "misspec-proj":  str(_MODELS_DIR / "tensor_shared_annotator_projection_sigmoid.stan"),
    "misspec-both":  str(_MODELS_DIR / "tensor_shared_annotator_projection_binning_raw_z_sigmoid.stan"),
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


# ── Stan data builder ──────────────────────────────────────────────────────────

def build_stan_data(
    observed: list,
    missing: list,
    config,
    mode: str,
    K: int,
    J: int,
    stan_type: str = None,
) -> dict:
    """
    Build the Stan data dict for inference. Works for both item-split
    (DataGenConfig) and annotator-split (AnnotatorSplitConfig) bundles — item
    and annotator IDs are already 1-indexed and contiguous in both, so no
    remapping is needed here.
    """
    C = config.C

    # Shared ratings arrays (all types)
    ratings_base = {
        "N_ratings":                 len(observed),
        "rating_attributes":         [r["attribute"] for r in observed],
        "rating_annotators":         [r["annotator"]  for r in observed],
        "rating_items":              [r["item"]        for r in observed],
        "rating_values":             [r["value"]       for r in observed],
        "N_missing_ratings":         len(missing),
        "missing_rating_attributes": [r["attribute"] for r in missing],
        "missing_rating_annotators": [r["annotator"]  for r in missing],
        "missing_rating_items":      [r["item"]        for r in missing],
    }

    if stan_type != "tensor":
        raise ValueError(
            f"Unsupported stan_type: {stan_type!r}. Only 'tensor' is supported "
            "(oracle and misspecified models all share this data schema)."
        )

    # Tensor-prototype model: different parameter set, no LLM fields.
    return {
        "K":                          K,
        "I":                          config.I,
        "J":                          J,
        "D":                          config.D,
        "C":                          C,
        "T":                          config.T,
        "use_dawid_skene_noise":      int(config.use_dawid_skene_noise),
        "derive_thresholds_from_annotator":
                                      int(config.derive_thresholds_from_annotator),
        "alpha_dirichlet_jt":         1.0,   # uniform Dirichlet over T prototypes
        "alpha_confusion":            config.alpha_confusion,
        **ratings_base,
    }


def _augment_fixed_tensor_hyperparams(stan_data: dict, config, stan_file: str) -> dict:
    """
    Ensure tensor fixed-hyperparameter Stan models receive their required data fields.
    """
    if Path(stan_file).name != "tensor_model_fixed_hyperparams.stan":
        return stan_data

    required = ("sigma_u", "sigma_v", "sigma_uit", "sigma_measurement", "kappa")
    missing = [name for name in required if getattr(config, name, None) is None]
    if missing:
        raise ValueError(
            "Fixed tensor Stan model requires hyperparameters in config data: "
            + ", ".join(missing)
        )

    stan_data.update({
        "sigma_u": float(config.sigma_u),
        "sigma_v": float(config.sigma_v),
        "sigma_uit": float(config.sigma_uit),
        "sigma_measurement": float(config.sigma_measurement),
        "kappa": float(config.kappa),
    })
    return stan_data


# ── MCMC runner ────────────────────────────────────────────────────────────────

def _run_sample(
    model: cmdstanpy.CmdStanModel,
    stan_data: dict,
    args: argparse.Namespace,
    label: str,
    inits=1.0,
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
        inits=inits,
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


def _save_tensor_hyperparameter_posterior(
    fit: cmdstanpy.CmdStanMCMC,
    output_dir: Path,
    stan_type: str,
    stan_file: str | None = None,
) -> None:
    """
    Save posterior draws for tensor hyperparameters needed for experiment auditing.
    """
    if stan_type != "tensor":
        return

    if stan_file is not None and Path(stan_file).name == "tensor_model_fixed_hyperparams.stan":
        return

    param_names = ["sigma_u", "sigma_v", "sigma_uit", "sigma_measurement", "kappa"]
    posterior_draws = {}
    for name in param_names:
        try:
            posterior_draws[name] = fit.stan_variable(name)
        except Exception:
            logger.warning("Could not extract posterior draws for %s", name)
            return

    draws_len = len(posterior_draws[param_names[0]])
    csv_path = output_dir / "posterior_hyperparameters.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(param_names)
        for i in range(draws_len):
            writer.writerow([float(posterior_draws[name][i]) for name in param_names])

    summary = {}
    for name in param_names:
        values = np.asarray(posterior_draws[name], dtype=float)
        summary[name] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "q05": float(np.quantile(values, 0.05)),
            "q50": float(np.quantile(values, 0.50)),
            "q95": float(np.quantile(values, 0.95)),
        }
    summary_path = output_dir / "posterior_hyperparameters_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved tensor hyperparameter posterior draws: {csv_path}")
    print(f"Saved tensor hyperparameter posterior summary: {summary_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MCMC inference with the tensor domain model (item-split and annotator-split)"
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
    parser.add_argument("--model-variant", default="oracle",
                        choices=list(_TENSOR_MODEL_VARIANTS),
                        help="Named tensor model variant (default: oracle, the "
                             "correctly-specified model). Ignored if --stan-file is set.")
    parser.add_argument("--stan-file", default=None,
                        help="Override path to Stan model file (overrides --model-variant).")
    parser.add_argument("--stan-arg", action="append", metavar="KEY=VALUE",
                        help="Extra Stan data fields (repeatable). "
                             "E.g. --stan-arg DEBUG_INIT=1 for tensor.")

    # ── MCMC ──────────────────────────────────────────────────────────────────
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--iter-warmup", type=int, default=500)
    parser.add_argument("--iter-sampling", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--adapt-delta", type=float, default=0.85)
    parser.add_argument("--max-treedepth", type=int, default=12)
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bar")
    parser.add_argument("--init-strategy", default="random",
                        choices=["random", "ground_truth"],
                        help="Initialisation strategy: 'random' (default, inits=1.0) or "
                             "'ground_truth' (oracle init from bundle; tensor model only)")

    # ── Model parameter overrides ─────────────────────────────────────────────
    parser.add_argument("--override-D", type=int, default=None,
                        help="Override embedding dimension D")
    parser.add_argument("--override-sigma-measurement", type=float, default=None)
    parser.add_argument("--override-alpha-dirichlet", type=float, default=None,
                        help="Override kappa / alpha_dirichlet")
    parser.add_argument("--override-temperature", type=float, default=None)

    # ── Dirichlet-specific ────────────────────────────────────────────────────
    # Accepted for backward compatibility with LLMRubric/CPMSigmoid callers;
    # unused by the tensor model's Stan data schema (no LLM soft-label fields).
    parser.add_argument("--alpha-llm", type=float, default=5.0,
                        help="Unused by stan_type=tensor; kept for caller compatibility.")

    # ── Transductive inference ────────────────────────────────────────────────
    parser.add_argument("--transductive-use-test-observed", dest="transductive_use_test_observed",
                        action="store_true", default=True,
                        help="Include observed test rows during Stan inference (default: on)")
    parser.add_argument("--no-transductive-use-test-observed", dest="transductive_use_test_observed",
                        action="store_false",
                        help="Exclude observed test rows during Stan inference")

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

    data_config, mode = load_config(configs_path)
    print(f"Bundle mode: {mode}")

    # Stan type: CLI override > configs.json
    stan_type = args.stan_type if args.stan_type is not None else data_config.stan_type
    print(f"Stan type: {stan_type}")

    # Apply parameter overrides
    if args.override_D is not None:
        print(f"  Overriding D: {data_config.D} → {args.override_D}")
        data_config.D = args.override_D
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
    if args.transductive_use_test_observed:
        observed = bundle.observed_ratings
    else:
        observed = [r for r in bundle.observed_ratings if r["instance"] != "test"]
    missing = bundle.missing_ratings

    observed_train = sum(1 for r in observed if r["instance"] == "train")
    observed_val = sum(1 for r in observed if r["instance"] == "val")
    observed_test = sum(1 for r in observed if r["instance"] == "test")
    missing_test = sum(1 for r in missing if r["instance"] == "test")

    n_llm   = sum(_is_llm(r) for r in observed)
    n_human = len(observed) - n_llm
    print(f"\nObserved used: {len(observed)} ({n_human} human, {n_llm} LLM)")
    print(f"  train={observed_train}, val={observed_val}, test={observed_test}")
    print(f"Missing total: {len(missing)}")
    print(f"Missing test:  {missing_test}")
    print(f"transductive_use_test_observed={args.transductive_use_test_observed}")

    # ── Resolve Stan file ──────────────────────────────────────────────────────
    stan_file = args.stan_file or _TENSOR_MODEL_VARIANTS[args.model_variant]

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
        "n_observed":    len(observed),
        "n_missing":     len(missing),
        "n_llm":         n_llm,
        "n_human":       n_human,
        "init_strategy":  args.init_strategy,
        "transductive_use_test_observed": args.transductive_use_test_observed,
        "observed_train": observed_train,
        "observed_val": observed_val,
        "observed_test": observed_test,
        "missing_test": missing_test,
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
        stan_type=stan_type,
    )
    stan_data.update(stan_arg)
    stan_data = _augment_fixed_tensor_hyperparams(stan_data, data_config, stan_file)

    # ── Resolve init values ────────────────────────────────────────────────────
    if args.init_strategy == "ground_truth":
        if stan_type != "tensor":
            print("WARNING: --init-strategy ground_truth is only supported for "
                  "stan_type=tensor; falling back to inits=1.0")
            computed_inits = 1.0
        else:
            init_dict = create_init_from_ground_truth_tensor(bundle)
            computed_inits = [init_dict] * args.chains
            print(f"Using oracle (ground-truth) init for {args.chains} chain(s)")
            print(f"  Initialised params: {list(init_dict.keys())}")
    else:
        computed_inits = 1.0

    # ── Run inference ──────────────────────────────────────────────────────────
    try:
        print(f"\nCompiling Stan model: {stan_file}")
        model = compile_domain_model(stan_file)
        fit = _run_sample(model, stan_data, args, label="MCMC inference", inits=computed_inits)

    except Exception as e:
        logger.error(f"MCMC failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ── Save outputs ───────────────────────────────────────────────────────────
    fit.save_csvfiles(str(output_dir))
    print(f"\nMCMC complete. Samples saved to {output_dir}")
    _save_tensor_hyperparameter_posterior(fit, output_dir, stan_type, stan_file)

    divergences = fit.divergences.sum()
    if divergences > 0:
        print(f"\033[93mWARNING: {divergences} divergent transitions in final fit.\033[0m")
    else:
        print("No divergent transitions — good mixing!")


if __name__ == "__main__":
    main()
