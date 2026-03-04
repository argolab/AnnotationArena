#!/usr/bin/env python3
"""
MCMC inference using the Dirichlet observation model for LLM distributional labels.

Trains on observed ratings:
  - Human annotators  → ordinal probit hard-label likelihood: log P(c | model)
  - LLM annotators    → Dirichlet likelihood: q_n ~ Dir(alpha_llm * pi_ijk)

Predicts missing ratings (test-set human annotations) via posterior predictive.

Output is compatible with stan/scripts/evaluate_predictions.py — the generated
quantities block has the same variable names as stan_dist_model.stan.

Two-round mode (--use-train-only):
  Round 1: Train on training-instance ratings only (K = K_train + K_test, but
           test item embeddings receive no gradient signal). Posterior means of
           mean_preferences, annotator_preferences, and rating_probs are extracted.
  Round 2: Train on test-instance ratings only, with the Round 1 posterior means
           of mean_preferences, annotator_preferences, and rating_probs frozen as
           fixed data. Only item embeddings are free; train item embeddings receive
           no gradient signal since no train ratings are present. Round 2 fit is
           saved as the final output.

Usage (from repo root):
    python stan/scripts/run_inference_dirichlet.py \\
        --data-bundle OUTPUT/generated_data/llm_rubric_dist/data_bundle.json \\
        --run-name    llmrubric_dir \\
        --alpha-llm   5.0

    # Two-round mode:
    python stan/scripts/run_inference_dirichlet.py \\
        --data-bundle OUTPUT/generated_data/llm_rubric_dist/data_bundle.json \\
        --run-name    llmrubric_dir_tworound \\
        --alpha-llm   5.0 \\
        --use-train-only
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
import sys

import numpy as np

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

# Default Stan model paths (relative to repo root)
_MODELS_DIR   = Path(__file__).resolve().parents[2] / "models"
_STAN_FILE_R1 = str(_MODELS_DIR / "stan_dirichlet_model.stan")
_STAN_FILE_R2 = str(_MODELS_DIR / "stan_dirichlet_model_round2.stan")


# ── Helpers ────────────────────────────────────────────────────────────────────

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

    Epsilon is added to every entry before renormalising so that LLM
    distributions passed to dirichlet_lpdf never contain exact zeros.
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


def _build_stan_data(observed: list, missing: list, data_config: DataGenConfig,
                     K: int, alpha_llm: float) -> dict:
    """Build the base Stan data dict shared by both rounds."""
    C = data_config.C
    return {
        "K": K,
        "I": data_config.I,
        "J": data_config.J,
        "D": data_config.D,
        "C": C,
        "N_ratings":         len(observed),
        "rating_attributes": [r["attribute"] for r in observed],
        "rating_annotators": [r["annotator"]  for r in observed],
        "rating_items":      [r["item"]        for r in observed],
        "rating_values":     [r["value"]        for r in observed],
        "rating_dists":      [_to_dist(r, C)    for r in observed],
        "is_llm_rating":     [_is_llm(r)        for r in observed],
        "N_missing_ratings":              len(missing),
        "missing_rating_attributes":      [r["attribute"] for r in missing],
        "missing_rating_annotators":      [r["annotator"]  for r in missing],
        "missing_rating_items":           [r["item"]        for r in missing],
        "sigma_annotator":   data_config.sigma_annotator,
        "sigma_measurement": data_config.sigma_measurement,
        "alpha_dirichlet":   data_config.alpha_dirichlet,
        "temperature":       data_config.temperature,
        "alpha_llm":         alpha_llm,
    }


def _extract_posterior_means(fit: cmdstanpy.CmdStanMCMC, I: int, J: int,
                              D: int, C: int) -> dict:
    """Extract posterior means of mean_preferences, annotator_preferences,
    and rating_probs from a Round 1 fit object.

    Returns a dict with keys:
      mean_preferences       : np.ndarray [I, D]
      annotator_preferences  : np.ndarray [I*J, D]
      rating_probs           : np.ndarray [I*J, C]
    """
    draws = fit.draws_pd()

    # ── mean_preferences [I, D] ───────────────────────────────────────────────
    mp = np.zeros((I, D))
    for i in range(1, I + 1):
        for d in range(1, D + 1):
            col = f"mean_preferences[{i},{d}]"
            mp[i - 1, d - 1] = draws[col].mean()

    # ── annotator_preferences [I*J, D] ───────────────────────────────────────
    ap = np.zeros((I * J, D))
    for ij in range(1, I * J + 1):
        for d in range(1, D + 1):
            col = f"annotator_preferences[{ij},{d}]"
            ap[ij - 1, d - 1] = draws[col].mean()

    # ── rating_probs [I*J, C] ─────────────────────────────────────────────────
    rp = np.zeros((I * J, C))
    for ij in range(1, I * J + 1):
        for c in range(1, C + 1):
            col = f"rating_probs[{ij},{c}]"
            rp[ij - 1, c - 1] = draws[col].mean()
        # Renormalise to a valid simplex (posterior mean may not sum to exactly 1)
        rp[ij - 1] /= rp[ij - 1].sum()

    return {
        "mean_preferences":      mp,
        "annotator_preferences": ap,
        "rating_probs":          rp,
    }


def _run_sample(model: cmdstanpy.CmdStanModel, stan_data: dict, args,
                label: str) -> cmdstanpy.CmdStanMCMC:
    """Thin wrapper around model.sample with shared MCMC settings."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
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
    divergences = fit.divergences.sum()
    if divergences > 0:
        print(f"\033[93mWARNING ({label}): {divergences} divergent transitions. "
              f"Consider raising --adapt-delta or --max-treedepth.\033[0m")
    else:
        print(f"No divergent transitions in {label} — good mixing!")
    return fit


# ── Main ───────────────────────────────────────────────────────────────────────

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

    # Two-round flag
    parser.add_argument("--use-train-only", action="store_true",
                        help="Two-round inference: Round 1 on train ratings, "
                             "Round 2 on test ratings with annotator params frozen "
                             "from Round 1 posterior means.")

    # MCMC
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--iter-warmup", type=int, default=500)
    parser.add_argument("--iter-sampling", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--adapt-delta", type=float, default=0.85)
    parser.add_argument("--max-treedepth", type=int, default=12)

    # Model overrides
    parser.add_argument("--stan-file", default=None,
                        help="Override path to Round 1 Stan file "
                             "(stan_dirichlet_model.stan)")
    parser.add_argument("--stan-file-round2", default="/home/stone/AnnotationArena/imputer/ranking/models/stan_dirichlet_model_freeze.stan",
                        help="Override path to Round 2 Stan file "
                             "(stan_dirichlet_model_round2.stan). "
                             "Only used with --use-train-only.")
    parser.add_argument("--override-D", type=int, default=None)
    parser.add_argument("--override-sigma-annotator", type=float, default=None)
    parser.add_argument("--override-sigma-measurement", type=float, default=None)
    parser.add_argument("--override-alpha-dirichlet", type=float, default=None)
    parser.add_argument("--override-temperature", type=float, default=None)

    # Dirichlet-specific
    parser.add_argument("--alpha-llm", type=float, default=5.0,
                        help="Dirichlet concentration for LLM observations. "
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

    K  = data_config.K_train + data_config.K_test
    C  = data_config.C
    I  = data_config.I
    J  = data_config.J
    D  = data_config.D

    print(f"\nData config: K={K} (train={data_config.K_train}, test={data_config.K_test}), "
          f"I={I}, J={J}, D={D}, C={C}")

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

    # ── Save augmented configs ─────────────────────────────────────────────────
    augmented_configs = {
        "datagen": {
            "K_train": data_config.K_train,
            "K_test":  data_config.K_test,
            "I":       I,
            "J":       J,
            "D":       D,
            "C":       C,
            "sigma_annotator":   data_config.sigma_annotator,
            "sigma_measurement": data_config.sigma_measurement,
            "alpha_dirichlet":   data_config.alpha_dirichlet,
            "temperature":       data_config.temperature,
        },
        "dirichlet": {
            "alpha_llm":        args.alpha_llm,
            "two_round_mode":   args.use_train_only,
        },
    }
    with open(output_dir / "augmented_configs.json", "w") as f:
        json.dump(augmented_configs, f, indent=2)

    # ── Branch: single-round vs two-round ─────────────────────────────────────
    try:
        if not args.use_train_only:
            # ── Single round (original behaviour) ─────────────────────────────
            observed = bundle.observed_ratings
            missing  = bundle.missing_ratings

            n_llm   = sum(_is_llm(r) for r in observed)
            n_human = len(observed) - n_llm
            print(f"Observed ratings: {len(observed)} total  ({n_human} human, {n_llm} LLM)")
            print(f"Missing ratings:  {len(missing)}")
            print(f"alpha_llm = {args.alpha_llm}")

            stan_data = _build_stan_data(observed, missing, data_config, K, args.alpha_llm)

            stan_file = args.stan_file or _STAN_FILE_R1
            print(f"\nCompiling Stan model (Round 1 / single): {stan_file}")
            model = compile_domain_model(stan_file)

            fit = _run_sample(model, stan_data, args, label="Single-round MCMC")

        else:
            # ── Two-round mode ─────────────────────────────────────────────────

            # ── Round 1: train ratings ─────────────────────────────────────────
            observed_train = [r for r in bundle.observed_ratings
                              if r["instance"] == "train"]
            missing_train  = [r for r in bundle.missing_ratings
                              if r["instance"] == "train"]

            n_llm_r1   = sum(_is_llm(r) for r in observed_train)
            n_human_r1 = len(observed_train) - n_llm_r1
            print(f"\n[Round 1] Train observed: {len(observed_train)} "
                  f"({n_human_r1} human, {n_llm_r1} LLM) | "
                  f"train missing: {len(missing_train)}")

            stan_data_r1 = _build_stan_data(
                observed_train, missing_train, data_config, K, args.alpha_llm
            )

            stan_file_r1 = args.stan_file or _STAN_FILE_R1
            print(f"Compiling Round 1 Stan model: {stan_file_r1}")
            model_r1 = compile_domain_model(stan_file_r1)

            fit_r1 = _run_sample(model_r1, stan_data_r1, args, label="Round 1 (train)")

            # ── Extract Round 1 posterior means ───────────────────────────────
            print("\nExtracting Round 1 posterior means for frozen parameters...")
            frozen = _extract_posterior_means(fit_r1, I, J, D, C)
            print(f"  mean_preferences      shape: {frozen['mean_preferences'].shape}")
            print(f"  annotator_preferences shape: {frozen['annotator_preferences'].shape}")
            print(f"  rating_probs          shape: {frozen['rating_probs'].shape}")

            # ── Round 2: test ratings, frozen annotator params ─────────────────
            observed_test = [r for r in bundle.observed_ratings
                             if r["instance"] == "test"]
            missing_test  = [r for r in bundle.missing_ratings
                             if r["instance"] == "test"]

            n_llm_r2   = sum(_is_llm(r) for r in observed_test)
            n_human_r2 = len(observed_test) - n_llm_r2
            print(f"\n[Round 2] Test observed: {len(observed_test)} "
                  f"({n_human_r2} human, {n_llm_r2} LLM) | "
                  f"test missing: {len(missing_test)}")

            stan_data_r2 = _build_stan_data(
                observed_test, missing_test, data_config, K, args.alpha_llm
            )

            # Add frozen parameters as data for Round 2 model
            stan_data_r2["mean_preferences"]      = frozen["mean_preferences"].tolist()
            stan_data_r2["annotator_preferences"]  = frozen["annotator_preferences"].tolist()
            stan_data_r2["rating_probs"]           = frozen["rating_probs"].tolist()

            stan_file_r2 = args.stan_file_round2 or _STAN_FILE_R2
            print(f"Compiling Round 2 Stan model: {stan_file_r2}")
            model_r2 = compile_domain_model(stan_file_r2)

            fit = _run_sample(model_r2, stan_data_r2, args, label="Round 2 (test, frozen params)")

            # Update augmented configs with rating counts per round
            augmented_configs["dirichlet"].update({
                "round1_n_human_ratings": n_human_r1,
                "round1_n_llm_ratings":   n_llm_r1,
                "round2_n_human_ratings": n_human_r2,
                "round2_n_llm_ratings":   n_llm_r2,
            })
            with open(output_dir / "augmented_configs.json", "w") as f:
                json.dump(augmented_configs, f, indent=2)

    except Exception as e:
        logger.error(f"MCMC failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ── Save final fit ─────────────────────────────────────────────────────────
    print("\nMCMC completed!")
    fit.save_csvfiles(str(output_dir))
    print(f"Samples saved to {output_dir}")


if __name__ == "__main__":
    main()
