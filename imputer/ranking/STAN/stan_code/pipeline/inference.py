"""
Inference utilities for Stan domain model.

Provides initialization strategies, sampling configuration, and MCMC execution
for the domain model inference.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import cmdstanpy

from .configs import DataGenConfig
from .bundle import GroundTruthBundle

# Set up logging
logger = logging.getLogger(__name__)


class InferenceConfig:
    """Configuration for MCMC inference."""
    
    def __init__(
        self,
        chains: int = 4,
        iter_warmup: int = 1000,
        iter_sampling: int = 1000,
        seed: Optional[int] = None,
        adapt_delta: float = 0.8,
        max_treedepth: int = 10,
        init_strategy: str = "random",  # "random", "ground_truth", "file"
        init_file: Optional[str] = None,
        show_progress: bool = True
    ):
        self.chains = chains
        self.iter_warmup = iter_warmup
        self.iter_sampling = iter_sampling
        self.seed = seed
        self.adapt_delta = adapt_delta
        self.max_treedepth = max_treedepth
        self.init_strategy = init_strategy
        self.init_file = init_file
        self.show_progress = show_progress


def compile_domain_model(stan_file: Optional[str] = None) -> cmdstanpy.CmdStanModel:
    """Compile the domain model for inference."""
    if stan_file is None:
        stan_file = str(Path(__file__).parent.parent.parent / "models" / "domain_model.stan")
    
    logger.info(f"Compiling Stan model from: {stan_file}")
    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    logger.info("Stan model compiled successfully")
    return model


def prepare_stan_data_for_inference(
    bundle: GroundTruthBundle,
    config: DataGenConfig,
    use_train_only: bool = False,
    use_test_only: bool = False,
    stan_arg: Optional[Dict[str, Any]] = None,
    use_dist: bool = False,
) -> Dict[str, Any]:
    """
    Prepare Stan data for inference.

    - When use_dist=False (default), builds Stan data for the standard domain
      model (hard-label ratings + pairwise rankings).
    - When use_dist=True, builds Stan data for the distributional labels model
      (`stan_dist_model.stan`), using `rating_dists` instead of `rating_values`
      and omitting pairwise rankings.

    Hyperparameters always come from `config.to_stan_data()`. Any extra keys
    (e.g. `M`, `S` for the discrete model when fitting on non-discrete data) are
    passed in via `stan_arg` and merged last so the caller controls model-specific
    data.
    """
    if use_train_only and use_test_only:
        raise ValueError("Cannot use both train_only and test_only")
    
    # Filter ratings and pairwise data based on instance
    # When using train_only or test_only, item indices must be re-mapped
    # to 1..K since the bundle uses global indices (test items offset by K_train).
    item_offset = 0  # subtracted from item indices to re-map
    if use_train_only:
        observed_ratings = [r for r in bundle.observed_ratings if r["instance"] == "train"]
        observed_pairwise = [p for p in bundle.observed_pairwise if p["instance"] == "train"]
        K = config.K_train
        item_offset = 0  # train items are already 1..K_train
    elif use_test_only:
        observed_ratings = [r for r in bundle.observed_ratings if r["instance"] == "test"]
        observed_pairwise = [p for p in bundle.observed_pairwise if p["instance"] == "test"]
        K = config.K_test
        item_offset = config.K_train  # test items are K_train+1..K_train+K_test -> 1..K_test
    else:
        # Use both train and test (transductive learning)
        observed_ratings = bundle.observed_ratings
        observed_pairwise = bundle.observed_pairwise
        K = config.K_train + config.K_test

    # Prepare missing variables (filter by instance, re-map item indices)
    if use_train_only:
        missing_ratings = [r for r in bundle.missing_ratings if r["instance"] == "train"]
        missing_pairwise = [p for p in bundle.missing_pairwise if p["instance"] == "train"]
    elif use_test_only:
        missing_ratings = [r for r in bundle.missing_ratings if r["instance"] == "test"]
        missing_pairwise = [p for p in bundle.missing_pairwise if p["instance"] == "test"]
    else:
        missing_ratings = bundle.missing_ratings
        missing_pairwise = bundle.missing_pairwise

    N_missing_ratings = len(missing_ratings)
    missing_rating_attributes = [r["attribute"] for r in missing_ratings]
    missing_rating_annotators = [r["annotator"] for r in missing_ratings]
    missing_rating_items = [r["item"] - item_offset for r in missing_ratings]

    # Dist-mode: build rating_dists and omit pairwise rankings.
    if use_dist:
        C = config.C

        def _to_dist(r: dict) -> list:
            if "rating_dist" in r:
                d = r["rating_dist"]
                s = sum(d)
                return [x / s for x in d]   # normalise for safety
            # Hard label → one-hot
            oh = [0.0] * C
            oh[r["value"] - 1] = 1.0
            return oh

        N_ratings = len(observed_ratings)
        rating_attributes = [r["attribute"] for r in observed_ratings]
        rating_annotators = [r["annotator"] for r in observed_ratings]
        rating_items = [r["item"] - item_offset for r in observed_ratings]
        rating_dists = [_to_dist(r) for r in observed_ratings]

        observed_part = {
            "K": K,
            "I": config.I,
            "J": config.J,
            "D": config.D,
            "C": C,
            "N_ratings": N_ratings,
            "rating_attributes": rating_attributes,
            "rating_annotators": rating_annotators,
            "rating_items": rating_items,
            "rating_dists": rating_dists,
            "N_missing_ratings": N_missing_ratings,
            "missing_rating_attributes": missing_rating_attributes,
            "missing_rating_annotators": missing_rating_annotators,
            "missing_rating_items": missing_rating_items,
        }
        stan_data = {**config.to_stan_data(), **observed_part}
        if stan_arg:
            stan_data.update(stan_arg)
        return stan_data

    # Standard hard-label + pairwise mode.
    N_pairwise_rankings = len(observed_pairwise)
    pairwise_ranking_attributes = [p["attribute"] for p in observed_pairwise]
    pairwise_ranking_annotators = [p["annotator"] for p in observed_pairwise]
    pairwise_ranking_items = [[item - item_offset for item in p["items"]] for p in observed_pairwise]
    pairwise_ranking_orders = [p["order"][0] for p in observed_pairwise]  # 1 if item1 > item2, 2 if item2 > item1

    N_ratings = len(observed_ratings)
    rating_attributes = [r["attribute"] for r in observed_ratings]
    rating_annotators = [r["annotator"] for r in observed_ratings]
    rating_items = [r["item"] - item_offset for r in observed_ratings]
    rating_values = [r["value"] for r in observed_ratings]

    observed_part = {
        "K": K,
        "N_ratings": N_ratings,
        "rating_attributes": rating_attributes,
        "rating_annotators": rating_annotators,
        "rating_items": rating_items,
        "rating_values": rating_values,
        # Observed pairwise rankings
        "N_pairwise_rankings": N_pairwise_rankings,
        "pairwise_ranking_attributes": pairwise_ranking_attributes,
        "pairwise_ranking_annotators": pairwise_ranking_annotators,
        "pairwise_ranking_items": pairwise_ranking_items,
        "pairwise_ranking_orders": pairwise_ranking_orders,
        # Missing variables
        "N_missing_ratings": N_missing_ratings,
        "missing_rating_attributes": missing_rating_attributes,
        "missing_rating_annotators": missing_rating_annotators,
        "missing_rating_items": missing_rating_items,
        "N_missing_pairwise_rankings": len(missing_pairwise),
        "missing_pairwise_ranking_attributes": [p["attribute"] for p in missing_pairwise],
        "missing_pairwise_ranking_annotators": [p["annotator"] for p in missing_pairwise],
        "missing_pairwise_ranking_items": [[item - item_offset for item in p["items"]] for p in missing_pairwise],
    }

    # Hyperparameters from config; then any --stan-arg overrides/adds (e.g. M, S for discrete model)
    stan_data = {**config.to_stan_data(), **observed_part}
    if stan_arg:
        stan_data.update(stan_arg)
    return stan_data


def create_init_from_ground_truth(
    bundle: GroundTruthBundle,
    config: DataGenConfig,
    use_train_only: bool = False,
    use_test_only: bool = False
) -> Dict[str, Any]:
    """
    Create initialization values from ground truth.
    
    Args:
        bundle: Generated data bundle with ground truth
        config: Data generation configuration
        use_train_only: If True, only use training instance data
        use_test_only: If True, only use test instance data
    
    Returns:
        Dictionary of initialization values for Stan
    """
    if use_train_only:
        K = config.K_train
        # Use training embeddings
        embeddings = bundle.embeddings[:config.K_train]
    elif use_test_only:
        K = config.K_test
        # Use test embeddings (offset by K_train)
        embeddings = bundle.embeddings[config.K_train:]
    else:
        # Use both train and test embeddings
        K = config.K_train + config.K_test
        embeddings = bundle.embeddings
    
    init_values = {
        "embeddings": embeddings,
        "mean_preferences": bundle.mean_preferences,
        "annotator_preferences": bundle.annotator_preferences,
        "rating_probs": bundle.rating_probs,
    }
    
    return init_values


def run_mcmc_inference(
    bundle: GroundTruthBundle,
    config: DataGenConfig,
    inference_config: InferenceConfig,
    stan_file: Optional[str] = None,
    stan_arg: Optional[Dict[str, Any]] = None,
    use_train_only: bool = False,
    use_test_only: bool = False,
    use_dist: bool = False,
) -> cmdstanpy.CmdStanMCMC:
    """
    Run MCMC inference using either the standard domain model or the
    distributional-labels model.

    Hyperparameters come from config.to_stan_data(); stan_arg (e.g. from
    --stan-arg) is merged so the caller can supply model-specific data (e.g.
    M, S for discrete, DEBUG_INIT for tensor, etc.).
    
    Args:
        bundle: Generated data bundle
        config: Data generation configuration (from the run that produced the bundle)
        inference_config: MCMC configuration
        stan_file: Path to the Stan model file (domain or dist)
        stan_arg: Optional Stan data to merge (e.g. {"M": 6, "S": 3} for discrete model)
        use_train_only: If True, only use training instance data
        use_test_only: If True, only use test instance data
        use_dist: If True, build Stan data for the dist model (rating_dists)
    
    Returns:
        CmdStanMCMC fit object
    """
    # Compile model
    model = compile_domain_model(stan_file)
    
    # Prepare Stan data (config + observed; stan_arg overrides/adds)
    stan_data = prepare_stan_data_for_inference(
        bundle,
        config,
        use_train_only=use_train_only,
        use_test_only=use_test_only,
        stan_arg=stan_arg,
        use_dist=use_dist,
    )

    if use_dist:
        logger.info(
            "Dist mode Stan data: K=%s, N_ratings=%s, N_missing_ratings=%s",
            stan_data.get("K"),
            stan_data.get("N_ratings"),
            stan_data.get("N_missing_ratings"),
        )
    
    # Prepare initialization
    logger.info(f"Preparing initialization with strategy: {inference_config.init_strategy}")
    if inference_config.init_strategy == "ground_truth":
        init_values = create_init_from_ground_truth(bundle, config, use_train_only, use_test_only)
        init = [init_values] * inference_config.chains
        logger.info(f"Using ground truth initialization for {inference_config.chains} chains")
    elif inference_config.init_strategy == "file" and inference_config.init_file and inference_config.init_file != "random":
        init = inference_config.init_file
        logger.info(f"Using initialization file: {inference_config.init_file}")
    else:
        # "random" or no valid file: use numeric init so CmdStanPy does not treat a path
        init = 1.0
        logger.info(f"Using default initialization (1.0) for {inference_config.chains} chains")
    
    # Run MCMC sampling
    logger.info(f"Starting MCMC sampling with {inference_config.chains} chains")
    logger.info(f"Warmup iterations: {inference_config.iter_warmup}, Sampling iterations: {inference_config.iter_sampling}")
    logger.info(f"Adapt delta: {inference_config.adapt_delta}, Max tree depth: {inference_config.max_treedepth}")
    logger.info(f"Seed: {inference_config.seed}")
    
    fit = model.sample(
        data=stan_data,
        chains=inference_config.chains,
        iter_warmup=inference_config.iter_warmup,
        iter_sampling=inference_config.iter_sampling,
        seed=inference_config.seed,
        adapt_delta=inference_config.adapt_delta,
        max_treedepth=inference_config.max_treedepth,
        inits=init,
        show_progress=inference_config.show_progress,
        show_console=True
    )
    
    logger.info("MCMC sampling completed successfully")
    logger.info(f"Divergent transitions: {fit.divergences}")
    
    return fit
