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
    use_test_only: bool = False
) -> Dict[str, Any]:
    """
    Prepare Stan data for domain model inference.
    
    Args:
        bundle: Generated data bundle
        config: Data generation configuration
        use_train_only: If True, only use training instance data
        use_test_only: If True, only use test instance data
    
    Returns:
        Dictionary of Stan data for domain model
    """
    if use_train_only and use_test_only:
        raise ValueError("Cannot use both train_only and test_only")
    
    # Filter ratings and pairwise data based on instance
    if use_train_only:
        observed_ratings = [r for r in bundle.observed_ratings if r["instance"] == "train"]
        observed_pairwise = [p for p in bundle.observed_pairwise if p["instance"] == "train"]
        K = config.K_train
    elif use_test_only:
        observed_ratings = [r for r in bundle.observed_ratings if r["instance"] == "test"]
        observed_pairwise = [p for p in bundle.observed_pairwise if p["instance"] == "test"]
        K = config.K_test
    else:
        # Use both train and test (transductive learning)
        observed_ratings = bundle.observed_ratings
        observed_pairwise = bundle.observed_pairwise
        K = config.K_train + config.K_test

    # Convert ratings to Stan format
    N_ratings = len(observed_ratings)
    rating_attributes = [r["attribute"] for r in observed_ratings]
    rating_annotators = [r["annotator"] for r in observed_ratings]
    rating_items = [r["item"] for r in observed_ratings]
    rating_values = [r["value"] for r in observed_ratings]

    # Convert pairwise rankings to Stan format
    N_pairwise_rankings = len(observed_pairwise)
    pairwise_ranking_attributes = [p["attribute"] for p in observed_pairwise]
    pairwise_ranking_annotators = [p["annotator"] for p in observed_pairwise]
    pairwise_ranking_items = [p["items"] for p in observed_pairwise]
    pairwise_ranking_orders = [p["order"][0] for p in observed_pairwise]  # 1 if item1 > item2, 2 if item2 > item1

    # Prepare missing variables (all missing ratings and pairwise)
    missing_ratings = bundle.missing_ratings
    missing_pairwise = bundle.missing_pairwise

    N_missing_ratings = len(missing_ratings)
    missing_rating_attributes = [r["attribute"] for r in missing_ratings]
    missing_rating_annotators = [r["annotator"] for r in missing_ratings]
    missing_rating_items = [r["item"] for r in missing_ratings]

    N_missing_pairwise_rankings = len(missing_pairwise)
    missing_pairwise_ranking_attributes = [p["attribute"] for p in missing_pairwise]
    missing_pairwise_ranking_annotators = [p["annotator"] for p in missing_pairwise]
    missing_pairwise_ranking_items = [p["items"] for p in missing_pairwise]
    
    stan_data = {
        # Dimensions
        "K": K,
        "I": config.I,
        "J": config.J,
        "D": config.D,
        "C": config.C,
        
        # Observed ratings
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
        
        "N_missing_pairwise_rankings": N_missing_pairwise_rankings,
        "missing_pairwise_ranking_attributes": missing_pairwise_ranking_attributes,
        "missing_pairwise_ranking_annotators": missing_pairwise_ranking_annotators,
        "missing_pairwise_ranking_items": missing_pairwise_ranking_items,
        
        # Hyperparameters
        "sigma_annotator": config.sigma_annotator,
        "sigma_measurement": config.sigma_measurement,
        "alpha_dirichlet": config.alpha_dirichlet,
        "temperature": config.temperature,
    }
    
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

    print('Initial values: ', init_values)
    
    return init_values


def run_mcmc_inference(
    bundle: GroundTruthBundle,
    config: DataGenConfig,
    inference_config: InferenceConfig,
    stan_file: Optional[str] = None,
    use_train_only: bool = False,
    use_test_only: bool = False
) -> cmdstanpy.CmdStanMCMC:
    """
    Run MCMC inference using the domain model.
    
    Args:
        bundle: Generated data bundle
        config: Data generation configuration
        inference_config: MCMC configuration
        stan_file: Path to domain_model.stan (defaults to models/domain_model.stan)
        use_train_only: If True, only use training instance data
        use_test_only: If True, only use test instance data
    
    Returns:
        CmdStanMCMC fit object
    """
    # Compile model
    model = compile_domain_model(stan_file)
    
    # Prepare Stan data (model hyperparameters and observed data)
    stan_data = prepare_stan_data_for_inference(bundle, config, use_train_only, use_test_only)
    
    # Prepare initialization
    logger.info(f"Preparing initialization with strategy: {inference_config.init_strategy}")
    if inference_config.init_strategy == "ground_truth":
        init_values = create_init_from_ground_truth(bundle, config, use_train_only, use_test_only)
        init = init_values
        logger.info(f"Using ground truth initialization for {inference_config.chains} chains")
    elif inference_config.init_strategy == "file" and inference_config.init_file:
        init = inference_config.init_file
        logger.info(f"Using initialization file: {inference_config.init_file}")
    else:  # "random"
        # Use range [-2, 2] for random initialization
        init = 1.0
        logger.info(f"Using random initialization with range [-2, 2] for {inference_config.chains} chains")
    
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
        show_console=False
    )
    
    logger.info("MCMC sampling completed successfully")
    logger.info(f"Divergent transitions: {fit.divergences}")
    
    return fit
