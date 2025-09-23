"""
Python wrapper for Stan data generation.

Interfaces with iclr_data_generation.stan to generate synthetic datasets
and convert Stan output into GroundTruthBundle format.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import cmdstanpy

from .configs import DataGenConfig
from .bundle import GroundTruthBundle


def compile_data_generation_model(stan_file: str) -> cmdstanpy.CmdStanModel:
    """Compile the data generation Stan model."""
    return cmdstanpy.CmdStanModel(stan_file=stan_file)


def generate_data(config: DataGenConfig, stan_file: Optional[str] = None) -> GroundTruthBundle:
    """
    Generate synthetic data using Stan data generation model.
    
    Args:
        config: Data generation configuration
        stan_file: Path to iclr_data_generation.stan (defaults to models/iclr_data_generation.stan)
    
    Returns:
        GroundTruthBundle with generated data and ground truth parameters
    """
    if stan_file is None:
        stan_file = str(Path(__file__).parent.parent.parent / "models" / "iclr_data_generation.stan")
    
    # Compile model
    model = compile_data_generation_model(stan_file)
    
    # Prepare Stan data
    stan_data = {
        "K": config.K,
        "I": config.I, 
        "J": config.J,
        "D": config.D,
        "C": config.C,
        "pairwise_cap_per_item": config.pairwise_cap_per_item,
        "sigma_annotator": config.sigma_annotator,
        "sigma_measurement": config.sigma_measurement,
        "alpha_dirichlet": config.alpha_dirichlet,
        "temperature": config.temperature,
    }
    
    # Sample with fixed parameters (data generation)
    fit = model.sample(
        data=stan_data,
        fixed_param=True,
        chains=1,
        iter_sampling=1,
        seed=config.seed
    )
    
    # Extract generated quantities
    bundle = extract_bundle_from_stan_output(fit, config)
    
    return bundle


def extract_bundle_from_stan_output(fit: cmdstanpy.CmdStanMCMC, config: DataGenConfig) -> GroundTruthBundle:
    """Extract GroundTruthBundle from Stan output."""
    
    # Get the single sample (since we used fixed_param=True with 1 sample)
    sample = fit.stan_variables()
    
    # Extract parameters
    embeddings = sample["embeddings"]  # Shape: [K, D]
    mean_preferences = sample["mean_preferences"]  # Shape: [I, D]
    annotator_preferences = sample["annotator_preferences"]  # Shape: [I*J, D]
    rating_probs = sample["rating_probs"]  # Shape: [I*J, C]
    rating_thresholds = sample["rating_thresholds"]  # Shape: [I*J, C]
    base_scores = sample["base_scores"]  # Shape: [I*J, K]
    
    # Extract annotations
    all_rating_values = sample["all_rating_values"]  # Shape: [I*J, K]
    all_rating_observed = sample["all_rating_observed"]  # Shape: [I*J, K]
    
    # Extract pairwise rankings
    num_pairwise_rankings = int(sample["num_pairwise_rankings"])
    pairwise_items = sample["pairwise_ranking_items"][:num_pairwise_rankings]  # Shape: [N, 2]
    pairwise_orders = sample["pairwise_ranking_orders"][:num_pairwise_rankings]  # Shape: [N]
    pairwise_annotators = sample["pairwise_ranking_annotator"][:num_pairwise_rankings]  # Shape: [N]
    pairwise_attributes = sample["pairwise_ranking_attribute"][:num_pairwise_rankings]  # Shape: [N]
    pairwise_tied_ratings = sample["pairwise_ranking_tied_rating"][:num_pairwise_rankings]  # Shape: [N]
    pairwise_observed = sample["pairwise_ranking_observed"][:num_pairwise_rankings]  # Shape: [N]
    
    # Convert to list format
    all_ratings = []
    observed_ratings = []
    missing_ratings = []
    
    for i in range(config.I):
        for j in range(config.J):
            ij_idx = i * config.J + j
            for k in range(config.K):
                rating_dict = {
                    "attribute": i + 1,
                    "annotator": j + 1,
                    "item": k + 1,
                    "value": int(all_rating_values[ij_idx, k])
                }
                all_ratings.append(rating_dict)
                
                if all_rating_observed[ij_idx, k] == 1:
                    observed_ratings.append(rating_dict)
                else:
                    missing_ratings.append(rating_dict)
    
    # Convert pairwise rankings to list format
    all_pairwise = []
    observed_pairwise = []
    missing_pairwise = []
    
    for n in range(num_pairwise_rankings):
        pairwise_dict = {
            "attribute": int(pairwise_attributes[n]),
            "annotator": int(pairwise_annotators[n]),
            "items": [int(pairwise_items[n, 0]), int(pairwise_items[n, 1])],
            "order": [1, 2] if pairwise_orders[n] == 1 else [2, 1],
            "tied_rating": int(pairwise_tied_ratings[n])
        }
        all_pairwise.append(pairwise_dict)
        
        if pairwise_observed[n] == 1:
            observed_pairwise.append(pairwise_dict)
        else:
            missing_pairwise.append(pairwise_dict)
    
    # Compute statistics
    stats = {
        "total_possible_ratings": config.I * config.J * config.K,
        "total_ratings": len(all_ratings),
        "observed_ratings": len(observed_ratings),
        "missing_ratings": len(missing_ratings),
        "total_pairwise": len(all_pairwise),
        "observed_pairwise": len(observed_pairwise),
        "missing_pairwise": len(missing_pairwise),
        "observation_rate": len(observed_ratings) / len(all_ratings) if all_ratings else 0
    }
    
    return GroundTruthBundle(
        embeddings=embeddings,
        mean_preferences=mean_preferences,
        annotator_preferences=annotator_preferences,
        rating_probs=rating_probs,
        rating_thresholds=rating_thresholds,
        base_scores=base_scores,
        all_ratings=all_ratings,
        all_pairwise=all_pairwise,
        observed_ratings=observed_ratings,
        missing_ratings=missing_ratings,
        observed_pairwise=observed_pairwise,
        missing_pairwise=missing_pairwise,
        stats=stats
    )
