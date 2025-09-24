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
        "K_train": config.K_train,
        "K_test": config.K_test,
        "I": config.I, 
        "J": config.J,
        "D": config.D,
        "C": config.C,
        "enable_third_annotator": 1 if config.enable_third_annotator else 0,
        "enable_pairwise_rankings": 1 if config.enable_pairwise_rankings else 0,
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
    
    # Extract shared parameters (same for train and test)
    mean_preferences = sample["mean_preferences"]  # Shape: [I, D] - SHARED
    annotator_preferences = sample["annotator_preferences"]  # Shape: [I*J, D] - SHARED
    rating_probs = sample["rating_probs"]  # Shape: [I*J, C] - SHARED
    rating_thresholds = sample["rating_thresholds"]  # Shape: [I*J, C] - SHARED
    
    # Extract training instance data
    train_embeddings = sample["train_embeddings"]  # Shape: [K_train, D]
    train_base_scores = sample["train_base_scores"]  # Shape: [I*J, K_train]
    train_rating_values = sample["train_rating_values"]  # Shape: [I*J, K_train]
    train_rating_observed = sample["train_rating_observed"]  # Shape: [I*J, K_train]
    
    # Extract test instance data
    test_embeddings = sample["test_embeddings"]  # Shape: [K_test, D]
    test_base_scores = sample["test_base_scores"]  # Shape: [I*J, K_test]
    test_rating_values = sample["test_rating_values"]  # Shape: [I*J, K_test]
    test_rating_observed = sample["test_rating_observed"]  # Shape: [I*J, K_test]
    
    # Extract pairwise rankings
    num_train_pairwise = int(sample["num_train_pairwise_rankings"])
    num_test_pairwise = int(sample["num_test_pairwise_rankings"])
    
    # Training pairwise rankings
    train_pairwise_items = sample["train_pairwise_items"][:num_train_pairwise]  # Shape: [N_train, 2]
    train_pairwise_orders = sample["train_pairwise_orders"][:num_train_pairwise]  # Shape: [N_train]
    train_pairwise_annotators = sample["train_pairwise_annotator"][:num_train_pairwise]  # Shape: [N_train]
    train_pairwise_attributes = sample["train_pairwise_attribute"][:num_train_pairwise]  # Shape: [N_train]
    train_pairwise_tied_ratings = sample["train_pairwise_tied_rating"][:num_train_pairwise]  # Shape: [N_train]
    train_pairwise_observed = sample["train_pairwise_observed"][:num_train_pairwise]  # Shape: [N_train]
    
    # Test pairwise rankings
    test_pairwise_items = sample["test_pairwise_items"][:num_test_pairwise]  # Shape: [N_test, 2]
    test_pairwise_orders = sample["test_pairwise_orders"][:num_test_pairwise]  # Shape: [N_test]
    test_pairwise_annotators = sample["test_pairwise_annotator"][:num_test_pairwise]  # Shape: [N_test]
    test_pairwise_attributes = sample["test_pairwise_attribute"][:num_test_pairwise]  # Shape: [N_test]
    test_pairwise_tied_ratings = sample["test_pairwise_tied_rating"][:num_test_pairwise]  # Shape: [N_test]
    test_pairwise_observed = sample["test_pairwise_observed"][:num_test_pairwise]  # Shape: [N_test]
    
    # Convert training ratings to list format
    train_ratings = []
    train_observed_ratings = []
    train_missing_ratings = []
    
    for i in range(config.I):
        for j in range(config.J):
            ij_idx = i * config.J + j
            for k in range(config.K_train):
                rating_dict = {
                    "attribute": i + 1,
                    "annotator": j + 1,
                    "item": k + 1,
                    "value": int(train_rating_values[ij_idx, k]),
                    "instance": "train"
                }
                train_ratings.append(rating_dict)
                
                if train_rating_observed[ij_idx, k] == 1:
                    train_observed_ratings.append(rating_dict)
                else:
                    train_missing_ratings.append(rating_dict)
    
    # Convert test ratings to list format
    test_ratings = []
    test_observed_ratings = []
    test_missing_ratings = []
    
    for i in range(config.I):
        for j in range(config.J):
            ij_idx = i * config.J + j
            for k in range(config.K_test):
                rating_dict = {
                    "attribute": i + 1,
                    "annotator": j + 1,
                    "item": k + 1 + config.K_train,  # Offset by training items
                    "value": int(test_rating_values[ij_idx, k]),
                    "instance": "test"
                }
                test_ratings.append(rating_dict)
                
                if test_rating_observed[ij_idx, k] == 1:
                    test_observed_ratings.append(rating_dict)
                else:
                    test_missing_ratings.append(rating_dict)
    
    # Combine all ratings
    all_ratings = train_ratings + test_ratings
    observed_ratings = train_observed_ratings + test_observed_ratings
    missing_ratings = train_missing_ratings + test_missing_ratings
    
    # Convert training pairwise rankings to list format
    train_pairwise = []
    train_observed_pairwise = []
    train_missing_pairwise = []
    
    for n in range(num_train_pairwise):
        pairwise_dict = {
            "attribute": int(train_pairwise_attributes[n]),
            "annotator": int(train_pairwise_annotators[n]),
            "items": [int(train_pairwise_items[n, 0]), int(train_pairwise_items[n, 1])],
            "order": [1, 2] if train_pairwise_orders[n] == 1 else [2, 1],
            "tied_rating": int(train_pairwise_tied_ratings[n]),
            "instance": "train"
        }
        train_pairwise.append(pairwise_dict)
        
        if train_pairwise_observed[n] == 1:
            train_observed_pairwise.append(pairwise_dict)
        else:
            train_missing_pairwise.append(pairwise_dict)
    
    # Convert test pairwise rankings to list format
    test_pairwise = []
    test_observed_pairwise = []
    test_missing_pairwise = []
    
    for n in range(num_test_pairwise):
        pairwise_dict = {
            "attribute": int(test_pairwise_attributes[n]),
            "annotator": int(test_pairwise_annotators[n]),
            "items": [int(test_pairwise_items[n, 0]) + config.K_train, 
                     int(test_pairwise_items[n, 1]) + config.K_train],  # Offset by training items
            "order": [1, 2] if test_pairwise_orders[n] == 1 else [2, 1],
            "tied_rating": int(test_pairwise_tied_ratings[n]),
            "instance": "test"
        }
        test_pairwise.append(pairwise_dict)
        
        if test_pairwise_observed[n] == 1:
            test_observed_pairwise.append(pairwise_dict)
        else:
            test_missing_pairwise.append(pairwise_dict)
    
    # Combine all pairwise rankings
    all_pairwise = train_pairwise + test_pairwise
    observed_pairwise = train_observed_pairwise + test_observed_pairwise
    missing_pairwise = train_missing_pairwise + test_missing_pairwise
    
    # Combine item embeddings (train + test), Notice this is not used for the imputer, just for debugging the domain model.
    all_embeddings = np.vstack([train_embeddings, test_embeddings])  # Shape: [K_train + K_test, D]
    
    # Combine base scores (train + test)
    all_base_scores = np.hstack([train_base_scores, test_base_scores])  # Shape: [I*J, K_train + K_test]
    
    # Compute statistics
    total_items = config.K_train + config.K_test
    stats = {
        "K_train": config.K_train,
        "K_test": config.K_test,
        "total_items": total_items,
        "total_possible_ratings": config.I * config.J * total_items,
        "total_ratings": len(all_ratings),
        "observed_ratings": len(observed_ratings),
        "missing_ratings": len(missing_ratings),
        "train_ratings": len(train_ratings),
        "test_ratings": len(test_ratings),
        "train_observed": len(train_observed_ratings),
        "test_observed": len(test_observed_ratings),
        "total_pairwise": len(all_pairwise),
        "observed_pairwise": len(observed_pairwise),
        "missing_pairwise": len(missing_pairwise),
        "train_pairwise": len(train_pairwise),
        "test_pairwise": len(test_pairwise),
        "observation_rate": len(observed_ratings) / len(all_ratings) if all_ratings else 0,
        "train_observation_rate": len(train_observed_ratings) / len(train_ratings) if train_ratings else 0,
        "test_observation_rate": len(test_observed_ratings) / len(test_ratings) if test_ratings else 0
    }
    
    return GroundTruthBundle(
        embeddings=all_embeddings,
        mean_preferences=mean_preferences,
        annotator_preferences=annotator_preferences,
        rating_probs=rating_probs,
        rating_thresholds=rating_thresholds,
        base_scores=all_base_scores,
        all_ratings=all_ratings,
        all_pairwise=all_pairwise,
        observed_ratings=observed_ratings,
        missing_ratings=missing_ratings,
        observed_pairwise=observed_pairwise,
        missing_pairwise=missing_pairwise,
        stats=stats
    )
