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
    # Determine which Stan file to use based on observation protocol
    if stan_file is None:
        if config.observation_protocol == "extended_rankings":
            stan_file = str(Path(__file__).parent.parent.parent / "models" / "extended_rankings_generation.stan")
        else:
            # Use default for both tie_breaking and mar (mar is post-processed)
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
        seed=config.seed,
        show_console=True
    )

    # Extract generated quantities
    bundle = extract_bundle_from_stan_output(fit, config)

    # Apply MAR protocol if specified
    if config.observation_protocol == "mar":
        print(f"\nApplying MAR protocol with missing rate: {config.mar_missing_rate:.2%}")
        bundle = apply_mar_protocol(bundle, config.mar_missing_rate, seed=config.seed)

    return bundle


def extract_bundle_from_stan_output(fit: cmdstanpy.CmdStanMCMC, config: DataGenConfig) -> GroundTruthBundle:
    """Extract GroundTruthBundle from Stan output."""
    
    # Get the single sample (since we used fixed_param=True with 1 sample)
    sample = fit.stan_variables()
    
    # Extract shared parameters (same for train and test)
    mean_preferences = sample["mean_preferences"][0]  # Shape: [I, D] - SHARED
    annotator_preferences = sample["annotator_preferences"][0]  # Shape: [I*J, D] - SHARED
    rating_probs = sample["rating_probs"][0]  # Shape: [I*J, C] - SHARED
    rating_cumprobs = sample["rating_cumprobs"][0]  # Shape: [I*J, C] - SHARED
    rating_thresholds_z = sample["rating_thresholds_z"][0]  # Shape: [I*J, C+1] - SHARED
    
    # Extract training instance data
    train_embeddings = sample["train_embeddings"][0]  # Shape: [K_train, D]
    train_base_scores = sample["train_base_scores"][0]  # Shape: [I*J, K_train]
    train_rating_values = sample["train_rating_values"][0]  # Shape: [I*J, K_train]
    train_rating_observed = sample["train_rating_observed"][0]  # Shape: [I*J, K_train]
    
    # Extract test instance data
    test_embeddings = sample["test_embeddings"][0]  # Shape: [K_test, D]
    test_base_scores = sample["test_base_scores"][0]  # Shape: [I*J, K_test]
    test_rating_values = sample["test_rating_values"][0]  # Shape: [I*J, K_test]
    test_rating_observed = sample["test_rating_observed"][0]  # Shape: [I*J, K_test]
    
    # Extract posterior rating probabilities (optional downstream use)
    train_posterior_rating_probs = sample.get("train_posterior_rating_probs")
    if train_posterior_rating_probs is not None:
        train_posterior_rating_probs = train_posterior_rating_probs[0]
    test_posterior_rating_probs = sample.get("test_posterior_rating_probs")
    if test_posterior_rating_probs is not None:
        test_posterior_rating_probs = test_posterior_rating_probs[0]

    # Extract pairwise rankings
    num_train_pairwise = int(sample["num_train_pairwise_rankings"])
    num_test_pairwise = int(sample["num_test_pairwise_rankings"])
    
    # Training pairwise rankings
    train_pairwise_items = sample["train_pairwise_items"][0, :num_train_pairwise]  # Shape: [N_train, 2]
    train_pairwise_orders = sample["train_pairwise_orders"][0, :num_train_pairwise]  # Shape: [N_train]
    train_pairwise_annotators = sample["train_pairwise_annotator"][0, :num_train_pairwise]  # Shape: [N_train]
    train_pairwise_attributes = sample["train_pairwise_attribute"][0, :num_train_pairwise]  # Shape: [N_train]
    train_pairwise_tied_ratings = sample["train_pairwise_tied_rating"][0, :num_train_pairwise]  # Shape: [N_train]
    train_pairwise_observed = sample["train_pairwise_observed"][0, :num_train_pairwise]  # Shape: [N_train]
    
    # Test pairwise rankings
    test_pairwise_items = sample["test_pairwise_items"][0, :num_test_pairwise]  # Shape: [N_test, 2]
    test_pairwise_orders = sample["test_pairwise_orders"][0, :num_test_pairwise]  # Shape: [N_test]
    test_pairwise_annotators = sample["test_pairwise_annotator"][0, :num_test_pairwise]  # Shape: [N_test]
    test_pairwise_attributes = sample["test_pairwise_attribute"][0, :num_test_pairwise]  # Shape: [N_test]
    test_pairwise_tied_ratings = sample["test_pairwise_tied_rating"][0, :num_test_pairwise]  # Shape: [N_test]
    test_pairwise_observed = sample["test_pairwise_observed"][0, :num_test_pairwise]  # Shape: [N_test]
    
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
    
    # Create missing_ratings_indexes_in_test_instance: indices of missing ratings that are from test set
    missing_ratings_indexes_in_test_instance = [i for i, rating in enumerate(missing_ratings) if rating["instance"] == "test"]
    
    return GroundTruthBundle(
        embeddings=all_embeddings,
        mean_preferences=mean_preferences,
        annotator_preferences=annotator_preferences,
        rating_probs=rating_probs,
        rating_cumprobs=rating_cumprobs,
        rating_thresholds_z=rating_thresholds_z,
        base_scores=all_base_scores,
        all_ratings=all_ratings,
        all_pairwise=all_pairwise,
        observed_ratings=observed_ratings,
        missing_ratings=missing_ratings,
        missing_ratings_indexes_in_test_instance=missing_ratings_indexes_in_test_instance,
        observed_pairwise=observed_pairwise,
        missing_pairwise=missing_pairwise,
        stats=stats,
        train_posterior_rating_probs=train_posterior_rating_probs,
        test_posterior_rating_probs=test_posterior_rating_probs,
    )


def apply_mar_protocol(bundle: GroundTruthBundle, missing_rate: float, seed: Optional[int] = None) -> GroundTruthBundle:
    """
    Apply Missing At Random (MAR) protocol to a bundle.

    Randomly selects missing_rate% of all ratings to mark as missing, regardless of
    the original observation protocol. This creates a completely random missingness pattern.

    Args:
        bundle: GroundTruthBundle with original observation protocol
        missing_rate: Fraction of ratings to mark as missing (e.g., 0.5 for 50%)
        seed: Random seed for reproducibility

    Returns:
        New GroundTruthBundle with MAR observation pattern
    """
    import random

    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Get all ratings
    all_ratings = bundle.all_ratings
    n_total = len(all_ratings)

    # Calculate how many ratings to mark as missing
    n_missing = int(n_total * missing_rate)
    n_observed = n_total - n_missing

    print(f"  Total ratings: {n_total}")
    print(f"  Target missing: {n_missing} ({n_missing / n_total:.2%})")
    print(f"  Target observed: {n_observed} ({n_observed / n_total:.2%})")

    # Randomly select indices to mark as missing
    all_indices = list(range(n_total))
    random.shuffle(all_indices)
    missing_indices = set(all_indices[:n_missing])
    observed_indices = set(all_indices[n_missing:])

    # Split ratings into observed and missing
    observed_ratings = []
    missing_ratings = []

    for idx, rating in enumerate(all_ratings):
        if idx in missing_indices:
            missing_ratings.append(rating)
        else:
            observed_ratings.append(rating)

    # Verify the split
    actual_missing_rate = len(missing_ratings) / n_total
    actual_observed_rate = len(observed_ratings) / n_total

    print(f"  Actual missing: {len(missing_ratings)} ({actual_missing_rate:.2%})")
    print(f"  Actual observed: {len(observed_ratings)} ({actual_observed_rate:.2%})")

    # Sanity check
    assert len(missing_ratings) + len(observed_ratings) == n_total, "Rating counts don't match!"
    assert abs(actual_missing_rate - missing_rate) < 0.01, f"Missing rate mismatch: {actual_missing_rate:.2%} != {missing_rate:.2%}"

    # Split by train/test for statistics
    train_observed = [r for r in observed_ratings if r["instance"] == "train"]
    train_missing = [r for r in missing_ratings if r["instance"] == "train"]
    test_observed = [r for r in observed_ratings if r["instance"] == "test"]
    test_missing = [r for r in missing_ratings if r["instance"] == "test"]

    train_ratings = [r for r in all_ratings if r["instance"] == "train"]
    test_ratings = [r for r in all_ratings if r["instance"] == "test"]

    # Keep all pairwise rankings as observed (MAR only applies to ratings)
    all_pairwise = bundle.all_pairwise
    observed_pairwise = bundle.observed_pairwise
    missing_pairwise = []  # No missing pairwise in MAR protocol

    # Update statistics
    stats = {
        "K_train": bundle.stats["K_train"],
        "K_test": bundle.stats["K_test"],
        "total_items": bundle.stats["total_items"],
        "total_possible_ratings": bundle.stats["total_possible_ratings"],
        "total_ratings": n_total,
        "observed_ratings": len(observed_ratings),
        "missing_ratings": len(missing_ratings),
        "train_ratings": len(train_ratings),
        "test_ratings": len(test_ratings),
        "train_observed": len(train_observed),
        "test_observed": len(test_observed),
        "total_pairwise": len(all_pairwise),
        "observed_pairwise": len(observed_pairwise),
        "missing_pairwise": len(missing_pairwise),  # Should be 0 for MAR protocol
        "train_pairwise": len([p for p in all_pairwise if p["instance"] == "train"]),
        "test_pairwise": len([p for p in all_pairwise if p["instance"] == "test"]),
        "observation_rate": actual_observed_rate,
        "train_observation_rate": len(train_observed) / len(train_ratings) if train_ratings else 0,
        "test_observation_rate": len(test_observed) / len(test_ratings) if test_ratings else 0,
        "protocol": "MAR",
        "mar_missing_rate": missing_rate,
    }

    # Create missing_ratings_indexes_in_test_instance
    missing_ratings_indexes_in_test_instance = [i for i, rating in enumerate(missing_ratings) if rating["instance"] == "test"]

    # Create new bundle with MAR observation pattern
    return GroundTruthBundle(
        embeddings=bundle.embeddings,
        mean_preferences=bundle.mean_preferences,
        annotator_preferences=bundle.annotator_preferences,
        rating_probs=bundle.rating_probs,
        rating_cumprobs=bundle.rating_cumprobs,
        rating_thresholds_z=bundle.rating_thresholds_z,
        base_scores=bundle.base_scores,
        all_ratings=all_ratings,
        all_pairwise=all_pairwise,
        observed_ratings=observed_ratings,
        missing_ratings=missing_ratings,
        missing_ratings_indexes_in_test_instance=missing_ratings_indexes_in_test_instance,
        observed_pairwise=observed_pairwise,
        missing_pairwise=missing_pairwise,
        stats=stats,
        train_posterior_rating_probs=bundle.train_posterior_rating_probs,
        test_posterior_rating_probs=bundle.test_posterior_rating_probs,
    )
