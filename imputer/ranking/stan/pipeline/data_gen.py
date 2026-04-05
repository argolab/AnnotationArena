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


def generate_data(
    config: DataGenConfig,
    stan_file: Optional[str] = None,
) -> GroundTruthBundle:
    """
    Generate synthetic data using Stan data generation model.

    Args:
        config: Data generation configuration; config must have stan_type and exactly the
                type-specific Stan fields set (CLI does this). Stan data is config.to_stan_data().
        stan_file: Path to .stan file (defaults from config.stan_type)

    Returns:
        GroundTruthBundle with generated data and ground truth parameters
    """
    stan_type = config.stan_type
    if stan_file is None:
        if stan_type == "discrete":
            stan_file = str(Path(__file__).parent.parent.parent / "stan_models" / "discrete_type_data_generation.stan")
        elif stan_type == "tensor":
            stan_file = str(Path(__file__).parent.parent.parent / "stan_models" / "tensor_generation.stan")
        elif stan_type in ("normal-noise-dot-product", "factored-dot-product"):
            stan_file = str(Path(__file__).parent.parent.parent / "stan_models" / "normal_noise_dot_product_generation.stan")
        elif getattr(config, "observation_protocol", None) == "extended_rankings":
            import warnings
            warnings.warn(
                "observation_protocol='extended_rankings' is deprecated; using iclr_data_generation.stan",
                UserWarning,
                stacklevel=2,
            )
            stan_file = str(Path(__file__).parent.parent.parent / "models" / "iclr_data_generation.stan")
        else:
            stan_file = str(Path(__file__).parent.parent.parent / "models" / "iclr_data_generation.stan")

    model = compile_data_generation_model(stan_file)
    stan_data = config.to_stan_data()  # Core + type-specific fields dump to json and passed to Stan

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

    # Apply MCAR protocol if specified
    if config.observation_protocol == "mcar":
        print(f"\nApplying MCAR protocol with missing rate: {config.mcar_missing_rate:.2%}")
        bundle = apply_mcar_protocol(bundle, config.mcar_missing_rate, seed=config.seed)

    # Apply pairwise observation rate if specified (for tie_breaking protocol)
    if config.observation_protocol == "tie_breaking" and config.pairwise_observation_rate < 1.0:
        print(f"\nApplying pairwise observation rate: {config.pairwise_observation_rate:.2%}")
        bundle = apply_pairwise_observation_rate(bundle, config.pairwise_observation_rate, seed=config.seed)

    return bundle


def _is_cross_instance(annotator: int, item: int, config: DataGenConfig) -> bool:
    """
    Check if a rating is cross-instance.
    
    For tie_breaking protocol:
    - Train-only annotators: 1..J/3 (can only rate train items)
    - Overlap annotators: J/3+1..2J/3 (can rate both train and test items)
    - Test-only annotators: 2J/3+1..J (can only rate test items)
    
    Cross-instance means:
    - Train-only annotator (1..J/3) rating test item (item > K_train)
    - Test-only annotator (2J/3+1..J) rating train item (item <= K_train)
    
    Overlap annotators (J/3+1..2J/3) rating either train or test items are NOT cross-instance.
    """
    if config.observation_protocol != "tie_breaking":
        return False
    
    if item < 1 or item > (config.K_train + config.K_test):
        raise ValueError(f"Item index {item} is out of range (1-{config.K_train + config.K_test})")
    
    train_only_end = config.J // 3
    test_only_start = (2 * config.J) // 3 + 1
    
    is_train_only_annotator = annotator <= train_only_end
    is_test_only_annotator = annotator >= test_only_start
    is_train_item = item <= config.K_train
    is_test_item = item > config.K_train
    
    # Cross-instance: train-only annotator rating test item OR test-only annotator rating train item
    return (is_train_only_annotator and is_test_item) or (is_test_only_annotator and is_train_item)


def _is_cross_instance_pairwise(annotator: int, items: List[int], config: DataGenConfig) -> bool:
    """
    Check if a pairwise ranking is cross-instance.
    
    For tie_breaking protocol:
    - Train-only annotators: 1..J/3 (can only rate train items)
    - Overlap annotators: J/3+1..2J/3 (can rate both train and test items)
    - Test-only annotators: 2J/3+1..J (can only rate test items)
    
    Cross-instance means:
    - Train-only annotator (1..J/3) with test items (all items > K_train)
    - Test-only annotator (2J/3+1..J) with train items (all items <= K_train)
    - Items spanning both instances (some <= K_train, some > K_train) - invalid case
    
    NOT cross-instance:
    - Train-only annotator with train items
    - Test-only annotator with test items
    - Overlap annotator (J/3+1..2J/3) with either train or test items
    """
    if config.observation_protocol != "tie_breaking":
        return False
    
    train_only_end = config.J // 3
    test_only_start = (2 * config.J) // 3 + 1
    
    is_train_only_annotator = annotator <= train_only_end
    is_test_only_annotator = annotator >= test_only_start
    
    # Check if items are train, test, or span both instances
    for item in items:
        if item < 1 or item > (config.K_train + config.K_test):
            raise ValueError(f"Item index {item} is out of range (1-{config.K_train + config.K_test})")
    
    # Cross-instance: train-only annotator with any test items OR test-only annotator with any train items
    return (is_train_only_annotator and any(item > config.K_train for item in items)) or \
           (is_test_only_annotator and any(item <= config.K_train for item in items))


def extract_bundle_from_stan_output(fit: cmdstanpy.CmdStanMCMC, config: DataGenConfig) -> GroundTruthBundle:
    """Extract GroundTruthBundle from Stan output.
    
    Dynamically extracts available fields based on what the generator outputs,
    supporting different generator types (iclr, discrete_type, etc.).
    """
    
    # Get the single sample (since we used fixed_param=True with 1 sample)
    sample = fit.stan_variables()
    
    # Extract core data structures (always present)
    train_rating_values = sample["train_rating_values"][0]  # Shape: [I*J, K_train]
    train_rating_observed = sample["train_rating_observed"][0]  # Shape: [I*J, K_train]
    test_rating_values = sample["test_rating_values"][0]  # Shape: [I*J, K_test]
    test_rating_observed = sample["test_rating_observed"][0]  # Shape: [I*J, K_test]
    
    num_classes = config.C
    # Extract optional standard embedding-world ground truth (if present)
    mean_preferences = sample.get("mean_preferences")
    if mean_preferences is not None:
        mean_preferences = mean_preferences[0]
    
    annotator_preferences = sample.get("annotator_preferences")
    if annotator_preferences is not None:
        annotator_preferences = annotator_preferences[0]
    
    rating_probs = sample.get("rating_probs")
    if rating_probs is not None:
        rating_probs = rating_probs[0]
    
    rating_cumprobs = sample.get("rating_cumprobs")
    if rating_cumprobs is not None:
        rating_cumprobs = rating_cumprobs[0]
    
    rating_thresholds_z = sample.get("rating_thresholds_z")
    if rating_thresholds_z is not None:
        rating_thresholds_z = rating_thresholds_z[0]
    
    # Extract embeddings and base scores (if present)
    train_embeddings = sample.get("train_embeddings")
    test_embeddings = sample.get("test_embeddings")
    train_base_scores = sample.get("train_base_scores")
    test_base_scores = sample.get("test_base_scores")
    
    if train_embeddings is not None:
        train_embeddings = train_embeddings[0]
    if test_embeddings is not None:
        test_embeddings = test_embeddings[0]
    if train_base_scores is not None:
        train_base_scores = train_base_scores[0]
    if test_base_scores is not None:
        test_base_scores = test_base_scores[0]
    
    # Extract posterior rating probabilities (optional downstream use)
    train_posterior_rating_probs = sample.get("train_posterior_rating_probs")
    if train_posterior_rating_probs is not None:
        train_posterior_rating_probs = train_posterior_rating_probs[0]
    test_posterior_rating_probs = sample.get("test_posterior_rating_probs")
    if test_posterior_rating_probs is not None:
        test_posterior_rating_probs = test_posterior_rating_probs[0]
    
    # Extract extra ground truth (generator-specific fields)
    extra_ground_truth = {}
    
    # Factored annotator model fields (iclr generator with use_factored_annotator=1)
    for key in ["annotator_embeddings", "attr_transforms", "threshold_transform_W", "threshold_attr_bias"]:
        if key in sample:
            extra_ground_truth[key] = sample[key][0]
    
    # Discrete prototype-style fields (discrete_type generator)
    for key in ["z_train", "z_test", "s_of_j", "a_attr", "u_proto", "v_style", "delta_ims", "mu_ims"]:
        if key in sample:
            extra_ground_truth[key] = sample[key][0]

    # Extract pairwise rankings
    num_train_pairwise = 0
    num_test_pairwise = 0


    # Training pairwise rankings
    if not num_train_pairwise == 0:
        train_pairwise_items = sample["train_pairwise_items"][0, :num_train_pairwise]  # Shape: [N_train, 2]
        train_pairwise_orders = sample["train_pairwise_orders"][0, :num_train_pairwise]  # Shape: [N_train]
        train_pairwise_annotators = sample["train_pairwise_annotator"][0, :num_train_pairwise]  # Shape: [N_train]
        train_pairwise_attributes = sample["train_pairwise_attribute"][0, :num_train_pairwise]  # Shape: [N_train]
        train_pairwise_tied_ratings = sample["train_pairwise_tied_rating"][0, :num_train_pairwise]  # Shape: [N_train]
        train_pairwise_observed = sample["train_pairwise_observed"][0, :num_train_pairwise]  # Shape: [N_train]

    # Test pairwise rankings
    if not num_test_pairwise == 0:
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
            annotator = j + 1
            for k in range(config.K_train):
                item = k + 1
                # Skip cross-instance: test-only annotator rating train item
                if _is_cross_instance(annotator, item, config):
                    continue
                
                rating_dict = {
                    "attribute": i + 1,
                    "annotator": annotator,
                    "item": item,
                    "value": int(train_rating_values[ij_idx, k]),
                    "instance": "train",
                    "rating_dist": [0.0] * num_classes
                }
                try:
                    rating_dict["rating_dist"][int(train_rating_values[ij_idx, k]) - 1] = 1.0
                    train_ratings.append(rating_dict)
                
                    if train_rating_observed[ij_idx, k] == 1:
                        train_observed_ratings.append(rating_dict)
                    else:
                        train_missing_ratings.append(rating_dict)
                except IndexError:
                    continue
    
    # Convert test ratings to list format
    test_ratings = []
    test_observed_ratings = []
    test_missing_ratings = []
    
    for i in range(config.I):
        for j in range(config.J):
            ij_idx = i * config.J + j
            annotator = j + 1
            for k in range(config.K_test):
                item = k + 1 + config.K_train  # Offset by training items
                # Skip cross-instance: train-only annotator rating test item
                if _is_cross_instance(annotator, item, config):
                    continue
                
                rating_dict = {
                    "attribute": i + 1,
                    "annotator": annotator,
                    "item": item,
                    "value": int(test_rating_values[ij_idx, k]),
                    "instance": "test"
                }
                if not int(test_rating_values[ij_idx, k]) in [1, 2, 3, 4]:
                    continue
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
        annotator = int(train_pairwise_annotators[n])
        items = [int(train_pairwise_items[n, 0]), int(train_pairwise_items[n, 1])]
        # Skip cross-instance pairwise rankings
        if _is_cross_instance_pairwise(annotator, items, config):
            continue
        
        pairwise_dict = {
            "attribute": int(train_pairwise_attributes[n]),
            "annotator": annotator,
            "items": items,
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
        annotator = int(test_pairwise_annotators[n])
        items = [int(test_pairwise_items[n, 0]) + config.K_train, 
                 int(test_pairwise_items[n, 1]) + config.K_train]  # Offset by training items
        # Skip cross-instance pairwise rankings
        if _is_cross_instance_pairwise(annotator, items, config):
            continue
        
        pairwise_dict = {
            "attribute": int(test_pairwise_attributes[n]),
            "annotator": annotator,
            "items": items,
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
    
    # Combine item embeddings (train + test) if both are present
    all_embeddings = None
    if train_embeddings is not None and test_embeddings is not None:
        all_embeddings = np.vstack([train_embeddings, test_embeddings])  # Shape: [K_train + K_test, D]
    
    # Combine base scores (train + test) if both are present
    all_base_scores = None
    if train_base_scores is not None and test_base_scores is not None:
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
    
    # Build bundle with only fields that are present
    bundle_kwargs = {
        "all_ratings": all_ratings,
        "all_pairwise": all_pairwise,
        "observed_ratings": observed_ratings,
        "missing_ratings": missing_ratings,
        "missing_ratings_indexes_in_test_instance": missing_ratings_indexes_in_test_instance,
        "observed_pairwise": observed_pairwise,
        "missing_pairwise": missing_pairwise,
        "stats": stats,
    }
    
    # Add optional standard fields if present
    if all_embeddings is not None:
        bundle_kwargs["embeddings"] = all_embeddings
    if mean_preferences is not None:
        bundle_kwargs["mean_preferences"] = mean_preferences
    if annotator_preferences is not None:
        bundle_kwargs["annotator_preferences"] = annotator_preferences
    if rating_probs is not None:
        bundle_kwargs["rating_probs"] = rating_probs
    if rating_cumprobs is not None:
        bundle_kwargs["rating_cumprobs"] = rating_cumprobs
    if rating_thresholds_z is not None:
        bundle_kwargs["rating_thresholds_z"] = rating_thresholds_z
    if all_base_scores is not None:
        bundle_kwargs["base_scores"] = all_base_scores
    if train_posterior_rating_probs is not None:
        bundle_kwargs["train_posterior_rating_probs"] = train_posterior_rating_probs
    if test_posterior_rating_probs is not None:
        bundle_kwargs["test_posterior_rating_probs"] = test_posterior_rating_probs
    
    # Add extra ground truth if any
    if extra_ground_truth:
        bundle_kwargs["extra_ground_truth"] = extra_ground_truth
    
    return GroundTruthBundle(**bundle_kwargs)


def apply_mcar_protocol(bundle: GroundTruthBundle, missing_rate: float, seed: Optional[int] = None) -> GroundTruthBundle:
    """
    Apply Missing Completely At Random (MCAR) protocol to a bundle.

    Randomly selects missing_rate% of all ratings to mark as missing, regardless of
    the original observation protocol. This creates a completely random missingness pattern (IID).

    Args:
        bundle: GroundTruthBundle with original observation protocol
        missing_rate: Fraction of ratings to mark as missing (e.g., 0.5 for 50%)
        seed: Random seed for reproducibility

    Returns:
        New GroundTruthBundle with MCAR observation pattern
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
        "protocol": "MCAR",
        "mcar_missing_rate": missing_rate,
    }

    # Create missing_ratings_indexes_in_test_instance
    missing_ratings_indexes_in_test_instance = [i for i, rating in enumerate(missing_ratings) if rating["instance"] == "test"]

    # Create new bundle with MCAR observation pattern
    # Preserve all optional fields dynamically
    bundle_kwargs = {
        "all_ratings": all_ratings,
        "all_pairwise": all_pairwise,
        "observed_ratings": observed_ratings,
        "missing_ratings": missing_ratings,
        "missing_ratings_indexes_in_test_instance": missing_ratings_indexes_in_test_instance,
        "observed_pairwise": observed_pairwise,
        "missing_pairwise": missing_pairwise,
        "stats": stats,
    }
    
    # Copy optional fields if present
    if bundle.embeddings is not None:
        bundle_kwargs["embeddings"] = bundle.embeddings
    if bundle.mean_preferences is not None:
        bundle_kwargs["mean_preferences"] = bundle.mean_preferences
    if bundle.annotator_preferences is not None:
        bundle_kwargs["annotator_preferences"] = bundle.annotator_preferences
    if bundle.rating_probs is not None:
        bundle_kwargs["rating_probs"] = bundle.rating_probs
    if bundle.rating_cumprobs is not None:
        bundle_kwargs["rating_cumprobs"] = bundle.rating_cumprobs
    if bundle.rating_thresholds_z is not None:
        bundle_kwargs["rating_thresholds_z"] = bundle.rating_thresholds_z
    if bundle.base_scores is not None:
        bundle_kwargs["base_scores"] = bundle.base_scores
    if bundle.train_posterior_rating_probs is not None:
        bundle_kwargs["train_posterior_rating_probs"] = bundle.train_posterior_rating_probs
    if bundle.test_posterior_rating_probs is not None:
        bundle_kwargs["test_posterior_rating_probs"] = bundle.test_posterior_rating_probs
    if bundle.extra_ground_truth is not None:
        bundle_kwargs["extra_ground_truth"] = bundle.extra_ground_truth
    
    return GroundTruthBundle(**bundle_kwargs)


def apply_pairwise_observation_rate(bundle: GroundTruthBundle, observation_rate: float, seed: Optional[int] = None) -> GroundTruthBundle:
    """
    Apply observation rate to missing pairwise rankings.

    Takes missing pairwise rankings and randomly marks observation_rate% of them as observed.
    This allows partial observation of pairwise comparisons in tie_breaking protocol.

    Args:
        bundle: GroundTruthBundle with original tie_breaking observation protocol
        observation_rate: Fraction of missing pairwise rankings to mark as observed (e.g., 0.3 for 30%)
        seed: Random seed for reproducibility

    Returns:
        New GroundTruthBundle with updated pairwise observation pattern
    """
    import random

    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed + 1000)  # Offset to avoid overlap with other random operations

    # Get current pairwise rankings
    observed_pairwise = list(bundle.observed_pairwise)
    missing_pairwise = list(bundle.missing_pairwise)

    n_missing = len(missing_pairwise)
    if n_missing == 0:
        print("  No missing pairwise rankings to apply observation rate to")
        return bundle

    # Calculate how many to newly observe
    n_to_observe = int(n_missing * observation_rate)

    print(f"  Missing pairwise rankings: {n_missing}")
    print(f"  Newly observing: {n_to_observe} ({observation_rate:.2%})")

    # Randomly select indices to mark as observed
    missing_indices = list(range(n_missing))
    random.shuffle(missing_indices)
    observe_indices = set(missing_indices[:n_to_observe])

    # Split missing pairwise into newly observed and still missing
    newly_observed = []
    still_missing = []

    for idx, pairwise in enumerate(missing_pairwise):
        if idx in observe_indices:
            newly_observed.append(pairwise)
        else:
            still_missing.append(pairwise)

    # Update observed and missing lists
    observed_pairwise.extend(newly_observed)
    missing_pairwise = still_missing

    # Verify
    print(f"  Final observed pairwise: {len(observed_pairwise)}")
    print(f"  Final missing pairwise: {len(missing_pairwise)}")

    # Update statistics
    stats = dict(bundle.stats)
    stats["observed_pairwise"] = len(observed_pairwise)
    stats["missing_pairwise"] = len(missing_pairwise)
    stats["pairwise_observation_rate"] = observation_rate

    # Create new bundle with updated pairwise observation pattern
    # Preserve all optional fields dynamically
    bundle_kwargs = {
        "all_ratings": bundle.all_ratings,
        "all_pairwise": bundle.all_pairwise,
        "observed_ratings": bundle.observed_ratings,
        "missing_ratings": bundle.missing_ratings,
        "missing_ratings_indexes_in_test_instance": bundle.missing_ratings_indexes_in_test_instance,
        "observed_pairwise": observed_pairwise,
        "missing_pairwise": missing_pairwise,
        "stats": stats,
    }
    
    # Copy optional fields if present
    if bundle.embeddings is not None:
        bundle_kwargs["embeddings"] = bundle.embeddings
    if bundle.mean_preferences is not None:
        bundle_kwargs["mean_preferences"] = bundle.mean_preferences
    if bundle.annotator_preferences is not None:
        bundle_kwargs["annotator_preferences"] = bundle.annotator_preferences
    if bundle.rating_probs is not None:
        bundle_kwargs["rating_probs"] = bundle.rating_probs
    if bundle.rating_cumprobs is not None:
        bundle_kwargs["rating_cumprobs"] = bundle.rating_cumprobs
    if bundle.rating_thresholds_z is not None:
        bundle_kwargs["rating_thresholds_z"] = bundle.rating_thresholds_z
    if bundle.base_scores is not None:
        bundle_kwargs["base_scores"] = bundle.base_scores
    if bundle.train_posterior_rating_probs is not None:
        bundle_kwargs["train_posterior_rating_probs"] = bundle.train_posterior_rating_probs
    if bundle.test_posterior_rating_probs is not None:
        bundle_kwargs["test_posterior_rating_probs"] = bundle.test_posterior_rating_probs
    if bundle.extra_ground_truth is not None:
        bundle_kwargs["extra_ground_truth"] = bundle.extra_ground_truth
    
    return GroundTruthBundle(**bundle_kwargs)
