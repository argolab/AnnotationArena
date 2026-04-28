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
        elif stan_type == "dawid-skene":
            stan_file = str(Path(__file__).parent.parent.parent / "stan_models" / "dawid_skene_generation.stan")
        elif stan_type == "dawid-skene-true":
            stan_file = str(Path(__file__).parent.parent.parent / "stan_models" / "true_dawid_skene_generation.stan")
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
        show_console=False
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

    Item ID ranges (global):
      train: 1 .. K_train
      val:   K_train+1 .. K_train+K_val
      test:  K_train+K_val+1 .. K_train+K_val+K_test

    For tie_breaking protocol:
    - Train-only annotators (1..J/3): can only rate train items
    - Overlap annotators (J/3+1..2J/3): can rate train and test items
    - Test-only annotators (2J/3+1..J): can only rate test items
    - Val items are always accessible to all annotators (no cross-instance filtering)
    """
    if config.observation_protocol != "tie_breaking":
        return False

    total_items = config.K_train + config.K_val + config.K_test
    if item < 1 or item > total_items:
        raise ValueError(f"Item index {item} is out of range (1-{total_items})")

    # Val items are always accessible — no cross-instance filtering
    val_start = config.K_train + 1
    val_end = config.K_train + config.K_val
    if val_start <= item <= val_end:
        return False

    train_only_end = config.J // 3
    test_only_start = (2 * config.J) // 3 + 1

    is_train_only_annotator = annotator <= train_only_end
    is_test_only_annotator = annotator >= test_only_start
    is_train_item = item <= config.K_train
    is_test_item = item > val_end

    return (is_train_only_annotator and is_test_item) or (is_test_only_annotator and is_train_item)


def _is_cross_instance_pairwise(annotator: int, items: List[int], config: DataGenConfig) -> bool:
    """
    Check if a pairwise ranking is cross-instance.

    Item ID ranges (global):
      train: 1 .. K_train
      val:   K_train+1 .. K_train+K_val
      test:  K_train+K_val+1 .. K_train+K_val+K_test

    Val items are always accessible — no cross-instance filtering applied to them.
    """
    if config.observation_protocol != "tie_breaking":
        return False

    total_items = config.K_train + config.K_val + config.K_test
    val_start = config.K_train + 1
    val_end = config.K_train + config.K_val

    for item in items:
        if item < 1 or item > total_items:
            raise ValueError(f"Item index {item} is out of range (1-{total_items})")

    # Val-only pairwise: never cross-instance
    if all(val_start <= item <= val_end for item in items):
        return False

    train_only_end = config.J // 3
    test_only_start = (2 * config.J) // 3 + 1

    is_train_only_annotator = annotator <= train_only_end
    is_test_only_annotator = annotator >= test_only_start

    return (is_train_only_annotator and any(item > val_end for item in items)) or \
           (is_test_only_annotator and any(item <= config.K_train for item in items))


def extract_bundle_from_stan_output(fit: cmdstanpy.CmdStanMCMC, config: DataGenConfig) -> GroundTruthBundle:
    """Extract GroundTruthBundle from Stan output.
    
    Dynamically extracts available fields based on what the generator outputs,
    supporting different generator types (iclr, discrete_type, etc.).
    """
    
    # Get the single sample (since we used fixed_param=True with 1 sample)
    sample = fit.stan_variables()
    
    # Extract core data structures (always present)
    train_rating_values = sample["train_rating_values"][0]    # Shape: [I*J, K_train]
    train_rating_observed = sample["train_rating_observed"][0]  # Shape: [I*J, K_train]
    test_rating_values = sample["test_rating_values"][0]      # Shape: [I*J, K_test]
    test_rating_observed = sample["test_rating_observed"][0]   # Shape: [I*J, K_test]
    val_rating_values = sample["val_rating_values"][0]        # Shape: [I*J, K_val]
    val_rating_observed = sample["val_rating_observed"][0]     # Shape: [I*J, K_val]

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
    val_embeddings = sample.get("val_embeddings")
    train_base_scores = sample.get("train_base_scores")
    test_base_scores = sample.get("test_base_scores")
    val_base_scores = sample.get("val_base_scores")
    
    if train_embeddings is not None:
        train_embeddings = train_embeddings[0]
    if test_embeddings is not None:
        test_embeddings = test_embeddings[0]
    if val_embeddings is not None:
        val_embeddings = val_embeddings[0]
    if train_base_scores is not None:
        train_base_scores = train_base_scores[0]
    if test_base_scores is not None:
        test_base_scores = test_base_scores[0]
    if val_base_scores is not None:
        val_base_scores = val_base_scores[0]
    
    # Extract posterior rating probabilities (optional downstream use)
    train_posterior_rating_probs = sample.get("train_posterior_rating_probs")
    if train_posterior_rating_probs is not None:
        train_posterior_rating_probs = train_posterior_rating_probs[0]
    val_posterior_rating_probs = sample.get("val_posterior_rating_probs")
    if val_posterior_rating_probs is not None:
        val_posterior_rating_probs = val_posterior_rating_probs[0]
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

    # Dawid-Skene confusion matrix
    if "confusion_matrix" in sample:
        extra_ground_truth["confusion_matrix"] = sample["confusion_matrix"][0]
    if "pi" in sample:
        extra_ground_truth["pi"] = sample["pi"][0]
    if "confusion" in sample:
        extra_ground_truth["confusion"] = sample["confusion"][0]
    for key in ["train_true_classes", "val_true_classes", "test_true_classes"]:
        if key in sample:
            extra_ground_truth[key] = sample[key][0]

    # Extract pairwise rankings counts
    num_train_pairwise = 0
    num_test_pairwise = 0
    num_val_pairwise = 0

    # Training pairwise rankings
    if not num_train_pairwise == 0:
        train_pairwise_items = sample["train_pairwise_items"][0, :num_train_pairwise]
        train_pairwise_orders = sample["train_pairwise_orders"][0, :num_train_pairwise]
        train_pairwise_annotators = sample["train_pairwise_annotator"][0, :num_train_pairwise]
        train_pairwise_attributes = sample["train_pairwise_attribute"][0, :num_train_pairwise]
        train_pairwise_tied_ratings = sample["train_pairwise_tied_rating"][0, :num_train_pairwise]
        train_pairwise_observed = sample["train_pairwise_observed"][0, :num_train_pairwise]

    # Test pairwise rankings
    if not num_test_pairwise == 0:
        test_pairwise_items = sample["test_pairwise_items"][0, :num_test_pairwise]
        test_pairwise_orders = sample["test_pairwise_orders"][0, :num_test_pairwise]
        test_pairwise_annotators = sample["test_pairwise_annotator"][0, :num_test_pairwise]
        test_pairwise_attributes = sample["test_pairwise_attribute"][0, :num_test_pairwise]
        test_pairwise_tied_ratings = sample["test_pairwise_tied_rating"][0, :num_test_pairwise]
        test_pairwise_observed = sample["test_pairwise_observed"][0, :num_test_pairwise]

    # Val pairwise rankings
    if not num_val_pairwise == 0:
        val_pairwise_items = sample["val_pairwise_items"][0, :num_val_pairwise]
        val_pairwise_orders = sample["val_pairwise_orders"][0, :num_val_pairwise]
        val_pairwise_annotators = sample["val_pairwise_annotator"][0, :num_val_pairwise]
        val_pairwise_attributes = sample["val_pairwise_attribute"][0, :num_val_pairwise]
        val_pairwise_tied_ratings = sample["val_pairwise_tied_rating"][0, :num_val_pairwise]
        val_pairwise_observed_arr = sample["val_pairwise_observed"][0, :num_val_pairwise]

    # ===== CONVERT TRAINING RATINGS =====
    train_ratings = []
    train_observed_ratings = []
    train_missing_ratings = []
    
    for i in range(config.I):
        for j in range(config.J):
            ij_idx = i * config.J + j
            annotator = j + 1
            for k in range(config.K_train):
                item = k + 1
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
                value = int(train_rating_values[ij_idx, k])
                if not (1 <= value <= num_classes):
                    continue
                rating_dict["rating_dist"][value - 1] = 1.0
                train_ratings.append(rating_dict)
                if train_rating_observed[ij_idx, k] == 1:
                    train_observed_ratings.append(rating_dict)
                else:
                    train_missing_ratings.append(rating_dict)

    # ===== CONVERT TEST RATINGS =====
    test_ratings = []
    test_observed_ratings = []
    test_missing_ratings = []
    
    for i in range(config.I):
        for j in range(config.J):
            ij_idx = i * config.J + j
            annotator = j + 1
            for k in range(config.K_test):
                item = k + 1 + config.K_train + config.K_val  # Test: K_train+K_val+1 .. end
                if _is_cross_instance(annotator, item, config):
                    continue
                
                rating_dict = {
                    "attribute": i + 1,
                    "annotator": annotator,
                    "item": item,
                    "value": int(test_rating_values[ij_idx, k]),
                    "instance": "test"
                }
                if not (1 <= int(test_rating_values[ij_idx, k]) <= num_classes):
                    continue
                test_ratings.append(rating_dict)
                if test_rating_observed[ij_idx, k] == 1:
                    test_observed_ratings.append(rating_dict)
                else:
                    test_missing_ratings.append(rating_dict)

    # ===== CONVERT VALIDATION RATINGS =====
    val_ratings = []
    val_observed_ratings = []
    val_missing_ratings = []

    for i in range(config.I):
        for j in range(config.J):
            ij_idx = i * config.J + j
            annotator = j + 1
            for k in range(config.K_val):
                item = k + 1 + config.K_train  # Val: K_train+1 .. K_train+K_val
                # Val items are never cross-instance (all annotators can rate them)

                rating_dict = {
                    "attribute": i + 1,
                    "annotator": annotator,
                    "item": item,
                    "value": int(val_rating_values[ij_idx, k]),
                    "instance": "val",
                    "rating_dist": [0.0] * num_classes
                }
                value = int(val_rating_values[ij_idx, k])
                if not (1 <= value <= num_classes):
                    continue
                rating_dict["rating_dist"][value - 1] = 1.0
                val_ratings.append(rating_dict)
                if val_rating_observed[ij_idx, k] == 1:
                    val_observed_ratings.append(rating_dict)
                else:
                    val_missing_ratings.append(rating_dict)

    # Combine all ratings in order: train, val, test
    all_ratings = train_ratings + val_ratings + test_ratings
    observed_ratings = train_observed_ratings + val_observed_ratings + test_observed_ratings
    missing_ratings = train_missing_ratings + val_missing_ratings + test_missing_ratings

    # ===== CONVERT TRAINING PAIRWISE =====
    train_pairwise = []
    train_observed_pairwise = []
    train_missing_pairwise = []
    
    for n in range(num_train_pairwise):
        annotator = int(train_pairwise_annotators[n])
        items = [int(train_pairwise_items[n, 0]), int(train_pairwise_items[n, 1])]
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

    # ===== CONVERT TEST PAIRWISE =====
    test_pairwise = []
    test_observed_pairwise = []
    test_missing_pairwise = []
    
    for n in range(num_test_pairwise):
        annotator = int(test_pairwise_annotators[n])
        items = [int(test_pairwise_items[n, 0]) + config.K_train + config.K_val,
                 int(test_pairwise_items[n, 1]) + config.K_train + config.K_val]  # Test: K_train+K_val+1..end
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

    # ===== CONVERT VAL PAIRWISE =====
    val_pairwise = []
    val_observed_pairwise = []
    val_missing_pairwise = []

    for n in range(num_val_pairwise):
        annotator = int(val_pairwise_annotators[n])
        items = [int(val_pairwise_items[n, 0]) + config.K_train,
                 int(val_pairwise_items[n, 1]) + config.K_train]  # Val: K_train+1..K_train+K_val
        # Val pairwise are never cross-instance

        pairwise_dict = {
            "attribute": int(val_pairwise_attributes[n]),
            "annotator": annotator,
            "items": items,
            "order": [1, 2] if val_pairwise_orders[n] == 1 else [2, 1],
            "tied_rating": int(val_pairwise_tied_ratings[n]),
            "instance": "val"
        }
        val_pairwise.append(pairwise_dict)
        if val_pairwise_observed_arr[n] == 1:
            val_observed_pairwise.append(pairwise_dict)
        else:
            val_missing_pairwise.append(pairwise_dict)

    # Combine all pairwise rankings in order: train, val, test
    all_pairwise = train_pairwise + val_pairwise + test_pairwise
    observed_pairwise = train_observed_pairwise + val_observed_pairwise + test_observed_pairwise
    missing_pairwise = train_missing_pairwise + val_missing_pairwise + test_missing_pairwise

    # Combine item embeddings in order: train, val, test
    all_embeddings = None
    if train_embeddings is not None and val_embeddings is not None and test_embeddings is not None:
        all_embeddings = np.vstack([train_embeddings, val_embeddings, test_embeddings])  # [K_train+K_val+K_test, D]
    elif train_embeddings is not None and test_embeddings is not None:
        all_embeddings = np.vstack([train_embeddings, test_embeddings])

    # Combine base scores in order: train, val, test
    all_base_scores = None
    if train_base_scores is not None and val_base_scores is not None and test_base_scores is not None:
        all_base_scores = np.hstack([train_base_scores, val_base_scores, test_base_scores])
    elif train_base_scores is not None and test_base_scores is not None:
        all_base_scores = np.hstack([train_base_scores, test_base_scores])

    # Compute statistics
    total_items = config.K_train + config.K_test + config.K_val
    stats = {
        "K_train": config.K_train,
        "K_test": config.K_test,
        "K_val": config.K_val,
        "total_items": total_items,
        "total_possible_ratings": config.I * config.J * total_items,
        "total_ratings": len(all_ratings),
        "observed_ratings": len(observed_ratings),
        "missing_ratings": len(missing_ratings),
        "train_ratings": len(train_ratings),
        "test_ratings": len(test_ratings),
        "val_ratings": len(val_ratings),
        "train_observed": len(train_observed_ratings),
        "test_observed": len(test_observed_ratings),
        "val_observed": len(val_observed_ratings),
        "total_pairwise": len(all_pairwise),
        "observed_pairwise": len(observed_pairwise),
        "missing_pairwise": len(missing_pairwise),
        "train_pairwise": len(train_pairwise),
        "test_pairwise": len(test_pairwise),
        "val_pairwise": len(val_pairwise),
        "observation_rate": len(observed_ratings) / len(all_ratings) if all_ratings else 0,
        "train_observation_rate": len(train_observed_ratings) / len(train_ratings) if train_ratings else 0,
        "test_observation_rate": len(test_observed_ratings) / len(test_ratings) if test_ratings else 0,
        "val_observation_rate": len(val_observed_ratings) / len(val_ratings) if val_ratings else 0,
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
    if val_posterior_rating_probs is not None:
        bundle_kwargs["val_posterior_rating_probs"] = val_posterior_rating_probs
    if test_posterior_rating_probs is not None:
        bundle_kwargs["test_posterior_rating_probs"] = test_posterior_rating_probs
    if extra_ground_truth:
        bundle_kwargs["extra_ground_truth"] = extra_ground_truth
    
    return GroundTruthBundle(**bundle_kwargs)


def apply_mcar_protocol(bundle: GroundTruthBundle, missing_rate: float, seed: Optional[int] = None) -> GroundTruthBundle:
    """
    Apply Missing Completely At Random (MCAR) protocol to a bundle.

    Randomly selects missing_rate% of all ratings to mark as missing, regardless of
    the original observation protocol. This creates a completely random missingness pattern (IID).
    Applied uniformly across train, test, and val instances.

    Args:
        bundle: GroundTruthBundle with original observation protocol
        missing_rate: Fraction of ratings to mark as missing (e.g., 0.5 for 50%)
        seed: Random seed for reproducibility

    Returns:
        New GroundTruthBundle with MCAR observation pattern
    """
    import random

    if seed is not None:
        random.seed(seed)

    all_ratings = bundle.all_ratings
    n_total = len(all_ratings)
    n_missing = int(n_total * missing_rate)
    n_observed = n_total - n_missing

    print(f"  Total ratings: {n_total}")
    print(f"  Target missing: {n_missing} ({n_missing / n_total:.2%})")
    print(f"  Target observed: {n_observed} ({n_observed / n_total:.2%})")

    all_indices = list(range(n_total))
    random.shuffle(all_indices)
    missing_indices = set(all_indices[:n_missing])

    observed_ratings = []
    missing_ratings = []
    for idx, rating in enumerate(all_ratings):
        if idx in missing_indices:
            missing_ratings.append(rating)
        else:
            observed_ratings.append(rating)

    actual_missing_rate = len(missing_ratings) / n_total
    actual_observed_rate = len(observed_ratings) / n_total

    print(f"  Actual missing: {len(missing_ratings)} ({actual_missing_rate:.2%})")
    print(f"  Actual observed: {len(observed_ratings)} ({actual_observed_rate:.2%})")

    assert len(missing_ratings) + len(observed_ratings) == n_total, "Rating counts don't match!"
    assert abs(actual_missing_rate - missing_rate) < 0.01, f"Missing rate mismatch: {actual_missing_rate:.2%} != {missing_rate:.2%}"

    # Split by instance for statistics
    train_observed = [r for r in observed_ratings if r["instance"] == "train"]
    train_missing = [r for r in missing_ratings if r["instance"] == "train"]
    test_observed = [r for r in observed_ratings if r["instance"] == "test"]
    test_missing = [r for r in missing_ratings if r["instance"] == "test"]
    val_observed = [r for r in observed_ratings if r["instance"] == "val"]
    val_missing = [r for r in missing_ratings if r["instance"] == "val"]

    train_ratings = [r for r in all_ratings if r["instance"] == "train"]
    test_ratings = [r for r in all_ratings if r["instance"] == "test"]
    val_ratings = [r for r in all_ratings if r["instance"] == "val"]

    all_pairwise = bundle.all_pairwise
    observed_pairwise = bundle.observed_pairwise
    missing_pairwise = []  # No missing pairwise in MCAR protocol

    # Carry forward all original stats keys, then overwrite with updated counts.
    # Using .get() with fallbacks makes this work for both item-split and annotator-split bundles.
    stats = {
        **{k: v for k, v in bundle.stats.items()},  # preserve K/J_train/J_val/J_test etc.
        "total_possible_ratings": bundle.stats["total_possible_ratings"],
        "total_ratings": n_total,
        "observed_ratings": len(observed_ratings),
        "missing_ratings": len(missing_ratings),
        "train_ratings": len(train_ratings),
        "test_ratings": len(test_ratings),
        "val_ratings": len(val_ratings),
        "train_observed": len(train_observed),
        "test_observed": len(test_observed),
        "val_observed": len(val_observed),
        "total_pairwise": len(all_pairwise),
        "observed_pairwise": len(observed_pairwise),
        "missing_pairwise": len(missing_pairwise),
        "train_pairwise": len([p for p in all_pairwise if p["instance"] == "train"]),
        "test_pairwise": len([p for p in all_pairwise if p["instance"] == "test"]),
        "val_pairwise": len([p for p in all_pairwise if p["instance"] == "val"]),
        "observation_rate": actual_observed_rate,
        "train_observation_rate": len(train_observed) / len(train_ratings) if train_ratings else 0,
        "test_observation_rate": len(test_observed) / len(test_ratings) if test_ratings else 0,
        "val_observation_rate": len(val_observed) / len(val_ratings) if val_ratings else 0,
        "protocol": "MCAR",
        "mcar_missing_rate": missing_rate,
    }

    missing_ratings_indexes_in_test_instance = [i for i, rating in enumerate(missing_ratings) if rating["instance"] == "test"]

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
    if bundle.val_posterior_rating_probs is not None:
        bundle_kwargs["val_posterior_rating_probs"] = bundle.val_posterior_rating_probs
    if bundle.test_posterior_rating_probs is not None:
        bundle_kwargs["test_posterior_rating_probs"] = bundle.test_posterior_rating_probs
    if bundle.extra_ground_truth is not None:
        bundle_kwargs["extra_ground_truth"] = bundle.extra_ground_truth
    
    return GroundTruthBundle(**bundle_kwargs)


def apply_pairwise_observation_rate(bundle: GroundTruthBundle, observation_rate: float, seed: Optional[int] = None) -> GroundTruthBundle:
    """
    Apply observation rate to missing pairwise rankings.

    Takes missing pairwise rankings and randomly marks observation_rate% of them as observed.
    Applied across train, test, and val instances.

    Args:
        bundle: GroundTruthBundle with original tie_breaking observation protocol
        observation_rate: Fraction of missing pairwise rankings to mark as observed (e.g., 0.3 for 30%)
        seed: Random seed for reproducibility

    Returns:
        New GroundTruthBundle with updated pairwise observation pattern
    """
    import random

    if seed is not None:
        random.seed(seed + 1000)  # Offset to avoid overlap with other random operations

    observed_pairwise = list(bundle.observed_pairwise)
    missing_pairwise = list(bundle.missing_pairwise)

    n_missing = len(missing_pairwise)
    if n_missing == 0:
        print("  No missing pairwise rankings to apply observation rate to")
        return bundle

    n_to_observe = int(n_missing * observation_rate)

    print(f"  Missing pairwise rankings: {n_missing}")
    print(f"  Newly observing: {n_to_observe} ({observation_rate:.2%})")

    missing_indices = list(range(n_missing))
    random.shuffle(missing_indices)
    observe_indices = set(missing_indices[:n_to_observe])

    newly_observed = []
    still_missing = []
    for idx, pairwise in enumerate(missing_pairwise):
        if idx in observe_indices:
            newly_observed.append(pairwise)
        else:
            still_missing.append(pairwise)

    observed_pairwise.extend(newly_observed)
    missing_pairwise = still_missing

    print(f"  Final observed pairwise: {len(observed_pairwise)}")
    print(f"  Final missing pairwise: {len(missing_pairwise)}")

    stats = dict(bundle.stats)
    stats["observed_pairwise"] = len(observed_pairwise)
    stats["missing_pairwise"] = len(missing_pairwise)
    stats["pairwise_observation_rate"] = observation_rate

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
    if bundle.val_posterior_rating_probs is not None:
        bundle_kwargs["val_posterior_rating_probs"] = bundle.val_posterior_rating_probs
    if bundle.test_posterior_rating_probs is not None:
        bundle_kwargs["test_posterior_rating_probs"] = bundle.test_posterior_rating_probs
    if bundle.extra_ground_truth is not None:
        bundle_kwargs["extra_ground_truth"] = bundle.extra_ground_truth
    
    return GroundTruthBundle(**bundle_kwargs)


# =============================================================================
# ANNOTATOR-SPLIT DATA GENERATION
# =============================================================================

def generate_data_annotator_split(
    config: "AnnotatorSplitConfig",
    stan_file: Optional[str] = None,
) -> GroundTruthBundle:
    """
    Generate synthetic data using the annotator-split Stan model.

    Items are shared across all instances. Annotators are partitioned into
    train/val/test groups; each group rates every item.

    Args:
        config: AnnotatorSplitConfig specifying K, J_train, J_val, J_test, etc.
        stan_file: Path to .stan file (defaults to annotator_split_generation.stan)

    Returns:
        GroundTruthBundle with train/val/test ratings labelled by annotator group.
    """
    if stan_file is None:
        if getattr(config, "stan_type", None) == "tensor":
            stan_file = str(
                Path(__file__).parent.parent.parent
                / "stan_models"
                / "tensor_generation_annotator.stan"
            )
        else:
            stan_file = str(
                Path(__file__).parent.parent.parent
                / "stan_models"
                / "normal_noise_dot_product_annotator_gen.stan"
            )

    model = compile_data_generation_model(stan_file)
    stan_data = config.to_stan_data()

    fit = model.sample(
        data=stan_data,
        fixed_param=True,
        chains=1,
        iter_sampling=1,
        seed=config.seed,
        show_console=False,
    )

    bundle = extract_bundle_from_annotator_split_stan_output(fit, config)

    if config.observation_protocol == "mcar":
        print(f"\nApplying MCAR protocol with missing rate: {config.mcar_missing_rate:.2%}")
        bundle = apply_mcar_protocol(bundle, config.mcar_missing_rate, seed=config.seed)

    return bundle


def extract_bundle_from_annotator_split_stan_output(
    fit: cmdstanpy.CmdStanMCMC,
    config: "AnnotatorSplitConfig",
) -> GroundTruthBundle:
    """
    Extract GroundTruthBundle from annotator-split Stan output.

    Stan produces a single rating array of shape [I*J, K] covering all annotators
    and all items. We split by annotator group to assign instance labels:
      - annotators 1..J_train            -> "train"
      - annotators J_train+1..J_train+J_val  -> "val"
      - annotators J_train+J_val+1..J    -> "test"

    All annotators rate all items, so there are no missing ratings from the
    observation protocol (every cell is observed).
    """
    from .configs import AnnotatorSplitConfig  # local import to avoid circular

    sample = fit.stan_variables()
    num_classes = config.C
    J_total = config.J

    # Annotator group boundaries (1-indexed)
    train_end = config.J_train
    val_start = config.J_train + 1
    val_end = config.J_train + config.J_val
    test_start = config.J_train + config.J_val + 1

    # Core rating arrays: shape [I*J, K]
    rating_values = sample["rating_values"][0]    # [I*J_total, K]
    rating_observed = sample["rating_observed"][0] # [I*J_total, K] — all 1s from Stan

    # Optional ground-truth arrays
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

    embeddings = sample.get("embeddings")
    if embeddings is not None:
        embeddings = embeddings[0]

    base_scores = sample.get("base_scores")
    if base_scores is not None:
        base_scores = base_scores[0]

    train_posterior_rating_probs = sample.get("train_posterior_rating_probs")
    if train_posterior_rating_probs is not None:
        train_posterior_rating_probs = train_posterior_rating_probs[0]

    val_posterior_rating_probs = sample.get("val_posterior_rating_probs")
    if val_posterior_rating_probs is not None:
        val_posterior_rating_probs = val_posterior_rating_probs[0]

    test_posterior_rating_probs = sample.get("test_posterior_rating_probs")
    if test_posterior_rating_probs is not None:
        test_posterior_rating_probs = test_posterior_rating_probs[0]

    extra_ground_truth = {}
    for key in ["annotator_embeddings", "attr_transforms", "threshold_transform_W", "threshold_attr_bias"]:
        if key in sample:
            extra_ground_truth[key] = sample[key][0]

    # ===== CONVERT RATINGS =====
    # Instance label determined by annotator group membership.
    train_ratings, train_observed_ratings, train_missing_ratings = [], [], []
    val_ratings,   val_observed_ratings,   val_missing_ratings   = [], [], []
    test_ratings,  test_observed_ratings,  test_missing_ratings  = [], [], []

    for i in range(config.I):
        for j in range(J_total):
            ij_idx = i * J_total + j
            annotator = j + 1  # 1-indexed

            if annotator <= train_end:
                instance = "train"
            elif annotator <= val_end:
                instance = "val"
            else:
                instance = "test"

            for k in range(config.K):
                item = k + 1  # All instances share item IDs 1..K
                value = int(rating_values[ij_idx, k])

                rating_dict = {
                    "attribute": i + 1,
                    "annotator": annotator,
                    "item": item,
                    "value": value,
                    "instance": instance,
                    "rating_dist": [0.0] * num_classes,
                }
                try:
                    rating_dict["rating_dist"][value - 1] = 1.0
                except IndexError:
                    continue

                observed = int(rating_observed[ij_idx, k]) == 1

                # In annotator-split mode, only cells selected by num_annotate_annotator
                # exist as ratings. Unselected cells are not ratings at all — skip them.
                # MCAR will subsequently mark some of these observed ratings as missing.
                if not observed:
                    continue

                if instance == "train":
                    train_ratings.append(rating_dict)
                    train_observed_ratings.append(rating_dict)
                elif instance == "val":
                    val_ratings.append(rating_dict)
                    val_observed_ratings.append(rating_dict)
                else:
                    test_ratings.append(rating_dict)
                    test_observed_ratings.append(rating_dict)

    all_ratings = train_ratings + val_ratings + test_ratings
    observed_ratings = train_observed_ratings + val_observed_ratings + test_observed_ratings
    missing_ratings = train_missing_ratings + val_missing_ratings + test_missing_ratings

    # No pairwise rankings in annotator-split mode
    all_pairwise: list = []
    observed_pairwise: list = []
    missing_pairwise: list = []

    missing_ratings_indexes_in_test_instance = [
        i for i, r in enumerate(missing_ratings) if r["instance"] == "test"
    ]

    stats = {
        "K": config.K,
        "J_train": config.J_train,
        "J_val": config.J_val,
        "J_test": config.J_test,
        "total_items": config.K,
        # total_possible_ratings = I * num_annotate_annotator * K (not I*J*K)
        # since only num_annotate_annotator annotators rate each item.
        # Use total_ratings as a proxy since all_ratings contains only selected cells.
        "total_possible_ratings": len(all_ratings),
        "total_ratings": len(all_ratings),
        "observed_ratings": len(observed_ratings),
        "missing_ratings": len(missing_ratings),
        "train_ratings": len(train_ratings),
        "val_ratings": len(val_ratings),
        "test_ratings": len(test_ratings),
        "train_observed": len(train_observed_ratings),
        "val_observed": len(val_observed_ratings),
        "test_observed": len(test_observed_ratings),
        "total_pairwise": 0,
        "observed_pairwise": 0,
        "missing_pairwise": 0,
        "observation_rate": len(observed_ratings) / len(all_ratings) if all_ratings else 0,
        "train_observation_rate": len(train_observed_ratings) / len(train_ratings) if train_ratings else 0,
        "val_observation_rate": len(val_observed_ratings) / len(val_ratings) if val_ratings else 0,
        "test_observation_rate": len(test_observed_ratings) / len(test_ratings) if test_ratings else 0,
    }

    bundle_kwargs: dict = {
        "all_ratings": all_ratings,
        "all_pairwise": all_pairwise,
        "observed_ratings": observed_ratings,
        "missing_ratings": missing_ratings,
        "missing_ratings_indexes_in_test_instance": missing_ratings_indexes_in_test_instance,
        "observed_pairwise": observed_pairwise,
        "missing_pairwise": missing_pairwise,
        "stats": stats,
    }

    if embeddings is not None:
        bundle_kwargs["embeddings"] = embeddings
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
    if base_scores is not None:
        bundle_kwargs["base_scores"] = base_scores
    if train_posterior_rating_probs is not None:
        bundle_kwargs["train_posterior_rating_probs"] = train_posterior_rating_probs
    if val_posterior_rating_probs is not None:
        bundle_kwargs["val_posterior_rating_probs"] = val_posterior_rating_probs
    if test_posterior_rating_probs is not None:
        bundle_kwargs["test_posterior_rating_probs"] = test_posterior_rating_probs
    if extra_ground_truth:
        bundle_kwargs["extra_ground_truth"] = extra_ground_truth

    return GroundTruthBundle(**bundle_kwargs)
