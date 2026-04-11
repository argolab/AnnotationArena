from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class GroundTruthBundle:
    # Core data structures (always present)
    all_ratings: List[Dict[str, Any]]

    # Each pairwise dict: {
    #   'attribute': int(1..I), 'annotator': int(1..J),
    #   'items': [int(k1), int(k2)] with 1..K,
    #   'order': [1,2] if k1>k2 else [2,1],
    #   'tied_rating': int(1..C) (rating bin that tied)  # stored but not used in the current implementation
    # }
    all_pairwise: List[Dict[str, Any]]

    # Subsets following same schemas as above
    observed_ratings: List[Dict[str, Any]]
    missing_ratings: List[Dict[str, Any]]
    observed_pairwise: List[Dict[str, Any]]
    missing_pairwise: List[Dict[str, Any]]
    # Indices of missing ratings that belong to test instances
    missing_ratings_indexes_in_test_instance: List[int]

    # Example keys: {'total_possible_ratings': int, 'total_pairwise_rankings': int, 'train_ratings': int, ...}
    stats: Dict[str, Any]

    # Standard embedding-world ground truth (optional - present for iclr/extended_rankings generators)
    embeddings: Optional[np.ndarray] = None  # shape [K, D]
    mean_preferences: Optional[np.ndarray] = None  # shape [I, D]
    annotator_preferences: Optional[np.ndarray] = None  # shape [I*J, D]
    rating_probs: Optional[np.ndarray] = None  # shape [I*J, C]
    rating_cumprobs: Optional[np.ndarray] = None  # shape [I*J, C]
    rating_thresholds_z: Optional[np.ndarray] = None  # shape [I*J, C+1]
    base_scores: Optional[np.ndarray] = None  # shape [I*J, K]

    # Posterior rating probabilities under measurement noise
    # train: shape [I*J, K_train, C], val: shape [I*J, K_val, C], test: shape [I*J, K_test, C]
    train_posterior_rating_probs: Optional[np.ndarray] = None
    val_posterior_rating_probs: Optional[np.ndarray] = None
    test_posterior_rating_probs: Optional[np.ndarray] = None

    # Generator-specific extra ground truth (e.g., factored annotator model, discrete prototypes)
    # Keys depend on generator type:
    # - Factored annotator: "annotator_embeddings", "attr_transforms", "threshold_transform_W", "threshold_attr_bias"
    # - Discrete prototype: "z_train", "z_test", "s_of_j", "a_attr", "u_proto", "v_style", "delta_ims", "mu_ims"
    extra_ground_truth: Optional[Dict[str, Any]] = None

    log_lik_ratings_obs: Optional[float] = None
    log_lik_ratings_missing: Optional[float] = None
    log_lik_rankings_obs: Optional[float] = None
    log_lik_rankings_missing: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroundTruthBundle':
        """Create GroundTruthBundle from dictionary (e.g., loaded from JSON).
        
        Dynamically loads fields that are present, allowing for different generator types.
        """
        # Core fields (always required)
        core_fields = {
            "all_ratings": data["all_ratings"],
            "all_pairwise": data["all_pairwise"],
            "observed_ratings": data["observed_ratings"],
            "missing_ratings": data["missing_ratings"],
            "observed_pairwise": data["observed_pairwise"],
            "missing_pairwise": data["missing_pairwise"],
            "missing_ratings_indexes_in_test_instance": [i for i in range(len(data["missing_ratings"])) if data["missing_ratings"][i]["instance"] == "test"],
            "stats": data["stats"],
        }
        
        # Optional standard embedding-world fields
        optional_fields = {}
        for key in ["embeddings", "mean_preferences", "annotator_preferences", 
                    "rating_probs", "rating_cumprobs", "rating_thresholds_z", "base_scores"]:
            if key in data and data[key] is not None:
                optional_fields[key] = np.array(data[key])
        
        # Optional posterior probs (train, val, test)
        for key in ["train_posterior_rating_probs", "val_posterior_rating_probs", "test_posterior_rating_probs"]:
            if key in data and data[key] is not None:
                optional_fields[key] = np.array(data[key])
        
        # Optional log likelihoods
        for key in ["log_lik_ratings_obs", "log_lik_ratings_missing", 
                    "log_lik_rankings_obs", "log_lik_rankings_missing"]:
            if key in data:
                optional_fields[key] = data[key]
        
        # Extra ground truth: collect all remaining keys that aren't core or standard optional fields
        standard_keys = {"all_ratings", "all_pairwise", "observed_ratings", "missing_ratings",
                        "observed_pairwise", "missing_pairwise", "stats",
                        "embeddings", "mean_preferences", "annotator_preferences",
                        "rating_probs", "rating_cumprobs", "rating_thresholds_z", "base_scores",
                        "train_posterior_rating_probs", "val_posterior_rating_probs", "test_posterior_rating_probs",
                        "log_lik_ratings_obs", "log_lik_ratings_missing",
                        "log_lik_rankings_obs", "log_lik_rankings_missing"}
        extra_gt = {}
        for key, value in data.items():
            if key not in standard_keys and value is not None:
                # Convert numpy arrays/lists to numpy arrays for consistency
                if isinstance(value, list):
                    try:
                        extra_gt[key] = np.array(value)
                    except (ValueError, TypeError):
                        extra_gt[key] = value  # Keep as-is if not convertible
                else:
                    extra_gt[key] = value
        
        if extra_gt:
            optional_fields["extra_ground_truth"] = extra_gt
        
        return cls(**core_fields, **optional_fields)


@dataclass
class ObservedSet:
    ratings: List[Dict[str, Any]]
    pairwise: List[Dict[str, Any]]


@dataclass
class MissingSet:
    ratings: List[Dict[str, Any]]
    pairwise: List[Dict[str, Any]]


