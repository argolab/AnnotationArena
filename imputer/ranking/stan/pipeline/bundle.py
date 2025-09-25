from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class GroundTruthBundle:
    embeddings: np.ndarray  # shape [K, D] (in practice we have K=K_train+K_test)
    mean_preferences: np.ndarray  # shape [I, D]
    annotator_preferences: np.ndarray  # shape [I*J, D] with ij_idx = (i-1)*J + j
    rating_probs: np.ndarray  # shape [I*J, C]
    rating_cumprobs: np.ndarray  # shape [I*J, C] cumulative probabilities
    rating_thresholds_z: np.ndarray  # shape [I*J, C+1] z-cutpoints [-inf, ..., +inf]
    base_scores: np.ndarray  # shape [I*J, K]

    # Each rating dict: {'attribute': int(1..I), 'annotator': int(1..J), 'item': int(1..K), 'value': int(1..C)}
    all_ratings: List[Dict[str, Any]]

    # Each pairwise dict: {
    #   'attribute': int(1..I), 'annotator': int(1..J),
    #   'items': [int(k1), int(k2)] with 1..K,
    #   'order': [1,2] if k1>k2 else [2,1],
    #   'tied_rating': int(1..C) (rating bin that tied)  # stored butnot used in the current implementation
    # }
    all_pairwise: List[Dict[str, Any]]

    # Subsets following same schemas as above
    observed_ratings: List[Dict[str, Any]]
    missing_ratings: List[Dict[str, Any]]
    observed_pairwise: List[Dict[str, Any]]
    missing_pairwise: List[Dict[str, Any]]
    missing_ratings_indexes: List[int]

    # Example keys: {'total_possible_ratings': int, 'total_pairwise_rankings': int, 'train_ratings': int, ...}
    stats: Dict[str, Any]

    log_lik_ratings_obs: Optional[float] = None
    log_lik_ratings_missing: Optional[float] = None
    log_lik_rankings_obs: Optional[float] = None
    log_lik_rankings_missing: Optional[float] = None

    # Posterior rating probabilities under measurement noise
    # train: shape [I*J, K_train, C], test: shape [I*J, K_test, C]
    train_posterior_rating_probs: Optional[np.ndarray] = None
    test_posterior_rating_probs: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroundTruthBundle':
        """Create GroundTruthBundle from dictionary (e.g., loaded from JSON)."""
        return cls(
            embeddings=np.array(data["embeddings"]),
            mean_preferences=np.array(data["mean_preferences"]),
            annotator_preferences=np.array(data["annotator_preferences"]),
            rating_probs=np.array(data["rating_probs"]),
            rating_cumprobs=np.array(data["rating_cumprobs"]),
            rating_thresholds_z=np.array(data["rating_thresholds_z"]),
            base_scores=np.array(data["base_scores"]),
            all_ratings=data["all_ratings"],
            all_pairwise=data["all_pairwise"],
            observed_ratings=data["observed_ratings"],
            missing_ratings=data["missing_ratings"],
            missing_ratings_indexes=[i for i in range(len(data["missing_ratings"])) if data["missing_ratings"][i]["instance"] == "test"],
            observed_pairwise=data["observed_pairwise"],
            missing_pairwise=data["missing_pairwise"],
            stats=data["stats"],
            log_lik_ratings_obs=data.get("log_lik_ratings_obs"),
            log_lik_ratings_missing=data.get("log_lik_ratings_missing"),
            log_lik_rankings_obs=data.get("log_lik_rankings_obs"),
            log_lik_rankings_missing=data.get("log_lik_rankings_missing"),
            train_posterior_rating_probs=(np.array(data["train_posterior_rating_probs"]) if "train_posterior_rating_probs" in data else None),
            test_posterior_rating_probs=(np.array(data["test_posterior_rating_probs"]) if "test_posterior_rating_probs" in data else None),
        )


@dataclass
class ObservedSet:
    ratings: List[Dict[str, Any]]
    pairwise: List[Dict[str, Any]]


@dataclass
class MissingSet:
    ratings: List[Dict[str, Any]]
    pairwise: List[Dict[str, Any]]


