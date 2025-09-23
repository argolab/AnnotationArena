from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class GroundTruthBundle:
    embeddings: np.ndarray  # shape [K, D]
    mean_preferences: np.ndarray  # shape [I, D]
    annotator_preferences: np.ndarray  # shape [I*J, D] with ij_idx = (i-1)*J + j
    rating_probs: np.ndarray  # shape [I*J, C]
    rating_thresholds: np.ndarray  # shape [I*J, C] (internal thresholds)
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

    # Example keys: {'total_possible_ratings': int, 'total_pairwise_rankings': int, 'train_ratings': int, ...}
    stats: Dict[str, Any]

    log_lik_ratings_obs: Optional[float] = None
    log_lik_ratings_missing: Optional[float] = None
    log_lik_rankings_obs: Optional[float] = None
    log_lik_rankings_missing: Optional[float] = None


@dataclass
class ObservedSet:
    ratings: List[Dict[str, Any]]
    pairwise: List[Dict[str, Any]]


@dataclass
class MissingSet:
    ratings: List[Dict[str, Any]]
    pairwise: List[Dict[str, Any]]


