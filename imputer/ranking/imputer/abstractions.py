from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn as nn


@dataclass
class RankingData:
    """Structured representation of a single variable for ranking/rating.

    Fields are 0-indexed for model consumption.
    - annotator_id: annotator index
    - attribute_id: attribute index
    - is_listwise: True if this variable is a listwise ranking, False if a rating
    - item_ids: for rating, a list with a single item id; for ranking, the list of item ids
    
    Optional fields carry supervision when available (training/eval time):
    - rating_value: class index [0..C-1] if rating observed
    - ranking_order: list of positions in [1..R] of same length as item_ids when ranking observed
    """
    annotator_id: int
    attribute_id: int
    is_listwise: bool
    item_ids: List[int]
    rating_value: Optional[int] = None
    ranking_order: Optional[List[int]] = None


class EmbeddingProviderBase(nn.Module):
    """Base class for producing per-variable embeddings from structured inputs."""

    def __init__(self):
        super().__init__()

    def encode_structured(
        self,
        variables: List[RankingData],
        num_likert_classes: int,
        max_rank_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert structured variables into model tensors and embeddings.

        Returns a 5-tuple of tensors (all shaped [1, V, ...] for V variables):
          - feature_embeddings: [1, V, D]
          - param_data:        [1, V, P] where P=max(C, R)
          - variable_types:    [1, V] 0=rating, 1=ranking
          - attribute_ids:     [1, V]
          - annotator_ids:     [1, V]
        Implementations may leverage internal learned tables to build features.
        """
        raise NotImplementedError

