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
    """Very abstract provider base for embedding components."""

    def __init__(self):
        super().__init__()


class RankingEmbeddingProviderBase(EmbeddingProviderBase):
    """Base for ranking/rating embedding providers.

    - Implements forward(List[RankingData]) to route rating vs ranking.
    - Subclasses implement get_rating_embedding and get_ranking_embedding to return a D-dim feature vector.
    - Returns only feature embeddings [1, V, D]; no parameter stream.
    """

    def __init__(self, *, num_likert_classes: int, max_rank_size: int, embedding_dim: int):
        super().__init__()
        self.num_likert_classes = int(num_likert_classes)
        self.max_rank_size = int(max_rank_size)
        self.embedding_dim = int(embedding_dim) # word_embedding dim

    # Abstract hooks for subclasses
    def get_rating_embedding(self, attribute_id: int, annotator_id: int, item_id: int) -> torch.Tensor:
        raise NotImplementedError

    def get_ranking_embedding(self, attribute_id: int, annotator_id: int, item_ids: List[int]) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def _ensure_device(self) -> torch.device:
        # Use parameters' device if any, else CPU
        try:
            p = next(self.parameters())
            return p.device
        except StopIteration:
            return torch.device('cpu')

    def forward(
        self,
        variables: List[RankingData],
    ) -> torch.Tensor:
        V = len(variables)  # this will be the input token length for transformer
        D = self.embedding_dim
        device = self._ensure_device()

        feature_embeddings = torch.zeros(1, V, D, device=device)

        for i, var in enumerate(variables):
            if var.is_listwise:
                feat = self.get_ranking_embedding(var.attribute_id, var.annotator_id, var.item_ids[: self.max_rank_size])
                feature_embeddings[0, i] = feat
            else:
                item_id = var.item_ids[0] if len(var.item_ids) > 0 else -1
                feat = self.get_rating_embedding(var.attribute_id, var.annotator_id, item_id)
                feature_embeddings[0, i] = feat

        return feature_embeddings
