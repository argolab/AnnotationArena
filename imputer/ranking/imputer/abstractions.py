from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from imputer.data import RankingData


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
    def get_rating_embedding(self, attribute_id: int, annotator_id: int, item_id: int, rating_value: Optional[int], is_masked: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def get_ranking_embedding(self, attribute_id: int, annotator_id: int, item_ids: List[int], ranking_order: Optional[List[int]], is_masked: bool = False) -> torch.Tensor:
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
        # Call on_forward_start hook for subclasses to perform initialization
        self.on_forward_start(variables)
        
        V = len(variables)  # this will be the input token length for transformer
        D = self.embedding_dim
        device = self._ensure_device()

        feature_embeddings = torch.zeros(1, V, D + 1 + max(self.max_rank_size, self.num_likert_classes), device=device)

        for i, var in enumerate(variables):
            # Determine if variable is masked (default to False if None)
            is_masked = var.is_masked if var.is_masked is not None else False
            is_missing = var.is_missing if var.is_missing is not None else False

            is_masked = is_masked or is_missing

            if var.is_listwise:
                feat = self.get_ranking_embedding(var.attribute_id, var.annotator_id, var.item_ids[: self.max_rank_size], var.ranking_order, is_masked)
                feature_embeddings[0, i] = feat
            else:
                item_id = var.item_ids[0] if len(var.item_ids) > 0 else -1
                feat = self.get_rating_embedding(var.attribute_id, var.annotator_id, item_id, var.rating_value, is_masked)
                feature_embeddings[0, i] = feat

        return feature_embeddings[:, :, :D], feature_embeddings[:, :, D+1:]

    def on_forward_start(self, variables: List[RankingData]):
        pass