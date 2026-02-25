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

    @property
    def parameter_dimension(self):
        return 1 + max(self.num_likert_classes, self.max_rank_size)  # 1 for the missing-status bit

    # Abstract hooks for subclasses
    def get_rating_embedding(self, attribute_id: int, annotator_id: int, item_id: int, rating_value: Optional[int], is_missing: bool = False, rating_dist=None) -> torch.Tensor:
        raise NotImplementedError

    def get_ranking_embedding(self, attribute_id: int, annotator_id: int, item_ids: List[int], ranking_order: Optional[List[int]], is_missing: bool = False) -> torch.Tensor:
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
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Call on_forward_start hook for subclasses to perform initialization (only for B=1)
        if batch_size == 1:
            self.on_forward_start(variables)
        
        V = len(variables)  # this will be the input token length for transformer
        D = self.embedding_dim
        B = batch_size
        device = self._ensure_device()

        if B == 1:
            # Original behavior: single batch
            # Collect embeddings in a list to preserve gradient flow
            # Using in-place assignment into a zero tensor breaks autograd graph
            embedding_list = []

            for i, var in enumerate(variables):
                is_missing = var.is_missing or var.is_masked
                assert not (var.is_missing and var.is_masked), "Variable cannot be both missing and masked"
                assert var.is_missing is not None and var.is_masked is not None, "Variable must have missing or masked status"

                if var.is_listwise:
                    feat = self.get_ranking_embedding(var.attribute_id, var.annotator_id, var.item_ids[: self.max_rank_size], var.ranking_order, is_missing)
                else:
                    item_id = var.item_ids[0] if len(var.item_ids) > 0 else -1
                    feat = self.get_rating_embedding(var.attribute_id, var.annotator_id, item_id, var.rating_value, is_missing, rating_dist=var.rating_dist)
                
                embedding_list.append(feat)

            # Stack embeddings to preserve gradient flow from embedding parameters
            # Shape: [V, D + parameter_dimension] -> [1, V, D + parameter_dimension]
            feature_embeddings = torch.stack(embedding_list, dim=0).unsqueeze(0)
        else:
            # Batched behavior: generate B embeddings (for different masking patterns)
            # Shape: [B, V, D + parameter_dimension]
            batch_embeddings = []
            for b in range(B):
                embedding_list = []
                for i, var in enumerate(variables):
                    is_missing = var.is_missing or var.is_masked
                    assert not (var.is_missing and var.is_masked), "Variable cannot be both missing and masked"
                    assert var.is_missing is not None and var.is_masked is not None, "Variable must have missing or masked status"

                    if var.is_listwise:
                        feat = self.get_ranking_embedding(var.attribute_id, var.annotator_id, var.item_ids[: self.max_rank_size], var.ranking_order, is_missing)
                    else:
                        item_id = var.item_ids[0] if len(var.item_ids) > 0 else -1
                        feat = self.get_rating_embedding(var.attribute_id, var.annotator_id, item_id, var.rating_value, is_missing, rating_dist=var.rating_dist)
                    
                    embedding_list.append(feat)
                
                batch_embeddings.append(torch.stack(embedding_list, dim=0))
            
            # Stack batches: [B, V, D + parameter_dimension]
            feature_embeddings = torch.stack(batch_embeddings, dim=0)

        # Return feature embeddings and full param stream including the missing-status bit
        return feature_embeddings[:, :, :D], feature_embeddings[:, :, D:]

    def on_forward_start(self, variables: List[RankingData]):
        pass