import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from imputer.abstractions import RankingEmbeddingProviderBase
from imputer.data import RankingData
import logging

logger = logging.getLogger(__name__)


class AtomCompositonalEmbeddingProvider(RankingEmbeddingProviderBase):

    def __init__(
        self,
        num_attributes: int,
        num_annotators: int,
        num_items: int,
        embedding_dim: int,
        num_likert_classes: int,
        max_rank_size: int,
        device: str,
        embedding_dropout: float = 0.0,
        use_concat_embedding: bool = False,
        enable_full_dropout: bool = False,
    ):

        super().__init__(
            num_likert_classes=num_likert_classes,
            max_rank_size=max_rank_size,
            embedding_dim=embedding_dim
        )
        self.use_concat_embedding = use_concat_embedding
        
        self.num_attributes = num_attributes
        self.num_annotators = num_annotators
        self.num_items = num_items
        self.device = device
        self.embedding_dropout_p = float(embedding_dropout)
        self.enable_full_dropout = enable_full_dropout

        # Dimension scaling for concat mode: each component vector should be embedding_dim // 3
        # to ensure concatenation yields embedding_dim
        if self.use_concat_embedding:
            # Each component vector dimension: 3 (one-hot) + internal_dimension = embedding_dim // 3
            self.component_dim = embedding_dim // 3
            self.internal_dimension = self.component_dim - 3
            assert embedding_dim % 3 == 0, f"embedding_dim must be divisible by 3 for concat mode, got {embedding_dim}"
        else:
            # Non-concat mode: use learned weight matrices, keep original dimension logic
            self.component_dim = embedding_dim
            self.internal_dimension = embedding_dim - 3
            self.W_I = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
            self.W_J = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
            self.W_K = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        
        # Pairwise relation matrix dimension matches component dimension
        # Initialize to near negative identity
        self.pairwise_relation = nn.Parameter(-torch.eye(self.component_dim) + 1e-3 * torch.randn(self.component_dim, self.component_dim))

        print(f"use_concat_embedding: {self.use_concat_embedding}")

        # Embeddings for each component (all learnable parameters)
        self.attribute_embedding = nn.Parameter(torch.randn(num_attributes, self.internal_dimension))
        self.annotator_embedding = nn.Parameter(torch.randn(num_annotators, self.internal_dimension))
        self.item_embedding = nn.Parameter(torch.randn(num_items, self.internal_dimension))

        # Initialize all learnable parameters with Kaiming normal initialization
        torch.nn.init.kaiming_normal_(self.attribute_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.annotator_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.item_embedding, mode='fan_out', nonlinearity='relu')

        # Class constant for the logit value
        self.LOGIT_HIGH = getattr(self, "LOGIT_HIGH", 20.0)


    def _full_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding_dropout_p <= 0.0 or not self.enable_full_dropout:
            return x
        keep_prob = 1.0 - self.embedding_dropout_p
        if keep_prob <= 0.0:
            return torch.zeros_like(x)
        mask = (torch.rand((), device=x.device) < keep_prob).item()
        if not mask:
            return torch.zeros_like(x)
        return x / keep_prob
    
    def set_full_dropout(self, enabled: bool):
        """Enable or disable full dropout. Should be True during training, False during evaluation."""
        self.enable_full_dropout = enabled

    def get_rating_embedding(self, attribute_id: int, annotator_id: int, item_id: int, rating_value, is_missing: bool = False) -> torch.Tensor:
        """Implementation for rating embeddings."""
        attr_core = self._full_dropout(self.attribute_embedding[attribute_id])
        annot_core = self._full_dropout(self.annotator_embedding[annotator_id])
        attr_vec = torch.cat((torch.tensor([1, 0, 0]).to(self.device), attr_core), dim=-1)
        
        # Build annot_vec and item_vec (all learnable now)
        annot_vec = torch.cat((torch.tensor([0, 1, 0]).to(self.device), annot_core), dim=-1)
        item_vec = torch.cat((torch.tensor([0, 0, 1]).to(self.device), self.item_embedding[item_id]), dim=-1)
        
        assert 0 <= item_id < self.num_items, f"Item ID {item_id} is out of bounds"
        
        # Combine embeddings based on use_concat_embedding flag
        if self.use_concat_embedding:
            total_embedding = torch.cat([attr_vec, annot_vec, item_vec], dim=-1)
        else:
            total_embedding = self.W_I @ attr_vec + self.W_J @ annot_vec + self.W_K @ item_vec
        
        parameter = torch.zeros(self.parameter_dimension).to(self.device)
        if is_missing:
            # Missing/masked items: encode uniform distribution (all zeros = uniform logits)
            parameter[0] = 1.0  # Mask bit
            # Leave remaining dimensions as zeros (uniform distribution)
        else:
            # Observed items: encode the known rating value
            assert rating_value is not None and 0 <= rating_value < self.num_likert_classes, f"rating_value {rating_value} must be in range [0, {self.num_likert_classes})"
            parameter[0] = 0.0  # Not masked
            parameter[rating_value + 1] = self.LOGIT_HIGH  # One-hot-like encoding of rating logit
        return torch.cat((total_embedding, parameter), dim=-1)

    # Get embedding for ranking variables
    def get_ranking_embedding(self, attribute_id: int, annotator_id: int, item_ids: List[int], ranking_order, is_missing: bool = False) -> torch.Tensor:
        """Implementation for ranking embeddings."""
        attr_core = self._full_dropout(self.attribute_embedding[attribute_id])
        annot_core = self._full_dropout(self.annotator_embedding[annotator_id])
        attr_vec = torch.cat((torch.tensor([1, 0, 0]).to(self.device), attr_core), dim=-1)
        annot_vec = torch.cat((torch.tensor([0, 1, 0]).to(self.device), annot_core), dim=-1)

        item_embedding_1 = torch.cat((torch.tensor([0, 0, 1]).to(self.device), self.item_embedding[item_ids[0]]), dim=-1)
        item_embedding_2 = torch.cat((torch.tensor([0, 0, 1]).to(self.device), self.item_embedding[item_ids[1]]), dim=-1)
        item_embedding = item_embedding_1 + item_embedding_2 @ self.pairwise_relation
        
        # Combine embeddings based on use_concat_embedding flag
        if self.use_concat_embedding:
            total_embedding = torch.cat([attr_vec, annot_vec, item_embedding], dim=-1)
        else:
            total_embedding = self.W_I @ attr_vec + self.W_J @ annot_vec + self.W_K @ item_embedding
        
        parameter = torch.zeros(self.parameter_dimension).to(self.device)
        if is_missing:
            # Missing/masked items: encode uniform distribution (50-50 chance for pairwise)
            parameter[0] = 1.0  # Mask bit
            # Leave remaining dimensions as zeros (uniform distribution)
        else:
            # Observed items: encode proper logit convention
            assert ranking_order is not None, "ranking_order cannot be None for observed rankings"
            assert len(ranking_order) == self.max_rank_size, f"ranking_order length {len(ranking_order)} must equal max_rank_size {self.max_rank_size}"
            parameter[0] = 0.0  # Not masked
            
            # Convert ranking order to logits: first item wins if ranking_order[0] < ranking_order[1]
            # Higher logit = better ranking, so first item gets higher logit if it wins
            # Use a class constant for the logit value
            if ranking_order[0] < ranking_order[1]:  # First item wins
                parameter[1] = self.LOGIT_HIGH  # First item gets higher logit
                parameter[2] = 0.0  # Second item gets lower logit
            else:  # Second item wins
                parameter[1] = 0.0  # First item gets lower logit
                parameter[2] = self.LOGIT_HIGH  # Second item gets higher logit
        return torch.cat((total_embedding, parameter), dim=-1)
    
