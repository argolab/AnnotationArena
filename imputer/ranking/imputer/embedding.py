import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .abstractions import RankingEmbeddingProviderBase
from .data import RankingData


class OuterProductRankingEmbeddingProvider(RankingEmbeddingProviderBase):
    """Embedding provider using sum of outer products for rankings.

    Implements get_rating_embedding/get_ranking_embedding and inherits the forward(List[RankingData])
    from RankingEmbeddingProviderBase.
    """

    def __init__(
        self,
        num_attributes: int,
        num_annotators: int,
        num_items: int,
        embedding_dim: int,
        num_likert_classes: int,
        max_rank_size: int,
    ):
        super().__init__(
            num_likert_classes=num_likert_classes,
            max_rank_size=max_rank_size,
            embedding_dim=embedding_dim,
        )
        self.num_attributes = num_attributes
        self.num_annotators = num_annotators
        self.num_items = num_items

        # Embeddings for each component (learned parameters)
        self.attribute_embedding = nn.Parameter(torch.randn(num_attributes, embedding_dim))
        self.annotator_embedding = nn.Parameter(torch.randn(num_annotators, embedding_dim))
        self.item_embedding = nn.Parameter(torch.randn(num_items, embedding_dim))

        torch.nn.init.kaiming_normal_(self.attribute_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.annotator_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.item_embedding, mode='fan_out', nonlinearity='relu')
        self.parameter_projection = nn.Linear(embedding_dim + self.num_likert_classes + self.max_rank_size + 1, embedding_dim)
        D = embedding_dim
        self.ranking_projection = nn.Sequential(
            nn.Linear(D * D, D * 2),
            nn.ReLU(),
            nn.Linear(D * 2, D),
        )

    # Abstract hook implementations
    def get_rating_embedding(self, attribute_id: int, annotator_id: int, item_id: int, rating_value) -> torch.Tensor:
        # print("WARNING: not using rating values")
        attr_vec = self.attribute_embedding[attribute_id]
        annot_vec = self.annotator_embedding[annotator_id]
        assert 0 <= item_id < self.num_items, f"Item ID {item_id} is out of bounds"
        parameter = torch.zeros(self.num_likert_classes + self.max_rank_size + 1).to("cpu")
        if rating_value is None:
            parameter[0] = 1.0
        else:
            parameter[rating_value + 1] = 1.0
        return self.parameter_projection(torch.cat((attr_vec + annot_vec + self.item_embedding[item_id], parameter), dim=-1))

    # Get embedding for ranking variables
    def get_ranking_embedding(self, attribute_id: int, annotator_id: int, item_ids: List[int], ranking_order) -> torch.Tensor:
        # print("WARNING: not using ranking order")
        attr_vec = self.attribute_embedding[attribute_id]
        annot_vec = self.annotator_embedding[annotator_id]
        valid = [i for i in item_ids if 0 <= i < self.num_items]
        assert len(valid) > 1, "At least two valid items are required for ranking"
        item_attr_embeddings: List[torch.Tensor] = [attr_vec + annot_vec + self.item_embedding[i] for i in valid]
        total_outer = torch.zeros(self.embedding_dim, self.embedding_dim, device=attr_vec.device)
        for i in range(len(item_attr_embeddings)):
            for j in range(i + 1, len(item_attr_embeddings)):
                total_outer += torch.outer(item_attr_embeddings[i], item_attr_embeddings[j])

        parameter = torch.zeros(self.num_likert_classes + self.max_rank_size + 1).to("cpu")
        if ranking_order is None:
            parameter[0] = 1.0
        else:
            parameter[self.num_likert_classes + 1:] = torch.tensor(ranking_order)
        return self.parameter_projection(torch.cat((self.ranking_projection(total_outer.flatten()), parameter), dim=-1))
    

class PairwiseRankingProjectionEmbeddingProvider(RankingEmbeddingProviderBase):
    """Embedding provider using sum of outer products for rankings.

    Implements get_rating_embedding/get_ranking_embedding and inherits the forward(List[RankingData])
    from RankingEmbeddingProviderBase.
    """

    def __init__(
        self,
        num_attributes: int,
        num_annotators: int,
        num_items: int,
        embedding_dim: int,
        num_likert_classes: int,
        max_rank_size: int,
        device: str,
    ):
        super().__init__(
            num_likert_classes=num_likert_classes,
            max_rank_size=max_rank_size,
            embedding_dim=embedding_dim,
        )
        self.num_attributes = num_attributes
        self.num_annotators = num_annotators
        self.num_items = num_items
        self.device = device
        # Embeddings for each component (learned parameters)
        self.attribute_embedding = nn.Parameter(torch.randn(num_attributes, embedding_dim))
        self.annotator_embedding = nn.Parameter(torch.randn(num_annotators, embedding_dim))
        self.item_embedding = nn.Parameter(torch.randn(num_items, embedding_dim))

        torch.nn.init.kaiming_normal_(self.attribute_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.annotator_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.item_embedding, mode='fan_out', nonlinearity='relu')
        self.parameter_projection = nn.Linear(embedding_dim + self.num_likert_classes + self.max_rank_size + 1, embedding_dim)
        self.pairwise_relation = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        D = embedding_dim
        self.ranking_projection = nn.Sequential(
            nn.Linear(D * D, D * 2),
            nn.ReLU(),
            nn.Linear(D * 2, D),
        )

    # Abstract hook implementations
    def get_rating_embedding(self, attribute_id: int, annotator_id: int, item_id: int, rating_value) -> torch.Tensor:
        # print("WARNING: not using rating values")
        attr_vec = self.attribute_embedding[attribute_id]
        annot_vec = self.annotator_embedding[annotator_id]
        assert 0 <= item_id < self.num_items, f"Item ID {item_id} is out of bounds"
        parameter = torch.zeros(self.num_likert_classes + self.max_rank_size + 1).to(self.device)
        if rating_value is None:
            parameter[0] = 1.0
        else:
            parameter[rating_value + 1] = 1.0
        return self.parameter_projection(torch.cat((attr_vec + annot_vec + self.item_embedding[item_id], parameter), dim=-1))

    # Get embedding for ranking variables
    def get_ranking_embedding(self, attribute_id: int, annotator_id: int, item_ids: List[int], ranking_order) -> torch.Tensor:
        # print("WARNING: not using ranking order")
        attr_vec = self.attribute_embedding[attribute_id]
        annot_vec = self.annotator_embedding[annotator_id]
        #assert len(item_ids) == 2, "Pairwise Ranking Embedding Provider only support two items ranking" should add this back when generated ranking with 2 items data

        item_embedding_1 = self.item_embedding[item_ids[0]]
        item_embedding_2 = self.item_embedding[item_ids[1]]
        item_embedding = item_embedding_1 + item_embedding_2 @ self.pairwise_relation
        total_embedding = attr_vec + annot_vec + item_embedding
        parameter = torch.zeros(self.num_likert_classes + self.max_rank_size + 1).to(self.device)
        if ranking_order is None:
            parameter[0] = 1.0
        else:
            parameter[self.num_likert_classes + 1:] = torch.tensor(ranking_order)
        return self.parameter_projection(torch.cat((total_embedding, parameter), dim=-1))
