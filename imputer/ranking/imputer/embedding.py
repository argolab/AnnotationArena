import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .abstractions import RankingEmbeddingProviderBase
from .data import RankingData

import logging

logger = logging.getLogger(__name__)

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

    @classmethod
    def _from_true_embedding(
        cls,
        attribute_embedding=None,
        annotator_embedding=None,
        item_embedding=None,
        *,
        attribute_embedding_size: tuple | None = None,
        annotator_embedding_size: tuple | None = None,
        item_embedding_size: tuple | None = None,
        num_likert_classes: int,
        max_rank_size: int,
        freeze: bool | dict = False,
    ) -> "OuterProductRankingEmbeddingProvider":
        f"""Factory: build a provider from ground-truth embedding matrices.

        Accepts any subset of the three matrices. Shapes are [count, D]. For any
        component not provided, pass its (count, D) via the corresponding *_embedding_size.
        All components must agree on D.

        Example:
            provider = OuterProductRankingEmbeddingProvider._from_true_embedding(
                attribute_embedding=attr_gt,            # [A, D]
                annotator_embedding_size=(U, D),      # if embedding not provided, give sizes
                item_embedding=item_gt,                 # [I, D]
                num_likert_classes=5,
                max_rank_size=3,
                freeze={'attribute': True, 'item': False, 'annotator': False}, # freeze only attribute embeddings
            )
        """
        # Convert to tensors to infer shapes and dtypes
        def to_tensor(x):
            if x is None:
                return None
            return x if torch.is_tensor(x) else torch.as_tensor(x)


        # assert either size or embedding is provided
        assert attribute_embedding_size is not None or attribute_embedding is not None, "attribute_embedding_size must be provided when attribute_embedding is None"
        assert annotator_embedding_size is not None or annotator_embedding is not None, "annotator_embedding_size must be provided when annotator_embedding is None"
        assert item_embedding_size is not None or item_embedding is not None, "item_embedding_size must be provided when item_embedding is None"

        attr = to_tensor(attribute_embedding)
        anno = to_tensor(annotator_embedding)
        item = to_tensor(item_embedding)

        def infer_shape(name: str, t: torch.Tensor | None, size_tuple: tuple | None):
            if t is not None:
                assert t.ndim == 2, f"{name} must be rank-2"
                return int(t.shape[0]), int(t.shape[1])
            assert size_tuple is not None and len(size_tuple) == 2, (
                f"{name}_size=(count, dim) must be provided when {name} is None"
            )
            return int(size_tuple[0]), int(size_tuple[1])

        A, D_attr = infer_shape('attribute_embedding', attr, attribute_embedding_size)
        U, D_anno = infer_shape('annotator_embedding', anno, annotator_embedding_size)
        I, D_item = infer_shape('item_embedding', item, item_embedding_size)

        # Validate and choose embedding dimension
        if not (D_attr == D_anno == D_item):
            raise ValueError(f"Inconsistent embedding dims: attr={D_attr}, annot={D_anno}, item={D_item}")
        D = int(D_attr)

        # Debug logs of inferred sizes
        logger.info(f"[EmbeddingProvider] attribute: count={A}, dim={D_attr}, provided={'yes' if attr is not None else 'no'}")
        logger.info(f"[EmbeddingProvider] annotator: count={U}, dim={D_anno}, provided={'yes' if anno is not None else 'no'}")
        logger.info(f"[EmbeddingProvider] item: count={I}, dim={D_item}, provided={'yes' if item is not None else 'no'}")
        logger.info(f"[EmbeddingProvider] final embedding_dim={D}")

        provider = cls(
            num_attributes=int(A),
            num_annotators=int(U),
            num_items=int(I),
            embedding_dim=D,
            num_likert_classes=int(num_likert_classes),
            max_rank_size=int(max_rank_size),
        )

        # Move tensors to provider device/dtype and copy (only if provided)
        with torch.no_grad():
            if attr is not None:
                provider.attribute_embedding.copy_(attr.to(device=provider.attribute_embedding.device, dtype=provider.attribute_embedding.dtype))
            if anno is not None:
                provider.annotator_embedding.copy_(anno.to(device=provider.annotator_embedding.device, dtype=provider.annotator_embedding.dtype))
            if item is not None:
                provider.item_embedding.copy_(item.to(device=provider.item_embedding.device, dtype=provider.item_embedding.dtype))

        # Handle optional freezing
        def get_flag(key: str) -> bool:
            if isinstance(freeze, dict):
                return bool(freeze.get(key, False))
            return bool(freeze)

        if get_flag('attribute'):
            provider.attribute_embedding.requires_grad = False
        if get_flag('annotator'):
            provider.annotator_embedding.requires_grad = False
        if get_flag('item'):
            provider.item_embedding.requires_grad = False

        return provider

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
