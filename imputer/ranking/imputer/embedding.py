import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from imputer.abstractions import RankingEmbeddingProviderBase
from imputer.data import RankingData
import random
import logging

logger = logging.getLogger(__name__)


class BaseRankingEmbeddingProvider(RankingEmbeddingProviderBase):
    """Intermediate base class that manages common embedding initialization and utilities.
    
    This class handles the common functionality shared between different ranking embedding
    providers, including embedding initialization, parameter projection, and the _from_true_embedding
    factory method.
    """

    def __init__(
        self,
        num_attributes: int,
        num_annotators: int,
        num_items: int,
        embedding_dim: int,
        num_likert_classes: int,
        max_rank_size: int,
        device: str = "cpu",
    ):
        super().__init__(
            num_likert_classes=num_likert_classes,
            max_rank_size=max_rank_size,
            embedding_dim=embedding_dim,
        )
        self.num_attributes = num_attributes
        self.num_annotators = num_annotators
        self.num_items = num_items
        self.device = torch.device(device)

        # Embeddings for each component (learned parameters)
        self.attribute_embedding = nn.Parameter(torch.randn(num_attributes, embedding_dim))
        self.annotator_embedding = nn.Parameter(torch.randn(num_annotators, embedding_dim))
        self.item_embedding = nn.Parameter(torch.randn(num_items, embedding_dim))

        torch.nn.init.kaiming_normal_(self.attribute_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.annotator_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.item_embedding, mode='fan_out', nonlinearity='relu')

    def reset_embedding(self, num_attributes: int, num_annotators: int, num_items: int, embedding_dim: int):
        """Reset embeddings to random values with specified dimensions."""
        self.num_attributes = num_attributes
        self.num_annotators = num_annotators
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # Reinitialize embeddings with new random values
        self.attribute_embedding = nn.Parameter(torch.randn(num_attributes, embedding_dim))
        self.annotator_embedding = nn.Parameter(torch.randn(num_annotators, embedding_dim))
        self.item_embedding = nn.Parameter(torch.randn(num_items, embedding_dim))
        
        # Apply kaiming initialization
        torch.nn.init.kaiming_normal_(self.attribute_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.annotator_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.item_embedding, mode='fan_out', nonlinearity='relu')
        
        # Set embeddings as non-trainable for randomized providers
        self.attribute_embedding.requires_grad = False
        self.annotator_embedding.requires_grad = False
        self.item_embedding.requires_grad = False

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
        device: str = "cuda",
    ) -> "BaseRankingEmbeddingProvider":
        f"""Factory: build a provider from ground-truth embedding matrices.

        Accepts any subset of the three matrices. Shapes are [count, D]. For any
        component not provided, pass its (count, D) via the corresponding *_embedding_size.
        All components must agree on D.

        Example:
            provider = BaseRankingEmbeddingProvider._from_true_embedding(
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
            device=device,
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


class OuterProductRankingEmbeddingProvider(BaseRankingEmbeddingProvider):
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
            num_attributes=num_attributes,
            num_annotators=num_annotators,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_likert_classes=num_likert_classes,
            max_rank_size=max_rank_size,
            device=device,
        )
        self.parameter_projection = nn.Linear(embedding_dim + 1 + self.num_likert_classes + self.max_rank_size, embedding_dim)
        D = embedding_dim
        self.ranking_projection = nn.Sequential(
            nn.Linear(D * D, D * 2),
            nn.ReLU(),
            nn.Linear(D * 2, D),
        )

    def get_rating_embedding(self, attribute_id: int, annotator_id: int, item_id: int, rating_value, is_missing: bool = False) -> torch.Tensor:
        """Implementation for rating embeddings."""
        attr_vec = self.attribute_embedding[attribute_id]
        annot_vec = self.annotator_embedding[annotator_id]
        assert 0 <= item_id < self.num_items, f"Item ID {item_id} is out of bounds"
        parameter = torch.zeros(1 + self.num_likert_classes + self.max_rank_size).to(self.device)
        if is_missing:
            parameter[0] = 1.0
        else:
            assert rating_value is not None and 0 <= rating_value < self.num_likert_classes, f"rating_value {rating_value} must be in range [0, {self.num_likert_classes})"
            parameter[rating_value + 1] = 1.0
        return self.parameter_projection(torch.cat((attr_vec + annot_vec + self.item_embedding[item_id], parameter), dim=-1))

    # Get embedding for ranking variables
    def get_ranking_embedding(self, attribute_id: int, annotator_id: int, item_ids: List[int], ranking_order, is_missing: bool = False) -> torch.Tensor:
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

        parameter = torch.zeros(1 + self.num_likert_classes + self.max_rank_size).to(self.device)
        if is_missing:
            parameter[0] = 1.0
        else:
            assert ranking_order is not None, "ranking_order cannot be None for observed rankings"
            ranking_tensor = torch.tensor(ranking_order)
            assert len(ranking_order) == self.max_rank_size, f"ranking_order length {len(ranking_order)} must equal max_rank_size {self.max_rank_size}"
            parameter[self.num_likert_classes + 1:] = ranking_tensor
        return self.parameter_projection(torch.cat((self.ranking_projection(total_outer.flatten()), parameter), dim=-1))
    

class CombineRandomTrainedEmbeddingProvider(BaseRankingEmbeddingProvider):
    """Embedding provider using pairwise relations for rankings.

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
            num_attributes=num_attributes,
            num_annotators=num_annotators,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_likert_classes=num_likert_classes,
            max_rank_size=max_rank_size,
            device=device,
        )
        self.parameter_projection = nn.Linear(embedding_dim + self.num_likert_classes + self.max_rank_size + 1, embedding_dim)
        self.pairwise_relation = nn.Parameter(torch.randn(embedding_dim, embedding_dim))


    def get_rating_embedding(self, attribute_id: int, annotator_id: int, item_id: int, rating_value, is_missing: bool = False) -> torch.Tensor:
        """Implementation for rating embeddings."""
        attr_vec = torch.cat((torch.randn(self.embedding_dim // 2), self.attribute_embedding[attribute_id][self.embedding_dim // 2:]), dim=-1)
        annot_vec = torch.cat((torch.randn(self.embedding_dim // 2), self.annotator_embedding[annotator_id][self.embedding_dim // 2:]), dim=-1)
        assert 0 <= item_id < self.num_items, f"Item ID {item_id} is out of bounds"
        parameter = torch.zeros(self.num_likert_classes + self.max_rank_size + 1).to(self.device)
        if is_missing:
            parameter[0] = 1.0
        else:
            assert rating_value is not None and 0 <= rating_value < self.num_likert_classes, f"rating_value {rating_value} must be in range [0, {self.num_likert_classes})"
            parameter[rating_value + 1] = 1.0
        item_embedding =torch.cat((torch.randn(self.embedding_dim // 2), self.item_embedding[item_id][self.embedding_dim // 2:]), dim=-1)
        return self.parameter_projection(torch.cat((attr_vec + annot_vec + item_embedding, parameter), dim=-1))

    # Get embedding for ranking variables
    def get_ranking_embedding(self, attribute_id: int, annotator_id: int, item_ids: List[int], ranking_order, is_missing: bool = False) -> torch.Tensor:
        # print("WARNING: not using ranking order")
        attr_vec = torch.cat((torch.randn(self.embedding_dim // 2), self.attribute_embedding[attribute_id][self.embedding_dim // 2:]), dim=-1)
        annot_vec = torch.cat((torch.randn(self.embedding_dim // 2), self.annotator_embedding[annotator_id][self.embedding_dim // 2:]), dim=-1)
        assert len(item_ids) == 2, "Pairwise Ranking Embedding Provider only support two items ranking"

        item_embedding_1 = torch.cat((torch.randn(self.embedding_dim // 2), self.item_embedding[item_ids[0]][self.embedding_dim // 2:]), dim=-1)
        item_embedding_2 = torch.cat((torch.randn(self.embedding_dim // 2), self.item_embedding[item_ids[1]][self.embedding_dim // 2:]), dim=-1)
        item_embedding = item_embedding_1 + item_embedding_2 @ self.pairwise_relation
        total_embedding = attr_vec + annot_vec + item_embedding
        parameter = torch.zeros(self.num_likert_classes + self.max_rank_size + 1).to(self.device)
        if is_missing:
            parameter[0] = 1.0
        else:
            assert ranking_order is not None, "ranking_order cannot be None for observed rankings"
            ranking_tensor = torch.tensor(ranking_order)
            assert len(ranking_order) == self.max_rank_size, f"ranking_order length {len(ranking_order)} must equal max_rank_size {self.max_rank_size}"
            parameter[self.num_likert_classes + 1:] = ranking_tensor
        return self.parameter_projection(torch.cat((total_embedding, parameter), dim=-1))

class PairwiseRankingProjectionEmbeddingProvider(BaseRankingEmbeddingProvider):
    """Embedding provider using pairwise relations for rankings.

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
            num_attributes=num_attributes,
            num_annotators=num_annotators,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_likert_classes=num_likert_classes,
            max_rank_size=max_rank_size,
            device=device,
        )
        self.parameter_projection = nn.Linear(embedding_dim + self.num_likert_classes + self.max_rank_size + 1, embedding_dim)
        self.pairwise_relation = nn.Parameter(torch.randn(embedding_dim, embedding_dim))


    def get_rating_embedding(self, attribute_id: int, annotator_id: int, item_id: int, rating_value, is_missing: bool = False) -> torch.Tensor:
        """Implementation for rating embeddings."""
        attr_vec = self.attribute_embedding[attribute_id]
        annot_vec = self.annotator_embedding[annotator_id]
        assert 0 <= item_id < self.num_items, f"Item ID {item_id} is out of bounds"
        parameter = torch.zeros(self.num_likert_classes + self.max_rank_size + 1).to(self.device)
        if is_missing:
            parameter[0] = 1.0
        else:
            assert rating_value is not None and 0 <= rating_value < self.num_likert_classes, f"rating_value {rating_value} must be in range [0, {self.num_likert_classes})"
            parameter[rating_value + 1] = 1.0
        return self.parameter_projection(torch.cat((attr_vec + annot_vec + self.item_embedding[item_id], parameter), dim=-1))

    # Get embedding for ranking variables
    def get_ranking_embedding(self, attribute_id: int, annotator_id: int, item_ids: List[int], ranking_order, is_missing: bool = False) -> torch.Tensor:
        # print("WARNING: not using ranking order")
        attr_vec = self.attribute_embedding[attribute_id]
        annot_vec = self.annotator_embedding[annotator_id]
        assert len(item_ids) == 2, "Pairwise Ranking Embedding Provider only support two items ranking"

        item_embedding_1 = self.item_embedding[item_ids[0]]
        item_embedding_2 = self.item_embedding[item_ids[1]]
        item_embedding = item_embedding_1 + item_embedding_2 @ self.pairwise_relation
        total_embedding = attr_vec + annot_vec + item_embedding
        parameter = torch.zeros(self.num_likert_classes + self.max_rank_size + 1).to(self.device)
        if is_missing:
            parameter[0] = 1.0
        else:
            assert ranking_order is not None, "ranking_order cannot be None for observed rankings"
            ranking_tensor = torch.tensor(ranking_order)
            assert len(ranking_order) == self.max_rank_size, f"ranking_order length {len(ranking_order)} must equal max_rank_size {self.max_rank_size}"
            parameter[self.num_likert_classes + 1:] = ranking_tensor
        return self.parameter_projection(torch.cat((total_embedding, parameter), dim=-1))


class FullyRandomizedEmbeddingProvider(PairwiseRankingProjectionEmbeddingProvider):
    """Fully randomized embedding provider that generates new random embeddings for each forward pass.
    
    This class extends PairwiseRankingProjectionEmbeddingProvider but overrides on_forward_start
    to generate completely random embeddings for all attributes, annotators, and items at the
    beginning of each forward pass. The get_rating_embedding and get_ranking_embedding methods
    remain the same as the parent class.
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
            num_attributes=num_attributes,
            num_annotators=num_annotators,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_likert_classes=num_likert_classes,
            max_rank_size=max_rank_size,
            device=device,
        )

    def on_forward_start(self, variables: List[RankingData]):
        """Generate new random embeddings for all components at the start of each forward pass."""
        # Calculate the required dimensions based on the variables in this batch
        max_attribute_id = max(var.attribute_id for var in variables) if variables else 0
        max_annotator_id = max(var.annotator_id for var in variables) if variables else 0
        max_item_id = max(max(var.item_ids) if var.item_ids else 0 for var in variables) if variables else 0
        
        # Add 1 to convert from 0-based indexing to count
        required_attributes = max_attribute_id + 1
        required_annotators = max_annotator_id + 1
        required_items = max_item_id + 1
        
        # Reset embeddings with the required dimensions
        self.reset_embedding(
            num_attributes=required_attributes,
            num_annotators=required_annotators,
            num_items=required_items,
            embedding_dim=self.embedding_dim
        )

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
        randomness: bool
    ):

        super().__init__(
            num_likert_classes=num_likert_classes,
            max_rank_size=max_rank_size,
            embedding_dim=embedding_dim
        )
        self.pairwise_relation = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.W_I = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.W_J = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.W_K = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.W_K1 = nn.Parameter(torch.randn(embedding_dim, embedding_dim))

        self.num_attributes = num_attributes
        self.num_annotators = num_annotators
        self.num_items = num_items
        self.device = device
        self.internal_dimension = embedding_dim - 3
        self.randomness = randomness

        print(f"Randomness set to {self.randomness}")

        # Embeddings for each component (learned parameters)
        self.attribute_embedding = nn.Parameter(torch.randn(num_attributes, self.internal_dimension))
        self.annotator_embedding_learnable = nn.Parameter(torch.randn(num_annotators, self.internal_dimension // 2))
        self.annotator_embedding_random = torch.rand(num_annotators, self.internal_dimension - self.internal_dimension // 2, device=self.device)
        self.item_embedding = torch.rand(num_items, self.internal_dimension, device=self.device)

        torch.nn.init.kaiming_normal_(self.attribute_embedding, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.annotator_embedding_learnable, mode='fan_out', nonlinearity='relu')

        # Class constant for the logit value
        self.LOGIT_HIGH = getattr(self, "LOGIT_HIGH", 20.0)

    def get_rating_embedding(self, attribute_id: int, annotator_id: int, item_id: int, rating_value, is_missing: bool = False) -> torch.Tensor:
        """Implementation for rating embeddings."""
        attr_vec = torch.cat((torch.tensor([1, 0, 0]).to(self.device), self.attribute_embedding[attribute_id]), dim=-1)
        annot_vec = torch.cat((torch.tensor([0, 1, 0]).to(self.device), self.annotator_embedding_learnable[annotator_id], self.annotator_embedding_random[annotator_id]), dim=-1)
        item_vec = torch.cat((torch.tensor([0, 0, 1]).to(self.device), self.item_embedding[item_id]), dim=-1)
        assert 0 <= item_id < self.num_items, f"Item ID {item_id} is out of bounds"
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
        return torch.cat((attr_vec + annot_vec + item_vec, parameter), dim=-1)

    # Get embedding for ranking variables
    def get_ranking_embedding(self, attribute_id: int, annotator_id: int, item_ids: List[int], ranking_order, is_missing: bool = False) -> torch.Tensor:
        attr_vec = torch.cat((torch.tensor([1, 0, 0]).to(self.device), self.attribute_embedding[attribute_id]), dim=-1)
        annot_vec = torch.cat((torch.tensor([0, 1, 0]).to(self.device), self.annotator_embedding_learnable[annotator_id], self.annotator_embedding_random[annotator_id]), dim=-1)

        item_embedding_1 = torch.cat((torch.tensor([0, 0, 1]).to(self.device), self.item_embedding[item_ids[0]]), dim=-1)
        item_embedding_2 = torch.cat((torch.tensor([0, 0, 1]).to(self.device), self.item_embedding[item_ids[1]]), dim=-1)
        item_embedding = item_embedding_1 + item_embedding_2 @ self.pairwise_relation
        total_embedding = attr_vec + annot_vec  + item_embedding
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
    
    def on_forward_start(self, variables: List[RankingData]):
        """Generate new random embeddings for all components at the start of each forward pass."""
        if self.randomness:
            self.partial_reset_embedding()

        return

    def partial_reset_embedding(self):
        self.annotator_embedding_random = torch.rand(self.num_annotators, self.internal_dimension - self.internal_dimension // 2, device=self.device)
        self.item_embedding = torch.rand(self.num_items, self.internal_dimension, device=self.device)

