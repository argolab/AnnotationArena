"""
Test script for embedding providers
"""

import torch
import sys
import os

# Add parent directory to path so we can import from imputer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imputer.embedding import AtomCompositonalEmbeddingProvider
from imputer.data import RankingData

def create_mock_ranking_data():
    """Create mock ranking data for testing."""
    return [
        RankingData(
            attribute_id=0,
            annotator_id=0,
            item_ids=[0, 1],
            ranking_order=[1, 2],
            rating_value=None,
            is_listwise=True,
            status=2,
            instance="train",
        ),
        RankingData(
            attribute_id=1,
            annotator_id=1,
            item_ids=[1, 2],
            ranking_order=[2, 1],
            rating_value=None,
            is_listwise=True,
            status=2,
            instance="train",
        ),
        RankingData(
            attribute_id=0,
            annotator_id=1,
            item_ids=[0],
            ranking_order=None,
            rating_value=3,
            is_listwise=False,
            status=2,
            instance="train",
        ),
        RankingData(
            attribute_id=1,
            annotator_id=0,
            item_ids=[2],
            ranking_order=None,
            rating_value=4,
            is_listwise=False,
            status=2,
            instance="train",
        ),
    ]

# NOTE: FullyRandomizedEmbeddingProvider has been removed as part of pointer mechanism renovation
# def test_fully_randomized_embedding_provider():
#     """Test FullyRandomizedEmbeddingProvider."""
#     # This test is disabled - FullyRandomizedEmbeddingProvider has been removed
#     pass

def test_atom_compositional_embedding_provider():
    """Test AtomCompositonalEmbeddingProvider."""
    provider = AtomCompositonalEmbeddingProvider(
        num_attributes=2,
        num_annotators=2,
        num_items=3,
        embedding_dim=32,
        num_likert_classes=5,
        max_rank_size=2,
        device="cpu",
        use_concat_embedding=False,
    )
    variables = create_mock_ranking_data()
    features, params = provider(variables)
    assert features.shape[0] == 1 and features.shape[1] == len(variables), (
        f"Expected (1, {len(variables)}, ...), got {features.shape}"
    )
    assert provider.attribute_embedding.requires_grad
    assert provider.annotator_embedding.requires_grad
    assert provider.item_embedding.requires_grad
    features2, params2 = provider(variables)
    assert torch.allclose(features, features2), "Atom embeddings should be deterministic"

# NOTE: FullyRandomizedEmbeddingProvider has been removed as part of pointer mechanism renovation
# def test_embedding_consistency():
#     """Test that both providers produce consistent shapes."""
#     # This test is disabled - FullyRandomizedEmbeddingProvider has been removed
#     pass

# NOTE: FullyRandomizedEmbeddingProvider has been removed as part of pointer mechanism renovation
# def test_max_rank_size_assertion():
#     """Test that max_rank_size assertion works correctly."""
#     # This test is disabled - FullyRandomizedEmbeddingProvider has been removed
#     pass

# NOTE: FullyRandomizedEmbeddingProvider has been removed as part of pointer mechanism renovation
# def test_num_likert_classes_assertion():
#     """Test that num_likert_classes assertion works correctly."""
#     # This test is disabled - FullyRandomizedEmbeddingProvider has been removed
#     pass