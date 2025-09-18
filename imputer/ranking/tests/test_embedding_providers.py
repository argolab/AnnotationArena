"""
Test script for embedding providers
"""

import torch
import sys
import os

# Add parent directory to path so we can import from imputer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imputer.embedding import FullyRandomizedEmbeddingProvider, CombineRandomTrainedEmbeddingProvider
from imputer.data import RankingData

def create_mock_ranking_data():
    """Create mock ranking data for testing."""
    return [
        RankingData(
            attribute_id=0,
            annotator_id=0,
            item_ids=[0, 1],  # Use 0-based indexing
            ranking_order=[1, 2],
            rating_value=None,
            is_listwise=True,
            is_masked=False
        ),
        RankingData(
            attribute_id=1,
            annotator_id=1,
            item_ids=[1, 2],  # Use 0-based indexing
            ranking_order=[2, 1],
            rating_value=None,
            is_listwise=True,
            is_masked=False
        ),
        RankingData(
            attribute_id=0,
            annotator_id=1,
            item_ids=[0],  # Use 0-based indexing
            ranking_order=None,
            rating_value=3,
            is_listwise=False,
            is_masked=False
        ),
        RankingData(
            attribute_id=1,
            annotator_id=0,
            item_ids=[2],  # Use 0-based indexing
            ranking_order=None,
            rating_value=4,
            is_listwise=False,
            is_masked=False
        )
    ]

def test_fully_randomized_embedding_provider():
    """Test FullyRandomizedEmbeddingProvider."""
    # Create provider
    provider = FullyRandomizedEmbeddingProvider(
        num_attributes=2,
        num_annotators=2,
        num_items=3,
        embedding_dim=32,
        num_likert_classes=5,
        max_rank_size=2,
        device="cpu"
    )
    
    # Create mock data
    variables = create_mock_ranking_data()
    
    # Test forward pass
    embeddings = provider(variables)
    
    # Check shape
    assert embeddings.shape == (1, len(variables), 32), f"Expected shape (1, {len(variables)}, 32), got {embeddings.shape}"
    
    # Check that embeddings are not trainable
    assert not provider.attribute_embedding.requires_grad, "Attribute embedding should not be trainable"
    assert not provider.annotator_embedding.requires_grad, "Annotator embedding should not be trainable"
    assert not provider.item_embedding.requires_grad, "Item embedding should not be trainable"
    
    # Test that embeddings change between forward passes (randomized)
    embeddings1 = provider(variables)
    embeddings2 = provider(variables)
    
    # They should be different due to randomization
    assert not torch.allclose(embeddings1, embeddings2), "Embeddings should be different between forward passes"

def test_combine_random_trained_embedding_provider():
    """Test CombineRandomTrainedEmbeddingProvider."""
    # Create provider
    provider = CombineRandomTrainedEmbeddingProvider(
        num_attributes=2,
        num_annotators=2,
        num_items=3,
        embedding_dim=32,
        num_likert_classes=5,
        max_rank_size=2,
        device="cpu"
    )
    
    # Create mock data
    variables = create_mock_ranking_data()
    
    # Test forward pass
    embeddings = provider(variables)
    
    # Check shape
    assert embeddings.shape == (1, len(variables), 32), f"Expected shape (1, {len(variables)}, 32), got {embeddings.shape}"
    
    # Check that embeddings are trainable
    assert provider.attribute_embedding.requires_grad, "Attribute embedding should be trainable"
    assert provider.annotator_embedding.requires_grad, "Annotator embedding should be trainable"
    assert provider.item_embedding.requires_grad, "Item embedding should be trainable"
    
    # Test that embeddings are consistent between forward passes (trained)
    embeddings1 = provider(variables)
    embeddings2 = provider(variables)
    
    # They should be different due to random components in CombineRandomTrainedEmbeddingProvider
    assert not torch.allclose(embeddings1, embeddings2), "Embeddings should be different due to random components"

def test_embedding_consistency():
    """Test that both providers produce consistent shapes."""
    # Create both providers
    randomized_provider = FullyRandomizedEmbeddingProvider(
        num_attributes=2, num_annotators=2, num_items=3,
        embedding_dim=32, num_likert_classes=5, max_rank_size=2,
        device="cpu"
    )
    
    trained_provider = CombineRandomTrainedEmbeddingProvider(
        num_attributes=2, num_annotators=2, num_items=3,
        embedding_dim=32, num_likert_classes=5, max_rank_size=2,
        device="cpu"
    )
    
    # Create mock data
    variables = create_mock_ranking_data()
    
    # Get embeddings from both providers
    randomized_embeddings = randomized_provider(variables)
    trained_embeddings = trained_provider(variables)
    
    # Check that shapes are consistent
    assert randomized_embeddings.shape == trained_embeddings.shape, "Embedding shapes should be consistent"

def test_max_rank_size_assertion():
    """Test that max_rank_size assertion works correctly."""
    provider = FullyRandomizedEmbeddingProvider(
        num_attributes=2, num_annotators=2, num_items=3,
        embedding_dim=32, num_likert_classes=5, max_rank_size=2,
        device="cpu"
    )
    
    # Create data with correct max_rank_size
    variables = create_mock_ranking_data()
    
    # This should work
    embeddings = provider(variables)
    assert embeddings.shape[0] == 1  # Should complete without error
    
    # Test with incorrect max_rank_size (this should fail)
    wrong_variables = [
        RankingData(
            attribute_id=0,
            annotator_id=0,
            item_ids=[1, 2, 3],  # 3 items, but max_rank_size=2
            ranking_order=[1, 2, 3],  # 3 elements, but max_rank_size=2
            rating_value=None,
            is_listwise=True,
            is_masked=False
        )
    ]
    
    # This should raise an AssertionError
    try:
        embeddings = provider(wrong_variables)
        assert False, "Should have failed with wrong max_rank_size"
    except AssertionError:
        pass  # Expected

def test_num_likert_classes_assertion():
    """Test that num_likert_classes assertion works correctly."""
    provider = FullyRandomizedEmbeddingProvider(
        num_attributes=2, num_annotators=2, num_items=3,
        embedding_dim=32, num_likert_classes=5, max_rank_size=2,
        device="cpu"
    )
    
    # Test with correct rating value
    variables = create_mock_ranking_data()
    embeddings = provider(variables)
    assert embeddings.shape[0] == 1  # Should complete without error
    
    # Test with incorrect rating value (this should fail)
    wrong_variables = [
        RankingData(
            attribute_id=0,
            annotator_id=0,
            item_ids=[1],
            ranking_order=None,
            rating_value=10,  # 10 > num_likert_classes=5
            is_listwise=False,
            is_masked=False
        )
    ]
    
    # This should raise an AssertionError
    try:
        embeddings = provider(wrong_variables)
        assert False, "Should have failed with wrong rating value"
    except AssertionError:
        pass  # Expected