"""
Test script for DataConverter
"""

import torch
import sys
import os

# Add parent directory to path so we can import from imputer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imputer.data import DataConverter, RankingData

def create_mock_data():
    """Create mock data for testing."""
    return {
        'ratings': [
            {'annotator': 1, 'attribute': 1, 'item': 1, 'value': 2},
            {'annotator': 2, 'attribute': 1, 'item': 2, 'value': 3},
            {'annotator': 1, 'attribute': 2, 'item': 1, 'value': 1},
            {'annotator': 2, 'attribute': 2, 'item': 2, 'value': 4},
        ],
        'pairwise_rankings': [
            {'annotator': 1, 'attribute': 1, 'items': [1, 2], 'order': [1, 2]},
            {'annotator': 2, 'attribute': 2, 'items': [1, 2], 'order': [2, 1]},
        ]
    }

def test_data_converter_initialization():
    """Test DataConverter initialization."""
    converter = DataConverter(
        num_attributes=2,
        num_annotators=2,
        num_items=3,
        num_likert_classes=5,
        max_rank_size=2
    )
    
    assert converter.num_attributes == 2
    assert converter.num_annotators == 2
    assert converter.num_items == 3
    assert converter.num_likert_classes == 5
    assert converter.max_rank_size == 2

def test_create_variables_from_actual_data():
    """Test create_variables_from_actual_data method."""
    converter = DataConverter(
        num_attributes=2, num_annotators=2, num_items=3,
        num_likert_classes=5, max_rank_size=2
    )
    
    data = create_mock_data()
    rating_variables, ranking_variables = converter.create_variables_from_actual_data(data, data)
    
    # Check that we get the expected number of variables (doubled because we pass same data for train and test)
    assert len(rating_variables) == 8, f"Expected 8 rating variables (4 train + 4 test), got {len(rating_variables)}"
    assert len(ranking_variables) == 4, f"Expected 4 ranking variables (2 train + 2 test), got {len(ranking_variables)}"
    
    # Check that variables are dictionaries (not RankingData objects)
    for var in rating_variables + ranking_variables:
        assert isinstance(var, dict), "Variables should be dictionaries"

def test_process_training_data():
    """Test process_training_data method."""
    converter = DataConverter(
        num_attributes=2, num_annotators=2, num_items=3,
        num_likert_classes=5, max_rank_size=2
    )
    
    data = create_mock_data()
    rating_data, ranking_data = converter.process_training_data(data)
    
    # Check that we get the expected data structures
    assert isinstance(rating_data, dict), "Rating data should be a dict"
    assert isinstance(ranking_data, list), "Ranking data should be a list"

def test_create_batch():
    """Test create_batch method."""
    converter = DataConverter(
        num_attributes=2, num_annotators=2, num_items=3,
        num_likert_classes=5, max_rank_size=2
    )
    
    data = create_mock_data()
    rating_variables, ranking_variables = converter.create_variables_from_actual_data(data, data)
    rating_data, ranking_data = converter.process_training_data(data)
    
    # Test with different masking rates
    for masking_rate in [0.0, 0.5, 1.0]:
        batch = converter.create_batch(
            rating_variables=rating_variables,
            ranking_variables=ranking_variables,
            rating_data=rating_data,
            ranking_data=ranking_data,
            mode="train",
            masking_rate=masking_rate
        )
        
        # Check that batch has the expected structure
        assert 'all_variables' in batch, "Batch should have 'all_variables' key"
        assert 'variable_data' in batch, "Batch should have 'variable_data' key"
        assert 'rating_targets' in batch, "Batch should have 'rating_targets' key"
        assert 'ranking_targets' in batch, "Batch should have 'ranking_targets' key"
        assert 'rating_masked' in batch, "Batch should have 'rating_masked' key"
        assert 'ranking_masked' in batch, "Batch should have 'ranking_masked' key"

def test_masking_behavior():
    """Test that masking works correctly."""
    converter = DataConverter(
        num_attributes=2, num_annotators=2, num_items=3,
        num_likert_classes=5, max_rank_size=2
    )
    
    data = create_mock_data()
    rating_variables, ranking_variables = converter.create_variables_from_actual_data(data, data)
    rating_data, ranking_data = converter.process_training_data(data)
    
    # Test with 100% masking
    batch = converter.create_batch(
        rating_variables=rating_variables,
        ranking_variables=ranking_variables,
        rating_data=rating_data,
        ranking_data=ranking_data,
        mode="train",
        masking_rate=1.0
    )
    
    # Check that masking is applied (some variables should be masked)
    rating_masked = batch['rating_masked']
    ranking_masked = batch['ranking_masked']
    
    # Check that at least some variables are masked (not all variables may be available for masking)
    assert torch.any(rating_masked), "Some rating variables should be masked with 100% masking"
    assert torch.any(ranking_masked), "Some ranking variables should be masked with 100% masking"

def test_batch_consistency():
    """Test that batches are consistent across runs."""
    converter = DataConverter(
        num_attributes=2, num_annotators=2, num_items=3,
        num_likert_classes=5, max_rank_size=2
    )
    
    data = create_mock_data()
    rating_variables, ranking_variables = converter.create_variables_from_actual_data(data, data)
    rating_data, ranking_data = converter.process_training_data(data)
    
    # Create two batches with the same parameters
    batch1 = converter.create_batch(
        rating_variables=rating_variables,
        ranking_variables=ranking_variables,
        rating_data=rating_data,
        ranking_data=ranking_data,
        mode="train",
        masking_rate=0.5
    )
    
    batch2 = converter.create_batch(
        rating_variables=rating_variables,
        ranking_variables=ranking_variables,
        rating_data=rating_data,
        ranking_data=ranking_data,
        mode="train",
        masking_rate=0.5
    )
    
    # Check that basic structure is consistent
    assert len(batch1['all_variables']) == len(batch2['all_variables']), "Batch sizes should be consistent"
    assert batch1['variable_data'].shape == batch2['variable_data'].shape, "Variable data shapes should be consistent"