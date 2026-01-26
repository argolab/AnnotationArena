"""
Test script for corrected understanding of masking
"""

import torch
import random
import sys
import os

# Add parent directory to path so we can import from imputer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imputer.legacy.multi_instance_trainer import MultiInstanceTrainerBase, SequentialMIT, MixedMIT
from imputer.data import DataConverter
from imputer.eval import EvaluationEngine
from imputer.ranking_imputer import MultiVariableImputer

def create_mock_config():
    """Create a mock config for testing."""
    class MockConfig:
        def __init__(self):
            self.total_batches = 10
            self.batch_size = 4
            self.masking_rates = [0.0, 0.2, 0.5, 0.8, 1.0]
            self.eval_frequency = 5
            self.learning_rate = 0.001
            self.device = "cpu"
            self.test_masking_rate = 0.3
    
    return MockConfig()

def create_mock_training_data():
    """Create mock training data for testing."""
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

def create_mock_test_data():
    """Create mock test data for testing."""
    return {
        'ratings': [
            {'annotator': 1, 'attribute': 1, 'item': 3, 'value': 3},
            {'annotator': 2, 'attribute': 2, 'item': 3, 'value': 2},
        ],
        'pairwise_rankings': [
            {'annotator': 1, 'attribute': 2, 'items': [2, 3], 'order': [1, 2]},
            {'annotator': 2, 'attribute': 1, 'items': [1, 3], 'order': [2, 1]},
        ]
    }

def test_training_batch_creation():
    """Test that training batches are created correctly."""
    config = create_mock_config()
    converter = DataConverter(num_attributes=2, num_annotators=2, num_items=3, num_likert_classes=5, max_rank_size=2)
    
    # Create mock model and eval engine
    model = MultiVariableImputer(
        num_attributes=2, num_annotators=2, num_items=3, 
        embedding_dim=32, num_likert_classes=5, max_rank_size=2, device="cpu"
    )
    eval_engine = EvaluationEngine(config)
    
    # Create MultiInstanceTrainerBase
    trainer = MultiInstanceTrainerBase(model, eval_engine, config, converter)
    
    # Create mock training data
    training_data = create_mock_training_data()
    
    # Test training batch creation
    batch = trainer.create_masked_batch(training_data, masking_rate=0.5, batch_size=4)
    
    # Check that batch has the expected structure
    assert 'all_variables' in batch, "Training batch should have 'all_variables' key"
    assert 'variable_data' in batch, "Training batch should have 'variable_data' key"
    assert 'rating_targets' in batch, "Training batch should have 'rating_targets' key"
    assert 'ranking_targets' in batch, "Training batch should have 'ranking_targets' key"

def test_evaluation_batch_creation():
    """Test that evaluation batches are created correctly."""
    config = create_mock_config()
    converter = DataConverter(num_attributes=2, num_annotators=2, num_items=3, num_likert_classes=5, max_rank_size=2)
    
    # Create mock model and eval engine
    model = MultiVariableImputer(
        num_attributes=2, num_annotators=2, num_items=3, 
        embedding_dim=32, num_likert_classes=5, max_rank_size=2, device="cpu"
    )
    eval_engine = EvaluationEngine(config)
    
    # Create MultiInstanceTrainerBase
    trainer = MultiInstanceTrainerBase(model, eval_engine, config, converter)
    
    # Create mock test data
    test_data = create_mock_test_data()
    
    # Test evaluation batch creation
    batch = trainer.create_evaluation_batch(test_data, masking_rate=0.3)
    
    # Check that batch has the expected structure
    assert 'all_variables' in batch, "Evaluation batch should have 'all_variables' key"
    assert 'variable_data' in batch, "Evaluation batch should have 'variable_data' key"

def test_sequential_mit_with_corrected_understanding():
    """Test SequentialMIT with corrected understanding."""
    config = create_mock_config()
    converter = DataConverter(num_attributes=2, num_annotators=2, num_items=3, num_likert_classes=5, max_rank_size=2)
    
    # Create mock model and eval engine
    model = MultiVariableImputer(
        num_attributes=2, num_annotators=2, num_items=3, 
        embedding_dim=32, num_likert_classes=5, max_rank_size=2, device="cpu"
    )
    eval_engine = EvaluationEngine(config)
    
    # Create SequentialMIT
    sequential_trainer = SequentialMIT(model, eval_engine, config, converter)
    
    # Create mock training instances (each contains training data)
    train_instances = [create_mock_training_data() for _ in range(2)]
    
    # Test generator
    generator = sequential_trainer.create_training_generator(train_instances, config.total_batches, config.batch_size)
    
    batches = list(generator)
    
    # Check that we get batches
    assert len(batches) > 0, "Should generate at least some batches"
    
    # Check that batches have the expected structure
    for i, batch in enumerate(batches):
        assert isinstance(batch, dict), f"Batch {i} should be a dict"
        assert 'all_variables' in batch, f"Batch {i} should have 'all_variables' key"

def test_mixed_mit():
    """Test MixedMIT with corrected understanding."""
    config = create_mock_config()
    converter = DataConverter(num_attributes=2, num_annotators=2, num_items=3, num_likert_classes=5, max_rank_size=2)
    
    # Create mock model and eval engine
    model = MultiVariableImputer(
        num_attributes=2, num_annotators=2, num_items=3, 
        embedding_dim=32, num_likert_classes=5, max_rank_size=2, device="cpu"
    )
    eval_engine = EvaluationEngine(config)
    
    # Create MixedMIT
    mixed_trainer = MixedMIT(model, eval_engine, config, converter)
    
    # Create mock training instances (each contains training data)
    train_instances = [create_mock_training_data() for _ in range(3)]
    
    # Test generator
    generator = mixed_trainer.create_training_generator(train_instances, config.total_batches, config.batch_size)
    
    batches = list(generator)
    
    # Check that we get the exact number of batches
    assert len(batches) == config.total_batches, f"Expected {config.total_batches} batches, got {len(batches)}"
    
    # Check that batches have the expected structure
    for i, batch in enumerate(batches):
        assert isinstance(batch, dict), f"Batch {i} should be a dict"
        assert 'all_variables' in batch, f"Batch {i} should have 'all_variables' key"

def test_data_flow_understanding():
    """Test that we understand the data flow correctly."""
    # Training data flow
    training_data = create_mock_training_data()
    
    # Test data flow
    test_data = create_mock_test_data()
    
    # The key insight: training and test data are separate
    assert training_data != test_data, "Training and test data should be different"

def test_masking_rates():
    """Test that different masking rates work correctly."""
    config = create_mock_config()
    converter = DataConverter(num_attributes=2, num_annotators=2, num_items=3, num_likert_classes=5, max_rank_size=2)
    
    # Create mock model and eval engine
    model = MultiVariableImputer(
        num_attributes=2, num_annotators=2, num_items=3, 
        embedding_dim=32, num_likert_classes=5, max_rank_size=2, device="cpu"
    )
    eval_engine = EvaluationEngine(config)
    
    # Create MultiInstanceTrainerBase
    trainer = MultiInstanceTrainerBase(model, eval_engine, config, converter)
    
    # Create mock training data
    training_data = create_mock_training_data()
    
    # Test different masking rates
    for masking_rate in [0.0, 0.5, 1.0]:
        batch = trainer.create_masked_batch(training_data, masking_rate=masking_rate, batch_size=4)
        
        # Check that batch is created successfully
        assert 'all_variables' in batch, f"Batch with masking_rate {masking_rate} should have 'all_variables' key"