"""
Comprehensive test script for EvaluationEngine

Tests all major functionality including:
- Evaluation masking and Test_M/Test_O splitting
- Loss computation for ratings and rankings
- Accuracy computation with proper ranking logic
- Edge cases and error handling
- Integration with trainer callbacks
"""

import torch
import sys
import os
import numpy as np
import random

# Add parent directory to path so we can import from imputer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imputer.eval import EvaluationEngine, EvaluationResults
from imputer.data import DataConverter, RankingData
from imputer.ranking_imputer import MultiVariableImputer
from imputer.legacy.trainer import ImputerTrainer, EvaluationCallback


def create_test_data():
    """Create comprehensive test data with ratings and rankings."""
    return {
        'ratings': [
            {'annotator': 1, 'attribute': 1, 'item': 1, 'value': 2},
            {'annotator': 2, 'attribute': 1, 'item': 2, 'value': 3},
            {'annotator': 1, 'attribute': 2, 'item': 1, 'value': 1},
            {'annotator': 2, 'attribute': 2, 'item': 2, 'value': 4},
            {'annotator': 1, 'attribute': 1, 'item': 3, 'value': 5},
            {'annotator': 2, 'attribute': 2, 'item': 3, 'value': 2},
        ],
        'pairwise_rankings': [
            {'annotator': 1, 'attribute': 1, 'items': [1, 2], 'order': [1, 2]},
            {'annotator': 2, 'attribute': 2, 'items': [1, 2], 'order': [2, 1]},
            {'annotator': 1, 'attribute': 2, 'items': [1, 3], 'order': [1, 2]},
            {'annotator': 2, 'attribute': 1, 'items': [2, 3], 'order': [2, 1]},
        ]
    }


def test_evaluation_engine_initialization():
    """Test basic initialization of EvaluationEngine."""
    print("Testing EvaluationEngine initialization...")

    engine = EvaluationEngine()
    assert engine.config is None
    assert engine.loss_strategy is not None

    # Test with config
    engine_with_config = EvaluationEngine(config={'test': True})
    assert engine_with_config.config == {'test': True}

    print("✅ EvaluationEngine initialization test passed")


def test_evaluation_mask_creation():
    """Test evaluation mask creation with various masking rates."""
    print("Testing evaluation mask creation...")

    engine = EvaluationEngine()
    converter = DataConverter(
        num_attributes=2, num_annotators=2, num_items=3,
        num_likert_classes=5, max_rank_size=2
    )

    data = create_test_data()
    variables = converter.create_variables(data)

    # Test different masking rates
    for masking_rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mask = engine.create_evaluation_mask(variables, masking_rate)

        assert len(mask) == len(variables)
        expected_masked = int(len(variables) * masking_rate)
        actual_masked = sum(mask)

        # Allow for rounding differences
        assert abs(actual_masked - expected_masked) <= 1

        print(f"  Masking rate {masking_rate}: {actual_masked}/{len(variables)} variables masked")

    # Test edge cases
    empty_variables = []
    empty_mask = engine.create_evaluation_mask(empty_variables, 0.5)
    assert empty_mask == []

    # Test invalid masking rates (should be clamped)
    mask_negative = engine.create_evaluation_mask(variables, -0.1)
    assert not any(mask_negative)  # No variables should be masked

    mask_over_one = engine.create_evaluation_mask(variables, 1.5)
    assert all(mask_over_one)  # All variables should be masked

    print("✅ Evaluation mask creation test passed")


def test_variable_splitting():
    """Test splitting variables into Test_M and Test_O."""
    print("Testing variable splitting...")

    engine = EvaluationEngine()
    converter = DataConverter(
        num_attributes=2, num_annotators=2, num_items=3,
        num_likert_classes=5, max_rank_size=2
    )

    data = create_test_data()
    variables = converter.create_variables(data)

    # Create a specific mask for testing
    mask = [True, False, True, False, True, False, False, True, True, False]
    mask = mask[:len(variables)]  # Truncate to actual length

    test_m, test_o = engine.split_variables(variables, mask)

    # Check that variables are correctly split
    assert len(test_m) + len(test_o) == len(variables)
    assert len(test_m) == sum(mask)
    assert len(test_o) == len(variables) - sum(mask)

    # Verify that masked variables are in test_m and observed in test_o
    masked_count = 0
    observed_count = 0
    for i, (var, is_masked) in enumerate(zip(variables, mask)):
        if is_masked:
            assert test_m[masked_count] == var
            masked_count += 1
        else:
            assert test_o[observed_count] == var
            observed_count += 1

    print(f"  Split {len(variables)} variables into {len(test_m)} masked and {len(test_o)} observed")
    print("✅ Variable splitting test passed")


def test_rmse_computation():
    """Test RMSE computation for rating predictions."""
    print("Testing RMSE computation...")

    engine = EvaluationEngine()

    # Test perfect predictions
    perfect_preds = [0, 1, 2, 3, 4]  # 0-indexed
    perfect_targets = [0, 1, 2, 3, 4]  # 0-indexed
    rmse_perfect = engine.compute_rmse(perfect_preds, perfect_targets)
    assert rmse_perfect == 0.0

    # Test with some error
    preds = [0, 1, 2, 3, 4]  # 0-indexed (1-5 scale: [1, 2, 3, 4, 5])
    targets = [1, 2, 3, 4, 0]  # 0-indexed (1-5 scale: [2, 3, 4, 5, 1])
    # Differences on 1-5 scale: [1-2, 2-3, 3-4, 4-5, 5-1] = [-1, -1, -1, -1, 4]
    # Squared: [1, 1, 1, 1, 16] = 20/5 = 4, sqrt(4) = 2.0
    rmse_error = engine.compute_rmse(preds, targets)
    expected_rmse = 2.0
    assert abs(rmse_error - expected_rmse) < 1e-6

    # Test empty predictions
    rmse_empty = engine.compute_rmse([], [])
    assert rmse_empty == 0.0

    print(f"  Perfect predictions RMSE: {rmse_perfect}")
    print(f"  With errors RMSE: {rmse_error} (expected: {expected_rmse})")
    print("✅ RMSE computation test passed")


def test_ranking_accuracy():
    """Test ranking accuracy computation for pairwise rankings."""
    print("Testing ranking accuracy computation...")

    engine = EvaluationEngine()

    # Test perfect pairwise ranking predictions
    perfect_preds = [[1, 2], [2, 1], [1, 2]]
    perfect_targets = [[1, 2], [2, 1], [1, 2]]

    result = engine._compute_subset_metrics(perfect_preds, perfect_targets, None, 'ranking')
    assert result['accuracy'] == 1.0
    assert result['count'] == 3

    # Test mixed accuracy
    mixed_preds = [[1, 2], [1, 2], [1, 2]]  # All predict first item wins
    mixed_targets = [[1, 2], [2, 1], [1, 2]]  # First and third are correct

    result = engine._compute_subset_metrics(mixed_preds, mixed_targets, None, 'ranking')
    expected_accuracy = 2.0 / 3.0  # 2 out of 3 correct
    assert abs(result['accuracy'] - expected_accuracy) < 1e-6
    assert result['count'] == 3

    # Test empty rankings
    result_empty = engine._compute_subset_metrics([], [], None, 'ranking')
    assert result_empty['accuracy'] is None
    assert result_empty['count'] == 0

    print(f"  Perfect ranking accuracy: {1.0}")
    print(f"  Mixed ranking accuracy: {expected_accuracy:.3f}")
    print("✅ Ranking accuracy computation test passed")


def test_evaluation_batch_creation():
    """Test creation of evaluation batches with masking."""
    print("Testing evaluation batch creation...")

    engine = EvaluationEngine()
    converter = DataConverter(
        num_attributes=2, num_annotators=2, num_items=3,
        num_likert_classes=5, max_rank_size=2
    )

    data = create_test_data()
    variables = converter.create_variables(data)

    # Create evaluation mask
    evaluation_mask = engine.create_evaluation_mask(variables, 0.5)

    # Create evaluation batch
    ranking_data_list = engine._create_evaluation_batch(variables, evaluation_mask)

    assert len(ranking_data_list) == len(variables)

    # Check that masked variables have is_masked=True
    for i, (original_var, masked_var, is_masked) in enumerate(zip(variables, ranking_data_list, evaluation_mask)):
        if is_masked:
            # Masked variables should have is_masked=True
            assert masked_var.is_masked is True
            # But keep original values for reference
            assert masked_var.rating_value == original_var.rating_value
            assert masked_var.ranking_order == original_var.ranking_order
            # And other attributes
            assert masked_var.annotator_id == original_var.annotator_id
            assert masked_var.attribute_id == original_var.attribute_id
            assert masked_var.item_ids == original_var.item_ids
        else:
            # Observed variables should have is_masked=False
            assert masked_var.is_masked is False
            assert masked_var.rating_value == original_var.rating_value
            assert masked_var.ranking_order == original_var.ranking_order

    print(f"  Created evaluation batch with {sum(evaluation_mask)} masked variables")
    print("✅ Evaluation batch creation test passed")


def test_comprehensive_evaluation():
    """Test comprehensive evaluation with model integration."""
    print("Testing comprehensive evaluation with model...")

    converter = DataConverter(
        num_attributes=2, num_annotators=2, num_items=3,
        num_likert_classes=5, max_rank_size=2
    )

    # Create model
    model = MultiVariableImputer(
        num_attributes=2,
        num_annotators=2,
        num_items=3,
        num_likert_classes=5,
        max_rank_size=2,
        encoder_layers_num=1,  # Smaller for testing
        attention_heads=2,
        embedding_dim=32,
        dropout=0.1,
        device="cpu"
    )

    # Create test data
    data = create_test_data()
    variables = converter.create_variables(data)

    # Create evaluation engine
    engine = EvaluationEngine()

    # Test evaluation with different masking rates
    for masking_rate in [0.0, 0.5, 1.0]:
        results = engine.evaluate_model(
            model=model,
            variables=variables,
            masking_rate=masking_rate,
            converter=converter,
            device='cpu'
        )

        # Check that results object is properly formed
        assert isinstance(results, EvaluationResults)
        assert isinstance(results.total_loss, float)
        assert isinstance(results.rating_loss, float)
        assert isinstance(results.ranking_loss, float)

        # Check that we have some evaluation counts
        assert results.num_rating_evaluations >= 0
        assert results.num_ranking_evaluations >= 0

        # Check masked/observed metrics exist
        assert results.masked_metrics is not None
        assert results.observed_metrics is not None

        print(f"  Masking rate {masking_rate}: Total loss = {results.total_loss:.4f}")
        print(f"    Rating evaluations: {results.num_rating_evaluations}, Ranking evaluations: {results.num_ranking_evaluations}")
        print(f"    Rating accuracy: {results.rating_accuracy}, Ranking accuracy: {results.ranking_accuracy}")

    print("✅ Comprehensive evaluation test passed")


def test_trainer_callback_integration():
    """Test integration with trainer callback system."""
    print("Testing trainer callback integration...")

    converter = DataConverter(
        num_attributes=2, num_annotators=2, num_items=3,
        num_likert_classes=5, max_rank_size=2
    )

    # Create model and trainer
    model = MultiVariableImputer(
        num_attributes=2,
        num_annotators=2,
        num_items=3,
        num_likert_classes=5,
        max_rank_size=2,
        encoder_layers_num=1,
        attention_heads=2,
        embedding_dim=32,
        dropout=0.1,
        device="cpu"
    )

    trainer = ImputerTrainer(model, learning_rate=1e-3, device='cpu')

    # Create evaluation callback
    data = create_test_data()
    test_variables = converter.create_variables(data)

    eval_engine = EvaluationEngine()
    callback = EvaluationCallback(
        eval_engine=eval_engine,
        test_variables=test_variables,
        test_data=data,
        converter=converter,
        masking_rate=0.5,
        device='cpu'
    )

    # Register callback
    trainer.register_callback(callback)

    # Test callback execution
    callback_result = callback.on_epoch_end(model, epoch=0)

    # Check callback result format
    assert isinstance(callback_result, dict)
    assert 'epoch' in callback_result
    assert 'total_loss' in callback_result
    assert 'rating_accuracy' in callback_result
    assert 'ranking_accuracy' in callback_result

    print(f"  Callback result keys: {list(callback_result.keys())}")
    print(f"  Epoch: {callback_result['epoch']}, Total loss: {callback_result['total_loss']:.4f}")

    # Test training with callback
    training_data = converter.create_training_batch(test_variables, 10)

    # Short training run to test callback integration
    history = trainer.train([training_data], epochs=2, call_callbacks_every=1, verbose=False)

    assert 'training_history' in history
    assert 'callback_history' in history
    assert len(history['callback_history']) > 0

    print(f"  Training completed with {len(history['callback_history'])} callback results")
    print("✅ Trainer callback integration test passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases and error handling...")

    engine = EvaluationEngine()

    # Test with empty data
    try:
        empty_variables = []
        mask = engine.create_evaluation_mask(empty_variables, 0.5)
        assert mask == []
        print("  ✓ Empty variables handled correctly")
    except Exception as e:
        print(f"  ✗ Empty variables failed: {e}")
        raise

    # Test with extreme masking rates
    converter = DataConverter(
        num_attributes=1, num_annotators=1, num_items=2,
        num_likert_classes=5, max_rank_size=2
    )

    minimal_data = {
        'ratings': [{'annotator': 1, 'attribute': 1, 'item': 1, 'value': 3}],
        'pairwise_rankings': []
    }

    variables = converter.create_variables(minimal_data)

    # Test 0% masking
    mask_zero = engine.create_evaluation_mask(variables, 0.0)
    assert not any(mask_zero)

    # Test 100% masking
    mask_full = engine.create_evaluation_mask(variables, 1.0)
    assert all(mask_full)

    print("  ✓ Extreme masking rates handled correctly")

    # Test variable splitting with empty lists
    empty_mask = []
    test_m, test_o = engine.split_variables([], empty_mask)
    assert test_m == []
    assert test_o == []

    print("  ✓ Edge cases handled correctly")
    print("✅ Edge cases test passed")


def run_all_tests():
    """Run all evaluation engine tests."""
    print("="*60)
    print("RUNNING COMPREHENSIVE EVALUATION ENGINE TESTS")
    print("="*60)

    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    try:
        test_evaluation_engine_initialization()
        print()

        test_evaluation_mask_creation()
        print()

        test_variable_splitting()
        print()

        test_rmse_computation()
        print()

        test_ranking_accuracy()
        print()

        test_evaluation_batch_creation()
        print()

        test_comprehensive_evaluation()
        print()

        test_trainer_callback_integration()
        print()

        test_edge_cases()
        print()

        print("="*60)
        print("🎉 ALL EVALUATION ENGINE TESTS PASSED!")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)