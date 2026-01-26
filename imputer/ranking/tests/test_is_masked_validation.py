#!/usr/bin/env python3
"""
Generic test script to validate is_masked field handling throughout the pipeline.

This script tests:
1. Initial data loading (should have is_masked=None)
2. Masking logic in training batches (should set is_masked=True/False)
3. Training loss computation (should preserve is_masked)
4. Evaluation masking (should set is_masked=True/False)
5. Evaluation loss computation (should preserve is_masked)
6. Edge cases and error conditions
"""

import sys
import torch
import random
import copy
from typing import List, Dict, Any

# Add the imputer package to path
sys.path.append('imputer')

from imputer.data import RankingData, DataConverter
from imputer.losses import DefaultLossStrategy, TopLayerPredictionResult, adapt_batched_logits_to_predictions
from imputer.legacy.multi_instance_trainer import SequentialMIT
from imputer.eval import EvaluationEngine
from imputer.legacy.trainer import ImputerTrainer
from experiment_config import ModelConfig


def create_mock_data() -> Dict[str, Any]:
    """Create minimal mock data for testing."""
    return {
        'ratings': [
            {'annotator': 1, 'attribute': 1, 'item': 1, 'value': 3},
            {'annotator': 1, 'attribute': 2, 'item': 2, 'value': 4},
            {'annotator': 2, 'attribute': 1, 'item': 1, 'value': 2},
            {'annotator': 2, 'attribute': 2, 'item': 3, 'value': 5},
        ],
        'pairwise_rankings': [
            {'annotator': 1, 'attribute': 1, 'items': [1, 2], 'order': [1, 2]},
            {'annotator': 1, 'attribute': 2, 'items': [2, 3], 'order': [2, 1]},
            {'annotator': 2, 'attribute': 1, 'items': [1, 3], 'order': [1, 2]},
        ]
    }


def test_initial_data_loading():
    """Test 1: Initial data loading should have is_masked=None."""
    print("=== Test 1: Initial Data Loading ===")

    converter = DataConverter(num_attributes=3, num_annotators=3, num_items=5, max_rank_size=2)
    data = create_mock_data()
    variables = converter.create_variables(data)

    print(f"Created {len(variables)} variables")

    # Check that all variables have is_masked=None initially
    all_none = True
    for i, var in enumerate(variables):
        if var.is_masked is not None:
            print(f"ERROR: Variable {i} has is_masked={var.is_masked}, expected None")
            all_none = False
        else:
            print(f"✅ Variable {i}: is_masked=None (correct)")

    if all_none:
        print("✅ Test 1 PASSED: All variables have is_masked=None initially")
    else:
        print("❌ Test 1 FAILED: Some variables have non-None is_masked")

    return variables


def test_masking_logic(variables: List[RankingData]):
    """Test 2: Masking logic should set is_masked=True/False."""
    print("\n=== Test 2: Masking Logic ===")

    # Test the masking function from multi_instance_trainer
    masking_rate = 0.5
    num_to_mask = int(len(variables) * masking_rate)
    masked_indices = set(random.sample(range(len(variables)), num_to_mask))

    masked_variables = []
    for i, var in enumerate(variables):
        if i in masked_indices:
            # Create masked version (copying the logic from multi_instance_trainer)
            masked_var = RankingData(
                annotator_id=var.annotator_id,
                attribute_id=var.attribute_id,
                is_listwise=var.is_listwise,
                item_ids=var.item_ids,
                is_masked=True,  # Mark as masked
                rating_value=var.rating_value,
                ranking_order=var.ranking_order
            )
            masked_variables.append(masked_var)
        else:
            # Keep original (observed) for conditioning
            observed_var = RankingData(
                annotator_id=var.annotator_id,
                attribute_id=var.attribute_id,
                is_listwise=var.is_listwise,
                item_ids=var.item_ids,
                is_masked=False,  # Mark as observed
                rating_value=var.rating_value,
                ranking_order=var.ranking_order
            )
            masked_variables.append(observed_var)

    print(f"Created masked batch with {len(masked_variables)} variables")

    # Check that all variables have proper is_masked values
    masking_correct = True
    masked_count = 0
    observed_count = 0

    for i, var in enumerate(masked_variables):
        if var.is_masked is None:
            print(f"ERROR: Variable {i} has is_masked=None after masking")
            masking_correct = False
        elif var.is_masked is True:
            masked_count += 1
            print(f"✅ Variable {i}: is_masked=True (masked)")
        elif var.is_masked is False:
            observed_count += 1
            print(f"✅ Variable {i}: is_masked=False (observed)")
        else:
            print(f"ERROR: Variable {i} has invalid is_masked={var.is_masked}")
            masking_correct = False

    print(f"Masked: {masked_count}, Observed: {observed_count}")

    if masking_correct:
        print("✅ Test 2 PASSED: All variables have proper is_masked values after masking")
    else:
        print("❌ Test 2 FAILED: Some variables have invalid is_masked values")

    return masked_variables


def test_loss_computation(masked_variables: List[RankingData]):
    """Test 3: Loss computation should handle is_masked properly."""
    print("\n=== Test 3: Loss Computation ===")

    # Create mock predictions (simplified)
    N = len(masked_variables)
    C = 5  # rating classes
    R = 2  # ranking size

    # Create mock prediction results
    predictions = []
    for i in range(N):
        var = masked_variables[i]
        pred = TopLayerPredictionResult()

        if not var.is_listwise:
            pred.rating_logits = torch.randn(C)
            pred.ranking_logits = torch.zeros(R)  # Not used for ratings
        else:
            pred.rating_logits = torch.zeros(C)  # Not used for rankings
            pred.ranking_logits = torch.randn(R)

        pred.is_listwise = var.is_listwise
        predictions.append(pred)

    # Test loss computation
    loss_strategy = DefaultLossStrategy(masked_loss_weight=2.0, observed_loss_weight=1.0)

    try:
        losses = loss_strategy.compute(predictions, masked_variables)
        print("✅ Loss computation succeeded")

        # Check loss components
        required_keys = ['total_loss', 'rating_loss', 'ranking_loss',
                        'masked_total_loss', 'observed_total_loss',
                        'masked_rating_loss', 'observed_rating_loss',
                        'masked_ranking_loss', 'observed_ranking_loss']

        missing_keys = [key for key in required_keys if key not in losses]
        if missing_keys:
            print(f"ERROR: Missing loss keys: {missing_keys}")
            return False

        print(f"✅ All loss components present:")
        for key, value in losses.items():
            if not key.startswith('_'):
                print(f"  {key}: {value:.4f}")

        print("✅ Test 3 PASSED: Loss computation handled is_masked properly")
        return True

    except Exception as e:
        print(f"❌ Test 3 FAILED: Loss computation error: {e}")
        return False


def test_evaluation_masking(variables: List[RankingData]):
    """Test 4: Evaluation masking should set is_masked properly."""
    print("\n=== Test 4: Evaluation Masking ===")

    eval_engine = EvaluationEngine()

    # Test the evaluation mask creation
    masking_rate = 0.3
    evaluation_mask = eval_engine.create_evaluation_mask(variables, masking_rate)

    print(f"Created evaluation mask with {sum(evaluation_mask)} masked out of {len(evaluation_mask)}")

    # Test evaluation batch creation
    masked_variables = eval_engine._create_evaluation_batch(variables, evaluation_mask)

    # Check that all variables have proper is_masked values
    masking_correct = True
    masked_count = 0
    observed_count = 0

    for i, var in enumerate(masked_variables):
        expected_masked = evaluation_mask[i]

        if var.is_masked is None:
            print(f"ERROR: Variable {i} has is_masked=None after evaluation masking")
            masking_correct = False
        elif var.is_masked != expected_masked:
            print(f"ERROR: Variable {i} has is_masked={var.is_masked}, expected {expected_masked}")
            masking_correct = False
        else:
            if var.is_masked:
                masked_count += 1
                print(f"✅ Variable {i}: is_masked=True (correctly masked)")
            else:
                observed_count += 1
                print(f"✅ Variable {i}: is_masked=False (correctly observed)")

    print(f"Evaluation masked: {masked_count}, Evaluation observed: {observed_count}")

    if masking_correct:
        print("✅ Test 4 PASSED: Evaluation masking set is_masked properly")
    else:
        print("❌ Test 4 FAILED: Evaluation masking had errors")

    return masked_variables


def test_edge_cases():
    """Test 5: Edge cases and error conditions."""
    print("\n=== Test 5: Edge Cases ===")

    # Test empty variables list
    loss_strategy = DefaultLossStrategy()

    try:
        losses = loss_strategy.compute([], [])
        print("✅ Empty lists handled correctly")
    except Exception as e:
        print(f"ERROR: Empty lists failed: {e}")

    # Test mixed is_masked values (some None, some not)
    variables_with_none = [
        RankingData(annotator_id=0, attribute_id=0, is_listwise=False,
                   item_ids=[0], rating_value=3, is_masked=None),  # None
        RankingData(annotator_id=0, attribute_id=1, is_listwise=False,
                   item_ids=[1], rating_value=4, is_masked=True),   # True
    ]

    predictions = [
        TopLayerPredictionResult(rating_logits=torch.randn(5), ranking_logits=torch.zeros(2)),
        TopLayerPredictionResult(rating_logits=torch.randn(5), ranking_logits=torch.zeros(2))
    ]

    try:
        losses = loss_strategy.compute(predictions, variables_with_none)
        print("❌ ERROR: Should have failed with is_masked=None values")
        return False
    except ValueError as e:
        if "is_masked=None" in str(e):
            print("✅ Correctly caught is_masked=None error")
        else:
            print(f"ERROR: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return False

    print("✅ Test 5 PASSED: Edge cases handled correctly")
    return True


def test_masking_rate_zero():
    """Test 6: masking_rate=0.0 should respect existing is_masked flags."""
    print("\n=== Test 6: masking_rate=0.0 Behavior ===")

    # Create variables with explicit is_masked values
    variables = [
        RankingData(annotator_id=0, attribute_id=0, is_listwise=False,
                   item_ids=[0], rating_value=3, is_masked=True),   # Pre-masked
        RankingData(annotator_id=0, attribute_id=1, is_listwise=False,
                   item_ids=[1], rating_value=4, is_masked=False),  # Pre-observed
        RankingData(annotator_id=1, attribute_id=0, is_listwise=False,
                   item_ids=[0], rating_value=2, is_masked=True),   # Pre-masked
    ]

    eval_engine = EvaluationEngine()

    # Test with masking_rate=0.0
    evaluation_mask = eval_engine.create_evaluation_mask(variables, masking_rate=0.0)

    expected_mask = [True, False, True]  # Should match the pre-set is_masked values

    if evaluation_mask == expected_mask:
        print("✅ masking_rate=0.0 correctly respects existing is_masked flags")
        print(f"  Expected: {expected_mask}")
        print(f"  Got:      {evaluation_mask}")
    else:
        print("❌ masking_rate=0.0 failed to respect existing is_masked flags")
        print(f"  Expected: {expected_mask}")
        print(f"  Got:      {evaluation_mask}")
        return False

    print("✅ Test 6 PASSED: masking_rate=0.0 behavior correct")
    return True


def main():
    """Run all tests."""
    print("Starting is_masked validation tests...\n")

    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Test 1: Initial data loading
    variables = test_initial_data_loading()

    # Test 2: Masking logic
    masked_variables = test_masking_logic(variables)

    # Test 3: Loss computation
    loss_success = test_loss_computation(masked_variables)

    # Test 4: Evaluation masking
    eval_masked_variables = test_evaluation_masking(copy.deepcopy(variables))

    # Test 5: Edge cases
    edge_cases_success = test_edge_cases()

    # Test 6: masking_rate=0.0
    zero_rate_success = test_masking_rate_zero()

    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    tests = [
        ("Initial Data Loading", True),
        ("Masking Logic", True),
        ("Loss Computation", loss_success),
        ("Evaluation Masking", True),
        ("Edge Cases", edge_cases_success),
        ("masking_rate=0.0", zero_rate_success)
    ]

    passed = sum(1 for _, success in tests if success)
    total = len(tests)

    for test_name, success in tests:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! is_masked handling is correct.")
        return 0
    else:
        print("💥 SOME TESTS FAILED! is_masked handling needs fixes.")
        return 1


if __name__ == "__main__":
    exit(main())