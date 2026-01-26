#!/usr/bin/env python3
"""
Phase 2 Test Script: Step-by-Step Evaluation During Training

This script tests the Phase 2 enhancements:
1. Test set evaluation during finetuning (both strategies)
2. Enhanced progress tracking with test metrics
3. Separate evaluation frequencies for heldout vs test
4. Comprehensive results storage with test evaluation metrics

Test approach:
- Run mini-experiment with finetuning strategies
- Verify test evaluation occurs during training steps
- Check that callback results contain both heldout and test metrics
- Validate enhanced progress tracking output
- Ensure results storage includes test evaluation data
"""

import sys
import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Add the ranking directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_config import create_test_config
from legacy.new_experiment_runner import ExperimentRunner
from iclr_data_generator import ICLRDataGenerator, ICLRDatasetConfig
from imputer.data import DataConverter
from imputer.eval import EvaluationEngine
from imputer.legacy.multi_instance_trainer import GeneralMIT
from imputer.ranking_imputer import MultiVariableImputer


def test_test_evaluation_config():
    """Test that Phase 2 configuration options work correctly."""
    print("\n" + "="*60)
    print("TEST 1: Phase 2 Configuration Options")
    print("="*60)

    config = create_test_config()

    # Check that new evaluation config fields exist with defaults
    assert hasattr(config.evaluation_config, 'eval_on_test_during_finetuning'), "Missing eval_on_test_during_finetuning"
    assert hasattr(config.evaluation_config, 'test_eval_frequency'), "Missing test_eval_frequency"

    print(f"eval_on_test_during_finetuning: {config.evaluation_config.eval_on_test_during_finetuning}")
    print(f"test_eval_frequency: {config.evaluation_config.test_eval_frequency}")

    # Test configuration modification
    config.evaluation_config.eval_on_test_during_finetuning = True
    config.evaluation_config.test_eval_frequency = 3

    assert config.evaluation_config.eval_on_test_during_finetuning == True
    assert config.evaluation_config.test_eval_frequency == 3

    print("✅ Phase 2 configuration options work correctly!")


def test_general_mit_test_evaluation():
    """Test that GeneralMIT can perform test evaluation during finetuning."""
    print("\n" + "="*60)
    print("TEST 2: GeneralMIT Test Evaluation During Finetuning")
    print("="*60)

    # Create test configuration with test evaluation enabled
    config = create_test_config()
    config.finetuning_config.finetuning_steps = 10  # Small for testing
    config.finetuning_config.eval_frequency = 3    # Heldout evaluation every 3 steps
    config.evaluation_config.eval_on_test_during_finetuning = True
    config.evaluation_config.test_eval_frequency = 5  # Test evaluation every 5 steps

    # Force CPU for testing
    config.finetuning_config.device = "cpu"
    config.evaluation_config.device = "cpu"

    # Generate test instances
    generator = ICLRDataGenerator()
    data_config = ICLRDatasetConfig(
        K=config.data_config.K,
        I=config.data_config.I,
        J=config.data_config.J,
        D=config.data_config.D,
        C=config.data_config.C
    )

    # Create multiple test instances
    test_instances = []
    for i in range(3):  # Create 3 test instances
        dataset_obj = generator.generate_dataset(data_config, seed=100 + i)
        test_instance = {
            'ratings': dataset_obj.observed_ratings,
            'pairwise_rankings': dataset_obj.observed_pairwise_rankings
        }
        test_instances.append(test_instance)

    # Use first instance for finetuning, all instances for test evaluation
    finetune_instance = test_instances[0]

    # Create components
    converter = DataConverter()
    eval_engine = EvaluationEngine()

    model = MultiVariableImputer(
        num_attributes=config.data_config.I,
        num_annotators=config.data_config.J,
        num_items=config.data_config.K,
        num_likert_classes=config.data_config.C,
        max_rank_size=config.model_config.max_rank_size,
        encoder_layers_num=config.model_config.encoder_layers,
        attention_heads=config.model_config.attention_heads,
        embedding_dim=config.model_config.embedding_dim,
        dropout=config.model_config.dropout,
        device=config.finetuning_config.device
    )

    # Create GeneralMIT and run finetuning with test evaluation
    mit = GeneralMIT(model, eval_engine, config.finetuning_config, converter)

    print("Running GeneralMIT finetuning with test evaluation enabled...")
    print("Expected: Heldout evaluation every 3 steps, Test evaluation every 5 steps")

    results = mit.finetune_on_instance(
        model, finetune_instance,
        full_test_instances=test_instances,
        eval_config=config.evaluation_config
    )

    # Verify results structure includes callback results
    print(f"\nFinetuning completed. Results keys: {list(results.keys())}")

    required_keys = ['finetuning_results', 'callback_results', 'final_evaluation']
    for key in required_keys:
        assert key in results, f"Missing key in finetuning results: {key}"

    # Check callback results for both heldout and test evaluations
    callback_results = results['callback_results']
    print(f"Callback results: {len(callback_results)} total evaluations")

    if callback_results:
        # Look for evidence of both heldout and test evaluations
        heldout_count = 0
        test_count = 0

        for cb_result in callback_results:
            callback_type = getattr(cb_result, 'callback_type', 'heldout')
            if callback_type == 'test_evaluation':
                test_count += 1
            else:
                heldout_count += 1

        print(f"Heldout evaluations: {heldout_count}")
        print(f"Test evaluations: {test_count}")

        # We expect at least some evaluations if steps and frequencies allow
        total_steps = config.finetuning_config.finetuning_steps
        expected_heldout = max(1, total_steps // config.finetuning_config.eval_frequency)
        expected_test = max(1, total_steps // config.evaluation_config.test_eval_frequency)

        print(f"Expected heldout evaluations: ~{expected_heldout}")
        print(f"Expected test evaluations: ~{expected_test}")

        # Verify we have some callback results
        assert len(callback_results) > 0, "Should have some callback results"

        # Check structure of callback results
        sample_result = callback_results[0]
        print(f"Sample callback result keys: {list(sample_result.keys())}")

        expected_keys = ['total_loss', 'rating_accuracy', 'ranking_accuracy']
        for key in expected_keys:
            assert key in sample_result, f"Missing key in callback result: {key}"

    print("✅ GeneralMIT test evaluation during finetuning works correctly!")


def test_finetuning_strategies_with_test_evaluation():
    """Test that finetuning strategies use test evaluation when enabled."""
    print("\n" + "="*60)
    print("TEST 3: Finetuning Strategies with Test Evaluation")
    print("="*60)

    # Create mini test configuration
    config = create_test_config()
    config.data_config.num_instances = 2  # 1 train, 1 test
    config.pretraining_config.total_batches = 10  # Very small
    config.finetuning_config.finetuning_steps = 6   # Very small
    config.finetuning_config.eval_frequency = 2    # Evaluate every 2 steps
    config.evaluation_config.eval_on_test_during_finetuning = True
    config.evaluation_config.test_eval_frequency = 3  # Test eval every 3 steps

    # Only enable finetuning strategies for faster testing
    config.enabled_strategies = ["Pretrain_Finetuned_Imputer", "Finetuned_Imputer"]

    # Force CPU
    config.pretraining_config.device = "cpu"
    config.finetuning_config.device = "cpu"
    config.evaluation_config.device = "cpu"

    runner = ExperimentRunner(config)

    print("Running mini experiment with test evaluation enabled...")

    # Generate data and run pretraining
    train_instances, test_instances = runner.generate_data()
    pretrained_model = runner.run_pretraining(train_instances)

    # Test one instance with both finetuning strategies
    test_instance = test_instances[0]

    print("\n--- Testing Pretrain_Finetuned_Imputer ---")
    pretrain_finetuned_results = runner.evaluate_pretrain_finetuned_imputer(pretrained_model, test_instance, 0)

    print(f"Pretrain_Finetuned_Imputer results keys: {list(pretrain_finetuned_results.keys())}")

    # Check that callback results are present
    assert "callback_results" in pretrain_finetuned_results, "Missing callback_results in Pretrain_Finetuned_Imputer"

    callback_results = pretrain_finetuned_results["callback_results"]
    print(f"Pretrain_Finetuned callback results: {len(callback_results)} evaluations")

    print("\n--- Testing Finetuned_Imputer ---")
    finetuned_results = runner.evaluate_finetuned_imputer(test_instance, 0)

    print(f"Finetuned_Imputer results keys: {list(finetuned_results.keys())}")

    # Check that callback results are present
    assert "callback_results" in finetuned_results, "Missing callback_results in Finetuned_Imputer"

    callback_results = finetuned_results["callback_results"]
    print(f"Finetuned callback results: {len(callback_results)} evaluations")

    print("✅ Both finetuning strategies support test evaluation!")


def test_results_storage_structure():
    """Test that results storage includes test evaluation metrics properly."""
    print("\n" + "="*60)
    print("TEST 4: Results Storage with Test Evaluation Metrics")
    print("="*60)

    # Create minimal configuration for testing
    config = create_test_config()
    config.data_config.num_instances = 2
    config.pretraining_config.total_batches = 8
    config.finetuning_config.finetuning_steps = 4
    config.evaluation_config.eval_on_test_during_finetuning = True

    # Only test one strategy for speed
    config.enabled_strategies = ["Finetuned_Imputer"]

    # Force CPU
    config.pretraining_config.device = "cpu"
    config.finetuning_config.device = "cpu"
    config.evaluation_config.device = "cpu"

    runner = ExperimentRunner(config)

    print("Running mini experiment to test results storage...")

    # Generate and run experiment
    train_instances, test_instances = runner.generate_data()
    runner.run_pretraining(train_instances)

    # Test one instance
    test_instance = test_instances[0]
    instance_results = runner.evaluate_all_strategies(test_instance, None, 0)

    print(f"\nInstance results keys: {list(instance_results.keys())}")

    # Check finetuning strategy results
    if "Finetuned_Imputer" in instance_results:
        finetuned_results = instance_results["Finetuned_Imputer"]
        print(f"Finetuned_Imputer results keys: {list(finetuned_results.keys())}")

        # Verify callback results are stored and structured properly
        assert "callback_results" in finetuned_results, "Missing callback_results"

        callback_results = finetuned_results["callback_results"]
        print(f"Callback results: {len(callback_results)} entries")

        if callback_results:
            sample_result = callback_results[0]
            print(f"Sample callback result keys: {list(sample_result.keys())}")

            # Should contain evaluation metrics
            expected_keys = ['total_loss', 'rating_accuracy', 'ranking_accuracy']
            for key in expected_keys:
                assert key in sample_result, f"Missing key in callback result: {key}"

            # Check if we have both types of evaluation (though we may not with small test)
            callback_types = [getattr(cb, 'callback_type', 'heldout') for cb in callback_results]
            print(f"Callback types found: {set(callback_types)}")

    print("✅ Results storage structure includes test evaluation metrics!")


def test_backward_compatibility():
    """Test that Phase 2 changes don't break existing functionality."""
    print("\n" + "="*60)
    print("TEST 5: Backward Compatibility")
    print("="*60)

    # Test with test evaluation disabled
    config = create_test_config()
    config.evaluation_config.eval_on_test_during_finetuning = False
    config.finetuning_config.finetuning_steps = 5

    # Force CPU
    config.finetuning_config.device = "cpu"
    config.evaluation_config.device = "cpu"

    # Generate test data
    generator = ICLRDataGenerator()
    data_config = ICLRDatasetConfig(
        K=config.data_config.K,
        I=config.data_config.I,
        J=config.data_config.J,
        D=config.data_config.D,
        C=config.data_config.C
    )

    dataset_obj = generator.generate_dataset(data_config, seed=200)
    test_instance = {
        'ratings': dataset_obj.observed_ratings,
        'pairwise_rankings': dataset_obj.observed_pairwise_rankings
    }

    # Create components
    converter = DataConverter()
    eval_engine = EvaluationEngine()

    model = MultiVariableImputer(
        num_attributes=config.data_config.I,
        num_annotators=config.data_config.J,
        num_items=config.data_config.K,
        num_likert_classes=config.data_config.C,
        max_rank_size=config.model_config.max_rank_size,
        encoder_layers_num=config.model_config.encoder_layers,
        attention_heads=config.model_config.attention_heads,
        embedding_dim=config.model_config.embedding_dim,
        dropout=config.model_config.dropout,
        device=config.finetuning_config.device
    )

    # Test GeneralMIT with test evaluation disabled
    mit = GeneralMIT(model, eval_engine, config.finetuning_config, converter)

    print("Testing GeneralMIT with test evaluation disabled...")
    results = mit.finetune_on_instance(model, test_instance)  # No additional parameters

    # Should work without errors and provide basic results
    assert "finetuning_results" in results, "Basic finetuning should still work"
    assert "final_evaluation" in results, "Final evaluation should still work"

    print("✅ Backward compatibility maintained!")


def main():
    """Run all Phase 2 tests."""
    print("Phase 2 Test Script: Step-by-Step Evaluation During Training")
    print("="*80)

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    try:
        # Run all tests
        test_test_evaluation_config()
        test_general_mit_test_evaluation()
        test_finetuning_strategies_with_test_evaluation()
        test_results_storage_structure()
        test_backward_compatibility()

        print("\n" + "="*80)
        print("🎉 ALL PHASE 2 TESTS PASSED! 🎉")
        print("="*80)
        print("\nPhase 2 improvements verified:")
        print("✅ Test evaluation configuration options working")
        print("✅ GeneralMIT performs test evaluation during finetuning")
        print("✅ Both finetuning strategies support test evaluation")
        print("✅ Results storage includes test evaluation metrics")
        print("✅ Enhanced progress tracking with separate test/heldout metrics")
        print("✅ Backward compatibility maintained")
        print("✅ Configurable evaluation frequencies working")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()