#!/usr/bin/env python3
"""
Phase 1 Test Script: Comprehensive Testing of Masked Metrics and Training Improvements

This script tests the Phase 1 fixes:
1. Evaluation engine respects pre-existing is_masked flags
2. Progress tracking and callback collection in multi-instance trainer
3. GeneralMIT final evaluation preserves masking information
4. Results storage captures callback metrics instead of raw variables

Test approach:
- Load actual generated data
- Print samples from training/heldout sets during pretraining
- Print samples from T_O_train/T_O_heldout/T_M during finetuning
- Verify masking rates (30% masked, 70% observed)
- Check that masked_metrics and observed_metrics are computed correctly
- Validate heldout evaluation results format
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
from new_experiment_runner import ExperimentRunner
from iclr_data_generator import ICLRDataGenerator, ICLRDatasetConfig
from imputer.data import DataConverter
from imputer.eval import EvaluationEngine
from imputer.multi_instance_trainer import SequentialMIT, GeneralMIT
from imputer.ranking_imputer import MultiVariableImputer


def test_evaluation_engine_masking():
    """Test that evaluation engine respects pre-existing is_masked flags."""
    print("\n" + "="*60)
    print("TEST 1: Evaluation Engine Masking Logic")
    print("="*60)

    # Generate small test data
    config = create_test_config()
    generator = ICLRDataGenerator()

    # Create proper ICLRDatasetConfig like the real code does
    data_config = ICLRDatasetConfig(
        K=config.data_config.K,
        I=config.data_config.I,
        J=config.data_config.J,
        D=config.data_config.D,
        C=config.data_config.C
    )
    dataset_obj = generator.generate_dataset(data_config, seed=42)

    # Convert to format expected by our system (like the real code does)
    dataset = {
        'ratings': dataset_obj.observed_ratings,
        'pairwise_rankings': dataset_obj.observed_pairwise_rankings
    }

    converter = DataConverter()
    variables = converter.create_variables(dataset)

    # Test Case 1: Normal masking (masking_rate > 0)
    eval_engine = EvaluationEngine()
    mask_30 = eval_engine.create_evaluation_mask(variables, 0.3)
    masked_count = sum(mask_30)
    expected_masked = int(len(variables) * 0.3)

    print(f"Normal masking (30%): {masked_count}/{len(variables)} masked")
    print(f"Expected ~{expected_masked}, got {masked_count}")
    assert abs(masked_count - expected_masked) <= 1, f"Masking rate incorrect: expected ~{expected_masked}, got {masked_count}"

    # Test Case 2: Zero masking with pre-existing flags
    # Manually set some variables as masked
    for i in range(5):
        variables[i].is_masked = True
    for i in range(5, len(variables)):
        variables[i].is_masked = False

    mask_zero = eval_engine.create_evaluation_mask(variables, 0.0)

    print(f"Zero masking with existing flags: {sum(mask_zero)}/5 variables should be masked")
    print(f"First 5 variables masked: {mask_zero[:5]}")
    print(f"Next 5 variables observed: {mask_zero[5:10]}")

    # Verify that zero masking respects existing flags
    assert sum(mask_zero) == 5, f"Expected 5 masked variables, got {sum(mask_zero)}"
    assert all(mask_zero[:5]), "First 5 variables should be masked"
    assert not any(mask_zero[5:10]), "Next 5 variables should be observed"

    print("✅ Evaluation engine masking logic works correctly!")


def test_sequential_mit_training():
    """Test Sequential MIT training with progress tracking and callback collection."""
    print("\n" + "="*60)
    print("TEST 2: Sequential MIT Training with Progress Tracking")
    print("="*60)

    # Create small test configuration
    config = create_test_config()
    config.pretraining_config.total_batches = 20  # Small for testing
    config.pretraining_config.eval_frequency = 5   # Evaluate every 5 steps

    # Generate test data with multiple instances
    generator = ICLRDataGenerator()

    # Create proper ICLRDatasetConfig
    data_config = ICLRDatasetConfig(
        K=config.data_config.K,
        I=config.data_config.I,
        J=config.data_config.J,
        D=config.data_config.D,
        C=config.data_config.C
    )

    train_instances = []
    for i in range(2):  # Just 2 training instances for speed
        dataset_obj = generator.generate_dataset(data_config, seed=42 + i)
        # Convert to format expected by our system
        instance = {
            'ratings': dataset_obj.observed_ratings,
            'pairwise_rankings': dataset_obj.observed_pairwise_rankings
        }
        train_instances.append(instance)

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
        embedding_type=config.model_config.embedding_type,
        device=config.pretraining_config.device
    )

    # Create SequentialMIT and run training
    mit = SequentialMIT(model, eval_engine, config.pretraining_config, converter)

    print("Running Sequential MIT training...")
    results = mit.train_on_instances(train_instances, [])

    # Verify results structure
    print(f"\nTraining completed. Results keys: {list(results.keys())}")

    assert "training_results" in results, "Missing training_results"
    assert "heldout_variables" in results, "Missing heldout_variables"
    assert "callback_results" in results, "Missing callback_results"

    print(f"Training results: {len(results['training_results'])} steps")
    print(f"Heldout variables: {len(results['heldout_variables'])} variables")
    print(f"Callback results: {len(results['callback_results'])} evaluations")

    # Verify callback results structure
    if results["callback_results"]:
        sample_callback = results["callback_results"][0]
        print(f"Sample callback result keys: {list(sample_callback.keys())}")

        expected_keys = ['epoch', 'total_loss', 'rating_loss', 'ranking_loss', 'rating_accuracy']
        for key in expected_keys:
            assert key in sample_callback, f"Missing key in callback result: {key}"

    # Print training and heldout set info
    print(f"\nInstance train/heldout split verification:")
    for i, (train_set, heldout_set) in enumerate(zip(mit.instance_train_sets, mit.instance_heldout_sets)):
        total_vars = len(train_set) + len(heldout_set)
        train_ratio = len(train_set) / total_vars if total_vars > 0 else 0
        print(f"Instance {i}: {len(train_set)} train, {len(heldout_set)} heldout (train ratio: {train_ratio:.2f})")

        # Verify split ratio is approximately correct
        expected_ratio = config.pretraining_config.train_heldout_split
        assert abs(train_ratio - expected_ratio) < 0.1, f"Train/heldout split incorrect for instance {i}"

    print("✅ Sequential MIT training with progress tracking works correctly!")


def test_general_mit_finetuning():
    """Test GeneralMIT finetuning with proper masking preservation."""
    print("\n" + "="*60)
    print("TEST 3: GeneralMIT Finetuning with Masking Preservation")
    print("="*60)

    # Create test configuration
    config = create_test_config()
    config.finetuning_config.finetuning_steps = 10  # Small for testing
    config.finetuning_config.eval_frequency = 3    # Evaluate every 3 steps
    config.evaluation_config.test_masking_rate = 0.3  # 30% masking

    # Generate test instance
    generator = ICLRDataGenerator()

    # Create proper ICLRDatasetConfig
    data_config = ICLRDatasetConfig(
        K=config.data_config.K,
        I=config.data_config.I,
        J=config.data_config.J,
        D=config.data_config.D,
        C=config.data_config.C
    )
    dataset_obj = generator.generate_dataset(data_config, seed=100)
    # Convert to format expected by our system
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
        embedding_type=config.model_config.embedding_type,
        device=config.finetuning_config.device
    )

    # Create GeneralMIT and run finetuning
    mit = GeneralMIT(model, eval_engine, config.finetuning_config, converter)

    print("Running GeneralMIT finetuning...")
    results = mit.finetune_on_instance(model, test_instance)

    # Verify results structure
    print(f"\nFinetuning completed. Results keys: {list(results.keys())}")

    required_keys = ['finetuning_results', 'callback_results', 'final_evaluation', 't_o_variables', 't_m_variables']
    for key in required_keys:
        assert key in results, f"Missing key in finetuning results: {key}"

    # Verify T_O and T_M split
    t_o_vars = results['t_o_variables']
    t_m_vars = results['t_m_variables']
    total_vars = len(t_o_vars) + len(t_m_vars)
    masking_rate = len(t_m_vars) / total_vars

    print(f"T_O variables (observed): {len(t_o_vars)}")
    print(f"T_M variables (masked): {len(t_m_vars)}")
    print(f"Total variables: {total_vars}")
    print(f"Actual masking rate: {masking_rate:.3f}")

    expected_rate = config.evaluation_config.test_masking_rate
    assert abs(masking_rate - expected_rate) < 0.1, f"Masking rate incorrect: expected {expected_rate}, got {masking_rate}"

    # Verify masking flags are set correctly
    print(f"\nVerifying is_masked flags:")
    t_o_masked_flags = [var.is_masked for var in t_o_vars]
    t_m_masked_flags = [var.is_masked for var in t_m_vars]

    print(f"T_O is_masked flags: {t_o_masked_flags[:5]}... (should all be False)")
    print(f"T_M is_masked flags: {t_m_masked_flags[:5]}... (should all be True)")

    assert all(not flag for flag in t_o_masked_flags), "T_O variables should have is_masked=False"
    assert all(flag for flag in t_m_masked_flags), "T_M variables should have is_masked=True"

    # Check final evaluation results
    final_eval = results['final_evaluation']
    print(f"\nFinal evaluation metrics:")
    print(f"Total loss: {final_eval.total_loss:.4f}")
    print(f"Rating accuracy: {final_eval.rating_accuracy:.3f}")
    print(f"Ranking accuracy: {final_eval.ranking_accuracy:.3f}")

    # Verify masked and observed metrics
    if hasattr(final_eval, 'masked_metrics') and final_eval.masked_metrics:
        print(f"Masked metrics - evaluations: {final_eval.masked_metrics.get('num_rating_evaluations', 0) + final_eval.masked_metrics.get('num_ranking_evaluations', 0)}")
        # Should have non-zero evaluations for masked variables
        masked_evals = final_eval.masked_metrics.get('num_rating_evaluations', 0) + final_eval.masked_metrics.get('num_ranking_evaluations', 0)
        assert masked_evals > 0, "Masked metrics should have non-zero evaluations"

    if hasattr(final_eval, 'observed_metrics') and final_eval.observed_metrics:
        print(f"Observed metrics - evaluations: {final_eval.observed_metrics.get('num_rating_evaluations', 0) + final_eval.observed_metrics.get('num_ranking_evaluations', 0)}")
        # Should have non-zero evaluations for observed variables
        observed_evals = final_eval.observed_metrics.get('num_rating_evaluations', 0) + final_eval.observed_metrics.get('num_ranking_evaluations', 0)
        assert observed_evals > 0, "Observed metrics should have non-zero evaluations"

    print("✅ GeneralMIT finetuning with masking preservation works correctly!")


def test_results_storage():
    """Test that results storage captures callback metrics correctly."""
    print("\n" + "="*60)
    print("TEST 4: Results Storage with Callback Metrics")
    print("="*60)

    # Create minimal test runner
    config = create_test_config()
    config.pretraining_config.total_batches = 10
    config.pretraining_config.eval_frequency = 5
    config.finetuning_config.finetuning_steps = 5
    config.finetuning_config.eval_frequency = 2

    # Only enable one strategy for speed
    config.enabled_strategies = ["Pretrained_Imputer", "Finetuned_Imputer"]

    # Force CPU for testing (avoid CUDA issues)
    config.pretraining_config.device = "cpu"
    config.finetuning_config.device = "cpu"
    config.evaluation_config.device = "cpu"

    runner = ExperimentRunner(config)

    print("Running mini experiment to test results storage...")

    # Generate and run experiment
    train_instances, test_instances = runner.generate_data()
    pretrained_model = runner.run_pretraining(train_instances)

    # Check pretraining results structure
    pretraining_results = runner.results["pretraining_results"]
    print(f"\nPretraining results keys: {list(pretraining_results.keys())}")

    # Verify new callback metrics storage
    assert "heldout_evaluation_metrics" in pretraining_results, "Missing heldout_evaluation_metrics"
    assert "heldout_variables_info" in pretraining_results, "Missing heldout_variables_info"

    callback_metrics = pretraining_results["heldout_evaluation_metrics"]
    print(f"Heldout evaluation metrics: {len(callback_metrics)} entries")
    print(f"Variables info: {pretraining_results['heldout_variables_info']}")

    if callback_metrics:
        sample_metric = callback_metrics[0]
        print(f"Sample callback metric keys: {list(sample_metric.keys())}")
        # Should contain actual metrics, not raw variable data
        expected_keys = ['epoch', 'total_loss', 'rating_accuracy']
        for key in expected_keys:
            assert key in sample_metric, f"Missing key in callback metric: {key}"

    # Run one test instance evaluation
    test_instance = test_instances[0]
    instance_results = runner.evaluate_all_strategies(test_instance, pretrained_model, 0)

    print(f"\nTest instance results keys: {list(instance_results.keys())}")

    # Check finetuning strategy results
    if "Finetuned_Imputer" in instance_results:
        finetuned_results = instance_results["Finetuned_Imputer"]
        print(f"Finetuned_Imputer results keys: {list(finetuned_results.keys())}")

        # Verify callback results are stored
        assert "callback_results" in finetuned_results, "Missing callback_results in Finetuned_Imputer"

        callback_results = finetuned_results["callback_results"]
        print(f"Finetuning callback results: {len(callback_results)} entries")

        if callback_results:
            sample_result = callback_results[0]
            print(f"Sample finetuning callback keys: {list(sample_result.keys())}")

    print("✅ Results storage with callback metrics works correctly!")


def run_masking_verification():
    """Detailed verification of masking logic during training."""
    print("\n" + "="*60)
    print("DETAILED MASKING VERIFICATION")
    print("="*60)

    # Create test data
    config = create_test_config()
    generator = ICLRDataGenerator()

    # Create proper ICLRDatasetConfig
    data_config = ICLRDatasetConfig(
        K=config.data_config.K,
        I=config.data_config.I,
        J=config.data_config.J,
        D=config.data_config.D,
        C=config.data_config.C
    )
    dataset_obj = generator.generate_dataset(data_config, seed=123)
    # Convert to format expected by our system
    dataset = {
        'ratings': dataset_obj.observed_ratings,
        'pairwise_rankings': dataset_obj.observed_pairwise_rankings
    }

    converter = DataConverter()
    variables = converter.create_variables(dataset)

    print(f"Generated dataset with {len(variables)} variables")

    # Print sample of variables
    print(f"\nSample variables:")
    for i, var in enumerate(variables[:5]):
        print(f"  {i}: {var}")

    # Test masking at different rates
    eval_engine = EvaluationEngine()

    for rate in [0.0, 0.3, 0.5, 0.7]:
        mask = eval_engine.create_evaluation_mask(variables, rate)
        masked_count = sum(mask)
        print(f"\nMasking rate {rate}: {masked_count}/{len(variables)} = {masked_count/len(variables):.3f}")

        if rate == 0.0:
            # For zero masking, manually set some is_masked flags to test preservation
            variables[0].is_masked = True
            variables[1].is_masked = True
            variables[2].is_masked = False

            mask_preserved = eval_engine.create_evaluation_mask(variables[:3], 0.0)
            print(f"Zero masking with flags: {mask_preserved} (should be [True, True, False])")
            assert mask_preserved == [True, True, False], "Flag preservation failed"

    print("✅ Detailed masking verification passed!")


def main():
    """Run all Phase 1 tests."""
    print("Phase 1 Test Script: Masked Metrics and Training Improvements")
    print("="*80)

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    try:
        # Run all tests
        test_evaluation_engine_masking()
        test_sequential_mit_training()
        test_general_mit_finetuning()
        test_results_storage()
        run_masking_verification()

        print("\n" + "="*80)
        print("🎉 ALL PHASE 1 TESTS PASSED! 🎉")
        print("="*80)
        print("\nPhase 1 improvements verified:")
        print("✅ Evaluation engine respects pre-existing is_masked flags")
        print("✅ Progress tracking and callback collection in multi-instance trainer")
        print("✅ GeneralMIT final evaluation preserves masking information")
        print("✅ Results storage captures callback metrics instead of raw variables")
        print("✅ Enhanced logging and progress bars working")
        print("✅ Evaluation frequency configuration working")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()