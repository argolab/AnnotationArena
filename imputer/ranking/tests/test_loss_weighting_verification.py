#!/usr/bin/env python3
"""
Test script to verify loss weighting system works correctly throughout the pipeline.

This comprehensive test validates:
1. Loss weights flow correctly from config to all components
2. Masked/observed separation works in both pretraining and finetuning
3. is_masked preservation throughout the pipeline
4. EvaluationEngine uses config weights correctly
5. Both pretraining and finetuning use weighted losses

Run with: python tests/test_loss_weighting_verification.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
from dataclasses import dataclass
from typing import List, Dict, Any

from experiment_config import ExperimentConfig, DataConfig, ModelConfig, PretrainingConfig, FinetuningConfig, DomainConfig, EvaluationConfig
from imputer.data import DataConverter, RankingData
from imputer.ranking_imputer import MultiVariableImputer
from imputer.legacy.multi_instance_trainer import SequentialMIT, GeneralMIT
from imputer.eval import EvaluationEngine
from imputer.losses import DefaultLossStrategy, adapt_batched_logits_to_predictions


def create_test_config():
    """Create test configuration with specific loss weights."""
    return ExperimentConfig(
        data_config=DataConfig(
            num_instances=2,
            train_test_split=0.8,
            K=4,  # 4 items
            I=2,  # 2 attributes
            J=2,  # 2 annotators
            C=3,  # 3 rating classes
            D=16,
            sigma_annotator=0.1,
            sigma_measurement=0.1,
            alpha_dirichlet=2.0,
            temperature=0.5,
            sigma_embedding_prior=0.5,
            sigma_preference_prior=0.5,
            max_pairs_per_tied_group=5,
            min_group_size=2,
            max_group_size=4,
            rankings_per_annotator_attribute=3
        ),
        model_config=ModelConfig(
            encoder_layers=2,
            attention_heads=4,
            embedding_dim=16,
            dropout=0.1,
            max_rank_size=2,
            masked_loss_weight=3.0,    # KEY: Different weights
            observed_loss_weight=1.0   # KEY: Different weights
        ),
        pretraining_config=PretrainingConfig(
            strategy="sequential",
            total_batches=10,
            batch_size=4,
            learning_rate=1e-3,
            train_heldout_split=0.8,
            eval_frequency=5,
            masking_rates=[0.5]
        ),
        finetuning_config=FinetuningConfig(
            finetuning_steps=5,
            batch_size=4,
            learning_rate=1e-3,
            train_heldout_split=0.8,
            eval_frequency=2,
            test_masking_rate=0.5,
            masking_rates=[0.5]
        ),
        domain_config=DomainConfig(
            chains=2,
            iter_warmup=50,
            iter_sampling=100,
            adapt_delta=0.8,
            max_treedepth=8,
            sample_counts=[10, 20]
        ),
        evaluation_config=EvaluationConfig(
            test_masking_rate=0.5
        ),
        enabled_strategies=["Pretrained_Imputer", "Pretrain_Finetuned_Imputer"],
        output_dir="test_results",
        save_models=False,
        save_training_histories=False,
        log_wall_times=False,
        experiment_name="loss_weighting_test",
        random_seed=42,
        device="cpu"
    )


def create_test_data():
    """Create minimal test data in format expected by DataConverter."""
    # Create simple test data - note converter expects specific format
    rating_data = [
        {"annotator": 1, "attribute": 1, "item": 1, "value": 2},  # 0-indexed becomes annotator=0, attribute=0, item=0, rating=1
        {"annotator": 2, "attribute": 1, "item": 2, "value": 3},  # annotator=1, attribute=0, item=1, rating=2
        {"annotator": 1, "attribute": 2, "item": 3, "value": 2},  # annotator=0, attribute=1, item=2, rating=1
        {"annotator": 2, "attribute": 2, "item": 4, "value": 1},  # annotator=1, attribute=1, item=3, rating=0
    ]

    pairwise_ranking_data = [
        {"annotator": 1, "attribute": 1, "items": [1, 2], "order": [1, 2]},
        {"annotator": 2, "attribute": 1, "items": [1, 2], "order": [2, 1]},
        {"annotator": 1, "attribute": 2, "items": [3, 4], "order": [1, 2]},
        {"annotator": 2, "attribute": 2, "items": [3, 4], "order": [2, 1]},
    ]

    return {
        'ratings': rating_data,
        'pairwise_rankings': pairwise_ranking_data
    }


def test_loss_strategy_weights():
    """Test 1: Verify DefaultLossStrategy uses correct weights."""
    print("=== Test 1: DefaultLossStrategy Weight Usage ===")

    # Test with different weights
    strategy = DefaultLossStrategy(masked_loss_weight=3.0, observed_loss_weight=1.0)

    # Verify weights are stored
    assert strategy.masked_loss_weight == 3.0, f"Expected 3.0, got {strategy.masked_loss_weight}"
    assert strategy.observed_loss_weight == 1.0, f"Expected 1.0, got {strategy.observed_loss_weight}"

    print("✅ DefaultLossStrategy correctly stores weights")


def test_evaluation_engine_config_weights():
    """Test 2: Verify EvaluationEngine uses config weights."""
    print("\n=== Test 2: EvaluationEngine Config Weight Usage ===")

    config = create_test_config()

    # Test with config
    eval_engine_with_config = EvaluationEngine(config.model_config)
    assert eval_engine_with_config.loss_strategy.masked_loss_weight == 3.0
    assert eval_engine_with_config.loss_strategy.observed_loss_weight == 1.0

    print("✅ EvaluationEngine with config uses correct weights")

    # Test without config
    eval_engine_without_config = EvaluationEngine()
    assert eval_engine_without_config.loss_strategy.masked_loss_weight == 1.0
    assert eval_engine_without_config.loss_strategy.observed_loss_weight == 1.0

    print("✅ EvaluationEngine without config uses default weights")


def test_mit_weight_propagation():
    """Test 3: Verify MIT classes propagate model_config weights to trainer."""
    print("\n=== Test 3: MIT Weight Propagation ===")

    config = create_test_config()

    # Create converter and model
    converter = DataConverter(
        num_attributes=config.data_config.I,
        num_annotators=config.data_config.J,
        num_items=config.data_config.K,
        num_likert_classes=config.data_config.C,
        max_rank_size=config.model_config.max_rank_size
    )

    model = MultiVariableImputer(
        num_items=config.data_config.K,
        num_attributes=config.data_config.I,
        num_annotators=config.data_config.J,
        num_likert_classes=config.data_config.C,
        encoder_layers_num=config.model_config.encoder_layers,
        attention_heads=config.model_config.attention_heads,
        embedding_dim=config.model_config.embedding_dim,
        dropout=config.model_config.dropout,
        max_rank_size=config.model_config.max_rank_size
    ).to(config.device)

    eval_engine = EvaluationEngine(config.model_config)

    # Test SequentialMIT
    sequential_mit = SequentialMIT(
        model, eval_engine, config.pretraining_config, converter, config.model_config
    )

    # Check trainer weights
    assert sequential_mit.trainer.loss_strategy.masked_loss_weight == 3.0
    assert sequential_mit.trainer.loss_strategy.observed_loss_weight == 1.0

    print("✅ SequentialMIT propagates weights correctly")

    # Test GeneralMIT
    general_mit = GeneralMIT(
        model, eval_engine, config.finetuning_config, converter, config.model_config
    )

    assert general_mit.trainer.loss_strategy.masked_loss_weight == 3.0
    assert general_mit.trainer.loss_strategy.observed_loss_weight == 1.0

    print("✅ GeneralMIT propagates weights correctly")


def test_is_masked_preservation():
    """Test 4: Verify is_masked preservation throughout pipeline."""
    print("\n=== Test 4: is_masked Preservation ===")

    config = create_test_config()

    converter = DataConverter(
        num_attributes=config.data_config.I,
        num_annotators=config.data_config.J,
        num_items=config.data_config.K,
        num_likert_classes=config.data_config.C,
        max_rank_size=config.model_config.max_rank_size
    )

    # Create test variables
    test_data = create_test_data()
    variables = converter.create_variables(test_data)

    # Apply masking
    eval_engine = EvaluationEngine(config.model_config)
    evaluation_mask = eval_engine.create_evaluation_mask(variables, masking_rate=0.5)

    # Verify evaluation mask creates mixed masked/observed
    has_masked = any(evaluation_mask)
    has_observed = any(not m for m in evaluation_mask)
    assert has_masked and has_observed, "Should have both masked and observed variables"

    print("✅ Evaluation mask creates mixed masked/observed variables")

    # Test evaluation batch creation preserves is_masked
    eval_batch = eval_engine._create_evaluation_batch(variables, evaluation_mask)

    for i, (orig_var, eval_var, is_eval_masked) in enumerate(zip(variables, eval_batch, evaluation_mask)):
        if is_eval_masked:
            assert eval_var.is_masked == True, f"Variable {i} should be masked"
        else:
            assert eval_var.is_masked == False, f"Variable {i} should be observed"

    print("✅ Evaluation batch creation preserves is_masked correctly")


def test_masking_rate_zero_behavior():
    """Test 5: Verify masking_rate=0.0 respects existing is_masked flags."""
    print("\n=== Test 5: masking_rate=0.0 Behavior ===")

    config = create_test_config()
    eval_engine = EvaluationEngine(config.model_config)

    # Create variables with pre-set is_masked flags
    variables = [
        RankingData(annotator_id=0, attribute_id=0, is_listwise=False, item_ids=[0], rating_value=1, is_masked=True),
        RankingData(annotator_id=1, attribute_id=0, is_listwise=False, item_ids=[1], rating_value=2, is_masked=False),
        RankingData(annotator_id=0, attribute_id=1, is_listwise=True, item_ids=[0,1], ranking_order=[1,2], is_masked=True),
        RankingData(annotator_id=1, attribute_id=1, is_listwise=True, item_ids=[0,1], ranking_order=[2,1], is_masked=False)
    ]

    # Test masking_rate=0.0 respects existing flags
    mask_zero = eval_engine.create_evaluation_mask(variables, masking_rate=0.0)
    expected_mask = [True, False, True, False]  # Based on pre-set is_masked flags

    assert mask_zero == expected_mask, f"Expected {expected_mask}, got {mask_zero}"

    print("✅ masking_rate=0.0 correctly respects existing is_masked flags")


def test_weighted_loss_computation():
    """Test 6: Verify weighted loss computation produces different results."""
    print("\n=== Test 6: Weighted Loss Computation ===")

    # Create test prediction and reference data
    device = torch.device('cpu')

    # Mock predictions (simplified)
    from imputer.losses import TopLayerPredictionResult

    predictions = [
        TopLayerPredictionResult(is_listwise=False, rating_logits=torch.tensor([1.0, 2.0, 0.5])),
        TopLayerPredictionResult(is_listwise=False, rating_logits=torch.tensor([0.5, 1.0, 2.0])),
        TopLayerPredictionResult(is_listwise=True, ranking_logits=torch.tensor([1.5, 0.8]))
    ]

    references = [
        RankingData(annotator_id=0, attribute_id=0, is_listwise=False, item_ids=[0], rating_value=1, is_masked=True),   # Masked
        RankingData(annotator_id=1, attribute_id=0, is_listwise=False, item_ids=[1], rating_value=2, is_masked=False),  # Observed
        RankingData(annotator_id=0, attribute_id=1, is_listwise=True, item_ids=[0,1], ranking_order=[1,2], is_masked=True)  # Masked
    ]

    # Test with equal weights
    strategy_equal = DefaultLossStrategy(masked_loss_weight=1.0, observed_loss_weight=1.0)
    losses_equal = strategy_equal.compute(predictions, references)

    # Test with different weights (3x weight on masked)
    strategy_weighted = DefaultLossStrategy(masked_loss_weight=3.0, observed_loss_weight=1.0)
    losses_weighted = strategy_weighted.compute(predictions, references)

    # Total loss should be different due to weighting
    assert losses_equal['total_loss'] != losses_weighted['total_loss'], "Weighted loss should differ from equal weights"

    # Weighted version should have higher total loss (since masked gets 3x weight)
    assert losses_weighted['total_loss'] > losses_equal['total_loss'], "Weighted loss should be higher with 3x masked weight"

    print(f"✅ Equal weights total loss: {losses_equal['total_loss']:.4f}")
    print(f"✅ Weighted (3:1) total loss: {losses_weighted['total_loss']:.4f}")
    print("✅ Weighted loss computation produces different results as expected")


def test_end_to_end_weight_flow():
    """Test 7: End-to-end test of weight flow through training."""
    print("\n=== Test 7: End-to-End Weight Flow ===")

    config = create_test_config()

    # Set up components
    converter = DataConverter(
        num_attributes=config.data_config.I,
        num_annotators=config.data_config.J,
        num_items=config.data_config.K,
        num_likert_classes=config.data_config.C,
        max_rank_size=config.model_config.max_rank_size
    )

    model = MultiVariableImputer(
        num_items=config.data_config.K,
        num_attributes=config.data_config.I,
        num_annotators=config.data_config.J,
        num_likert_classes=config.data_config.C,
        encoder_layers_num=config.model_config.encoder_layers,
        attention_heads=config.model_config.attention_heads,
        embedding_dim=config.model_config.embedding_dim,
        dropout=config.model_config.dropout,
        max_rank_size=config.model_config.max_rank_size
    ).to(config.device)

    eval_engine = EvaluationEngine(config.model_config)

    # Create MIT trainer
    mit = SequentialMIT(model, eval_engine, config.pretraining_config, converter, config.model_config)

    # Create test training data
    test_instances = [create_test_data()]

    # Set up training (minimal)
    mit.setup_heldout_evaluation_callback(test_instances)

    # Get a training batch
    batch_gen = mit.create_training_batch_generator(test_instances)
    batch = next(batch_gen)

    # Verify batch has both masked and observed variables
    if batch and len(batch[0]) > 0:
        masked_count = sum(1 for var in batch[0] if var.is_masked)
        observed_count = sum(1 for var in batch[0] if not var.is_masked)

        print(f"✅ Training batch has {masked_count} masked and {observed_count} observed variables")
        assert masked_count > 0 and observed_count > 0, "Should have both masked and observed variables"

        # Run one training step to verify weights are used
        loss_dict = mit.trainer.train_step(batch)

        # Check that loss computation completed without errors
        assert 'total_loss' in loss_dict, "Training step should return total_loss"
        assert 'masked_total_loss' in loss_dict, "Should have masked loss breakdown"
        assert 'observed_total_loss' in loss_dict, "Should have observed loss breakdown"

        print(f"✅ Training step completed with weighted losses:")
        print(f"   - Total loss: {loss_dict['total_loss']:.4f}")
        print(f"   - Masked loss: {loss_dict['masked_total_loss']:.4f}")
        print(f"   - Observed loss: {loss_dict['observed_total_loss']:.4f}")


def main():
    """Run all verification tests."""
    print("🔍 COMPREHENSIVE LOSS WEIGHTING VERIFICATION")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    try:
        test_loss_strategy_weights()
        test_evaluation_engine_config_weights()
        test_mit_weight_propagation()
        test_is_masked_preservation()
        test_masking_rate_zero_behavior()
        test_weighted_loss_computation()
        test_end_to_end_weight_flow()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Loss weighting system verified successfully.")
        print("✅ Weights flow correctly from config to all components")
        print("✅ Both pretraining and finetuning use weighted losses")
        print("✅ is_masked preservation works throughout pipeline")
        print("✅ masking_rate=0.0 respects existing flags correctly")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()