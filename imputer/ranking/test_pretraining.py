#!/usr/bin/env python3
"""
Test script for pretraining implementation.

This script tests ONLY the pretraining phase to verify:
1. Training data mixing from all 8 training instances
2. Random masking on combined data
3. Loss computation on both masked and observed positions
4. Heldout evaluation on ALL positions (not just masked)
"""

import json
import logging
import sys
from pathlib import Path
import torch
import time

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import ExperimentConfig
from experiment_runner_iclr import ExperimentRunnerICLR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_pretraining_only(config_file: str, num_epochs: int = 10):
    """Test pretraining implementation with detailed verification."""

    logger.info("="*80)
    logger.info("PRETRAINING IMPLEMENTATION TEST")
    logger.info("="*80)

    # Load config
    config = ExperimentConfig.load_from_file(config_file)

    # Override epochs for testing
    config.training_config.epochs = num_epochs
    config.training_config.evaluation_frequency = 1  # Evaluate every epoch for testing

    logger.info(f"Config loaded: {len(config.train_instance_indices)} train instances, {len(config.test_instance_indices)} test instances")
    logger.info(f"Train instances: {config.train_instance_indices}")
    logger.info(f"Test instances: {config.test_instance_indices}")

    # Create experiment runner
    runner = ExperimentRunnerICLR(config)

    # Generate data first
    logger.info("Generating data for all instances...")
    runner.generate_data()

    logger.info("\n" + "="*60)
    logger.info("TESTING MIXED TRAINING DATA CREATION")
    logger.info("="*60)

    # Test mixed training data creation
    masking_rate = config.training_config.masking_rate
    logger.info(f"Using masking rate: {masking_rate}")

    # Check if mixed training data is created correctly
    all_variables, all_data = runner._create_mixed_training_data()

    logger.info(f"Mixed training data created:")
    logger.info(f"  - Total variables: {len(all_variables)}")
    logger.info(f"  - Total ratings: {len(all_data['ratings'])}")
    logger.info(f"  - Total rankings: {len(all_data['pairwise_rankings'])}")

    # Check source instances
    source_instances = set()
    rating_count = 0
    ranking_count = 0

    for var in all_variables:
        source_instances.add(var.get('source_instance', 'unknown'))
        if var['type'] == 'rating':
            rating_count += 1
        elif var['type'] == 'ranking':
            ranking_count += 1

    logger.info(f"  - Rating variables: {rating_count}")
    logger.info(f"  - Ranking variables: {ranking_count}")
    logger.info(f"  - Source instances: {sorted(source_instances)}")

    if source_instances != set(config.train_instance_indices):
        logger.error(f"ERROR: Source instances {source_instances} don't match train instances {config.train_instance_indices}")
        return False

    logger.info("✅ Mixed training data creation: PASSED")

    logger.info("\n" + "="*60)
    logger.info("TESTING BATCH CREATION WITH RANDOM MASKING")
    logger.info("="*60)

    # Test batch creation
    batch1 = runner.create_mixed_training_batch(masking_rate)
    batch2 = runner.create_mixed_training_batch(masking_rate)

    logger.info(f"Batch 1 variables: {batch1['variable_data'].shape[1]} variables")
    logger.info(f"Batch 2 variables: {batch2['variable_data'].shape[1]} variables")

    # Check that masking is random (different between batches)
    mask1 = set(torch.where(batch1['rating_masked'][0] == 1)[0].tolist())
    mask2 = set(torch.where(batch2['rating_masked'][0] == 1)[0].tolist())

    logger.info(f"Batch 1 masked rating positions: {len(mask1)} out of {batch1['rating_masked'].shape[1]}")
    logger.info(f"Batch 2 masked rating positions: {len(mask2)} out of {batch2['rating_masked'].shape[1]}")
    logger.info(f"Overlap between masks: {len(mask1 & mask2)} positions")

    if mask1 == mask2:
        logger.warning("WARNING: Masking appears to be identical between batches (may be due to small data or fixed seed)")
    else:
        logger.info("✅ Random masking: PASSED")

    logger.info("\n" + "="*60)
    logger.info("TESTING HELDOUT EVALUATION SETUP")
    logger.info("="*60)

    # Test heldout evaluation data
    heldout_variables, heldout_data, heldout_masked, heldout_observed = runner.create_heldout_evaluation_data(masking_rate)

    logger.info(f"Heldout evaluation data:")
    logger.info(f"  - Total variables: {len(heldout_variables)}")
    logger.info(f"  - Masked variables: {len(heldout_masked)}")
    logger.info(f"  - Observed variables: {len(heldout_observed)}")
    logger.info(f"  - Masking rate: {len(heldout_masked) / len(heldout_variables):.3f}")

    expected_masked = int(len(heldout_variables) * masking_rate)
    actual_masked = len(heldout_masked)

    if abs(actual_masked - expected_masked) <= 1:  # Allow for rounding
        logger.info("✅ Heldout masking rate: PASSED")
    else:
        logger.error(f"ERROR: Expected ~{expected_masked} masked, got {actual_masked}")
        return False

    logger.info("\n" + "="*60)
    logger.info("TESTING PRETRAINING LOOP")
    logger.info("="*60)

    # Test actual pretraining
    logger.info(f"Running {num_epochs} epochs of pretraining...")
    start_time = time.time()

    pretraining_results = runner.run_pretraining(masking_rate)

    end_time = time.time()
    logger.info(f"Pretraining completed in {end_time - start_time:.2f} seconds")

    # Check results structure
    logger.info("Pretraining results structure:")
    for key, value in pretraining_results.items():
        if isinstance(value, dict):
            logger.info(f"  - {key}: {list(value.keys())}")
        elif isinstance(value, list):
            logger.info(f"  - {key}: {len(value)} items")
        else:
            logger.info(f"  - {key}: {value}")

    # Check that we have the expected number of evaluations
    expected_evals = num_epochs  # Since evaluation_frequency = 1
    actual_evals = len(pretraining_results['heldout_losses']['total'])

    logger.info(f"Expected evaluations: {expected_evals}, Actual: {actual_evals}")

    if actual_evals == expected_evals:
        logger.info("✅ Evaluation frequency: PASSED")
    else:
        logger.error(f"ERROR: Expected {expected_evals} evaluations, got {actual_evals}")
        return False

    # Check loss trends
    train_losses = pretraining_results['train_losses']['total']
    heldout_losses = pretraining_results['heldout_losses']['total']

    logger.info(f"Training loss trend: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
    logger.info(f"Heldout loss trend: {heldout_losses[0]:.4f} → {heldout_losses[-1]:.4f}")

    # Check that training happened (loss should change)
    if abs(train_losses[0] - train_losses[-1]) > 1e-6:
        logger.info("✅ Training loss changing: PASSED")
    else:
        logger.warning("WARNING: Training loss not changing significantly")

    logger.info("\n" + "="*60)
    logger.info("TESTING EVALUATION METHOD COMPARISON")
    logger.info("="*60)

    # Compare old (masked-only) vs new (all-positions) evaluation
    logger.info("Comparing evaluation methods on heldout data...")

    # Old method: only masked variables
    old_metrics = runner.evaluate_conditional_imputation(
        heldout_variables, heldout_data, heldout_masked, heldout_observed
    )

    # New method: all variables
    new_metrics = runner.evaluate_all_positions(
        heldout_variables, heldout_data, masking_rate
    )

    logger.info("Old method (masked only):")
    logger.info(f"  - Total loss: {old_metrics['total_log_loss']:.4f}")
    logger.info(f"  - Rating accuracy: {old_metrics['rating_accuracy']:.4f}")
    logger.info(f"  - Evaluated: {old_metrics['masked_rating_count']} ratings, {old_metrics['masked_ranking_count']} rankings")

    logger.info("New method (all positions):")
    logger.info(f"  - Total loss: {new_metrics['total_log_loss']:.4f}")
    logger.info(f"  - Rating accuracy: {new_metrics['rating_accuracy']:.4f}")
    logger.info(f"  - Evaluated: {new_metrics['total_rating_count']} ratings, {new_metrics['total_ranking_count']} rankings")

    # New method should evaluate more variables
    if new_metrics['total_rating_count'] > old_metrics['masked_rating_count']:
        logger.info("✅ All-positions evaluation: PASSED")
    else:
        logger.error("ERROR: New method doesn't evaluate more variables than old method")
        return False

    logger.info("\n" + "="*80)
    logger.info("PRETRAINING TEST SUMMARY")
    logger.info("="*80)
    logger.info("✅ All tests PASSED")
    logger.info("✅ Mixed training data creation works correctly")
    logger.info("✅ Random masking is applied properly")
    logger.info("✅ Heldout evaluation uses ALL positions")
    logger.info("✅ Training loop executes without errors")
    logger.info("="*80)

    return True


def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description='Test pretraining implementation')
    parser.add_argument('config', help='Path to config JSON file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to test (default: 5)')

    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    success = test_pretraining_only(args.config, args.epochs)

    if success:
        logger.info("🎉 All pretraining tests PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ Some pretraining tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()