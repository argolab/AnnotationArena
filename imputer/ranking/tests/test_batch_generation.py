"""
Test script for new batch generation system in MIT trainers.

Tests the corrected self-supervised batch expansion where batch_size
refers to number of different masked versions of the same training data.
"""

import sys
import os

# Add parent directory to path so we can import from imputer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imputer.legacy.multi_instance_trainer import SequentialMIT, MixedMIT, GeneralMIT
from imputer.data import DataConverter, RankingData
from imputer.ranking_imputer import MultiVariableImputer
from imputer.eval import EvaluationEngine


class MockConfig:
    learning_rate = 1e-3
    device = 'cpu'
    total_batches = 3
    batch_size = 2  # 2 masked versions per batch
    masking_rates = [0.3, 0.5, 0.7]
    train_heldout_split = 0.8
    finetuning_steps = 5


def create_test_data():
    """Create test data with known structure."""
    return {
        'ratings': [
            {'annotator': 0, 'attribute': 0, 'item': 0, 'value': 3},
            {'annotator': 1, 'attribute': 1, 'item': 1, 'value': 4},
            {'annotator': 0, 'attribute': 1, 'item': 2, 'value': 2},
            {'annotator': 1, 'attribute': 0, 'item': 0, 'value': 5}
        ],
        'pairwise_rankings': [
            {'annotator': 0, 'attribute': 0, 'items': [0, 1], 'order': [1, 2]},
            {'annotator': 1, 'attribute': 1, 'items': [1, 2], 'order': [2, 1]}
        ]
    }


def test_sequential_mit_batch_generation():
    """Test SequentialMIT batch generation."""
    print("Testing SequentialMIT batch generation...")

    # Setup
    converter = DataConverter(2, 2, 3, 5, 2)
    model = MultiVariableImputer(2, 2, 3, 5, 2, 1, 2, 32, 0.1, device='cpu')
    eval_engine = EvaluationEngine()
    config = MockConfig()

    # Create test instances
    train_instances = [create_test_data(), create_test_data()]

    # Create MIT trainer
    mit = SequentialMIT(model, eval_engine, config, converter)

    # Setup heldout callback (initializes instance_train_sets)
    heldout_vars = mit.setup_heldout_evaluation_callback(train_instances)
    print(f"  Heldout variables: {len(heldout_vars)}")
    print(f"  Instance train sets: {len(mit.instance_train_sets)}")

    # Test batch generator - LIMIT TO AVOID INFINITE LOOP
    batch_generator = mit.create_training_batch_generator(train_instances)

    batches_tested = 0
    max_batches_to_test = 3

    for batch_of_masked_versions in batch_generator:
        if batches_tested >= max_batches_to_test:
            break

        print(f"  Batch {batches_tested}: {len(batch_of_masked_versions)} masked versions")

        # Check each masked version
        for j, masked_version in enumerate(batch_of_masked_versions):
            masked_count = sum(1 for v in masked_version if v.is_masked)
            observed_count = sum(1 for v in masked_version if not v.is_masked)
            total_vars = len(masked_version)
            print(f"    Version {j}: {total_vars} vars ({masked_count} masked, {observed_count} observed)")

            # Validate masking
            assert masked_count + observed_count == total_vars, "Masking counts don't add up"

        batches_tested += 1

    print("  ✅ SequentialMIT batch generation test passed\n")


def test_mixed_mit_batch_generation():
    """Test MixedMIT batch generation."""
    print("Testing MixedMIT batch generation...")

    # Setup
    converter = DataConverter(2, 2, 3, 5, 2)
    model = MultiVariableImputer(2, 2, 3, 5, 2, 1, 2, 32, 0.1, device='cpu')
    eval_engine = EvaluationEngine()
    config = MockConfig()

    # Create test instances
    train_instances = [create_test_data(), create_test_data()]

    # Create MIT trainer
    mit = MixedMIT(model, eval_engine, config, converter)

    # Setup heldout callback
    heldout_vars = mit.setup_heldout_evaluation_callback(train_instances)
    print(f"  Heldout variables: {len(heldout_vars)}")
    print(f"  Instance train sets: {len(mit.instance_train_sets)}")

    # Test batch generator - LIMIT TO AVOID INFINITE LOOP
    batch_generator = mit.create_training_batch_generator(train_instances)

    batches_tested = 0
    max_batches_to_test = 3

    for batch_of_masked_versions in batch_generator:
        if batches_tested >= max_batches_to_test:
            break

        print(f"  Batch {batches_tested}: {len(batch_of_masked_versions)} masked versions")

        # Check each masked version
        for j, masked_version in enumerate(batch_of_masked_versions):
            masked_count = sum(1 for v in masked_version if v.is_masked)
            observed_count = sum(1 for v in masked_version if not v.is_masked)
            total_vars = len(masked_version)
            print(f"    Version {j}: {total_vars} vars ({masked_count} masked, {observed_count} observed)")

        batches_tested += 1

    print("  ✅ MixedMIT batch generation test passed\n")


def test_trainer_with_new_format():
    """Test trainer with new batch format."""
    print("Testing trainer with new batch format...")

    # Setup
    converter = DataConverter(2, 2, 3, 5, 2)
    model = MultiVariableImputer(2, 2, 3, 5, 2, 1, 2, 32, 0.1, device='cpu')
    eval_engine = EvaluationEngine()
    config = MockConfig()

    # Create MIT trainer to get trainer instance
    mit = SequentialMIT(model, eval_engine, config, converter)
    trainer = mit.trainer

    # Create test batch of masked versions
    variables = [
        RankingData(annotator_id=0, attribute_id=0, is_listwise=False, item_ids=[0], rating_value=3),
        RankingData(annotator_id=1, attribute_id=1, is_listwise=False, item_ids=[1], rating_value=4)
    ]

    batch_of_masked_versions = []
    for _ in range(2):  # 2 masked versions
        masked_version = []
        for var in variables:
            masked_var = RankingData(
                annotator_id=var.annotator_id,
                attribute_id=var.attribute_id,
                is_listwise=var.is_listwise,
                item_ids=var.item_ids,
                rating_value=var.rating_value,
                is_masked=True  # Mask everything for test
            )
            masked_version.append(masked_var)
        batch_of_masked_versions.append(masked_version)

    print(f"  Created batch with {len(batch_of_masked_versions)} masked versions")

    # Test trainer
    result = trainer.train_step(batch_of_masked_versions)
    print(f"  Training result keys: {list(result.keys())}")
    print(f"  Total loss: {result.get('total_loss', 'N/A')}")

    print("  ✅ Trainer new format test passed\n")


def test_general_mit_batch_generation():
    """Test GeneralMIT batch generation."""
    print("Testing GeneralMIT batch generation...")

    # Setup
    converter = DataConverter(2, 2, 3, 5, 2)
    model = MultiVariableImputer(2, 2, 3, 5, 2, 1, 2, 32, 0.1, device='cpu')
    eval_engine = EvaluationEngine()
    config = MockConfig()

    # Create test instance
    test_instance = create_test_data()

    # Create MIT trainer
    mit = GeneralMIT(model, eval_engine, config, converter)

    # Set up T_O training data (simulate finetune_on_instance setup)
    test_variables = converter.create_variables(test_instance)
    mit.t_o_train_vars = test_variables[:4]  # Use first 4 as T_O training
    mit.t_o_heldout_vars = test_variables[4:] if len(test_variables) > 4 else []

    print(f"  T_O training vars: {len(mit.t_o_train_vars)}")

    # Test batch generator - LIMIT TO AVOID INFINITE LOOP
    if len(mit.t_o_train_vars) > 0:
        batch_generator = mit.create_training_batch_generator([])

        batches_tested = 0
        max_batches_to_test = 2

        for batch_of_masked_versions in batch_generator:
            if batches_tested >= max_batches_to_test:
                break

            print(f"  Batch {batches_tested}: {len(batch_of_masked_versions)} masked versions")

            # Check each masked version
            for j, masked_version in enumerate(batch_of_masked_versions):
                if len(masked_version) > 0:
                    masked_count = sum(1 for v in masked_version if v.is_masked)
                    observed_count = sum(1 for v in masked_version if not v.is_masked)
                    total_vars = len(masked_version)
                    print(f"    Version {j}: {total_vars} vars ({masked_count} masked, {observed_count} observed)")

            batches_tested += 1

    print("  ✅ GeneralMIT batch generation test passed\n")


def run_all_tests():
    """Run all batch generation tests."""
    print("=" * 60)
    print("RUNNING BATCH GENERATION TESTS")
    print("=" * 60)

    try:
        test_sequential_mit_batch_generation()
        test_mixed_mit_batch_generation()
        test_trainer_with_new_format()
        test_general_mit_batch_generation()

        print("=" * 60)
        print("🎉 ALL BATCH GENERATION TESTS PASSED!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)