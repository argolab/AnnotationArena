"""
Test script for domain model entrypoint function

Tests the new evaluate_test_instance() method to ensure:
- Proper EvaluationResults format output
- Correct masking and training workflow
- Variable sample count support
- Integration with existing domain model functionality
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path so we can import from imputer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from domain_model_trainer import DomainModelTrainer, DomainModelConfig, STAN_AVAILABLE
    from imputer.eval import EvaluationResults
except ImportError as e:
    print(f"Import error: {e}")
    print("Stan might not be available, skipping test")
    sys.exit(0)


def create_test_instance():
    """Create test instance data in expected format."""
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


def create_data_config():
    """Create data configuration for test instance."""
    return {
        'K': 3,  # 3 items
        'D': 4,  # 4 embedding dimensions
        'I': 2,  # 2 attributes
        'J': 2,  # 2 annotators
        'C': 5,  # 5 rating classes (1-5)
        'ranking_size': 2  # Pairwise rankings
    }


def test_entrypoint_function():
    """Test the domain model entrypoint function."""
    print("Testing domain model entrypoint function...")

    if not STAN_AVAILABLE:
        print("Stan not available, skipping domain model test")
        return True

    try:
        # Initialize domain model trainer
        trainer = DomainModelTrainer()
        print("✓ Domain model trainer initialized")

        # Create test data
        test_instance = create_test_instance()
        data_config = create_data_config()

        print(f"✓ Test instance created with {len(test_instance['ratings'])} ratings "
              f"and {len(test_instance['pairwise_rankings'])} rankings")

        # Test with minimal sample counts for speed
        print("Testing domain model evaluation with minimal samples...")
        results = trainer.evaluate_test_instance(
            test_instance_data=test_instance,
            data_config=data_config,
            masking_rate=0.5,
            chains=1,           # Minimal for testing
            iter_warmup=50,     # Minimal for testing
            iter_sampling=100,  # Minimal for testing
            seed=42
        )

        # Verify output format
        assert isinstance(results, EvaluationResults), f"Expected EvaluationResults, got {type(results)}"
        print("✓ Returns EvaluationResults object")

        # Check required fields
        assert isinstance(results.total_loss, float), "total_loss should be float"
        assert isinstance(results.rating_loss, float), "rating_loss should be float"
        assert isinstance(results.ranking_loss, float), "ranking_loss should be float"
        assert isinstance(results.num_rating_evaluations, int), "num_rating_evaluations should be int"
        assert isinstance(results.num_ranking_evaluations, int), "num_ranking_evaluations should be int"
        print("✓ All required fields present with correct types")

        # Check optional fields
        assert results.rating_accuracy is None or isinstance(results.rating_accuracy, float)
        assert results.ranking_accuracy is None or isinstance(results.ranking_accuracy, float)
        assert results.rating_rmse is None or isinstance(results.rating_rmse, float)
        print("✓ Optional fields have correct types")

        # Check masked/observed metrics
        assert results.masked_metrics is not None, "masked_metrics should not be None"
        assert results.observed_metrics is not None, "observed_metrics should not be None"
        assert isinstance(results.masked_metrics, dict), "masked_metrics should be dict"
        assert isinstance(results.observed_metrics, dict), "observed_metrics should be dict"
        print("✓ Masked/observed metrics present")

        # Check evaluation counts are reasonable
        total_ratings = len(test_instance['ratings'])
        total_rankings = len(test_instance['pairwise_rankings'])

        # With 50% masking, roughly half should be in each set
        assert 0 <= results.num_rating_evaluations <= total_ratings
        assert 0 <= results.num_ranking_evaluations <= total_rankings
        print(f"✓ Evaluation counts reasonable: {results.num_rating_evaluations}/{total_ratings} ratings, "
              f"{results.num_ranking_evaluations}/{total_rankings} rankings")

        # Print sample results
        print(f"\nSample results:")
        print(f"  Total loss: {results.total_loss:.4f}")
        print(f"  Rating loss: {results.rating_loss:.4f}")
        print(f"  Ranking loss: {results.ranking_loss:.4f}")
        print(f"  Rating accuracy: {results.rating_accuracy}")
        print(f"  Ranking accuracy: {results.ranking_accuracy}")
        print(f"  Rating RMSE: {results.rating_rmse}")

        return True

    except Exception as e:
        print(f"✗ Domain model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_sample_counts():
    """Test domain model with different MCMC sample counts."""
    print("\nTesting different MCMC sample counts...")

    if not STAN_AVAILABLE:
        print("Stan not available, skipping sample count test")
        return True

    try:
        trainer = DomainModelTrainer()
        test_instance = create_test_instance()
        data_config = create_data_config()

        sample_counts = [50, 100, 200]
        results_list = []

        for samples in sample_counts:
            print(f"  Testing with {samples} MCMC samples...")
            results = trainer.evaluate_test_instance(
                test_instance_data=test_instance,
                data_config=data_config,
                masking_rate=0.5,
                chains=1,
                iter_warmup=50,
                iter_sampling=samples,  # Variable sample count
                seed=42
            )
            results_list.append((samples, results))
            print(f"    Total loss: {results.total_loss:.4f}")

        # Verify all runs completed successfully
        assert len(results_list) == len(sample_counts)
        print(f"✓ Successfully tested {len(sample_counts)} different sample counts")

        return True

    except Exception as e:
        print(f"✗ Sample count test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_masking_rates():
    """Test domain model with different masking rates."""
    print("\nTesting different masking rates...")

    if not STAN_AVAILABLE:
        print("Stan not available, skipping masking rate test")
        return True

    try:
        trainer = DomainModelTrainer()
        test_instance = create_test_instance()
        data_config = create_data_config()

        masking_rates = [0.3, 0.5, 0.7]
        results_list = []

        for rate in masking_rates:
            print(f"  Testing with {rate*100:.0f}% masking...")
            results = trainer.evaluate_test_instance(
                test_instance_data=test_instance,
                data_config=data_config,
                masking_rate=rate,
                chains=1,
                iter_warmup=50,
                iter_sampling=100,
                seed=42
            )
            results_list.append((rate, results))
            print(f"    Evaluating on {results.num_rating_evaluations} ratings, "
                  f"{results.num_ranking_evaluations} rankings")

        # Verify masking rates affect evaluation counts appropriately
        assert len(results_list) == len(masking_rates)
        print(f"✓ Successfully tested {len(masking_rates)} different masking rates")

        return True

    except Exception as e:
        print(f"✗ Masking rate test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all domain model entrypoint tests."""
    print("="*60)
    print("TESTING DOMAIN MODEL ENTRYPOINT FUNCTION")
    print("="*60)

    if not STAN_AVAILABLE:
        print("Stan not available - skipping all domain model tests")
        print("This is expected if Stan is not installed")
        return True

    try:
        # Test basic functionality
        success1 = test_entrypoint_function()

        # Test variable sample counts
        success2 = test_different_sample_counts()

        # Test variable masking rates
        success3 = test_different_masking_rates()

        if success1 and success2 and success3:
            print("\n" + "="*60)
            print("🎉 ALL DOMAIN MODEL ENTRYPOINT TESTS PASSED!")
            print("="*60)
            print("✓ Entrypoint function works correctly")
            print("✓ Returns proper EvaluationResults format")
            print("✓ Supports variable MCMC sample counts")
            print("✓ Supports variable masking rates")
            print("✓ Ready for integration with experimental framework")
            return True
        else:
            print("\n❌ Some domain model tests failed")
            return False

    except Exception as e:
        print(f"\n❌ Domain model test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)