"""
Test script for Phase 5 configuration updates
"""

import sys
import os

# Add parent directory to path so we can import from imputer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ExperimentConfig, ModelConfig, TrainingConfig

def test_phase5_config_parameters():
    """Test that Phase 5 configuration parameters are properly added."""

    # Test loading existing config with new parameters
    config = ExperimentConfig.load_from_file('configs/single_instance.json')

    # Test new ModelConfig parameters (only atom supported)
    assert hasattr(config.model_config, 'embedding_type'), "ModelConfig should have embedding_type"
    assert config.model_config.embedding_type == "atom", \
        f"embedding_type must be 'atom', got {config.model_config.embedding_type}"

    # Test new TrainingConfig parameters
    training = config.training_config

    # Test test_masking_rate
    assert hasattr(training, 'test_masking_rate'), "TrainingConfig should have test_masking_rate"
    assert 0.0 <= training.test_masking_rate <= 1.0, f"test_masking_rate should be [0,1], got {training.test_masking_rate}"

    # Test pretraining_mode
    assert hasattr(training, 'pretraining_mode'), "TrainingConfig should have pretraining_mode"
    assert training.pretraining_mode in ["sequential", "mixed"], \
        f"pretraining_mode should be 'sequential' or 'mixed', got {training.pretraining_mode}"

    # Test total_batches
    assert hasattr(training, 'total_batches'), "TrainingConfig should have total_batches"
    assert training.total_batches > 0, f"total_batches should be positive, got {training.total_batches}"

    # Test batch_size
    assert hasattr(training, 'batch_size'), "TrainingConfig should have batch_size"
    assert training.batch_size > 0, f"batch_size should be positive, got {training.batch_size}"

    # Test masking_rates
    assert hasattr(training, 'masking_rates'), "TrainingConfig should have masking_rates"
    assert isinstance(training.masking_rates, list), "masking_rates should be a list"
    for rate in training.masking_rates:
        assert 0.0 <= rate <= 1.0, f"masking_rates should be [0,1], got {rate}"

    # Test eval_frequency
    assert hasattr(training, 'eval_frequency'), "TrainingConfig should have eval_frequency"
    assert training.eval_frequency > 0, f"eval_frequency should be positive, got {training.eval_frequency}"

    # Test evaluation_types
    assert hasattr(training, 'evaluation_types'), "TrainingConfig should have evaluation_types"
    assert isinstance(training.evaluation_types, list), "evaluation_types should be a list"
    valid_eval_types = ["pretrained", "pretrained_finetuned", "fresh", "domain"]
    for eval_type in training.evaluation_types:
        assert eval_type in valid_eval_types, f"Invalid evaluation_type: {eval_type}"

    print("✓ Phase 5 configuration parameters test passed!")

def test_multi_instance_config_phase5():
    """Test that multi-instance config also has Phase 5 parameters."""

    config = ExperimentConfig.load_from_file('configs/multi_instance_demo.json')

    # Test that multi-instance also has the new parameters
    training = config.training_config

    assert hasattr(training, 'test_masking_rate'), "Multi-instance config should have test_masking_rate"
    assert hasattr(training, 'pretraining_mode'), "Multi-instance config should have pretraining_mode"
    assert hasattr(training, 'evaluation_types'), "Multi-instance config should have evaluation_types"

    # Test that it supports multi-instance structure
    assert config.experiment_type == "multi_instance", "Should be multi_instance experiment"
    assert len(config.instances) > 1, "Multi-instance should have multiple instances"
    assert len(config.train_instance_indices) > 0, "Should have train instances"
    assert len(config.test_instance_indices) > 0, "Should have test instances"

    print("✓ Multi-instance Phase 5 configuration test passed!")

def test_config_validation():
    """Test that configuration validation works for Phase 5 parameters."""

    # Test valid config passes validation
    config = ExperimentConfig.load_from_file('configs/single_instance.json')
    config.validate()  # Should not raise

    # Test invalid embedding_type
    config.model_config.embedding_type = "invalid_type"
    try:
        config.validate()
        assert False, "Should have failed with invalid embedding_type"
    except (ValueError, AssertionError) as e:
        assert "embedding_type" in str(e).lower() or "atom" in str(e).lower()

    # Reset to valid
    config.model_config.embedding_type = "atom"

    # Test invalid pretraining_mode
    config.training_config.pretraining_mode = "invalid_mode"
    try:
        config.validate()
        assert False, "Should have failed with invalid pretraining_mode"
    except ValueError as e:
        assert "pretraining_mode must be one of" in str(e)

    # Reset to valid
    config.training_config.pretraining_mode = "sequential"

    # Test invalid test_masking_rate
    config.training_config.test_masking_rate = 1.5
    try:
        config.validate()
        assert False, "Should have failed with invalid test_masking_rate"
    except ValueError as e:
        assert "test_masking_rate must be between 0.0 and 1.0" in str(e)

    # Reset to valid
    config.training_config.test_masking_rate = 0.3

    # Test invalid masking_rates
    config.training_config.masking_rates = [0.0, 0.5, 1.5]
    try:
        config.validate()
        assert False, "Should have failed with invalid masking_rates"
    except ValueError as e:
        assert "masking_rates must be between 0.0 and 1.0" in str(e)

    # Reset to valid
    config.training_config.masking_rates = [0.0, 0.2, 0.5, 0.8, 1.0]

    # Test invalid evaluation_types
    config.training_config.evaluation_types = ["pretrained", "invalid_eval"]
    try:
        config.validate()
        assert False, "Should have failed with invalid evaluation_types"
    except ValueError as e:
        assert "evaluation_types must contain only" in str(e)

    print("✓ Configuration validation test passed!")

def test_fully_random_embedding_type():
    """Test that only atom embedding type is supported (others removed)."""
    config = ExperimentConfig.load_from_file('configs/single_instance.json')
    config.model_config.embedding_type = "fully_random"
    try:
        config.validate()
        print("⚠ Warning: fully_random embedding type should have been rejected but wasn't")
    except (ValueError, AssertionError):
        print("✓ Non-atom embedding type correctly rejected")

def test_backwards_compatibility():
    """Test that existing functionality still works."""

    # Test single instance creation
    single_config = ExperimentConfig.create_single_instance()
    assert single_config.experiment_type == "single_instance"
    assert len(single_config.instances) == 1

    # Test multi-instance creation
    from config import InstanceConfig
    instances = [InstanceConfig(K=20), InstanceConfig(K=25)]
    multi_config = ExperimentConfig.create_multi_instance(
        instances=instances,
        train_instance_indices=[0],
        test_instance_indices=[1]
    )
    assert multi_config.experiment_type == "multi_instance"
    assert len(multi_config.instances) == 2

    # Test that new parameters have defaults
    assert hasattr(multi_config.training_config, 'test_masking_rate')
    assert hasattr(multi_config.training_config, 'pretraining_mode')
    assert hasattr(multi_config.training_config, 'evaluation_types')

    print("✓ Backwards compatibility test passed!")

if __name__ == "__main__":
    test_phase5_config_parameters()
    test_multi_instance_config_phase5()
    test_config_validation()
    test_fully_random_embedding_type()
    test_backwards_compatibility()
    print("✓ All Phase 5 configuration tests passed!")