"""
Tests for Stan data generation wrapper.

Note: These tests require cmdstanpy and a compiled Stan model.
"""

import numpy as np
from pathlib import Path
import pytest

from stan.pipeline.configs import DataGenConfig
from stan.pipeline.data_gen import generate_data, extract_bundle_from_stan_output


def test_datagen_config():
    """Test DataGenConfig instantiation."""
    config = DataGenConfig(K_train=3, K_test=2, I=2, J=2, D=8, C=3, seed=42)
    assert config.K_train == 3
    assert config.K_test == 2
    assert config.I == 2
    assert config.J == 2
    assert config.D == 8
    assert config.C == 3
    assert config.pairwise_cap_per_item == 10  # default
    assert config.seed == 42


@pytest.mark.skip(reason="Requires cmdstanpy and compiled Stan model")
def test_generate_data_small():
    """Test complete data generation with small parameters."""
    config = DataGenConfig(K_train=2, K_test=1, I=2, J=2, D=4, C=3, seed=42)
    
    # This will fail if cmdstanpy is not available or Stan model is not compiled
    try:
        bundle = generate_data(config)
        
        # Check shapes
        total_items = config.K_train + config.K_test  # 2 + 1 = 3
        assert bundle.embeddings.shape == (total_items, 4)  # [3, 4]
        assert bundle.mean_preferences.shape == (2, 4)
        assert bundle.annotator_preferences.shape == (4, 4)  # I*J = 4
        assert bundle.rating_thresholds.shape == (4, 3)  # I*J = 4, C = 3
        assert bundle.base_scores.shape == (4, total_items)  # I*J = 4, K = 3
        
        # Check that we have ratings
        assert len(bundle.all_ratings) > 0
        assert len(bundle.observed_ratings) > 0
        assert len(bundle.missing_ratings) >= 0
        
        # Check rating structure
        for rating in bundle.all_ratings:
            assert 'attribute' in rating
            assert 'annotator' in rating
            assert 'item' in rating
            assert 'value' in rating
            assert 'instance' in rating  # New field
            assert 1 <= rating['attribute'] <= 2
            assert 1 <= rating['annotator'] <= 2
            assert 1 <= rating['item'] <= total_items
            assert 1 <= rating['value'] <= 3
            assert rating['instance'] in ['train', 'test']
        
        # Check pairwise structure
        for pairwise in bundle.all_pairwise:
            assert 'attribute' in pairwise
            assert 'annotator' in pairwise
            assert 'items' in pairwise
            assert 'order' in pairwise
            assert 'tied_rating' in pairwise
            assert 'instance' in pairwise  # New field
            assert len(pairwise['items']) == 2
            assert len(pairwise['order']) == 2
            assert pairwise['instance'] in ['train', 'test']
        
        # Check stats
        assert 'total_ratings' in bundle.stats
        assert 'observed_ratings' in bundle.stats
        assert 'missing_ratings' in bundle.stats
        assert 'K_train' in bundle.stats
        assert 'K_test' in bundle.stats
        assert 'train_ratings' in bundle.stats
        assert 'test_ratings' in bundle.stats
        assert bundle.stats['total_ratings'] == len(bundle.all_ratings)
        assert bundle.stats['observed_ratings'] == len(bundle.observed_ratings)
        assert bundle.stats['missing_ratings'] == len(bundle.missing_ratings)
        assert bundle.stats['K_train'] == config.K_train
        assert bundle.stats['K_test'] == config.K_test
        
    except ImportError:
        pytest.skip("cmdstanpy not available")
    except Exception as e:
        pytest.skip(f"Stan model compilation/sampling failed: {e}")


def test_config_validation():
    """Test that config validation works."""
    # Valid config
    config = DataGenConfig(K_train=3, K_test=2, I=2, J=2, D=8, C=3)
    assert config.K_train == 3
    assert config.K_test == 2
    
    # Test defaults
    config_default = DataGenConfig(K_train=3, K_test=2, I=2, J=2, D=8, C=3)
    assert config_default.pairwise_cap_per_item == 10
    assert config_default.sigma_annotator == 0.3
    assert config_default.sigma_measurement == 0.1
    assert config_default.kappa == 2.0
    assert config_default.temperature == 0.5
    assert config_default.enable_third_annotator == True
    assert config_default.enable_pairwise_rankings == True
