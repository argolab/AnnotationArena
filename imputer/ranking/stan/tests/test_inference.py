"""
Tests for inference utilities.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stan.pipeline.inference import InferenceConfig, prepare_stan_data_for_inference, create_init_from_ground_truth
from stan.pipeline.configs import DataGenConfig
from stan.pipeline.bundle import GroundTruthBundle


def test_inference_config():
    """Test InferenceConfig creation and defaults."""
    config = InferenceConfig()
    
    assert config.chains == 4
    assert config.iter_warmup == 1000
    assert config.iter_sampling == 1000
    assert config.seed is None
    assert config.adapt_delta == 0.8
    assert config.max_treedepth == 10
    assert config.init_strategy == "random"
    assert config.init_file is None
    assert config.show_progress is True


def test_inference_config_custom():
    """Test InferenceConfig with custom values."""
    config = InferenceConfig(
        chains=2,
        iter_warmup=500,
        iter_sampling=500,
        seed=42,
        adapt_delta=0.9,
        max_treedepth=12,
        init_strategy="ground_truth",
        show_progress=False
    )
    
    assert config.chains == 2
    assert config.iter_warmup == 500
    assert config.iter_sampling == 500
    assert config.seed == 42
    assert config.adapt_delta == 0.9
    assert config.max_treedepth == 12
    assert config.init_strategy == "ground_truth"
    assert config.show_progress is False


def test_prepare_stan_data_train_only():
    """Test preparing Stan data for train-only inference."""
    # Create mock data
    config = DataGenConfig(K_train=3, K_test=2, I=2, J=3, D=4, C=3)
    
    bundle = GroundTruthBundle(
        embeddings=np.random.randn(5, 4),  # K_train + K_test = 5
        mean_preferences=np.random.randn(2, 4),
        annotator_preferences=np.random.randn(6, 4),  # I*J = 6
        rating_probs=[np.random.dirichlet([1, 1, 1]) for _ in range(6)],
        rating_thresholds=[np.random.randn(3) for _ in range(6)],
        base_scores=np.random.randn(6, 5),
        all_ratings=[
            {"attribute": 1, "annotator": 1, "item": 1, "value": 2, "instance": "train"},
            {"attribute": 1, "annotator": 1, "item": 2, "value": 3, "instance": "train"},
            {"attribute": 1, "annotator": 2, "item": 1, "value": 1, "instance": "test"},
        ],
        all_pairwise=[
            {"attribute": 1, "annotator": 1, "items": [1, 2], "order": [1, 2], "tied_rating": 2, "instance": "train"},
            {"attribute": 1, "annotator": 1, "items": [3, 4], "order": [2, 1], "tied_rating": 3, "instance": "test"},
        ],
        observed_ratings=[
            {"attribute": 1, "annotator": 1, "item": 1, "value": 2, "instance": "train"},
            {"attribute": 1, "annotator": 2, "item": 1, "value": 1, "instance": "test"},
        ],
        missing_ratings=[
            {"attribute": 1, "annotator": 1, "item": 2, "value": 3, "instance": "train"},
        ],
        observed_pairwise=[
            {"attribute": 1, "annotator": 1, "items": [1, 2], "order": [1, 2], "tied_rating": 2, "instance": "train"},
        ],
        missing_pairwise=[
            {"attribute": 1, "annotator": 1, "items": [3, 4], "order": [2, 1], "tied_rating": 3, "instance": "test"},
        ],
        stats={"K_train": 3, "K_test": 2}
    )
    
    # Test train-only data preparation
    stan_data = prepare_stan_data_for_inference(bundle, config, use_train_only=True)
    
    assert stan_data["K"] == 3  # Only training items
    assert stan_data["I"] == 2
    assert stan_data["J"] == 3
    assert stan_data["D"] == 4
    assert stan_data["C"] == 3
    
    # Should only have training observed ratings
    assert stan_data["N_ratings"] == 1
    assert stan_data["rating_attributes"] == [1]
    assert stan_data["rating_annotators"] == [1]
    assert stan_data["rating_items"] == [1]
    assert stan_data["rating_values"] == [2]
    
    # Should only have training observed pairwise
    assert stan_data["N_pairwise_rankings"] == 1
    assert stan_data["pairwise_ranking_attributes"] == [1]
    assert stan_data["pairwise_ranking_annotators"] == [1]
    assert stan_data["pairwise_ranking_items"] == [[1, 2]]
    assert stan_data["pairwise_ranking_orders"] == [1]


def test_prepare_stan_data_test_only():
    """Test preparing Stan data for test-only inference."""
    # Create mock data (same as above)
    config = DataGenConfig(K_train=3, K_test=2, I=2, J=3, D=4, C=3)
    
    bundle = GroundTruthBundle(
        embeddings=np.random.randn(5, 4),
        mean_preferences=np.random.randn(2, 4),
        annotator_preferences=np.random.randn(6, 4),
        rating_probs=[np.random.dirichlet([1, 1, 1]) for _ in range(6)],
        rating_thresholds=[np.random.randn(3) for _ in range(6)],
        base_scores=np.random.randn(6, 5),
        all_ratings=[
            {"attribute": 1, "annotator": 1, "item": 1, "value": 2, "instance": "train"},
            {"attribute": 1, "annotator": 2, "item": 1, "value": 1, "instance": "test"},
        ],
        all_pairwise=[
            {"attribute": 1, "annotator": 1, "items": [1, 2], "order": [1, 2], "tied_rating": 2, "instance": "train"},
            {"attribute": 1, "annotator": 1, "items": [3, 4], "order": [2, 1], "tied_rating": 3, "instance": "test"},
        ],
        observed_ratings=[
            {"attribute": 1, "annotator": 1, "item": 1, "value": 2, "instance": "train"},
            {"attribute": 1, "annotator": 2, "item": 1, "value": 1, "instance": "test"},
        ],
        missing_ratings=[],
        observed_pairwise=[
            {"attribute": 1, "annotator": 1, "items": [1, 2], "order": [1, 2], "tied_rating": 2, "instance": "train"},
            {"attribute": 1, "annotator": 1, "items": [3, 4], "order": [2, 1], "tied_rating": 3, "instance": "test"},
        ],
        missing_pairwise=[],
        stats={"K_train": 3, "K_test": 2}
    )
    
    # Test test-only data preparation
    stan_data = prepare_stan_data_for_inference(bundle, config, use_test_only=True)
    
    assert stan_data["K"] == 2  # Only test items
    assert stan_data["N_ratings"] == 1
    assert stan_data["rating_attributes"] == [1]
    assert stan_data["rating_annotators"] == [2]
    assert stan_data["rating_items"] == [1]  # Item 1 in test instance
    assert stan_data["rating_values"] == [1]


def test_prepare_stan_data_both():
    """Test preparing Stan data for both train and test (transductive)."""
    # Create mock data
    config = DataGenConfig(K_train=3, K_test=2, I=2, J=3, D=4, C=3)
    
    bundle = GroundTruthBundle(
        embeddings=np.random.randn(5, 4),
        mean_preferences=np.random.randn(2, 4),
        annotator_preferences=np.random.randn(6, 4),
        rating_probs=[np.random.dirichlet([1, 1, 1]) for _ in range(6)],
        rating_thresholds=[np.random.randn(3) for _ in range(6)],
        base_scores=np.random.randn(6, 5),
        all_ratings=[
            {"attribute": 1, "annotator": 1, "item": 1, "value": 2, "instance": "train"},
            {"attribute": 1, "annotator": 2, "item": 1, "value": 1, "instance": "test"},
        ],
        all_pairwise=[
            {"attribute": 1, "annotator": 1, "items": [1, 2], "order": [1, 2], "tied_rating": 2, "instance": "train"},
            {"attribute": 1, "annotator": 1, "items": [3, 4], "order": [2, 1], "tied_rating": 3, "instance": "test"},
        ],
        observed_ratings=[
            {"attribute": 1, "annotator": 1, "item": 1, "value": 2, "instance": "train"},
            {"attribute": 1, "annotator": 2, "item": 1, "value": 1, "instance": "test"},
        ],
        missing_ratings=[],
        observed_pairwise=[
            {"attribute": 1, "annotator": 1, "items": [1, 2], "order": [1, 2], "tied_rating": 2, "instance": "train"},
            {"attribute": 1, "annotator": 1, "items": [3, 4], "order": [2, 1], "tied_rating": 3, "instance": "test"},
        ],
        missing_pairwise=[],
        stats={"K_train": 3, "K_test": 2}
    )
    
    # Test both train and test data preparation
    stan_data = prepare_stan_data_for_inference(bundle, config, use_train_only=False, use_test_only=False)
    
    assert stan_data["K"] == 5  # Both train and test items
    assert stan_data["N_ratings"] == 2
    assert stan_data["rating_attributes"] == [1, 1]
    assert stan_data["rating_annotators"] == [1, 2]
    assert stan_data["rating_items"] == [1, 1]  # Item 1 in train, item 1 in test
    assert stan_data["rating_values"] == [2, 1]


def test_create_init_from_ground_truth():
    """Test creating initialization from ground truth."""
    config = DataGenConfig(K_train=3, K_test=2, I=2, J=3, D=4, C=3)
    
    bundle = GroundTruthBundle(
        embeddings=np.random.randn(5, 4),
        mean_preferences=np.random.randn(2, 4),
        annotator_preferences=np.random.randn(6, 4),
        rating_probs=[np.random.dirichlet([1, 1, 1]) for _ in range(6)],
        rating_thresholds=[np.random.randn(3) for _ in range(6)],
        base_scores=np.random.randn(6, 5),
        all_ratings=[],
        all_pairwise=[],
        observed_ratings=[],
        missing_ratings=[],
        observed_pairwise=[],
        missing_pairwise=[],
        stats={"K_train": 3, "K_test": 2}
    )
    
    # Test train-only initialization
    init_values = create_init_from_ground_truth(bundle, config, use_train_only=True)
    
    assert "embeddings" in init_values
    assert "mean_preferences" in init_values
    assert "annotator_preferences" in init_values
    assert "rating_probs" in init_values
    
    assert init_values["embeddings"].shape == (3, 4)  # Only training embeddings
    assert init_values["mean_preferences"].shape == (2, 4)
    assert init_values["annotator_preferences"].shape == (6, 4)
    assert len(init_values["rating_probs"]) == 6
    
    # Test test-only initialization
    init_values = create_init_from_ground_truth(bundle, config, use_test_only=True)
    assert init_values["embeddings"].shape == (2, 4)  # Only test embeddings
    
    # Test both train and test initialization
    init_values = create_init_from_ground_truth(bundle, config, use_train_only=False, use_test_only=False)
    assert init_values["embeddings"].shape == (5, 4)  # Both train and test embeddings


if __name__ == "__main__":
    pytest.main([__file__])
