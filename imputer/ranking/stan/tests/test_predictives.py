"""
Tests for predictives extraction and evaluation utilities.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stan.pipeline.predictives import (
    PredictiveResults, 
    extract_predictives_from_fit,
    evaluate_rating_predictions,
    evaluate_pairwise_predictions,
    evaluate_predictives
)
from stan.pipeline.bundle import GroundTruthBundle


class MockFit:
    """Mock CmdStanMCMC fit object for testing."""
    
    def __init__(self, samples):
        self.samples = samples
    
    def stan_variables(self):
        return self.samples


def test_predictive_results():
    """Test PredictiveResults creation."""
    n_samples = 100
    n_missing_ratings = 5
    n_missing_pairwise = 3
    C = 5
    
    results = PredictiveResults(
        missing_rating_predictions=np.random.randint(1, C+1, (n_samples, n_missing_ratings)),
        missing_rating_probs=np.random.rand(n_samples, n_missing_ratings, C),
        missing_pairwise_predictions=np.random.randint(1, 3, (n_samples, n_missing_pairwise)),
        missing_pairwise_logits=np.random.randn(n_samples, n_missing_pairwise),
        log_lik_ratings_obs=np.random.randn(n_samples),
        log_lik_pairwise_obs=np.random.randn(n_samples),
        total_log_lik=np.random.randn(n_samples),
        metrics={"accuracy": 0.8, "mae": 0.5}
    )
    
    assert results.missing_rating_predictions.shape == (n_samples, n_missing_ratings)
    assert results.missing_rating_probs.shape == (n_samples, n_missing_ratings, C)
    assert results.missing_pairwise_predictions.shape == (n_samples, n_missing_pairwise)
    assert results.missing_pairwise_logits.shape == (n_samples, n_missing_pairwise)
    assert results.log_lik_ratings_obs.shape == (n_samples,)
    assert results.log_lik_pairwise_obs.shape == (n_samples,)
    assert results.total_log_lik.shape == (n_samples,)
    assert results.metrics["accuracy"] == 0.8


def test_extract_predictives_from_fit():
    """Test extracting predictives from mock fit."""
    n_samples = 50
    n_missing_ratings = 3
    n_missing_pairwise = 2
    C = 5
    
    mock_samples = {
        "missing_rating_predictions": np.random.randint(1, C+1, (n_samples, n_missing_ratings)),
        "missing_rating_probs": np.random.rand(n_samples, n_missing_ratings, C),
        "missing_pairwise_ranking_predictions": np.random.randint(1, 3, (n_samples, n_missing_pairwise)),
        "missing_pairwise_logits": np.random.randn(n_samples, n_missing_pairwise),
        "log_lik_ratings_obs": np.random.randn(n_samples),
        "log_lik_pairwise_obs": np.random.randn(n_samples),
        "total_log_lik": np.random.randn(n_samples),
    }
    
    mock_fit = MockFit(mock_samples)
    predictives = extract_predictives_from_fit(mock_fit)
    
    assert "missing_rating_predictions" in predictives
    assert "missing_rating_probs" in predictives
    assert "missing_pairwise_ranking_predictions" in predictives
    assert "missing_pairwise_logits" in predictives
    assert "log_lik_ratings_obs" in predictives
    assert "log_lik_pairwise_obs" in predictives
    assert "total_log_lik" in predictives
    
    assert predictives["missing_rating_predictions"].shape == (n_samples, n_missing_ratings)
    assert predictives["missing_rating_probs"].shape == (n_samples, n_missing_ratings, C)


def test_evaluate_rating_predictions():
    """Test rating prediction evaluation."""
    n_samples = 100
    n_missing_ratings = 3
    C = 5
    
    # Create mock predictions
    predictions = np.random.randint(1, C+1, (n_samples, n_missing_ratings))
    probabilities = np.random.rand(n_samples, n_missing_ratings, C)
    # Normalize probabilities
    probabilities = probabilities / probabilities.sum(axis=2, keepdims=True)
    
    # Create mock ground truth
    ground_truth = [
        {"attribute": 1, "annotator": 1, "item": 1, "value": 3, "instance": "train"},
        {"attribute": 1, "annotator": 2, "item": 2, "value": 2, "instance": "train"},
        {"attribute": 2, "annotator": 1, "item": 1, "value": 4, "instance": "test"},
    ]
    
    config = {"C": C}
    
    metrics = evaluate_rating_predictions(predictions, probabilities, ground_truth, config)
    
    assert "rating_accuracy" in metrics
    assert "rating_mae" in metrics
    assert "rating_log_likelihood" in metrics
    assert "rating_calibration_error" in metrics
    assert "n_missing_ratings" in metrics
    
    assert 0 <= metrics["rating_accuracy"] <= 1
    assert metrics["rating_mae"] >= 0
    assert metrics["n_missing_ratings"] == 3


def test_evaluate_pairwise_predictions():
    """Test pairwise prediction evaluation."""
    n_samples = 100
    n_missing_pairwise = 2
    
    # Create mock predictions
    predictions = np.random.randint(1, 3, (n_samples, n_missing_pairwise))
    logits = np.random.randn(n_samples, n_missing_pairwise)
    
    # Create mock ground truth
    ground_truth = [
        {"attribute": 1, "annotator": 1, "items": [1, 2], "order": [1, 2], "tied_rating": 3, "instance": "train"},
        {"attribute": 1, "annotator": 2, "items": [2, 3], "order": [2, 1], "tied_rating": 2, "instance": "test"},
    ]
    
    metrics = evaluate_pairwise_predictions(predictions, logits, ground_truth)
    
    assert "pairwise_accuracy" in metrics
    assert "pairwise_log_likelihood" in metrics
    assert "pairwise_auc" in metrics
    assert "n_missing_pairwise" in metrics
    
    assert 0 <= metrics["pairwise_accuracy"] <= 1
    assert 0 <= metrics["pairwise_auc"] <= 1
    assert metrics["n_missing_pairwise"] == 2


def test_evaluate_predictives():
    """Test end-to-end predictive evaluation."""
    n_samples = 50
    n_missing_ratings = 2
    n_missing_pairwise = 1
    C = 5
    
    # Create mock fit
    mock_samples = {
        "missing_rating_predictions": np.random.randint(1, C+1, (n_samples, n_missing_ratings)),
        "missing_rating_probs": np.random.rand(n_samples, n_missing_ratings, C),
        "missing_pairwise_ranking_predictions": np.random.randint(1, 3, (n_samples, n_missing_pairwise)),
        "missing_pairwise_logits": np.random.randn(n_samples, n_missing_pairwise),
        "log_lik_ratings_obs": np.random.randn(n_samples),
        "log_lik_pairwise_obs": np.random.randn(n_samples),
        "total_log_lik": np.random.randn(n_samples),
    }
    
    mock_fit = MockFit(mock_samples)
    
    # Create mock bundle
    bundle = GroundTruthBundle(
        embeddings=np.random.randn(5, 4),
        mean_preferences=np.random.randn(2, 4),
        annotator_preferences=np.random.randn(6, 4),
        rating_probs=[np.random.dirichlet([1, 1, 1, 1, 1]) for _ in range(6)],
        rating_thresholds=[np.random.randn(5) for _ in range(6)],
        base_scores=np.random.randn(6, 5),
        all_ratings=[],
        all_pairwise=[],
        observed_ratings=[],
        missing_ratings=[
            {"attribute": 1, "annotator": 1, "item": 1, "value": 3, "instance": "train"},
            {"attribute": 1, "annotator": 2, "item": 2, "value": 2, "instance": "train"},
        ],
        observed_pairwise=[],
        missing_pairwise=[
            {"attribute": 1, "annotator": 1, "items": [1, 2], "order": [1, 2], "tied_rating": 3, "instance": "train"},
        ],
        stats={"K_train": 3, "K_test": 2}
    )
    
    config = {"C": C}
    
    results = evaluate_predictives(mock_fit, bundle, config)
    
    assert isinstance(results, PredictiveResults)
    assert results.missing_rating_predictions.shape == (n_samples, n_missing_ratings)
    assert results.missing_pairwise_predictions.shape == (n_samples, n_missing_pairwise)
    assert "rating_accuracy" in results.metrics
    assert "pairwise_accuracy" in results.metrics
    assert "log_lik_ratings_obs_mean" in results.metrics


if __name__ == "__main__":
    pytest.main([__file__])
