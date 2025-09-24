"""
Posterior predictive extraction and evaluation utilities.

Extracts predictive samples from MCMC output and evaluates predictions
against ground truth for missing ratings and pairwise rankings.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cmdstanpy

from .bundle import GroundTruthBundle


class PredictiveResults:
    """Container for predictive results and evaluation metrics."""
    
    def __init__(
        self,
        missing_rating_predictions: np.ndarray,
        missing_rating_probs: np.ndarray,
        missing_pairwise_predictions: np.ndarray,
        missing_pairwise_logits: np.ndarray,
        log_lik_ratings_obs: np.ndarray,
        log_lik_pairwise_obs: np.ndarray,
        total_log_lik: np.ndarray,
        metrics: Dict[str, float]
    ):
        self.missing_rating_predictions = missing_rating_predictions
        self.missing_rating_probs = missing_rating_probs
        self.missing_pairwise_predictions = missing_pairwise_predictions
        self.missing_pairwise_logits = missing_pairwise_logits
        self.log_lik_ratings_obs = log_lik_ratings_obs
        self.log_lik_pairwise_obs = log_lik_pairwise_obs
        self.total_log_lik = total_log_lik
        self.metrics = metrics


def extract_predictives_from_fit(fit: cmdstanpy.CmdStanMCMC) -> Dict[str, np.ndarray]:
    """
    Extract posterior predictive samples from MCMC fit.
    
    Args:
        fit: CmdStanMCMC fit object
    
    Returns:
        Dictionary of predictive arrays
    """
    # Get all samples (shape: [chains * samples, variables])
    samples = fit.stan_variables()
    
    predictives = {
        "missing_rating_predictions": samples["missing_rating_predictions"],
        "missing_rating_probs": samples["missing_rating_probs"],
        "log_lik_ratings_obs": samples["log_lik_ratings_obs"],
        "log_lik_pairwise_obs": samples["log_lik_pairwise_obs"],
        "total_log_lik": samples["total_log_lik"],
    }
    
    # Handle optional pairwise variables (may not exist if N_missing_pairwise_rankings = 0)
    if "missing_pairwise_ranking_predictions" in samples:
        predictives["missing_pairwise_ranking_predictions"] = samples["missing_pairwise_ranking_predictions"]
        predictives["missing_pairwise_logits"] = samples["missing_pairwise_logits"]
    else:
        # Create empty arrays for consistency
        n_samples = len(samples["log_lik_ratings_obs"])
        predictives["missing_pairwise_ranking_predictions"] = np.array([]).reshape(n_samples, 0)
        predictives["missing_pairwise_logits"] = np.array([]).reshape(n_samples, 0)
    
    return predictives


def evaluate_rating_predictions(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    ground_truth: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Evaluate rating predictions against ground truth.
    
    Args:
        predictions: Array of predicted ratings [n_samples, n_missing_ratings]
        probabilities: Array of rating probabilities [n_samples, n_missing_ratings, C]
        ground_truth: List of ground truth missing ratings
        config: Configuration dict with C (number of rating categories)
    
    Returns:
        Dictionary of evaluation metrics
    """
    C = config["C"]
    n_samples, n_missing = predictions.shape
    
    # Convert ground truth to array
    gt_ratings = np.array([r["value"] for r in ground_truth])
    
    # Compute accuracy (mode of posterior predictions)
    posterior_mode = np.zeros(n_missing, dtype=int)
    for i in range(n_missing):
        # Find most frequent prediction across samples
        unique, counts = np.unique(predictions[:, i], return_counts=True)
        posterior_mode[i] = unique[np.argmax(counts)]
    
    accuracy = np.mean(posterior_mode == gt_ratings)
    
    # Compute mean absolute error
    mae = np.mean(np.abs(predictions - gt_ratings[np.newaxis, :]))
    
    # Compute log-likelihood of ground truth under posterior (average per rating)
    log_lik = 0.0
    for i, gt_rating in enumerate(gt_ratings):
        # Average log probability across samples
        avg_log_prob = np.mean(np.log(probabilities[:, i, gt_rating - 1] + 1e-10))
        log_lik += avg_log_prob
    
    # Normalize per rating to report average log-likelihood
    if n_missing > 0:
        log_lik /= n_missing

    # Compute calibration metrics
    # For each rating category, check if predicted probabilities are well-calibrated
    calibration_errors = []
    for c in range(C):
        mask = gt_ratings == (c + 1)
        if np.sum(mask) > 0:
            # Average predicted probability for this category
            avg_pred_prob = np.mean(probabilities[:, mask, c])
            # Actual frequency
            actual_freq = np.mean(mask)
            calibration_errors.append(abs(avg_pred_prob - actual_freq))
    
    mean_calibration_error = np.mean(calibration_errors) if calibration_errors else 0.0
    
    return {
        "rating_accuracy": accuracy,
        "rating_mae": mae,
        "rating_log_likelihood": log_lik,
        "rating_calibration_error": mean_calibration_error,
        "n_missing_ratings": n_missing,
    }


def evaluate_pairwise_predictions(
    predictions: np.ndarray,
    logits: np.ndarray,
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Evaluate pairwise ranking predictions against ground truth.
    
    Args:
        predictions: Array of predicted pairwise orders [n_samples, n_missing_pairwise]
        logits: Array of Bradley-Terry logits [n_samples, n_missing_pairwise]
        ground_truth: List of ground truth missing pairwise rankings
    
    Returns:
        Dictionary of evaluation metrics
    """
    n_samples, n_missing = predictions.shape
    
    # Convert ground truth to array
    gt_orders = np.array([p["order"][0] for p in ground_truth])  # 1 if item1 > item2, 2 if item2 > item1
    
    # Compute accuracy (mode of posterior predictions)
    posterior_mode = np.zeros(n_missing, dtype=int)
    for i in range(n_missing):
        # Find most frequent prediction across samples
        unique, counts = np.unique(predictions[:, i], return_counts=True)
        posterior_mode[i] = unique[np.argmax(counts)]
    
    accuracy = np.mean(posterior_mode == gt_orders)
    
    # Compute log-likelihood of ground truth under posterior
    log_lik = 0.0
    for i, gt_order in enumerate(gt_orders):
        # Bradley-Terry log-likelihood
        if gt_order == 1:  # item1 > item2
            avg_log_prob = np.mean(np.log(1 / (1 + np.exp(-logits[:, i])) + 1e-10))
        else:  # item2 > item1
            avg_log_prob = np.mean(np.log(1 / (1 + np.exp(logits[:, i])) + 1e-10))
        log_lik += avg_log_prob
    
    # Compute AUC-like metric for pairwise predictions
    # For each sample, compute the probability that item1 > item2
    prob_item1_better = 1 / (1 + np.exp(-logits))  # sigmoid of logits
    avg_prob_item1_better = np.mean(prob_item1_better, axis=0)
    
    # Convert ground truth to binary: 1 if item1 > item2, 0 if item2 > item1
    gt_binary = (gt_orders == 1).astype(float)
    
    # Compute AUC (simple implementation without sklearn)
    try:
        # Sort by predicted probabilities
        sorted_indices = np.argsort(avg_prob_item1_better)
        sorted_labels = gt_binary[sorted_indices]
        
        # Count positive and negative examples
        n_pos = np.sum(sorted_labels)
        n_neg = len(sorted_labels) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            auc = 0.5  # Random performance
        else:
            # Calculate AUC using trapezoidal rule
            ranks = np.arange(1, len(sorted_labels) + 1)
            pos_ranks = ranks[sorted_labels == 1]
            auc = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    except:
        auc = 0.5  # Fallback to random performance
    
    return {
        "pairwise_accuracy": accuracy,
        "pairwise_log_likelihood": log_lik,
        "pairwise_auc": auc,
        "n_missing_pairwise": n_missing,
    }


def evaluate_predictives(
    fit: cmdstanpy.CmdStanMCMC,
    bundle: GroundTruthBundle,
    config: Dict[str, Any]
) -> PredictiveResults:
    """
    Extract and evaluate posterior predictives against ground truth.
    
    Args:
        fit: CmdStanMCMC fit object
        bundle: Ground truth data bundle
        config: Configuration dict
    
    Returns:
        PredictiveResults object with predictions and metrics
    """
    # Extract predictives
    predictives = extract_predictives_from_fit(fit)
    
    # Evaluate rating predictions
    rating_metrics = evaluate_rating_predictions(
        predictives["missing_rating_predictions"],
        predictives["missing_rating_probs"],
        bundle.missing_ratings,
        config
    )
    
    # Evaluate pairwise predictions (if any exist)
    if len(bundle.missing_pairwise) > 0 and predictives["missing_pairwise_ranking_predictions"].shape[1] > 0:
        pairwise_metrics = evaluate_pairwise_predictions(
            predictives["missing_pairwise_ranking_predictions"],
            predictives["missing_pairwise_logits"],
            bundle.missing_pairwise
        )
    else:
        # No missing pairwise rankings to evaluate
        pairwise_metrics = {
            "pairwise_accuracy": 0.0,
            "pairwise_log_likelihood": 0.0,
            "pairwise_auc": 0.5,
            "n_missing_pairwise": 0,
        }
    
    # Combine metrics (namespace missing-data metrics)
    rating_missing_metrics = {
        "rating_missing_accuracy": rating_metrics.get("rating_accuracy", 0.0),
        "rating_missing_mae": rating_metrics.get("rating_mae", 0.0),
        "rating_missing_log_likelihood": rating_metrics.get("rating_log_likelihood", 0.0),
        "rating_missing_calibration_error": rating_metrics.get("rating_calibration_error", 0.0),
        "n_missing_ratings": rating_metrics.get("n_missing_ratings", 0),
    }
    pairwise_missing_metrics = {
        "pairwise_missing_accuracy": pairwise_metrics.get("pairwise_accuracy", 0.0),
        "pairwise_missing_log_likelihood": pairwise_metrics.get("pairwise_log_likelihood", 0.0),
        "pairwise_missing_auc": pairwise_metrics.get("pairwise_auc", 0.5),
        "n_missing_pairwise": pairwise_metrics.get("n_missing_pairwise", 0),
    }
    all_metrics = {**rating_missing_metrics, **pairwise_missing_metrics}
    
    # Add log-likelihood metrics
    all_metrics.update({
        "log_lik_ratings_obs_mean": np.mean(predictives["log_lik_ratings_obs"]),
        "log_lik_ratings_obs_std": np.std(predictives["log_lik_ratings_obs"]),
        "log_lik_pairwise_obs_mean": np.mean(predictives["log_lik_pairwise_obs"]),
        "log_lik_pairwise_obs_std": np.std(predictives["log_lik_pairwise_obs"]),
        "total_log_lik_mean": np.mean(predictives["total_log_lik"]),
        "total_log_lik_std": np.std(predictives["total_log_lik"]),
    })

    # Normalized observed log-likelihoods (per observation)
    n_obs_ratings = max(1, len(bundle.observed_ratings))
    n_obs_pairwise = max(1, len(bundle.observed_pairwise))
    all_metrics.update({
        "log_lik_ratings_obs_per_rating_mean": float(np.mean(predictives["log_lik_ratings_obs"]) / n_obs_ratings),
        "log_lik_ratings_obs_per_rating_std": float(np.std(predictives["log_lik_ratings_obs"]) / n_obs_ratings),
        "log_lik_pairwise_obs_per_pair_mean": float(np.mean(predictives["log_lik_pairwise_obs"]) / n_obs_pairwise),
        "log_lik_pairwise_obs_per_pair_std": float(np.std(predictives["log_lik_pairwise_obs"]) / n_obs_pairwise),
    })

    # Observed pairwise accuracy (sanity check) using posterior base_scores
    try:
        samples = fit.stan_variables()
        if "base_scores" in samples and len(bundle.observed_pairwise) > 0:
            base_scores = samples["base_scores"]  # shape: [draws, I*J, K]
            temperature = float(config.get("temperature", 1.0))
            draws = base_scores.shape[0]
            gt_orders = np.array([p["order"][0] for p in bundle.observed_pairwise])
            ij_indices = np.array([( (p["attribute"]-1) * config["J"] + p["annotator"]) - 1 for p in bundle.observed_pairwise], dtype=int)
            item1_idx = np.array([p["items"][0]-1 for p in bundle.observed_pairwise], dtype=int)
            item2_idx = np.array([p["items"][1]-1 for p in bundle.observed_pairwise], dtype=int)

            # For efficiency, compute logits per draw in a loop to limit memory
            predicted_orders_per_draw = np.empty((draws, len(bundle.observed_pairwise)), dtype=int)
            for d in range(draws):
                bs = base_scores[d]  # [I*J, K]
                logits = (bs[ij_indices, item1_idx] - bs[ij_indices, item2_idx]) / temperature
                predicted_orders_per_draw[d] = (logits > 0).astype(int) + 1  # 1 if item1>item2 else 2

            # Posterior mode per observation
            posterior_mode = np.zeros(len(bundle.observed_pairwise), dtype=int)
            for n in range(len(bundle.observed_pairwise)):
                unique, counts = np.unique(predicted_orders_per_draw[:, n], return_counts=True)
                posterior_mode[n] = unique[np.argmax(counts)]

            observed_pairwise_accuracy = float(np.mean(posterior_mode == gt_orders))
            all_metrics["pairwise_observed_accuracy"] = observed_pairwise_accuracy
        else:
            all_metrics["pairwise_observed_accuracy"] = 0.0
    except Exception:
        # Be conservative; don't fail evaluation due to this diagnostic metric
        all_metrics["pairwise_observed_accuracy"] = 0.0

    # Add observed-data sanity counts
    all_metrics.update({
        "n_observed_ratings": len(bundle.observed_ratings),
        "n_observed_pairwise": len(bundle.observed_pairwise),
    })
    
    return PredictiveResults(
        missing_rating_predictions=predictives["missing_rating_predictions"],
        missing_rating_probs=predictives["missing_rating_probs"],
        missing_pairwise_predictions=predictives["missing_pairwise_ranking_predictions"],
        missing_pairwise_logits=predictives["missing_pairwise_logits"],
        log_lik_ratings_obs=predictives["log_lik_ratings_obs"],
        log_lik_pairwise_obs=predictives["log_lik_pairwise_obs"],
        total_log_lik=predictives["total_log_lik"],
        metrics=all_metrics
    )


def save_predictives(run_dir: Path, results: PredictiveResults) -> None:
    """Save predictive results to files."""
    # Save metrics
    from .io import save_json
    save_json(results.metrics, run_dir / "predictive_metrics.json")
    
    # Save predictions as CSV
    n_samples, n_missing_ratings = results.missing_rating_predictions.shape
    n_samples, n_missing_pairwise = results.missing_pairwise_predictions.shape
    
    # Create DataFrames for easier analysis
    rating_df = pd.DataFrame({
        'sample': np.repeat(range(n_samples), n_missing_ratings),
        'missing_rating_idx': np.tile(range(n_missing_ratings), n_samples),
        'predicted_rating': results.missing_rating_predictions.flatten(),
    })
    
    pairwise_df = pd.DataFrame({
        'sample': np.repeat(range(n_samples), n_missing_pairwise),
        'missing_pairwise_idx': np.tile(range(n_missing_pairwise), n_samples),
        'predicted_order': results.missing_pairwise_predictions.flatten(),
        'logit': results.missing_pairwise_logits.flatten(),
    })
    
    # Save to CSV
    rating_df.to_csv(run_dir / "rating_predictions.csv", index=False)
    pairwise_df.to_csv(run_dir / "pairwise_predictions.csv", index=False)
    
    # Save probability distributions
    prob_df = pd.DataFrame(
        results.missing_rating_probs.reshape(n_samples * n_missing_ratings, -1),
        columns=[f'prob_cat_{c+1}' for c in range(results.missing_rating_probs.shape[2])]
    )
    prob_df['sample'] = np.repeat(range(n_samples), n_missing_ratings)
    prob_df['missing_rating_idx'] = np.tile(range(n_missing_ratings), n_samples)
    prob_df.to_csv(run_dir / "rating_probabilities.csv", index=False)
