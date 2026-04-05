"""
Posterior predictive extraction and evaluation utilities.

Extracts predictive samples from MCMC output and evaluates predictions
against ground truth for missing ratings and pairwise rankings.

Missing rating log-loss is always evaluated on test missing ratings only,
indexed via bundle.missing_ratings_indexes_in_test_instance.
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
    """Extract posterior predictive samples from MCMC fit."""
    samples = fit.stan_variables()

    predictives = {
        "missing_rating_predictions": samples["missing_rating_predictions"],
        "missing_rating_probs":       samples["missing_rating_probs"],
        "log_lik_ratings_obs":        samples["log_lik_ratings_obs"],
        "log_lik_pairwise_obs":       samples["log_lik_pairwise_obs"],
        "total_log_lik":              samples["total_log_lik"],
    }

    if "missing_pairwise_ranking_predictions" in samples:
        predictives["missing_pairwise_ranking_predictions"] = samples["missing_pairwise_ranking_predictions"]
        predictives["missing_pairwise_logits"]              = samples["missing_pairwise_logits"]
    else:
        n_samples = len(samples["log_lik_ratings_obs"])
        predictives["missing_pairwise_ranking_predictions"] = np.array([]).reshape(n_samples, 0)
        predictives["missing_pairwise_logits"]              = np.array([]).reshape(n_samples, 0)

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
        predictions:  [n_samples, n_test_missing]  — posterior predictive draws
        probabilities:[n_samples, n_test_missing, C]
        ground_truth: list of test missing rating dicts (instance == "test")
        config:       dict with at least "C"
    """
    C = config["C"]
    n_samples, n_missing = predictions.shape

    gt_ratings = np.array([r["value"] for r in ground_truth])
    assert gt_ratings.shape == (n_missing,), \
        f"gt_ratings shape mismatch: {gt_ratings.shape} != ({n_missing},)"

    # Posterior mode accuracy
    posterior_mode = np.zeros(n_missing, dtype=int)
    for i in range(n_missing):
        unique, counts = np.unique(predictions[:, i], return_counts=True)
        posterior_mode[i] = unique[np.argmax(counts)]
    accuracy = float(np.mean(posterior_mode == gt_ratings))

    # Mean absolute error
    mae = float(np.mean(np.abs(predictions - gt_ratings[np.newaxis, :])))

    # Per-rating log-likelihood: log E_posterior[p(c* | theta)]
    assert probabilities.shape == (n_samples, n_missing, C), \
        f"probabilities shape mismatch: {probabilities.shape}"
    log_lik = 0.0
    for i, gt_rating in enumerate(gt_ratings):
        avg_prob = np.mean(probabilities[:, i, gt_rating - 1])
        log_lik += np.log(avg_prob + 1e-10)
    log_lik = log_lik / n_missing if n_missing > 0 else 0.0

    # Calibration error
    calibration_errors = []
    for c in range(C):
        mask = gt_ratings == (c + 1)
        avg_pred_prob = np.mean(probabilities[:, mask, c]) if mask.any() else 0.0
        actual_freq   = float(np.mean(mask))
        calibration_errors.append(abs(avg_pred_prob - actual_freq))
    mean_calibration_error = float(np.mean(calibration_errors)) if calibration_errors else 0.0

    return {
        "rating_accuracy":           accuracy,
        "rating_mae":                mae,
        "rating_log_likelihood":     log_lik,
        "rating_calibration_error":  mean_calibration_error,
        "n_missing_ratings":         n_missing,
    }


def evaluate_pairwise_predictions(
    predictions: np.ndarray,
    logits: np.ndarray,
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Evaluate pairwise ranking predictions against ground truth.

    Args:
        predictions: [n_samples, n_missing_pairwise]
        logits:      [n_samples, n_missing_pairwise]
        ground_truth: list of missing pairwise dicts
    """
    n_samples, n_missing = predictions.shape
    gt_orders = np.array([p["order"][0] for p in ground_truth])

    # Posterior mode accuracy
    posterior_mode = np.zeros(n_missing, dtype=int)
    for i in range(n_missing):
        unique, counts = np.unique(predictions[:, i], return_counts=True)
        posterior_mode[i] = unique[np.argmax(counts)]
    accuracy = float(np.mean(posterior_mode == gt_orders))

    # Log-likelihood
    log_lik = 0.0
    for i, gt_order in enumerate(gt_orders):
        if gt_order == 1:
            avg_log_prob = float(np.mean(np.log(1 / (1 + np.exp(-logits[:, i])) + 1e-10)))
        else:
            avg_log_prob = float(np.mean(np.log(1 / (1 + np.exp( logits[:, i])) + 1e-10)))
        log_lik += avg_log_prob

    # AUC
    prob_item1_better = 1 / (1 + np.exp(-logits))
    avg_prob_item1_better = np.mean(prob_item1_better, axis=0)
    gt_binary = (gt_orders == 1).astype(float)
    try:
        sorted_indices = np.argsort(avg_prob_item1_better)
        sorted_labels  = gt_binary[sorted_indices]
        n_pos = np.sum(sorted_labels)
        n_neg = len(sorted_labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            auc = 0.5
        else:
            ranks     = np.arange(1, len(sorted_labels) + 1)
            pos_ranks = ranks[sorted_labels == 1]
            auc = float((np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    except Exception:
        auc = 0.5

    return {
        "pairwise_accuracy":        accuracy,
        "pairwise_log_likelihood":  log_lik,
        "pairwise_auc":             auc,
        "n_missing_pairwise":       n_missing,
    }


def evaluate_predictives(
    fit: cmdstanpy.CmdStanMCMC,
    bundle: GroundTruthBundle,
    config: Dict[str, Any],
) -> PredictiveResults:
    """
    Extract and evaluate posterior predictives against ground truth.

    Stan was given:
      - observed ratings: instance in ("train", "val")
      - missing ratings:  instance == "test" only

    So Stan's missing_rating_predictions[n] corresponds 1-to-1 with
    the test missing ratings passed at inference time. We use
    bundle.missing_ratings_indexes_in_test_instance to identify which
    entries of bundle.missing_ratings are test ratings, giving us the
    ground truth values to compare against.

    Args:
        fit:    CmdStanMCMC fit object.
        bundle: GroundTruthBundle (full bundle, all instances).
        config: Flat dict with K, J, I, D, C, temperature.
    """
    predictives = extract_predictives_from_fit(fit)

    # ── Test missing ratings (the only ones Stan predicted) ───────────────────
    # missing_ratings_indexes_in_test_instance gives indices into bundle.missing_ratings
    # that belong to the test instance.
    test_missing_indices = bundle.missing_ratings_indexes_in_test_instance
    test_missing_ratings = [bundle.missing_ratings[i] for i in test_missing_indices]
    n_test_missing = len(test_missing_ratings)

    # Stan predictions are indexed 0..N_test_missing-1 (passed in that order)
    stan_indices = list(range(n_test_missing))

    rating_metrics = evaluate_rating_predictions(
        predictives["missing_rating_predictions"][:, stan_indices],
        predictives["missing_rating_probs"][:, stan_indices],
        test_missing_ratings,
        config,
    )

    # ── Pairwise ──────────────────────────────────────────────────────────────
    if len(bundle.missing_pairwise) > 0 and predictives["missing_pairwise_ranking_predictions"].shape[1] > 0:
        pairwise_metrics = evaluate_pairwise_predictions(
            predictives["missing_pairwise_ranking_predictions"],
            predictives["missing_pairwise_logits"],
            bundle.missing_pairwise,
        )
    else:
        pairwise_metrics = {
            "pairwise_accuracy":       0.0,
            "pairwise_log_likelihood": 0.0,
            "pairwise_auc":            0.5,
            "n_missing_pairwise":      0,
        }

    # ── Combine metrics ───────────────────────────────────────────────────────
    all_metrics = {
        "rating_missing_accuracy":          rating_metrics["rating_accuracy"],
        "rating_missing_mae":               rating_metrics["rating_mae"],
        "rating_missing_log_likelihood":    rating_metrics["rating_log_likelihood"],
        "rating_missing_calibration_error": rating_metrics["rating_calibration_error"],
        "n_missing_ratings":                rating_metrics["n_missing_ratings"],
        "pairwise_missing_accuracy":        pairwise_metrics["pairwise_accuracy"],
        "pairwise_missing_log_likelihood":  pairwise_metrics["pairwise_log_likelihood"],
        "pairwise_missing_auc":             pairwise_metrics["pairwise_auc"],
        "n_missing_pairwise":               pairwise_metrics["n_missing_pairwise"],
    }

    # ── Observed log-likelihoods (from Stan generated quantities) ─────────────
    # These are SUMs across all observed ratings per MCMC draw.
    n_obs_ratings  = max(1, len(bundle.observed_ratings))
    n_obs_pairwise = max(1, len(bundle.observed_pairwise))
    all_metrics.update({
        "log_lik_ratings_obs_mean":            float(np.mean(predictives["log_lik_ratings_obs"])),
        "log_lik_ratings_obs_std":             float(np.std( predictives["log_lik_ratings_obs"])),
        "log_lik_pairwise_obs_mean":           float(np.mean(predictives["log_lik_pairwise_obs"])),
        "log_lik_pairwise_obs_std":            float(np.std( predictives["log_lik_pairwise_obs"])),
        "total_log_lik_mean":                  float(np.mean(predictives["total_log_lik"])),
        "total_log_lik_std":                   float(np.std( predictives["total_log_lik"])),
        "log_lik_ratings_obs_per_rating_mean": float(np.mean(predictives["log_lik_ratings_obs"]) / n_obs_ratings),
        "log_lik_ratings_obs_per_rating_std":  float(np.std( predictives["log_lik_ratings_obs"]) / n_obs_ratings),
        "log_lik_pairwise_obs_per_pair_mean":  float(np.mean(predictives["log_lik_pairwise_obs"]) / n_obs_pairwise),
        "log_lik_pairwise_obs_per_pair_std":   float(np.std( predictives["log_lik_pairwise_obs"]) / n_obs_pairwise),
    })

    # ── Observed pairwise accuracy (diagnostic) ───────────────────────────────
    try:
        samples = fit.stan_variables()
        if "base_scores" in samples and len(bundle.observed_pairwise) > 0:
            base_scores = samples["base_scores"]   # [draws, I*J, K]
            temperature = float(config.get("temperature", 1.0))
            J           = config["J"]
            draws       = base_scores.shape[0]

            gt_orders  = np.array([p["order"][0] for p in bundle.observed_pairwise])
            ij_indices = np.array(
                [((p["attribute"] - 1) * J + p["annotator"]) - 1 for p in bundle.observed_pairwise],
                dtype=int,
            )
            item1_idx = np.array([p["items"][0] - 1 for p in bundle.observed_pairwise], dtype=int)
            item2_idx = np.array([p["items"][1] - 1 for p in bundle.observed_pairwise], dtype=int)

            predicted_orders_per_draw = np.empty((draws, len(bundle.observed_pairwise)), dtype=int)
            for d in range(draws):
                bs     = base_scores[d]   # [I*J, K]
                logits = (bs[ij_indices, item1_idx] - bs[ij_indices, item2_idx]) / temperature
                predicted_orders_per_draw[d] = (logits < 0).astype(int) + 1

            posterior_mode = np.zeros(len(bundle.observed_pairwise), dtype=int)
            for n in range(len(bundle.observed_pairwise)):
                unique, counts = np.unique(predicted_orders_per_draw[:, n], return_counts=True)
                posterior_mode[n] = unique[np.argmax(counts)]

            all_metrics["pairwise_observed_accuracy"] = float(np.mean(posterior_mode == gt_orders))
        else:
            all_metrics["pairwise_observed_accuracy"] = 0.0
    except Exception:
        all_metrics["pairwise_observed_accuracy"] = 0.0

    all_metrics.update({
        "n_observed_ratings":  len(bundle.observed_ratings),
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
        metrics=all_metrics,
    )


def save_predictives(run_dir: Path, results: PredictiveResults) -> None:
    """Save predictive results to files."""
    from .io import save_json
    save_json(results.metrics, run_dir / "predictive_metrics.json")

    n_samples, n_missing_ratings  = results.missing_rating_predictions.shape
    n_samples, n_missing_pairwise = results.missing_pairwise_predictions.shape

    rating_df = pd.DataFrame({
        "sample":            np.repeat(range(n_samples), n_missing_ratings),
        "missing_rating_idx": np.tile(range(n_missing_ratings), n_samples),
        "predicted_rating":  results.missing_rating_predictions.flatten(),
    })

    pairwise_df = pd.DataFrame({
        "sample":               np.repeat(range(n_samples), n_missing_pairwise),
        "missing_pairwise_idx": np.tile(range(n_missing_pairwise), n_samples),
        "predicted_order":      results.missing_pairwise_predictions.flatten(),
        "logit":                results.missing_pairwise_logits.flatten(),
    })

    rating_df.to_csv(run_dir / "rating_predictions.csv", index=False)
    pairwise_df.to_csv(run_dir / "pairwise_predictions.csv", index=False)

    prob_df = pd.DataFrame(
        results.missing_rating_probs.reshape(n_samples * n_missing_ratings, -1),
        columns=[f"prob_cat_{c+1}" for c in range(results.missing_rating_probs.shape[2])],
    )
    prob_df["sample"]            = np.repeat(range(n_samples), n_missing_ratings)
    prob_df["missing_rating_idx"] = np.tile(range(n_missing_ratings), n_samples)
    prob_df.to_csv(run_dir / "rating_probabilities.csv", index=False)
