#!/usr/bin/env python3
"""Domain model for ICLR experiments with incremental MCMC sampling."""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import cmdstanpy as stan
    STAN_AVAILABLE = True
    logger.info("Using cmdstanpy for Stan interface")
except ImportError:
    STAN_AVAILABLE = False
    logger.error("cmdstanpy not available - please install: conda install -c conda-forge cmdstanpy")

@dataclass
class DomainModelResults:
    """Results from domain model evaluation."""
    total_log_loss: float
    rating_log_loss: float
    ranking_log_loss: float
    rating_accuracy: float
    ranking_accuracy: float
    rating_rmse: float
    num_rating_predictions: int
    num_ranking_predictions: int
    mcmc_samples: int
    wall_time: float

class DomainModelICLR:
    """Domain model for ICLR experiments with incremental MCMC evaluation."""

    def __init__(self, config):
        self.config = config
        # MCMC sample points for incremental evaluation
        self.sample_points = [100, 200, 500, 1000, 2000, 3000]

    def evaluate_test_instance(self, test_idx: int, observed_vars: List, masked_vars: List) -> Dict:
        """Evaluate domain model on test instance with incremental MCMC sampling."""
        logger.info(f"Domain model evaluation on test instance {test_idx}")

        # Load test instance data
        instance_dir = self.config.get_instance_data_dir(test_idx)
        train_file = instance_dir / "iclr_dataset_train.json"
        test_file = instance_dir / "iclr_dataset_test.json"

        with open(train_file, 'r') as f:
            train_data = json.load(f)
        with open(test_file, 'r') as f:
            test_data = json.load(f)

        # Combine all data
        full_data = {
            'ratings': train_data['ratings'] + test_data['ratings'],
            'pairwise_rankings': train_data['pairwise_rankings'] + test_data['pairwise_rankings']
        }

        # Extract observed data only
        observed_data = self._extract_observed_data(full_data, observed_vars)

        # Setup Stan model and data
        stan_data = self._prepare_stan_data(observed_data, test_idx)

        # Run MCMC with incremental evaluation
        results = {}
        cumulative_time = 0.0

        for num_samples in self.sample_points:
            start_time = time.time()

            # Run MCMC for this sample count
            logger.info(f"Running MCMC with {num_samples} samples...")
            stan_results = self._run_mcmc_samples(stan_data, num_samples)

            sample_time = time.time() - start_time
            cumulative_time += sample_time

            # Evaluate predictions on masked variables
            metrics = self._evaluate_predictions(
                stan_results, full_data, masked_vars, test_idx
            )

            results[num_samples] = DomainModelResults(
                total_log_loss=metrics['total_log_loss'],
                rating_log_loss=metrics['rating_log_loss'],
                ranking_log_loss=metrics['ranking_log_loss'],
                rating_accuracy=metrics['rating_accuracy'],
                ranking_accuracy=metrics['ranking_accuracy'],
                rating_rmse=metrics['rating_rmse'],
                num_rating_predictions=metrics['num_rating_predictions'],
                num_ranking_predictions=metrics['num_ranking_predictions'],
                mcmc_samples=num_samples,
                wall_time=cumulative_time
            )

            logger.info(f"Domain model {num_samples} samples: "
                       f"Total loss={metrics['total_log_loss']:.4f}, "
                       f"Time={cumulative_time:.2f}s")

        return results

    def _extract_observed_data(self, full_data: Dict, observed_vars: List) -> Dict:
        """Extract data corresponding to observed variables only."""
        observed_data = {'ratings': [], 'pairwise_rankings': []}

        for var in observed_vars:
            if var['type'] == 'rating':
                # Find matching rating
                for rating in full_data['ratings']:
                    if (rating['attribute'] == var['attribute'] and
                        rating['annotator'] == var['annotator'] and
                        rating['item'] == var['item']):
                        observed_data['ratings'].append(rating)
                        break
            else:
                # Find matching ranking
                for ranking in full_data['pairwise_rankings']:
                    if (ranking['attribute'] == var['attribute'] and
                        ranking['annotator'] == var['annotator'] and
                        ranking['items'] == var['items']):
                        observed_data['pairwise_rankings'].append(ranking)
                        break

        logger.info(f"Observed data: {len(observed_data['ratings'])} ratings, "
                   f"{len(observed_data['pairwise_rankings'])} rankings")
        return observed_data

    def _prepare_stan_data(self, data: Dict, instance_idx: int) -> Dict:
        """Prepare data for Stan model."""
        # Get instance configuration
        instance_config = self.config.instances[instance_idx]

        ratings = data['ratings']
        rankings = data['pairwise_rankings']

        # Calculate ranking_size
        ranking_size = len(rankings[0]['items']) if rankings else 2

        # Build Stan data structure (following legacy format exactly)
        stan_data = {
            # Dimensions
            'K': instance_config.K,
            'I': instance_config.I,
            'J': instance_config.J,
            'D': instance_config.D,
            'C': instance_config.C,
            'ranking_size': ranking_size,

            # Ratings
            'N_ratings': len(ratings),
            'rating_attributes': [r['attribute'] for r in ratings],
            'rating_annotators': [r['annotator'] for r in ratings],
            'rating_items': [r['item'] for r in ratings],
            'rating_values': [r['value'] for r in ratings],

            # Comparisons (not used but required by Stan model)
            'N_comparisons': 0,
            'comparison_attributes': [],
            'comparison_annotators': [],
            'comparison_items_a': [],
            'comparison_items_b': [],
            'comparison_results': [],

            # Rankings
            'N_rankings': len(rankings),
            'ranking_attributes': [r['attribute'] for r in rankings],
            'ranking_annotators': [r['annotator'] for r in rankings],
            'ranking_items': [r['items'] for r in rankings],
            'ranking_orders': [r['order'] for r in rankings],

            # Hyperparameters
            'sigma_annotator': instance_config.sigma_annotator,
            'sigma_measurement': instance_config.sigma_measurement,
            'alpha_dirichlet': instance_config.alpha_dirichlet,
            'temperature': instance_config.temperature,
            'sigma_embedding_prior': instance_config.sigma_embedding_prior,
            'sigma_preference_prior': instance_config.sigma_preference_prior
        }

        return stan_data

    def _run_mcmc_samples(self, stan_data: Dict, num_samples: int) -> Dict:
        """Run MCMC sampling for specified number of samples using actual Stan model."""
        if not STAN_AVAILABLE:
            raise RuntimeError("cmdstanpy not available - cannot run domain model")

        # Use the actual Stan model for domain inference
        model_path = Path(__file__).parent / "models" / "domain_model.stan"
        if not model_path.exists():
            raise FileNotFoundError(f"Stan model not found at {model_path}")

        # Compile and run Stan model
        model = stan.CmdStanModel(stan_file=str(model_path))

        logger.info(f"Running Stan MCMC with {num_samples} samples...")
        fit = model.sample(
            data=stan_data,
            chains=1,
            iter_warmup=500,
            iter_sampling=num_samples,
            adapt_delta=0.95,
            max_treedepth=10,
            show_progress=False
        )

        # Extract posterior samples
        results = {
            'item_embeddings': fit.stan_variable('embeddings'),
            'annotator_preferences': fit.stan_variable('annotator_preferences'),
            'thresholds': fit.stan_variable('rating_thresholds'),
            'measurement_noise': np.full(num_samples, stan_data['sigma_measurement'])  # Use fixed noise from input
        }

        return results

    def _evaluate_predictions(self, stan_results: Dict, full_data: Dict,
                            masked_vars: List, instance_idx: int) -> Dict:
        """Evaluate model predictions on masked variables."""
        total_rating_log_loss = 0.0
        total_ranking_log_loss = 0.0
        rating_correct = 0
        ranking_correct = 0
        rating_mse = 0.0
        num_rating_preds = 0
        num_ranking_preds = 0

        for var in masked_vars:
            if var['type'] == 'rating':
                # Find true rating value
                true_value = None
                for rating in full_data['ratings']:
                    if (rating['attribute'] == var['attribute'] and
                        rating['annotator'] == var['annotator'] and
                        rating['item'] == var['item']):
                        true_value = rating['value']
                        break

                if true_value is not None:
                    # Predict rating using posterior samples
                    pred_probs = self._predict_rating(
                        stan_results, var['item'], var['attribute'], var['annotator']
                    )

                    # Calculate log loss
                    log_loss = -np.log(pred_probs[true_value - 1] + 1e-10)
                    total_rating_log_loss += log_loss

                    # Calculate accuracy
                    pred_rating = np.argmax(pred_probs) + 1
                    if pred_rating == true_value:
                        rating_correct += 1

                    # Calculate MSE for RMSE
                    rating_mse += (pred_rating - true_value) ** 2

                    num_rating_preds += 1

            else:  # ranking
                # Find true ranking
                true_order = None
                for ranking in full_data['pairwise_rankings']:
                    if (ranking['attribute'] == var['attribute'] and
                        ranking['annotator'] == var['annotator'] and
                        ranking['items'] == var['items']):
                        true_order = ranking['order']
                        break

                if true_order is not None:
                    # Predict ranking using posterior samples
                    pred_prob = self._predict_ranking(
                        stan_results, var['items'], var['attribute'], var['annotator']
                    )

                    # Calculate log loss
                    true_first_wins = true_order[0] < true_order[1]
                    if true_first_wins:
                        log_loss = -np.log(pred_prob + 1e-10)
                    else:
                        log_loss = -np.log(1 - pred_prob + 1e-10)
                    total_ranking_log_loss += log_loss

                    # Calculate accuracy
                    pred_first_wins = pred_prob > 0.5
                    if pred_first_wins == true_first_wins:
                        ranking_correct += 1

                    num_ranking_preds += 1

        # Calculate averages
        avg_rating_log_loss = total_rating_log_loss / max(num_rating_preds, 1)
        avg_ranking_log_loss = total_ranking_log_loss / max(num_ranking_preds, 1)
        total_log_loss = avg_rating_log_loss + avg_ranking_log_loss

        rating_accuracy = rating_correct / max(num_rating_preds, 1)
        ranking_accuracy = ranking_correct / max(num_ranking_preds, 1)
        rating_rmse = np.sqrt(rating_mse / max(num_rating_preds, 1))

        return {
            'total_log_loss': total_log_loss,
            'rating_log_loss': avg_rating_log_loss,
            'ranking_log_loss': avg_ranking_log_loss,
            'rating_accuracy': rating_accuracy,
            'ranking_accuracy': ranking_accuracy,
            'rating_rmse': rating_rmse,
            'num_rating_predictions': num_rating_preds,
            'num_ranking_predictions': num_ranking_preds
        }

    def _predict_rating(self, stan_results: Dict, item: int, attribute: int, annotator: int) -> np.ndarray:
        """Predict rating probabilities using posterior samples."""
        num_samples = stan_results['item_embeddings'].shape[0]
        C = stan_results['thresholds'].shape[3] + 1

        # Get posterior samples for this prediction
        item_emb = stan_results['item_embeddings'][:, item-1, :]  # [num_samples, D]
        annotator_pref = stan_results['annotator_preferences'][:, annotator-1, attribute-1, :]  # [num_samples, D]
        thresholds = stan_results['thresholds'][:, annotator-1, attribute-1, :]  # [num_samples, C-1]
        measurement_noise = stan_results['measurement_noise']  # [num_samples]

        # Calculate utility for each sample
        utilities = np.sum(item_emb * annotator_pref, axis=1)  # [num_samples]

        # Convert to probabilities using Gaussian CDF (like old code)
        probs_per_sample = []
        for s in range(num_samples):
            base_score = utilities[s]
            thresh = thresholds[s]
            sigma_m = measurement_noise[s]

            # Create full threshold boundaries: [-∞, Q_1, Q_2, ..., Q_{C-1}, +∞]
            full_thresholds = np.concatenate([
                [-np.inf],
                thresh,
                [np.inf]
            ])

            # Calculate probabilities for each category using Gaussian CDF
            from scipy.stats import norm
            probs = np.zeros(C)

            for c in range(C):
                # For category c (0-indexed), we want boundaries at indices c and c+1
                upper_thresh = full_thresholds[c + 1]
                lower_thresh = full_thresholds[c]

                if upper_thresh == np.inf:
                    upper_prob = 1.0
                else:
                    upper_prob = norm.cdf((upper_thresh - base_score) / sigma_m)

                if lower_thresh == -np.inf:
                    lower_prob = 0.0
                else:
                    lower_prob = norm.cdf((lower_thresh - base_score) / sigma_m)

                probs[c] = max(upper_prob - lower_prob, 1e-10)  # Numerical stability

            # Normalize
            probs = probs / np.sum(probs)
            probs_per_sample.append(probs)

        # Average over samples
        avg_probs = np.mean(probs_per_sample, axis=0)
        return avg_probs

    def _predict_ranking(self, stan_results: Dict, items: List[int], attribute: int, annotator: int) -> float:
        """Predict ranking probability (first item wins) using posterior samples."""
        num_samples = stan_results['item_embeddings'].shape[0]

        item1_emb = stan_results['item_embeddings'][:, items[0]-1, :]  # [num_samples, D]
        item2_emb = stan_results['item_embeddings'][:, items[1]-1, :]  # [num_samples, D]
        annotator_pref = stan_results['annotator_preferences'][:, annotator-1, attribute-1, :]  # [num_samples, D]

        # Calculate utilities
        utility1 = np.sum(item1_emb * annotator_pref, axis=1)  # [num_samples]
        utility2 = np.sum(item2_emb * annotator_pref, axis=1)  # [num_samples]

        # Probability that item1 > item2
        prob_per_sample = 1 / (1 + np.exp(utility2 - utility1))

        # Average over samples
        avg_prob = np.mean(prob_per_sample)
        return avg_prob