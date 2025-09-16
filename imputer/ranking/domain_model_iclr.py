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
        self.chains = config.chains
        self.item_warmup = config.iter_warmup
        self.num_samples = config.iter_sampling
        self.adapt_delta = config.adapt_delta
        self.max_treedepth = config.max_treedepth
        self.evaluation_interval = config.evaluation_interval

    def extract_final_state_as_init(self, fit):
        """Extract the final state from a fit to use as initial values for next iteration."""
        
        final_inits = []

        embeddings_raw_final = fit.stan_variable('embeddings_raw')[-1, :, :]  # [K, D]
        mean_preferences_final = fit.stan_variable('mean_preferences')[-1, :, :]  # [I, D] 
        annotator_preferences_final = fit.stan_variable('annotator_preferences')[-1, :, :]  # [I*J, D]
        rating_thresholds_increments_final = fit.stan_variable('rating_thresholds_increments')[-1, :, :]  # [I*J, C-2]

            
        chain_init = {
            'embeddings_raw': embeddings_raw_final.tolist(),
            'mean_preferences': mean_preferences_final.tolist(),
            'annotator_preferences': annotator_preferences_final.tolist(),
            'rating_thresholds_increments': rating_thresholds_increments_final.tolist()
        }
        final_inits.append(chain_init)
        
        return final_inits

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
        num_samples = self.num_samples
        evaluate_interval = self.evaluation_interval
        current_sample = 0
        fit = None
        while current_sample < num_samples:
            current_sample += evaluate_interval
            start_time = time.time()

            # Run MCMC for this sample count
            logger.info(f"Running MCMC with {num_samples} samples...")

            if fit is None:
                stan_results, fit = self._run_mcmc_samples(stan_data, evaluate_interval)
            else:
                init_values = self.extract_final_state_as_init(fit)
                stan_results, fit = self._run_mcmc_samples(stan_data, evaluate_interval, init=True, init_values=init_values)

            sample_time = time.time() - start_time
            cumulative_time += sample_time

            # Evaluate predictions on masked variables
            metrics = self._evaluate_predictions(
                stan_results, full_data, masked_vars, test_idx
            )

            results[current_sample] = DomainModelResults(
                total_log_loss=metrics['total_log_loss'],
                rating_log_loss=metrics['rating_log_loss'],
                ranking_log_loss=metrics['ranking_log_loss'],
                rating_accuracy=metrics['rating_accuracy'],
                ranking_accuracy=metrics['ranking_accuracy'],
                rating_rmse=metrics['rating_rmse'],
                num_rating_predictions=metrics['num_rating_predictions'],
                num_ranking_predictions=metrics['num_ranking_predictions'],
                mcmc_samples=current_sample,
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

    def _run_mcmc_samples(self, stan_data: Dict, num_samples: int, init: bool=False, init_values=None) -> Dict:
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
        if not init:
            fit = model.sample(
                data=stan_data,
                chains=self.chains,
                iter_warmup=self.item_warmup,
                iter_sampling=num_samples,
                adapt_delta=self.adapt_delta,
                max_treedepth=self.max_treedepth,
                show_progress=True
            )
        else:
            fit = model.sample(
                data=stan_data,
                chains=self.chains,
                iter_warmup=1,
                iter_sampling=num_samples-1,
                adapt_delta=self.adapt_delta,
                inits=init_values,
                max_treedepth=self.max_treedepth,
                show_progress=True
            )

        # Extract posterior samples
        results = {
            'item_embeddings': fit.stan_variable('embeddings'),
            'annotator_preferences': fit.stan_variable('annotator_preferences'),
            'thresholds': fit.stan_variable('rating_thresholds'),
            'measurement_noise': np.full(num_samples, stan_data['sigma_measurement'])  # Use fixed noise from input
        }

        # Clean up compiled Stan executable
        try:
            exe_path = model.exe_file
            if exe_path and Path(exe_path).exists():
                Path(exe_path).unlink()
                logger.debug(f"Cleaned up Stan executable: {exe_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up Stan executable: {e}")

        return results, fit

    def _evaluate_predictions(self, stan_results: Dict, full_data: Dict,
                            masked_vars: List, instance_idx: int) -> Dict:
        """Evaluate model predictions on masked variables (using legacy approach)."""
        # Extract posterior means (like legacy code)
        embeddings = np.mean(stan_results['item_embeddings'], axis=0)  # [K, D]
        preferences = np.mean(stan_results['annotator_preferences'], axis=0)  # [I*J, D]
        thresholds = np.mean(stan_results['thresholds'], axis=0)  # [I*J, C+1]

        # Get config for dimensions
        instance_config = self.config.instances[instance_idx]
        J = instance_config.J
        C = instance_config.C
        sigma_measurement = instance_config.sigma_measurement

        # Compute base scores
        base_scores = preferences @ embeddings.T  # [I*J, K]

        total_rating_log_loss = 0.0
        total_ranking_log_loss = 0.0
        rating_correct = 0
        ranking_correct = 0
        rating_mse = 0.0
        num_rating_preds = 0
        num_ranking_preds = 0

        # Process masked ratings (following legacy code exactly)
        masked_ratings = [var for var in masked_vars if var['type'] == 'rating']
        if masked_ratings:
            rating_predictions = []
            rating_targets = []
            for var in masked_ratings:
                # Find true rating value
                true_value = None
                for rating in full_data['ratings']:
                    if (rating['attribute'] == var['attribute'] and
                        rating['annotator'] == var['annotator'] and
                        rating['item'] == var['item']):
                        true_value = rating['value']
                        break

                if true_value is not None:
                    i, j, k, c_true = var['attribute'], var['annotator'], var['item'], true_value
                    ij_idx = (i-1)*J + (j-1)

                    # Predict rating using posterior mean (like legacy)
                    base_score = base_scores[ij_idx, k-1]
                    pref_norm = np.linalg.norm(preferences[ij_idx])
                    total_std = np.sqrt(pref_norm**2 + sigma_measurement**2)
                    standardized_score = base_score / total_std

                    from scipy.stats import norm
                    cdf_val = norm.cdf(standardized_score)

                    # Find most likely category (using thresholds without -inf/+inf)
                    category_probs = np.zeros(C)
                    thresh_clean = thresholds[ij_idx, 1:-1]  # Remove -inf and +inf boundaries
                    for c in range(C):
                        if c == 0:
                            category_probs[c] = thresh_clean[0] - cdf_val if len(thresh_clean) > 0 else 1.0 - cdf_val
                        elif c == C-1:
                            category_probs[c] = cdf_val - thresh_clean[c-1] if len(thresh_clean) > c-1 else cdf_val
                        else:
                            category_probs[c] = thresh_clean[c] - thresh_clean[c-1] if len(thresh_clean) > c else 0.0

                    c_pred = np.argmax(category_probs) + 1
                    if c_pred == c_true:
                        rating_correct += 1

                    rating_predictions.append(c_pred)
                    rating_targets.append(c_true)
                    rating_mse += (c_pred - c_true) ** 2

                    # Simple log loss approximation
                    total_rating_log_loss += -np.log(max(category_probs[c_true-1], 1e-10))
                    num_rating_preds += 1

        # Process masked rankings (following legacy approach)
        masked_rankings = [var for var in masked_vars if var['type'] == 'ranking']
        for var in masked_rankings:
            # Find true ranking
            true_order = None
            for ranking in full_data['pairwise_rankings']:
                if (ranking['attribute'] == var['attribute'] and
                    ranking['annotator'] == var['annotator'] and
                    ranking['items'] == var['items']):
                    true_order = ranking['order']
                    break

            if true_order is not None:
                i, j = var['attribute'], var['annotator']
                ij_idx = (i-1)*J + (j-1)
                items = var['items']

                # Predict preference based on scores (like legacy)
                item1, item2 = items[0], items[1]
                score1 = base_scores[ij_idx, item1-1]
                score2 = base_scores[ij_idx, item2-1]

                # Predict: item1 > item2 if score1 > score2
                pred_first_wins = score1 > score2
                true_first_wins = true_order[0] == 1  # item1 ranks first

                if pred_first_wins == true_first_wins:
                    ranking_correct += 1

                # Simple log loss approximation
                score_diff = abs(score1 - score2)
                prob = 1 / (1 + np.exp(-score_diff))  # Sigmoid approximation
                if not true_first_wins:
                    prob = 1 - prob
                total_ranking_log_loss += -np.log(max(prob, 1e-10))
                num_ranking_preds += 1

        # Calculate final metrics
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

