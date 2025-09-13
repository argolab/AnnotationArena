#!/usr/bin/env python3
"""Domain model training and evaluation for mixed annotation types using Stan MCMC."""

import numpy as np
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from scipy.stats import entropy

logger = logging.getLogger(__name__)

try:
    import cmdstanpy as stan
    STAN_AVAILABLE = True
    logger.info("Using cmdstanpy for Stan interface")
except ImportError:
    STAN_AVAILABLE = False
    logger.error("cmdstanpy not available - please install: conda install -c conda-forge cmdstanpy")

@dataclass
class DomainModelConfig:
    """Configuration for domain model training."""
    chains: int = 4          # MCMC chains
    iter_warmup: int = 2000  # warmup iterations
    iter_sampling: int = 5000 # sampling iterations
    adapt_delta: float = 0.8  # target acceptance rate
    max_treedepth: int = 15   # maximum tree depth
    
    sigma_annotator: float = 0.3    # annotator preference variance
    sigma_measurement: float = 0.1  # measurement noise variance
    alpha_dirichlet: float = 2.0    # Dirichlet concentration
    temperature: float = 0.5        # ranking temperature
    
    sigma_embedding_prior: float = 1.0   # embedding prior scale
    sigma_preference_prior: float = 1.0  # preference prior scale
    

@dataclass
class DomainModelResults:
    """Results from domain model training."""
    training_log_likelihood: float
    test_rating_accuracy: float
    test_ranking_accuracy: float
    test_rating_log_loss: float
    test_ranking_log_loss: float
    training_time: float
    n_observations: int

class DomainModelTrainer:
    """MCMC trainer for mixed annotation domain model following 'train on all' paradigm."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize trainer with Stan model."""
        if not STAN_AVAILABLE:
            raise ImportError("Stan not available - cannot train domain model")
            
        if model_path is None:
            model_path = Path(__file__).parent / "models" / "domain_model.stan"
        
        logger.info(f"Compiling Stan model: {model_path}")
        self.model = stan.CmdStanModel(stan_file=str(model_path))
        logger.info("Stan model compiled successfully")
    
    def load_data(self, data_path: Path) -> Dict[str, Any]:
        """Load train/test annotation data."""
        
        with open(data_path / "iclr_complete_train.json", 'r') as f:
            train_data = json.load(f)
        
        with open(data_path / "iclr_complete_test.json", 'r') as f:
            test_data = json.load(f)
            
        with open(data_path / "iclr_complete_ground_truth.json", 'r') as f:
            ground_truth = json.load(f)
        
        with open(data_path / "iclr_complete_stats.json", 'r') as f:
            stats = json.load(f)
        
        # Need to infer I,J from data since ground truth doesn't store them directly
        ratings_sample = train_data['ratings'][0] if train_data['ratings'] else test_data['ratings'][0]
        pairwise_rankings_sample = train_data.get('pairwise_rankings', [])
        if not pairwise_rankings_sample:
            pairwise_rankings_sample = test_data.get('pairwise_rankings', [])
        
        data_config = {
            'K': len(ground_truth['embeddings']),
            'D': len(ground_truth['embeddings'][0]), 
            'I': max([r['attribute'] for r in train_data['ratings'] + test_data['ratings']] + 
                    [r['attribute'] for r in train_data.get('pairwise_rankings', []) + test_data.get('pairwise_rankings', [])]),
            'J': max([r['annotator'] for r in train_data['ratings'] + test_data['ratings']] + 
                    [r['annotator'] for r in train_data.get('pairwise_rankings', []) + test_data.get('pairwise_rankings', [])]),
            'C': max([r['value'] for r in train_data['ratings'] + test_data['ratings']]),
            'ranking_size': 2  # Hardcode to 2 for pairwise rankings
        }
        
        return {
            'train': train_data,
            'test': test_data,
            'ground_truth': ground_truth,
            'stats': stats,
            'config': data_config
        }
    
    def prepare_stan_data(self, observed_data: Dict[str, Any], config: DomainModelConfig, data_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Convert observed data to Stan format."""
        
        ratings = observed_data['ratings']
        N_ratings = len(ratings)
        
        rating_attributes = [r['attribute'] for r in ratings]
        rating_annotators = [r['annotator'] for r in ratings]
        rating_items = [r['item'] for r in ratings]
        rating_values = [r['value'] for r in ratings]
        
        rankings = observed_data.get('pairwise_rankings', [])
        N_rankings = len(rankings)
        
        ranking_attributes = [r['attribute'] for r in rankings]
        ranking_annotators = [r['annotator'] for r in rankings]
        ranking_items = [r['items'] for r in rankings]
        ranking_orders = [r['order'] for r in rankings]
        
        if data_config is not None:
            K = data_config['K']
            I = data_config['I'] 
            J = data_config['J']
            D = data_config['D']
            C = data_config['C']
            ranking_size = data_config['ranking_size']
        else:
            K = max([max(rating_items)] + [max(items) for items in ranking_items])
            I = max(rating_attributes + ranking_attributes)
            J = max(rating_annotators + ranking_annotators)
            C = max(rating_values)
            ranking_size = len(ranking_items[0]) if ranking_items else 4
            D = 32  # fallback, but should use data_config
        
        stan_data = {
            # Dimensions
            'K': K, 'I': I, 'J': J, 'D': D, 'C': C, 'ranking_size': ranking_size,
            
            # Ratings
            'N_ratings': N_ratings,
            'rating_attributes': rating_attributes,
            'rating_annotators': rating_annotators, 
            'rating_items': rating_items,
            'rating_values': rating_values,
            
            # Comparisons
            'N_comparisons': 0,  # No comparisons used
            # Comparison data removed
            'comparison_attributes': [],
            'comparison_annotators': [],
            'comparison_items_a': [],
            'comparison_items_b': [],
            'comparison_results': [],
            
            # Rankings
            'N_rankings': N_rankings,
            'ranking_attributes': ranking_attributes,
            'ranking_annotators': ranking_annotators,
            'ranking_items': ranking_items,
            'ranking_orders': ranking_orders,
            
            # Hyperparameters
            'sigma_annotator': config.sigma_annotator,
            'sigma_measurement': config.sigma_measurement,
            'alpha_dirichlet': config.alpha_dirichlet,
            'temperature': config.temperature,
            'sigma_embedding_prior': config.sigma_embedding_prior,
            'sigma_preference_prior': config.sigma_preference_prior
        }
        
        return stan_data
    
    def sample_training_subset(self, train_data: Dict[str, Any], fraction: float, seed: int = 42) -> Dict[str, Any]:
        """Sample a fraction of COMPLETE training data for progressive experiments"""
        
        np.random.seed(seed)
        
        # Sample ratings
        ratings = train_data['ratings']
        n_ratings = int(len(ratings) * fraction)
        sampled_ratings = np.random.choice(ratings, size=n_ratings, replace=False).tolist()
        
        # Sample pairwise rankings
        rankings = train_data.get('pairwise_rankings', [])
        n_rankings = max(1, int(len(rankings) * fraction)) if rankings else 0
        sampled_rankings = np.random.choice(rankings, size=n_rankings, replace=False).tolist() if rankings else []
        
        return {
            'ratings': sampled_ratings,
            # 'comparisons': removed
            'pairwise_rankings': sampled_rankings
        }
    
    def mask_test_data_for_evaluation(self, test_data: Dict[str, Any], mask_fraction: float = 0.5, seed: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Mask a fraction of test annotations to evaluate imputation capability"""
        
        np.random.seed(seed)
        
        observed_data = {'ratings': [], 'pairwise_rankings': []}
        missing_data = {'ratings': [], 'pairwise_rankings': []}
        
        # Mask ratings
        for rating in test_data['ratings']:
            if np.random.random() < mask_fraction:
                missing_data['ratings'].append(rating)
            else:
                observed_data['ratings'].append(rating)
        
        # Mask pairwise rankings  
        for ranking in test_data.get('pairwise_rankings', []):
            if np.random.random() < mask_fraction:
                missing_data['pairwise_rankings'].append(ranking)
            else:
                observed_data['pairwise_rankings'].append(ranking)
        
        return observed_data, missing_data
    
    def compute_kl_divergence(self, learned_embeddings: np.ndarray, 
                            true_embeddings: np.ndarray) -> float:
        """Compute KL divergence between learned and true embeddings"""
        
        # Flatten embeddings and add small epsilon to avoid log(0)
        epsilon = 1e-10
        
        # Convert to probability distributions (softmax over items for each dimension)
        learned_probs = []
        true_probs = []
        
        for d in range(learned_embeddings.shape[1]):  # For each dimension
            learned_d = np.exp(learned_embeddings[:, d])
            learned_d = learned_d / (np.sum(learned_d) + epsilon)
            learned_probs.extend(learned_d)
            
            true_d = np.exp(true_embeddings[:, d])
            true_d = true_d / (np.sum(true_d) + epsilon)
            true_probs.extend(true_d)
        
        learned_probs = np.array(learned_probs) + epsilon
        true_probs = np.array(true_probs) + epsilon
        
        # Normalize to ensure valid probability distributions
        learned_probs = learned_probs / np.sum(learned_probs)
        true_probs = true_probs / np.sum(true_probs)
        
        return entropy(true_probs, learned_probs)
    
    def _create_initial_values(self, stan_data: Dict[str, Any], seed: int, config: DomainModelConfig) -> List[Dict[str, Any]]:
        """Create reasonable initial values for Stan parameters"""
        
        np.random.seed(seed)
        
        K, I, J, D, C = stan_data['K'], stan_data['I'], stan_data['J'], stan_data['D'], stan_data['C']
        
        def create_init():
            # Small random embeddings
            embeddings = np.random.normal(0, 0.5, (K, D))
            
            # Small random mean preferences
            mean_preferences = np.random.normal(0, 0.5, (I, D))
            
            # Annotator preferences close to mean preferences
            annotator_preferences = np.zeros((I*J, D))
            for i in range(I):
                for j in range(J):
                    idx = i*J + j
                    annotator_preferences[idx] = mean_preferences[i] + np.random.normal(0, 0.1, D)
            
            # Initialize ordered rating thresholds
            rating_thresholds_raw = []
            for ij in range(I*J):
                # Create ordered thresholds: Q_1 < Q_2 < ... < Q_{C-1}
                thresholds = np.sort(np.random.normal(0, 1, C-1))
                rating_thresholds_raw.append(thresholds.tolist())
            
            return {
                'embeddings': embeddings.tolist(),
                'mean_preferences': mean_preferences.tolist(),
                'annotator_preferences': annotator_preferences.tolist(),
                'rating_thresholds_raw': rating_thresholds_raw
            }
        
        # Return list of initial values for each chain
        return [create_init() for _ in range(config.chains)]
    
    def compute_log_loss_on_missing(self, fit, observed_test: Dict[str, Any], missing_test: Dict[str, Any], stan_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute log-loss on missing test annotations using posterior predictive distribution"""
        
        # Get dimensions and parameters
        J = stan_data['J']
        C = stan_data['C']
        sigma_measurement = stan_data['sigma_measurement']
        temperature = stan_data['temperature']
        
        # Extract posterior means for prediction
        embeddings = np.mean(fit.stan_variable('embeddings'), axis=0)  # [K, D]
        preferences = np.mean(fit.stan_variable('annotator_preferences'), axis=0)  # [I*J, D]  
        threshold_samples = fit.stan_variable('rating_thresholds_raw')  # [samples, I*J, C-1]
        thresholds_mean = np.mean(threshold_samples, axis=0)  # [I*J, C-1]
        
        # Compute base scores
        base_scores = preferences @ embeddings.T  # [I*J, K]
        
        results = {'ratings': 0.0, 'rankings': 0.0, 'total': 0.0}
        total_log_loss = 0.0
        
        # 1. RATING LOG-LOSS
        if missing_test['ratings']:
            rating_log_loss = 0.0
            for r in missing_test['ratings']:
                i, j, k, c_true = r['attribute'], r['annotator'], r['item'], r['value'] 
                ij_idx = (i-1)*J + (j-1)
                
                # Base score z_ijk = v_ij · e_k
                base_score = base_scores[ij_idx, k-1]
                
                # Compute rating probabilities using corrected likelihood
                # P(rating = c) = Φ((Q_c - z)/σ_m) - Φ((Q_{c-1} - z)/σ_m)
                from scipy.stats import norm
                
                # Create threshold boundaries: [-∞, Q_1, Q_2, ..., Q_{C-1}, +∞]
                full_thresholds = np.concatenate([
                    [-np.inf], 
                    thresholds_mean[ij_idx], 
                    [np.inf]
                ])
                
                # Compute probability for true category c_true
                upper_thresh = full_thresholds[c_true]  # c_true is 1-indexed
                lower_thresh = full_thresholds[c_true-1]
                
                upper_prob = 1.0 if upper_thresh == np.inf else norm.cdf((upper_thresh - base_score) / sigma_measurement)
                lower_prob = 0.0 if lower_thresh == -np.inf else norm.cdf((lower_thresh - base_score) / sigma_measurement)
                
                prob = upper_prob - lower_prob
                prob = max(prob, 1e-10)  # Numerical stability
                
                rating_log_loss += -np.log(prob)
            
            rating_log_loss /= len(missing_test['ratings'])
            results['ratings'] = rating_log_loss
            total_log_loss += rating_log_loss
        
        # 2. PAIRWISE RANKING LOG-LOSS (Simplified)
        pairwise_rankings = missing_test.get('pairwise_rankings', [])
        if pairwise_rankings:
            ranking_log_loss = 0.0
            for r in pairwise_rankings:
                i, j = r['attribute'], r['annotator']
                items = r['items']  # [item1, item2]
                true_order = r['order']  # [1, 2] or [2, 1]
                ij_idx = (i-1)*J + (j-1)
                
                # Get item scores and apply temperature scaling
                item1, item2 = items[0], items[1]
                score1 = base_scores[ij_idx, item1-1] / temperature
                score2 = base_scores[ij_idx, item2-1] / temperature
                
                # Compute pairwise log-likelihood
                if true_order[0] == 1:  # item1 ranks first
                    # P(item1 > item2) = sigmoid(score1 - score2)
                    prob = 1.0 / (1.0 + np.exp(-(score1 - score2)))
                else:  # item2 ranks first
                    # P(item2 > item1) = sigmoid(score2 - score1)
                    prob = 1.0 / (1.0 + np.exp(-(score2 - score1)))
                
                prob = max(prob, 1e-10)  # Numerical stability
                ranking_log_loss += -np.log(prob)
            
            ranking_log_loss /= len(pairwise_rankings)
            results['rankings'] = ranking_log_loss
            total_log_loss += ranking_log_loss
        
        results['total'] = total_log_loss / (len([k for k in ['ratings', 'pairwise_rankings'] if missing_test.get(k, [])]))
        return results
    
    def evaluate_imputation_accuracy(self, fit, visible_test: Dict[str, Any], masked_test: Dict[str, Any], stan_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate imputation accuracy on masked test positions"""
        
        # Get dimensions from stan_data
        J = stan_data['J']
        C = stan_data['C']
        sigma_measurement = stan_data['sigma_measurement']
        temperature = stan_data['temperature']
        
        # Extract posterior means for prediction
        embeddings = np.mean(fit.stan_variable('embeddings'), axis=0)  # [K, D]
        preferences = np.mean(fit.stan_variable('annotator_preferences'), axis=0)  # [I*J, D]
        thresholds = np.mean(fit.stan_variable('rating_thresholds'), axis=0)  # [I*J, C]
        
        # Compute base scores
        base_scores = preferences @ embeddings.T  # [I*J, K]
        
        results = {'rating_accuracy': 0.0, 'ranking_accuracy': 0.0}
        
        # 1. RATING ACCURACY
        if masked_test['ratings']:
            correct_ratings = 0
            for r in masked_test['ratings']:
                i, j, k, c_true = r['attribute'], r['annotator'], r['item'], r['value']
                ij_idx = (i-1)*J + (j-1)
                
                # Predict rating using posterior mean
                base_score = base_scores[ij_idx, k-1]
                pref_norm = np.linalg.norm(preferences[ij_idx])
                total_std = np.sqrt(pref_norm**2 + sigma_measurement**2)
                standardized_score = base_score / total_std
                
                from scipy.stats import norm
                cdf_val = norm.cdf(standardized_score)
                
                # Find most likely category
                category_probs = np.zeros(C)
                for c in range(C):
                    if c == 0:
                        category_probs[c] = thresholds[ij_idx, c] - cdf_val
                    elif c == C-1:
                        category_probs[c] = cdf_val - thresholds[ij_idx, c-1]
                    else:
                        category_probs[c] = thresholds[ij_idx, c] - thresholds[ij_idx, c-1]
                
                c_pred = np.argmax(category_probs) + 1
                if c_pred == c_true:
                    correct_ratings += 1
            
            results['rating_accuracy'] = correct_ratings / len(masked_test['ratings'])
        
        # 2. PAIRWISE RANKING ACCURACY (Binary prediction accuracy)
        pairwise_rankings = masked_test.get('pairwise_rankings', [])
        if pairwise_rankings:
            correct_predictions = 0
            for rank in pairwise_rankings:
                i, j = rank['attribute'], rank['annotator']
                ij_idx = (i-1)*J + (j-1)
                items = rank['items']  # [item1, item2]
                true_order = rank['order']  # [1, 2] or [2, 1]
                
                # Predict preference based on scores
                item1, item2 = items[0], items[1]
                score1 = base_scores[ij_idx, item1-1]
                score2 = base_scores[ij_idx, item2-1]
                
                # Predict: item1 > item2 if score1 > score2
                pred_first_wins = score1 > score2
                true_first_wins = true_order[0] == 1  # item1 ranks first
                
                if pred_first_wins == true_first_wins:
                    correct_predictions += 1
            
            results['ranking_accuracy'] = correct_predictions / len(pairwise_rankings)
        
        return results
    
    def compute_test_accuracy(self, fit, test_data: Dict[str, Any], stan_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute simple accuracy metrics on ALL test data."""
        J = stan_data['J']
        sigma_measurement = stan_data['sigma_measurement']
        temperature = stan_data['temperature']
        
        # Extract posterior means for prediction
        embeddings = np.mean(fit.stan_variable('embeddings'), axis=0)  # [K, D]
        preferences = np.mean(fit.stan_variable('annotator_preferences'), axis=0)  # [I*J, D]
        threshold_samples = fit.stan_variable('rating_thresholds_raw')  # [samples, I*J, C-1]
        thresholds_mean = np.mean(threshold_samples, axis=0)  # [I*J, C-1]
        
        # Compute base scores
        base_scores = preferences @ embeddings.T  # [I*J, K]
        
        results = {'rating_accuracy': 0.0, 'ranking_accuracy': 0.0}
        
        # 1. RATING ACCURACY
        test_ratings = test_data['ratings']
        if test_ratings:
            correct_ratings = 0
            for r in test_ratings:
                i, j, k, c_true = r['attribute'], r['annotator'], r['item'], r['value']
                ij_idx = (i-1)*J + (j-1)
                
                # Base score z_ijk = v_ij · e_k
                base_score = base_scores[ij_idx, k-1]
                
                # Predict rating using thresholds
                from scipy.stats import norm
                
                # Create threshold boundaries: [-∞, Q_1, Q_2, ..., Q_{C-1}, +∞]
                full_thresholds = np.concatenate([
                    [-np.inf], 
                    thresholds_mean[ij_idx], 
                    [np.inf]
                ])
                
                # Find most likely category
                category_probs = []
                for c in range(1, len(full_thresholds)):
                    upper_thresh = full_thresholds[c]
                    lower_thresh = full_thresholds[c-1]
                    
                    upper_prob = 1.0 if upper_thresh == np.inf else norm.cdf((upper_thresh - base_score) / sigma_measurement)
                    lower_prob = 0.0 if lower_thresh == -np.inf else norm.cdf((lower_thresh - base_score) / sigma_measurement)
                    
                    prob = upper_prob - lower_prob
                    category_probs.append(prob)
                
                c_pred = np.argmax(category_probs) + 1
                if c_pred == c_true:
                    correct_ratings += 1
            
            results['rating_accuracy'] = correct_ratings / len(test_ratings)
        
        # 2. PAIRWISE RANKING ACCURACY
        pairwise_rankings = test_data.get('pairwise_rankings', [])
        if pairwise_rankings:
            correct_predictions = 0
            for rank in pairwise_rankings:
                i, j = rank['attribute'], rank['annotator']
                ij_idx = (i-1)*J + (j-1)
                items = rank['items']  # [item1, item2]
                true_order = rank['order']  # [1, 2] or [2, 1]
                
                # Predict preference based on scores
                item1, item2 = items[0], items[1]
                score1 = base_scores[ij_idx, item1-1]
                score2 = base_scores[ij_idx, item2-1]
                
                # Predict: item1 > item2 if score1 > score2
                pred_first_wins = score1 > score2
                true_first_wins = true_order[0] == 1  # item1 ranks first
                
                if pred_first_wins == true_first_wins:
                    correct_predictions += 1
            
            results['ranking_accuracy'] = correct_predictions / len(pairwise_rankings)
        
        return results
    
    def train_and_evaluate(self, data_path: Path, config: DomainModelConfig, 
                          seed: int = 42, output_dir: Path = None) -> DomainModelResults:
        """Train domain model on ALL training data and evaluate on ALL test data."""
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path("domain_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Loading ICLR pairwise data...")
        data = self.load_data(data_path)
        
        # Use ALL training data (no progressive sampling)
        train_data = data['train']
        test_data = data['test']
        
        logger.info(f"Training on {len(train_data['ratings'])} ratings + {len(train_data.get('pairwise_rankings', []))} pairwise rankings")
        logger.info(f"Testing on {len(test_data['ratings'])} ratings + {len(test_data.get('pairwise_rankings', []))} pairwise rankings")
        
        # Prepare Stan data
        stan_data = self.prepare_stan_data(train_data, config, data['config'])
        
        # Create initial values
        init_values = self._create_initial_values(stan_data, seed, config)
        
        # Train model
        import time
        start_time = time.time()
        
        logger.info("Running MCMC training...")
        fit = self.model.sample(
            data=stan_data,
            chains=config.chains,
            iter_warmup=config.iter_warmup,
            iter_sampling=config.iter_sampling,
            adapt_delta=config.adapt_delta,
            max_treedepth=config.max_treedepth,
            seed=seed,
            inits=init_values,
            show_progress=True
        )
        
        training_time = time.time() - start_time
        
        # Extract training log-likelihood
        training_log_lik = np.mean(fit.stan_variable('total_log_lik'))
        
        # Compute test accuracies on ALL test data (pure imputation)
        test_accuracy_results = self.compute_test_accuracy(fit, test_data, stan_data)
        
        # Compute test log-losses on ALL test data
        test_log_loss_results = self.compute_log_loss_on_missing(
            fit, {'ratings': [], 'pairwise_rankings': []}, test_data, stan_data
        )
        
        # Create results
        results = DomainModelResults(
            training_log_likelihood=training_log_lik,
            test_rating_accuracy=test_accuracy_results.get('rating_accuracy', 0.0),
            test_ranking_accuracy=test_accuracy_results.get('ranking_accuracy', 0.0),
            test_rating_log_loss=test_log_loss_results.get('ratings', 0.0),
            test_ranking_log_loss=test_log_loss_results.get('rankings', 0.0),
            training_time=training_time,
            n_observations=len(train_data['ratings']) + len(train_data.get('pairwise_rankings', []))
        )
        
        logger.info(f"Training completed in {training_time:.1f}s")
        logger.info(f"Training log-likelihood: {results.training_log_likelihood:.3f}")
        logger.info(f"Test rating accuracy: {results.test_rating_accuracy:.3f} ({results.test_rating_accuracy*100:.1f}%)")
        logger.info(f"Test ranking accuracy: {results.test_ranking_accuracy:.3f} ({results.test_ranking_accuracy*100:.1f}%)")
        logger.info(f"Test rating log-loss: {results.test_rating_log_loss:.3f}")
        logger.info(f"Test ranking log-loss: {results.test_ranking_log_loss:.3f}")
        
        return results
    
    def plot_results(self, results: DomainModelResults, output_dir: Path):
        """Create simple summary plot of domain model results"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple summary plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Bar plot of accuracy metrics
        metrics = ['Rating Accuracy', 'Ranking Accuracy']
        values = [results.test_rating_accuracy, results.test_ranking_accuracy]
        
        bars = ax.bar(metrics, values, color=['blue', 'green'], alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_title('Domain Model Test Accuracy')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        # Add training log-likelihood as text
        ax.text(0.02, 0.98, f'Training Log-Likelihood: {results.training_log_likelihood:.2f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'domain_model_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved to {output_dir}/domain_model_results.png")

def main():
    """Train and evaluate domain model on ICLR pairwise data."""
    
    logging.basicConfig(level=logging.INFO)
    
    # Load centralized configuration for ICLR pairwise experiment
    from config import ExperimentConfig
    exp_config = ExperimentConfig()
    
    # Create domain model configuration
    config = DomainModelConfig(
        chains=2,
        iter_warmup=500,
        iter_sampling=2000,
        sigma_annotator=exp_config.sigma_annotator,
        sigma_measurement=exp_config.sigma_measurement,
        alpha_dirichlet=exp_config.alpha_dirichlet,
        temperature=exp_config.temperature
    )
    
    # Train model on ICLR data
    data_path = Path(__file__).parent / "generated_data"
    trainer = DomainModelTrainer()
    output_dir = Path(__file__).parent / "domain_results"
    results = trainer.train_and_evaluate(data_path, config, seed=12345, output_dir=output_dir)
    
    # Print final summary
    print("\n" + "="*50)
    print("DOMAIN MODEL RESULTS")
    print("="*50)
    print(f"Training observations: {results.n_observations}")
    print(f"Training time: {results.training_time:.1f}s")
    print(f"Training log-likelihood: {results.training_log_likelihood:.3f}")
    print(f"")
    print(f"TEST ACCURACY (Pure Imputation):")
    print(f"  Rating accuracy: {results.test_rating_accuracy:.3f} ({results.test_rating_accuracy*100:.1f}%)")
    print(f"  Ranking accuracy: {results.test_ranking_accuracy:.3f} ({results.test_ranking_accuracy*100:.1f}%)")
    print(f"")
    print(f"TEST LOG-LOSS:")
    print(f"  Rating log-loss: {results.test_rating_log_loss:.3f}")
    print(f"  Ranking log-loss: {results.test_ranking_log_loss:.3f}")
    print("="*50)

if __name__ == "__main__":
    main()