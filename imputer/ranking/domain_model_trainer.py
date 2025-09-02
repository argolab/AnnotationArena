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
    
    budget_fractions: List[float] = None  # progressive training fractions
    
    def __post_init__(self):
        if self.budget_fractions is None:
            self.budget_fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

@dataclass
class ProgressiveResults:
    """Results from progressive training."""
    budget_fractions: List[float]
    training_log_likelihoods: List[float]
    test_log_likelihoods: List[float]
    kl_divergences: List[float]
    training_times: List[float]
    n_observations: List[int]
    ratings_log_lik: List[float]  # per-annotation-type metrics
    rankings_log_lik: List[float]

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
        
        with open(data_path / "test_complete_train.json", 'r') as f:
            train_data = json.load(f)
        
        with open(data_path / "test_complete_test.json", 'r') as f:
            test_data = json.load(f)
            
        with open(data_path / "test_complete_ground_truth.json", 'r') as f:
            ground_truth = json.load(f)
        
        with open(data_path / "test_complete_stats.json", 'r') as f:
            stats = json.load(f)
        
        # Need to infer I,J from data since ground truth doesn't store them directly
        ratings_sample = train_data['ratings'][0] if train_data['ratings'] else test_data['ratings'][0]
        rankings_sample = train_data['rankings'][0] if train_data['rankings'] else test_data['rankings'][0]
        
        data_config = {
            'K': len(ground_truth['embeddings']),
            'D': len(ground_truth['embeddings'][0]), 
            'I': max([r['attribute'] for r in train_data['ratings'] + test_data['ratings']] + 
                    [r['attribute'] for r in train_data['rankings'] + test_data['rankings']]),
            'J': max([r['annotator'] for r in train_data['ratings'] + test_data['ratings']] + 
                    [r['annotator'] for r in train_data['rankings'] + test_data['rankings']]),
            'C': max([r['value'] for r in train_data['ratings'] + test_data['ratings']]),
            'ranking_size': len(rankings_sample['items']) if train_data['rankings'] or test_data['rankings'] else 5
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
        
        rankings = observed_data['rankings']
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
        
        # Sample rankings
        rankings = train_data['rankings']
        n_rankings = max(1, int(len(rankings) * fraction))  # At least 1 ranking
        sampled_rankings = np.random.choice(rankings, size=n_rankings, replace=False).tolist()
        
        return {
            'ratings': sampled_ratings,
            # 'comparisons': removed
            'rankings': sampled_rankings
        }
    
    def mask_test_data_for_evaluation(self, test_data: Dict[str, Any], mask_fraction: float = 0.5, seed: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Mask a fraction of test annotations to evaluate imputation capability"""
        
        np.random.seed(seed)
        
        observed_data = {'ratings': [], 'rankings': []}
        missing_data = {'ratings': [], 'rankings': []}
        
        # Mask ratings
        for rating in test_data['ratings']:
            if np.random.random() < mask_fraction:
                missing_data['ratings'].append(rating)
            else:
                observed_data['ratings'].append(rating)
        
        # Mask rankings  
        for ranking in test_data['rankings']:
            if np.random.random() < mask_fraction:
                missing_data['rankings'].append(ranking)
            else:
                observed_data['rankings'].append(ranking)
        
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
    
    def _create_initial_values(self, stan_data: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
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
        return [create_init() for _ in range(4)]  # Create 4 different initializations
    
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
        
        # 2. RANKING LOG-LOSS (Plackett-Luce)
        if missing_test['rankings']:
            ranking_log_loss = 0.0
            for r in missing_test['rankings']:
                i, j = r['attribute'], r['annotator']
                items = r['items']
                true_order = r['order']
                ij_idx = (i-1)*J + (j-1)
                
                # Get item scores and apply temperature scaling
                item_scores = np.array([base_scores[ij_idx, k-1] / temperature for k in items])
                
                # Compute Plackett-Luce log-likelihood
                ranking_log_lik = 0.0
                for pos in range(len(items)):
                    chosen_idx = true_order[pos] - 1  # Convert to 0-indexed
                    remaining_scores = item_scores[pos:]  # Remaining items
                    
                    log_sum_exp_remaining = np.logaddexp.reduce(remaining_scores)
                    ranking_log_lik += item_scores[chosen_idx] - log_sum_exp_remaining
                
                ranking_log_loss += -ranking_log_lik
            
            ranking_log_loss /= len(missing_test['rankings'])
            results['rankings'] = ranking_log_loss
            total_log_loss += ranking_log_loss
        
        results['total'] = total_log_loss / (len([k for k in ['ratings', 'rankings'] if missing_test[k]]))
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
        
        # 2. RANKING ACCURACY (Kendall's tau correlation)
        if masked_test['rankings']:
            kendall_taus = []
            for rank in masked_test['rankings']:
                i, j = rank['attribute'], rank['annotator']
                ij_idx = (i-1)*J + (j-1)
                items = rank['items']
                true_order = rank['order']
                
                # Predict ranking based on scores
                item_scores = [base_scores[ij_idx, k-1] for k in items]
                pred_order = np.argsort(-np.array(item_scores)) + 1  # Convert to 1-indexed
                
                # Compute Kendall's tau
                from scipy.stats import kendalltau
                tau, _ = kendalltau(true_order, pred_order)
                kendall_taus.append(tau)
            
            results['ranking_accuracy'] = np.mean(kendall_taus)
        
        return results
    
    def progressive_training(self, data_path: Path, config: DomainModelConfig, 
                           seed: int = 42, output_dir: Path = None) -> ProgressiveResults:
        """Perform progressive training with increasing data budgets"""
        
        # Setup output directory for logging
        if output_dir is None:
            output_dir = Path("domain_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup comprehensive logging
        log_dir = output_dir / "stan_logs"
        log_dir.mkdir(exist_ok=True)
        
        logger.info(f"Stan logs will be saved to: {log_dir}")
        logger.info("Loading data for progressive training...")
        data = self.load_data(data_path)
        
        results = ProgressiveResults(
            budget_fractions=config.budget_fractions,
            training_log_likelihoods=[],
            test_log_likelihoods=[],
            kl_divergences=[],
            training_times=[],
            n_observations=[],
            ratings_log_lik=[],
            # comparisons_log_lik=[],  # Removed
            rankings_log_lik=[]
        )
        
        true_embeddings = np.array(data['ground_truth']['embeddings'])
        
        for i, fraction in enumerate(config.budget_fractions):
            logger.info(f"Training with budget fraction {fraction:.1%} ({i+1}/{len(config.budget_fractions)})...")
            
            # Sample subset of COMPLETE training data (no masking during training)
            subset_data = self.sample_training_subset(data['train'], fraction, seed + i)
            
            # Train on ALL available annotations in the subset
            stan_data = self.prepare_stan_data(subset_data, config, data['config'])
            
            # Create reasonable initial values
            init_values = self._create_initial_values(stan_data, seed + i)
            
            # Train model with comprehensive logging
            import time
            start_time = time.time()
            
            # Setup detailed Stan output files
            stan_output_dir = log_dir / f"fraction_{fraction:.1f}"
            stan_output_dir.mkdir(exist_ok=True)
            
            logger.info(f"Running MCMC for fraction {fraction:.1%}...")
            logger.info(f"Stan files: {stan_output_dir}")
            
            fit = self.model.sample(
                data=stan_data,
                chains=config.chains,
                iter_warmup=config.iter_warmup,
                iter_sampling=config.iter_sampling,
                adapt_delta=config.adapt_delta,
                max_treedepth=config.max_treedepth,
                seed=seed + i,
                inits=init_values,
                output_dir=str(stan_output_dir),
                save_warmup=True,
                show_progress=True
            )
            
            # Save Stan diagnostics and summaries
            try:
                # Save fit summary
                with open(stan_output_dir / "fit_summary.txt", 'w') as f:
                    f.write(str(fit.summary()))
                
                # Save diagnostics
                diagnostics = fit.diagnose()
                with open(stan_output_dir / "diagnostics.txt", 'w') as f:
                    f.write(str(diagnostics))
                
                # Save sample metadata
                import json
                sample_metadata = {
                    'chains': config.chains,
                    'iter_warmup': config.iter_warmup,
                    'iter_sampling': config.iter_sampling,
                    'adapt_delta': config.adapt_delta,
                    'max_treedepth': config.max_treedepth,
                    'seed': seed + i,
                    'budget_fraction': fraction,
                    'n_observations': len(subset_data['ratings']) + len(subset_data['rankings']),
                    'stan_data_dims': {k: v for k, v in stan_data.items() if isinstance(v, (int, float))}
                }
                with open(stan_output_dir / "sample_metadata.json", 'w') as f:
                    json.dump(sample_metadata, f, indent=2)
                
                logger.info(f"Stan output saved to {stan_output_dir}")
                
            except Exception as e:
                logger.warning(f"Failed to save some Stan diagnostics: {e}")
            
            training_time = time.time() - start_time
            
            # Extract results
            training_log_lik = np.mean(fit.stan_variable('total_log_lik'))
            learned_embeddings = np.mean(fit.stan_variable('embeddings'), axis=0)
            
            # Compute KL divergence
            kl_div = self.compute_kl_divergence(learned_embeddings, true_embeddings)
            
            # Evaluate imputation on test set: artificially mask 50% and predict them
            observed_test, missing_test = self.mask_test_data_for_evaluation(data['test'], mask_fraction=0.5, seed=seed + i + 200)
            
            # Compute log-loss on missing test annotations
            test_log_loss = self.compute_log_loss_on_missing(fit, observed_test, missing_test, stan_data)
            
            # Extract per-annotation-type log-losses
            rating_log_loss = test_log_loss.get('ratings', 0.0)
            ranking_log_loss = test_log_loss.get('rankings', 0.0)
            total_test_log_loss = test_log_loss.get('total', 0.0)
            
            # Store results
            results.training_log_likelihoods.append(training_log_lik)
            results.test_log_likelihoods.append(total_test_log_loss)  # Log-loss on test set
            results.kl_divergences.append(kl_div)
            results.training_times.append(training_time)
            results.n_observations.append(len(subset_data['ratings']) + len(subset_data['rankings']))
            results.ratings_log_lik.append(rating_log_loss)
            results.rankings_log_lik.append(ranking_log_loss)
            
            logger.info(f"  Training log-likelihood: {training_log_lik:.3f}")
            logger.info(f"  Test log-loss: {total_test_log_loss:.3f}")
            logger.info(f"  Rating log-loss: {rating_log_loss:.3f}")
            logger.info(f"  Ranking log-loss: {ranking_log_loss:.3f}")
            logger.info(f"  KL divergence: {kl_div:.3f}")
            logger.info(f"  Training time: {training_time:.1f}s")
            logger.info(f"  Observations used: {len(subset_data['ratings']) + len(subset_data['rankings'])}")
        
        return results
    
    def plot_results(self, results: ProgressiveResults, output_dir: Path):
        """Create plots of progressive training results"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Training log-likelihood and test log-loss curves
        ax = axes[0, 0]
        ax.plot(results.budget_fractions, results.training_log_likelihoods, 'b-o', label='Training Log-Likelihood')
        ax2 = ax.twinx()
        ax2.plot(results.budget_fractions, results.test_log_likelihoods, 'r-o', label='Test Log-Loss')
        ax.set_xlabel('Budget Fraction')
        ax.set_ylabel('Training Log-Likelihood', color='b')
        ax2.set_ylabel('Test Log-Loss (lower = better)', color='r')
        ax.set_title('Domain Model Performance')
        ax.grid(True)
        
        # 2. KL divergence
        ax = axes[0, 1]
        ax.plot(results.budget_fractions, results.kl_divergences, 'g-o')
        ax.set_xlabel('Budget Fraction')
        ax.set_ylabel('KL Divergence')
        ax.set_title('Embedding KL Divergence')
        ax.grid(True)
        
        # 3. Training time
        ax = axes[1, 0]
        ax.plot(results.budget_fractions, results.training_times, 'm-o')
        ax.set_xlabel('Budget Fraction')
        ax.set_ylabel('Training Time (s)')
        ax.set_title('MCMC Training Time')
        ax.grid(True)
        
        # 4. Per-annotation-type log-losses
        ax = axes[1, 1]
        ax.plot(results.budget_fractions, results.ratings_log_lik, 'b-o', label='Rating Log-Loss')
        ax.plot(results.budget_fractions, results.rankings_log_lik, 'g-o', label='Ranking Log-Loss')
        ax.set_xlabel('Budget Fraction')
        ax.set_ylabel('Log-Loss (lower = better)')
        ax.set_title('Per-Annotation-Type Log-Loss')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'domain_model_progressive_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Results plots saved to {output_dir}")

def main():
    """Test the domain model trainer"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration for testing
    config = DomainModelConfig(
        chains=2,  # Reduced for testing
        iter_warmup=500,
        iter_sampling=500,
        budget_fractions=[0.2, 0.5, 1.0]  # Smaller set for testing
    )
    
    # Train model
    data_path = Path(__file__).parent / "generated_data"
    trainer = DomainModelTrainer()
    output_dir = Path(__file__).parent / "domain_results"
    results = trainer.progressive_training(data_path, config, seed=12345, output_dir=output_dir)
    
    # Create plots  
    trainer.plot_results(results, output_dir)
    
    # Print summary
    print("Domain Model Training Complete!")
    print(f"Final test log-loss: {results.test_log_likelihoods[-1]:.3f}")
    print(f"Final rating log-loss: {results.ratings_log_lik[-1]:.3f}")
    print(f"Final ranking log-loss: {results.rankings_log_lik[-1]:.3f}")
    print(f"Final KL divergence: {results.kl_divergences[-1]:.3f}")
    print(f"Total training time: {sum(results.training_times):.1f}s")

if __name__ == "__main__":
    main()