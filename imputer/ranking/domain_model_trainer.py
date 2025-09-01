#!/usr/bin/env python3
"""
Domain model training and evaluation for mixed annotation types using Stan MCMC.

This module provides progressive training similar to the Gaussian experiments,
computing log-likelihood and KL divergence metrics.
"""

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
    """Configuration for domain model training"""
    # MCMC parameters
    chains: int = 4
    iter_warmup: int = 1000
    iter_sampling: int = 1000
    adapt_delta: float = 0.8
    max_treedepth: int = 10
    
    # Model hyperparameters (should match data generation)
    sigma_annotator: float = 0.3
    sigma_measurement: float = 0.1
    alpha_dirichlet: float = 2.0
    temperature: float = 0.5
    
    # Prior scales
    sigma_embedding_prior: float = 1.0
    sigma_preference_prior: float = 1.0
    
    # Progressive training
    budget_fractions: List[float] = None  # [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    def __post_init__(self):
        if self.budget_fractions is None:
            self.budget_fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

@dataclass
class ProgressiveResults:
    """Results from progressive training"""
    budget_fractions: List[float]
    training_log_likelihoods: List[float]
    test_log_likelihoods: List[float]
    kl_divergences: List[float]
    training_times: List[float]
    n_observations: List[int]
    
    # Per-annotation-type metrics
    ratings_log_lik: List[float]
    # Removed: comparisons_log_lik: List[float]
    rankings_log_lik: List[float]

class DomainModelTrainer:
    """MCMC trainer for mixed annotation domain model"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize trainer with Stan model"""
        if not STAN_AVAILABLE:
            raise ImportError("Stan not available - cannot train domain model")
            
        if model_path is None:
            model_path = Path(__file__).parent / "models" / "domain_model.stan"
        
        logger.info(f"Compiling Stan model: {model_path}")
        self.model = stan.CmdStanModel(stan_file=str(model_path))
        logger.info("Stan model compiled successfully")
    
    def load_data(self, data_path: Path) -> Dict[str, Any]:
        """Load train/test annotation data"""
        
        with open(data_path / "test_complete_train.json", 'r') as f:
            train_data = json.load(f)
        
        with open(data_path / "test_complete_test.json", 'r') as f:
            test_data = json.load(f)
            
        with open(data_path / "test_complete_ground_truth.json", 'r') as f:
            ground_truth = json.load(f)
        
        with open(data_path / "test_complete_stats.json", 'r') as f:
            stats = json.load(f)
        
        return {
            'train': train_data,
            'test': test_data,
            'ground_truth': ground_truth,
            'stats': stats
        }
    
    def prepare_stan_data(self, observed_data: Dict[str, Any], config: DomainModelConfig) -> Dict[str, Any]:
        """Convert observed data to Stan format"""
        
        # Extract ratings
        ratings = observed_data['ratings']
        N_ratings = len(ratings)
        
        rating_attributes = [r['attribute'] for r in ratings]
        rating_annotators = [r['annotator'] for r in ratings]
        rating_items = [r['item'] for r in ratings]
        rating_values = [r['value'] for r in ratings]
        
        # Comparisons removed - no longer used
        N_comparisons = 0
        comparison_attributes = []
        comparison_annotators = []
        comparison_items_a = []
        comparison_items_b = []
        comparison_results = []
        
        # Extract rankings
        rankings = observed_data['rankings']
        N_rankings = len(rankings)
        
        ranking_attributes = [r['attribute'] for r in rankings]
        ranking_annotators = [r['annotator'] for r in rankings]
        ranking_items = [r['items'] for r in rankings]
        ranking_orders = [r['order'] for r in rankings]
        
        # Infer dimensions from data
        K = max([max(rating_items)] + 
                [max(items) for items in ranking_items])
        I = max(rating_attributes + ranking_attributes)
        J = max(rating_annotators + ranking_annotators)
        C = max(rating_values)
        ranking_size = len(ranking_items[0]) if ranking_items else 4
        D = 32  # Default embedding dimension
        
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
    
    def sample_subset(self, train_data: Dict[str, Any], fraction: float, seed: int = 42) -> Dict[str, Any]:
        """Sample a fraction of training data for progressive training"""
        
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
    
    def mask_data_for_training(self, data: Dict[str, Any], mask_fraction: float = 0.5, seed: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Mask a fraction of data positions for imputation training"""
        
        np.random.seed(seed)
        
        visible_data = {'ratings': [], 'rankings': []}
        masked_data = {'ratings': [], 'rankings': []}
        
        # Mask ratings
        for rating in data['ratings']:
            if np.random.random() < mask_fraction:
                masked_data['ratings'].append(rating)
            else:
                visible_data['ratings'].append(rating)
        
        # Mask rankings
        for ranking in data['rankings']:
            if np.random.random() < mask_fraction:
                masked_data['rankings'].append(ranking)
            else:
                visible_data['rankings'].append(ranking)
        
        return visible_data, masked_data
    
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
            
            # Uniform rating probabilities (slightly perturbed)
            rating_probs = []
            for ij in range(I*J):
                # Start with uniform, add small noise
                probs = np.ones(C) / C + np.random.normal(0, 0.01, C)
                probs = np.maximum(probs, 0.01)  # Ensure positive
                probs = probs / np.sum(probs)  # Normalize
                rating_probs.append(probs.tolist())
            
            return {
                'embeddings': embeddings.tolist(),
                'mean_preferences': mean_preferences.tolist(),
                'annotator_preferences': annotator_preferences.tolist(),
                'rating_probs': rating_probs
            }
        
        # Return list of initial values for each chain
        return [create_init() for _ in range(4)]  # Create 4 different initializations
    
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
                           seed: int = 42) -> ProgressiveResults:
        """Perform progressive training with increasing data budgets"""
        
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
            
            # Sample subset of training data
            subset_data = self.sample_subset(data['train'], fraction, seed + i)
            
            # Mask 50% for imputation training
            visible_train, masked_train = self.mask_data_for_training(subset_data, mask_fraction=0.5, seed=seed + i + 100)
            
            # Train on visible data only
            stan_data = self.prepare_stan_data(visible_train, config)
            
            # Create reasonable initial values
            init_values = self._create_initial_values(stan_data, seed + i)
            
            # Train model
            import time
            start_time = time.time()
            
            fit = self.model.sample(
                data=stan_data,
                chains=config.chains,
                iter_warmup=config.iter_warmup,
                iter_sampling=config.iter_sampling,
                adapt_delta=config.adapt_delta,
                max_treedepth=config.max_treedepth,
                seed=seed + i,
                inits=init_values
            )
            
            training_time = time.time() - start_time
            
            # Extract results
            training_log_lik = np.mean(fit.stan_variable('total_log_lik'))
            learned_embeddings = np.mean(fit.stan_variable('embeddings'), axis=0)
            
            # Compute KL divergence
            kl_div = self.compute_kl_divergence(learned_embeddings, true_embeddings)
            
            # Evaluate imputation accuracy on test set (mask 50% of test)
            visible_test, masked_test = self.mask_data_for_training(data['test'], mask_fraction=0.5, seed=seed + i + 200)
            accuracy_results = self.evaluate_imputation_accuracy(fit, visible_test, masked_test, stan_data)
            
            # Per-annotation-type accuracies
            rating_acc = accuracy_results['rating_accuracy']
            comparison_acc = 0.0  # No comparisons to evaluate
            ranking_acc = accuracy_results['ranking_accuracy']
            
            # Store results
            results.training_log_likelihoods.append(training_log_lik)
            results.test_log_likelihoods.append(np.mean([rating_acc, ranking_acc]))  # Average accuracy
            results.kl_divergences.append(kl_div)
            results.training_times.append(training_time)
            results.n_observations.append(len(visible_train['ratings']) + len(visible_train['rankings']))
            results.ratings_log_lik.append(rating_acc)
            # results.comparisons_log_lik.append(0.0)  # Removed comparison tracking
            results.rankings_log_lik.append(ranking_acc)
            
            logger.info(f"  Training log-likelihood: {training_log_lik:.3f}")
            logger.info(f"  Rating accuracy: {rating_acc:.3f}")
            logger.info(f"  Ranking accuracy (Kendall tau): {ranking_acc:.3f}")
            logger.info(f"  KL divergence: {kl_div:.3f}")
            logger.info(f"  Training time: {training_time:.1f}s")
        
        return results
    
    def plot_results(self, results: ProgressiveResults, output_dir: Path):
        """Create plots of progressive training results"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Training log-likelihood and test accuracy curves
        ax = axes[0, 0]
        ax.plot(results.budget_fractions, results.training_log_likelihoods, 'b-o', label='Training Log-Likelihood')
        ax2 = ax.twinx()
        ax2.plot(results.budget_fractions, results.test_log_likelihoods, 'r-o', label='Test Accuracy')
        ax.set_xlabel('Budget Fraction')
        ax.set_ylabel('Training Log-Likelihood', color='b')
        ax2.set_ylabel('Test Imputation Accuracy', color='r')
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
        
        # 4. Per-annotation-type accuracies
        ax = axes[1, 1]
        ax.plot(results.budget_fractions, results.ratings_log_lik, 'b-o', label='Rating Accuracy')
        ax.plot(results.budget_fractions, results.rankings_log_lik, 'g-o', label='Ranking Accuracy (τ)')
        ax.set_xlabel('Budget Fraction')
        ax.set_ylabel('Imputation Accuracy')
        ax.set_title('Per-Annotation-Type Accuracy')
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
    results = trainer.progressive_training(data_path, config, seed=12345)
    
    # Create plots
    output_dir = Path(__file__).parent / "domain_results"
    trainer.plot_results(results, output_dir)
    
    # Print summary
    print("Domain Model Training Complete!")
    print(f"Final test accuracy: {results.test_log_likelihoods[-1]:.3f}")
    print(f"Final rating accuracy: {results.ratings_log_lik[-1]:.3f}")
    print(f"Final ranking accuracy: {results.rankings_log_lik[-1]:.3f}")
    print(f"Final KL divergence: {results.kl_divergences[-1]:.3f}")
    print(f"Total training time: {sum(results.training_times):.1f}s")

if __name__ == "__main__":
    main()