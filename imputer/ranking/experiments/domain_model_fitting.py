#!/usr/bin/env python3
"""
Clean Domain Model Fitting Script

Core functionality:
1. Load config
2. Train model with MCMC
3. Evaluate with Stan vs Python comparison
"""

import numpy as np
import json
import logging
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

try:
    import cmdstanpy as stan
    STAN_AVAILABLE = True
    logger.info("Using cmdstanpy for Stan interface")
except ImportError:
    STAN_AVAILABLE = False
    logger.error("cmdstanpy not available - please install: conda install -c conda-forge cmdstanpy")
    exit(1)


@dataclass
class DomainModelConfig:
    """Configuration for domain model training."""
    chains: int = 4
    iter_warmup: int = 1000
    iter_sampling: int = 2000
    adapt_delta: float = 0.8
    max_treedepth: int = 15
    test_masking_rate: float = 0.5


@dataclass
class DomainModelResults:
    """Results from domain model training."""
    training_time: float
    n_observations: int
    stan_log_prob: float
    python_log_prob: float
    rhat_max: float


class DomainModelFitter:
    """Clean domain model fitter."""
    
    def __init__(self, model_path: str):
        """Initialize with Stan model."""
        self.model_path = Path(model_path)
        self.model = stan.CmdStanModel(stan_file=str(self.model_path))
        logger.info(f"Loaded Stan model from {self.model_path}")
    
    def load_test_data(self, test_data_file: str) -> Dict[str, Any]:
        """Load test data from file."""
        logger.info(f"Loading test data from {test_data_file}")
        with open(test_data_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data[0]  # Take first instance
        elif isinstance(data, dict) and 'test_instances' in data:
            return data['test_instances'][0]
        else:
            return data
    
    def prepare_stan_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for Stan model."""
        logger.info("Preparing Stan data...")
        
        # Extract dimensions
        rating_items = [r['item'] for r in data.get('ratings', [])]
        ranking_items = [r['items'] for r in data.get('pairwise_rankings', [])]
        rating_annotators = [r['annotator'] for r in data.get('ratings', [])]
        ranking_annotators = [r['annotator'] for r in data.get('pairwise_rankings', [])]
        rating_values = [r['value'] for r in data.get('ratings', [])]
        
        K = max(rating_items + [item for sublist in ranking_items for item in sublist]) if rating_items or ranking_items else 1
        I = max(rating_annotators + ranking_annotators) if rating_annotators or ranking_annotators else 1
        J = max(rating_annotators + ranking_annotators) if rating_annotators or ranking_annotators else 1
        C = max(rating_values) if rating_values else 1
        D = 32  # Default embedding dimension
        ranking_size = len(ranking_items[0]) if ranking_items else 2
        
        stan_data = {
            'K': K, 'I': I, 'J': J, 'D': D, 'C': C, 'ranking_size': ranking_size,
            'N_ratings': len(data.get('ratings', [])),
            'N_rankings': len(data.get('pairwise_rankings', [])),
            'rating_attributes': [r['attribute'] for r in data.get('ratings', [])],
            'rating_annotators': rating_annotators,
            'rating_items': rating_items,
            'rating_values': rating_values,
            'ranking_attributes': [r['attribute'] for r in data.get('pairwise_rankings', [])],
            'ranking_annotators': ranking_annotators,
            'ranking_items': ranking_items,
            'ranking_orders': [[i+1 for i in range(ranking_size)] for _ in data.get('pairwise_rankings', [])],
            'sigma_annotator': 0.3,
            'sigma_measurement': 0.1,
            'alpha_dirichlet': 2.0,
            'temperature': 0.5,
            'sigma_embedding_prior': 1.0,
            'sigma_preference_prior': 1.0,
        }
        
        logger.info(f"Stan data prepared: K={K}, I={I}, J={J}, D={D}, C={C}")
        logger.info(f"Ratings: {len(data.get('ratings', []))}, Rankings: {len(data.get('pairwise_rankings', []))}")
        
        return stan_data
    
    def create_initial_values(self, stan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create initial values for MCMC."""
        K = stan_data['K']
        I = stan_data['I']
        J = stan_data['J']
        D = stan_data['D']
        C = stan_data['C']
        
        def create_init():
            return {
                'embeddings_raw': np.random.normal(0, 1, (K, D)).tolist(),
                'mean_preferences': np.random.normal(0, 1, (I, D)).tolist(),
                'annotator_preferences': np.random.normal(0, 1, (I * J, D)).tolist(),
                'rating_thresholds_increments': np.random.exponential(1, (I * J, C - 2)).tolist()
            }
        
        return [create_init() for _ in range(4)]  # 4 chains
    
    def run_mcmc(self, stan_data: Dict[str, Any], config: DomainModelConfig, 
                 initial_values: List[Dict[str, Any]]) -> tuple:
        """Run MCMC sampling."""
        logger.info(f"Running MCMC: {config.chains} chains, {config.iter_sampling} samples")
        
        start_time = time.time()
        fit = self.model.sample(
            data=stan_data,
            chains=config.chains,
            iter_warmup=config.iter_warmup,
            iter_sampling=config.iter_sampling,
            adapt_delta=config.adapt_delta,
            max_treedepth=config.max_treedepth,
            inits=initial_values,
            seed=42
        )
        training_time = time.time() - start_time
        logger.info(f"MCMC completed in {training_time:.2f} seconds")
        
        return fit, training_time
    
    def compute_stan_log_prob(self, fit: stan.CmdStanMCMC, stan_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute log probability using Stan's log_prob method and extract individual components."""
        logger.info("Computing Stan log probability...")
        
        # Get posterior samples
        samples = {}
        for var_name in ['embeddings_raw', 'mean_preferences', 'annotator_preferences', 'rating_thresholds_increments']:
            samples[var_name] = fit.stan_variable(var_name)
        
        # Use first sample for simplicity
        params = {
            'embeddings_raw': samples['embeddings_raw'][0].tolist(),
            'mean_preferences': samples['mean_preferences'][0].tolist(),
            'annotator_preferences': samples['annotator_preferences'][0].tolist(),
            'rating_thresholds_increments': samples['rating_thresholds_increments'][0].tolist()
        }
        
        # Compute total log probability
        log_prob_result = self.model.log_prob(params, stan_data)
        if hasattr(log_prob_result, 'iloc'):
            total_log_prob = log_prob_result.iloc[0, 0]
        else:
            total_log_prob = float(log_prob_result)
        
        # Extract individual components from generated quantities
        log_lik_ratings = fit.stan_variable('log_lik_ratings')[0]
        log_lik_rankings = fit.stan_variable('log_lik_rankings')[0]
        total_log_lik = fit.stan_variable('total_log_lik')[0]
        
        results = {
            'total_log_prob': total_log_prob,
            'log_lik_ratings': log_lik_ratings,
            'log_lik_rankings': log_lik_rankings,
            'total_log_lik': total_log_lik,
            'log_prior': total_log_prob - total_log_lik  # Prior = Total - Likelihood
        }
        
        logger.info(f"Stan log probability breakdown:")
        logger.info(f"  Total log prob: {total_log_prob:.2f}")
        logger.info(f"  Log lik ratings: {log_lik_ratings:.2f}")
        logger.info(f"  Log lik rankings: {log_lik_rankings:.2f}")
        logger.info(f"  Total log lik: {total_log_lik:.2f}")
        logger.info(f"  Log prior: {results['log_prior']:.2f}")
        
        return results
    
    def compute_python_log_prob(self, fit: stan.CmdStanMCMC, stan_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute log probability using Python implementation - EXACT match to Stan model."""
        logger.info("Computing Python log probability...")
        
        # Get posterior samples
        samples = {}
        for var_name in ['embeddings_raw', 'mean_preferences', 'annotator_preferences', 'rating_thresholds_increments']:
            samples[var_name] = fit.stan_variable(var_name)
        
        # Use first sample
        embeddings_raw = samples['embeddings_raw'][0]
        mean_preferences = samples['mean_preferences'][0]
        annotator_preferences = samples['annotator_preferences'][0]
        rating_thresholds_increments = samples['rating_thresholds_increments'][0]
        
        # Extract dimensions
        K = stan_data['K']
        I = stan_data['I']
        J = stan_data['J']
        D = stan_data['D']
        C = stan_data['C']
        sigma_annotator = stan_data['sigma_annotator']
        sigma_measurement = stan_data['sigma_measurement']
        sigma_embedding_prior = stan_data['sigma_embedding_prior']
        sigma_preference_prior = stan_data['sigma_preference_prior']
        
        # ===== TRANSFORMED PARAMETERS (EXACT COPY FROM STAN) =====
        
        # 1. Normalize embeddings to unit norm
        embeddings = np.zeros_like(embeddings_raw)
        for k in range(K):
            norm = np.sqrt(np.sum(embeddings_raw[k] ** 2))
            if norm > 1e-10:
                embeddings[k] = embeddings_raw[k] / norm
            else:
                embeddings[k] = embeddings_raw[k]
        
        # 2. Compute base scores: z_ij_k = v_ij · e_k
        base_scores = np.zeros((I * J, K))
        for i in range(I):
            for j in range(J):
                idx = i * J + j
                for k in range(K):
                    base_scores[idx, k] = np.dot(annotator_preferences[idx], embeddings[k])
        
        # 3. Construct rating thresholds
        rating_thresholds = np.zeros((I * J, C + 1))
        for ij in range(I * J):
            rating_thresholds[ij, 0] = float('-inf')  # -∞ for category 1
            rating_thresholds[ij, 1] = 0.0  # First threshold FIXED at 0
            for c in range(2, C):
                rating_thresholds[ij, c] = rating_thresholds[ij, c-1] + abs(rating_thresholds_increments[ij, c-2])
            rating_thresholds[ij, C] = float('inf')  # +∞ for category C
        
        # ===== PRIORS (EXACT COPY FROM STAN) =====
        
        log_prior = 0.0
        
        # Raw embeddings: e_k ~ N(0, σ_e²I)
        for k in range(K):
            log_prior += np.sum(-0.5 * (embeddings_raw[k] / sigma_embedding_prior) ** 2)
        
        # Mean preferences: v_i ~ N(0, σ_v²I)
        for i in range(I):
            log_prior += np.sum(-0.5 * (mean_preferences[i] / sigma_preference_prior) ** 2)
        
        # Annotator preferences: v_ij ~ N(v_i, σ_a²I)
        for i in range(I):
            for j in range(J):
                idx = i * J + j
                diff = annotator_preferences[idx] - mean_preferences[i]
                log_prior += np.sum(-0.5 * (diff / sigma_annotator) ** 2)
        
        # Rating threshold increments
        for ij in range(I * J):
            for c in range(C - 2):
                log_prior += -0.5 * (rating_thresholds_increments[ij, c] / 0.5) ** 2
        
        # ===== LIKELIHOODS (EXACT COPY FROM STAN) =====
        
        log_lik_ratings = 0.0
        log_lik_rankings = 0.0
        
        # 1. RATING LIKELIHOOD
        N_ratings = stan_data['N_ratings']
        for n in range(N_ratings):
            i = stan_data['rating_attributes'][n] - 1  # Convert to 0-based
            j = stan_data['rating_annotators'][n] - 1
            k = stan_data['rating_items'][n] - 1
            c = stan_data['rating_values'][n] - 1
            ij_idx = i * J + j
            
            # Base score: z_ijk = v_ij · e_k
            base_score = base_scores[ij_idx, k]
            
            # Rating likelihood: P(rating = c) = Φ((Q_c - z)/σ_m) - Φ((Q_{c-1} - z)/σ_m)
            upper_threshold = rating_thresholds[ij_idx, c + 1]
            lower_threshold = rating_thresholds[ij_idx, c]
            
            if upper_threshold == float('inf'):
                upper_prob = 1.0
            else:
                upper_prob = 0.5 * (1 + np.tanh((upper_threshold - base_score) / sigma_measurement / np.sqrt(2)))
            
            if lower_threshold == float('-inf'):
                lower_prob = 0.0
            else:
                lower_prob = 0.5 * (1 + np.tanh((lower_threshold - base_score) / sigma_measurement / np.sqrt(2)))
            
            bin_prob = upper_prob - lower_prob
            
            # Numerical stability
            if bin_prob > 1e-8:
                log_lik_ratings += np.log(bin_prob)
            else:
                log_lik_ratings += np.log(1e-8)
        
        # 2. PAIRWISE RANKING LIKELIHOOD
        N_rankings = stan_data['N_rankings']
        temperature = stan_data['temperature']
        for n in range(N_rankings):
            i = stan_data['ranking_attributes'][n] - 1
            j = stan_data['ranking_annotators'][n] - 1
            ij_idx = i * J + j
            
            # For pairwise: ranking_items[n] = [item1, item2], ranking_orders[n] = [1, 2] or [2, 1]
            item1 = stan_data['ranking_items'][n][0] - 1  # Convert to 0-based
            item2 = stan_data['ranking_items'][n][1] - 1
            score1 = base_scores[ij_idx, item1] / temperature
            score2 = base_scores[ij_idx, item2] / temperature
            
            # If order = [1, 2], item1 > item2, so P(item1 > item2) = sigmoid(score1 - score2)
            # If order = [2, 1], item2 > item1, so P(item2 > item1) = sigmoid(score2 - score1)
            if stan_data['ranking_orders'][n][0] == 1:  # item1 ranks first
                log_lik_rankings += np.log(1 / (1 + np.exp(-(score1 - score2))))
            else:  # item2 ranks first
                log_lik_rankings += np.log(1 / (1 + np.exp(-(score2 - score1))))
        
        total_log_lik = log_lik_ratings + log_lik_rankings
        total_log_prob = log_prior + total_log_lik
        
        results = {
            'total_log_prob': total_log_prob,
            'log_lik_ratings': log_lik_ratings,
            'log_lik_rankings': log_lik_rankings,
            'total_log_lik': total_log_lik,
            'log_prior': log_prior
        }
        
        logger.info(f"Python log probability breakdown:")
        logger.info(f"  Total log prob: {total_log_prob:.2f}")
        logger.info(f"  Log lik ratings: {log_lik_ratings:.2f}")
        logger.info(f"  Log lik rankings: {log_lik_rankings:.2f}")
        logger.info(f"  Total log lik: {total_log_lik:.2f}")
        logger.info(f"  Log prior: {log_prior:.2f}")
        
        return results
    
    def compute_convergence_diagnostics(self, fit: stan.CmdStanMCMC) -> Dict[str, float]:
        """Compute basic convergence diagnostics."""
        logger.info("Computing convergence diagnostics...")
        
        summary = fit.summary()
        
        # Rhat (only available for multiple chains)
        if 'Rhat' in summary.columns:
            rhat_values = summary['Rhat'].values
            rhat_max = np.max(rhat_values)
        else:
            rhat_max = 1.0  # Single chain, no Rhat available
        
        diagnostics = {
            'rhat_max': float(rhat_max)
        }
        
        logger.info(f"Rhat max: {rhat_max:.3f}")
        return diagnostics
    
    def fit_domain_model(self, test_data_file: str, config: DomainModelConfig,
                        output_dir: str = "domain_model_results") -> DomainModelResults:
        """Main fitting function."""
        logger.info("="*60)
        logger.info("STARTING DOMAIN MODEL FITTING")
        logger.info("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        test_instance = self.load_test_data(test_data_file)
        
        # Prepare data
        stan_data = self.prepare_stan_data(test_instance)
        
        # Create initial values
        initial_values = self.create_initial_values(stan_data)
        
        # Run MCMC
        fit, training_time = self.run_mcmc(stan_data, config, initial_values)
        
        # Compute diagnostics
        diagnostics = self.compute_convergence_diagnostics(fit)
        
        # Compute log probabilities
        stan_results = self.compute_stan_log_prob(fit, stan_data)
        python_results = self.compute_python_log_prob(fit, stan_data)
        
        # Compile results
        results = DomainModelResults(
            training_time=training_time,
            n_observations=stan_data['N_ratings'] + stan_data['N_rankings'],
            stan_log_prob=stan_results['total_log_prob'],
            python_log_prob=python_results['total_log_prob'],
            rhat_max=diagnostics['rhat_max']
        )
        
        # Save results
        results_file = output_path / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        
        # Print results
        logger.info("="*60)
        logger.info("DOMAIN MODEL FITTING COMPLETED")
        logger.info("="*60)
        logger.info(f"Training time: {training_time:.2f}s")
        logger.info(f"Observations: {results.n_observations}")
        logger.info("")
        logger.info("DETAILED COMPARISON:")
        logger.info(f"Stan vs Python differences:")
        logger.info(f"  Total log prob: {stan_results['total_log_prob']:.2f} vs {python_results['total_log_prob']:.2f} (diff: {stan_results['total_log_prob'] - python_results['total_log_prob']:.2f})")
        logger.info(f"  Log lik ratings: {stan_results['log_lik_ratings']:.2f} vs {python_results['log_lik_ratings']:.2f} (diff: {stan_results['log_lik_ratings'] - python_results['log_lik_ratings']:.2f})")
        logger.info(f"  Log lik rankings: {stan_results['log_lik_rankings']:.2f} vs {python_results['log_lik_rankings']:.2f} (diff: {stan_results['log_lik_rankings'] - python_results['log_lik_rankings']:.2f})")
        logger.info(f"  Log prior: {stan_results['log_prior']:.2f} vs {python_results['log_prior']:.2f} (diff: {stan_results['log_prior'] - python_results['log_prior']:.2f})")
        logger.info("")
        logger.info(f"Rhat max: {results.rhat_max:.3f}")
        logger.info(f"Results saved to {results_file}")
        
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Clean Domain Model Fitting')
    parser.add_argument('--test_data', required=True, help='Path to test data file')
    parser.add_argument('--model_path', default='models/domain_model.stan', help='Path to Stan model')
    parser.add_argument('--chains', type=int, default=4, help='Number of chains')
    parser.add_argument('--iter_warmup', type=int, default=1000, help='Warmup iterations')
    parser.add_argument('--iter_sampling', type=int, default=2000, help='Sampling iterations')
    parser.add_argument('--output_dir', default='domain_model_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('domain_model_fitting.log')
        ]
    )
    
    # Create config
    config = DomainModelConfig(
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling
    )
    
    # Run fitting
    fitter = DomainModelFitter(args.model_path)
    results = fitter.fit_domain_model(args.test_data, config, args.output_dir)
    
    return results


if __name__ == "__main__":
    main()