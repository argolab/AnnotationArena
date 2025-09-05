#!/usr/bin/env python3
"""Unified experiment runner with centralized configuration."""

import logging
from pathlib import Path
from config import ExperimentConfig, DEFAULT_CONFIG

def run_full_experiment(config: ExperimentConfig = None, regenerate_data: bool = False):
    """Run complete experiment: data generation + domain model training."""
    if config is None:
        config = DEFAULT_CONFIG
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(output_dir / "experiment.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("="*60)
    logger.info("STARTING RANKING ANNOTATION EXPERIMENT")
    logger.info("="*60)
    logger.info(f"Configuration: {config}")
    
    data_dir = Path("generated_data")
    if regenerate_data or not (data_dir / "test_complete_train.json").exists():
        logger.info("Generating synthetic data...")
        
        from complete_data_generator import CompleteDataGenerator
        generator = CompleteDataGenerator()
        
        data_config = config.to_data_generation_config()
        logger.info(f"Data generation config: {data_config}")
        
        dataset = generator.generate_dataset(data_config, seed=12345)
        generator.save_dataset(dataset, data_dir, "test_complete")
        
        logger.info(f"Data generation complete. Files saved to {data_dir}")
        logger.info(f"Train: {len(dataset.observed_ratings)} ratings, {len(dataset.observed_rankings)} rankings")
        logger.info(f"Test: {len(dataset.missing_ratings)} ratings, {len(dataset.missing_rankings)} rankings")
    else:
        logger.info("Using existing synthetic data")
    
    logger.info("Training domain model...")
    
    from domain_model_trainer import DomainModelTrainer
    trainer = DomainModelTrainer()
    
    domain_config = config.to_domain_model_config()
    logger.info(f"Domain model config: {domain_config}")
    
    results = trainer.progressive_training(
        data_path=data_dir, config=domain_config, seed=12345, output_dir=output_dir
    )
    
    trainer.plot_results(results, output_dir)
    
    logger.info("="*60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*60)
    logger.info(f"Final test log-loss: {results.test_log_likelihoods[-1]:.3f}")
    logger.info(f"Final rating log-loss: {results.ratings_log_lik[-1]:.3f}")
    logger.info(f"Final ranking log-loss: {results.rankings_log_lik[-1]:.3f}")
    logger.info(f"Final KL divergence: {results.kl_divergences[-1]:.3f}")
    logger.info(f"Total training time: {sum(results.training_times):.1f}s")
    logger.info(f"Results saved to: {output_dir}")
    
    import json
    with open(output_dir / "experiment_config.json", 'w') as f:
        config_dict = {
            'K': config.K, 'I': config.I, 'J': config.J, 'D': config.D, 'C': config.C,
            'ranking_size': config.ranking_size, 'rankings_per_annotator_attribute': config.rankings_per_annotator_attribute,
            'train_fraction': config.train_fraction, 'test_fraction': config.test_fraction,
            'sigma_annotator': config.sigma_annotator, 'sigma_measurement': config.sigma_measurement,
            'alpha_dirichlet': config.alpha_dirichlet, 'temperature': config.temperature,
            'sigma_embedding_prior': config.sigma_embedding_prior, 'sigma_preference_prior': config.sigma_preference_prior,
            'chains': config.chains, 'iter_warmup': config.iter_warmup, 'iter_sampling': config.iter_sampling,
            'adapt_delta': config.adapt_delta, 'max_treedepth': config.max_treedepth, 'budget_fractions': config.budget_fractions
        }
        json.dump(config_dict, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = run_full_experiment()
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Final test log-loss: {results.test_log_likelihoods[-1]:.3f}")
    print(f"Final rating log-loss: {results.ratings_log_lik[-1]:.3f}")
    print(f"Final ranking log-loss: {results.rankings_log_lik[-1]:.3f}")
    print(f"Final KL divergence: {results.kl_divergences[-1]:.3f}")
    print(f"Total training time: {sum(results.training_times):.1f}s")