#!/usr/bin/env python3
"""ICLR data generator for tie-breaking pairwise rankings."""

import numpy as np
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

try:
    import cmdstanpy as stan
    STAN_AVAILABLE = True
    logger.info("Using cmdstanpy for Stan interface")
except ImportError:
    STAN_AVAILABLE = False
    logger.error("cmdstanpy not available - please install: conda install -c conda-forge cmdstanpy")

@dataclass
class ICLRDatasetConfig:
    K: int = 30  # number of items
    I: int = 10  # number of attributes  
    J: int = 5   # number of annotators
    D: int = 64  # embedding dimension
    C: int = 5   # number of rating categories
    
    # Pairwise ranking limits
    max_pairs_per_tied_group: int = 10  # Maximum pairwise comparisons per tied group
    min_group_size: int = 2             # Minimum group size to generate pairs
    max_group_size: int = 6             # Maximum group size for all pairs
    
    train_fraction: float = 0.80
    test_fraction: float = 0.20

    sigma_annotator: float = 0.3
    sigma_measurement: float = 0.1
    alpha_dirichlet: float = 2.0
    temperature: float = 0.5

@dataclass  
class ICLRAnnotationDataset:
    embeddings: np.ndarray
    annotator_preferences: np.ndarray
    base_scores: np.ndarray
    all_ratings: np.ndarray
    all_pairwise_rankings: List[Dict]
    observed_ratings: List[Dict]
    missing_ratings: List[Dict]
    observed_pairwise_rankings: List[Dict]
    missing_pairwise_rankings: List[Dict]
    stats: Dict[str, Any]


class ICLRDataGenerator:
    
    def __init__(self, model_path: Optional[str] = None):
        if not STAN_AVAILABLE:
            raise ImportError("Stan not available - cannot generate data")
            
        if model_path is None:
            model_path = Path(__file__).parent / "models" / "iclr_data_generation.stan"
        
        logger.info(f"Compiling Stan model: {model_path}")
        # Force recompilation to avoid cache issues with random data generation
        self.model = stan.CmdStanModel(stan_file=str(model_path), force_compile=True)
        logger.info("Stan model compiled successfully")
    
    def generate_dataset(self, config: ICLRDatasetConfig, seed: Optional[int] = None) -> ICLRAnnotationDataset:
        """Generate synthetic annotation dataset using Stan."""
        logger.info(f"Generating dataset with config: {config}")
        logger.info(f"Expected annotations - Ratings: {config.I*config.J*config.K}")
        
        stan_data = {
            'K': config.K, 'I': config.I, 'J': config.J, 'D': config.D, 'C': config.C,
            'max_pairs_per_tied_group': config.max_pairs_per_tied_group,
            'min_group_size': config.min_group_size,
            'max_group_size': config.max_group_size,
            'sigma_annotator': config.sigma_annotator, 'sigma_measurement': config.sigma_measurement,
            'alpha_dirichlet': config.alpha_dirichlet, 'temperature': config.temperature
        }
        
        logger.info("Running Stan data generation...")
        fit = self.model.sample(
            data=stan_data, chains=1, iter_sampling=1, iter_warmup=0,
            adapt_engaged=False, fixed_param=True, seed=seed
        )
        logger.info("Stan generation completed")
        
        return self._extract_dataset(fit, config, seed)
    
    def _extract_dataset(self, fit, config: ICLRDatasetConfig, seed: Optional[int] = None) -> ICLRAnnotationDataset:
        """Extract and organize generated data into train/test splits."""
        logger.info("Extracting and organizing generated data...")
        
        embeddings = fit.stan_variable('embeddings')[0]
        annotator_preferences = fit.stan_variable('annotator_preferences')[0] 
        base_scores = fit.stan_variable('base_scores')[0]
        all_ratings = fit.stan_variable('all_rating_values')[0]
        
        # Extract pairwise ranking data
        num_pairwise_rankings = int(fit.stan_variable('num_pairwise_rankings')[0])
        pairwise_ranking_items = fit.stan_variable('pairwise_ranking_items')[0]
        pairwise_ranking_orders = fit.stan_variable('pairwise_ranking_orders')[0]
        pairwise_ranking_annotator = fit.stan_variable('pairwise_ranking_annotator')[0]
        pairwise_ranking_attribute = fit.stan_variable('pairwise_ranking_attribute')[0]
        pairwise_ranking_tied_rating = fit.stan_variable('pairwise_ranking_tied_rating')[0]
        
        logger.info(f"Generated {num_pairwise_rankings} pairwise rankings")
        
        logger.info("Performing deterministic train/test split...")
        
        if seed is not None:
            np.random.seed(seed + 1000)
        
        # Create rating list
        all_rating_list = []
        for ij_idx in range(config.I * config.J):
            i = (ij_idx // config.J) + 1  # Convert to 1-indexed
            j = (ij_idx % config.J) + 1   # Convert to 1-indexed
            for k in range(config.K):
                rating = {
                    'attribute': i, 'annotator': j, 'item': k + 1,  # Convert to 1-indexed
                    'value': int(all_ratings[ij_idx, k])
                }
                all_rating_list.append(rating)
        
        # Create pairwise ranking list
        all_pairwise_ranking_list = []
        for ranking_idx in range(num_pairwise_rankings):
            pairwise_ranking = {
                'attribute': int(pairwise_ranking_attribute[ranking_idx]),
                'annotator': int(pairwise_ranking_annotator[ranking_idx]),
                'items': [
                    int(pairwise_ranking_items[ranking_idx, 0]),
                    int(pairwise_ranking_items[ranking_idx, 1])
                ],
                'order': [1, 2] if pairwise_ranking_orders[ranking_idx] == 1 else [2, 1],
                'tied_rating': int(pairwise_ranking_tied_rating[ranking_idx])
            }
            all_pairwise_ranking_list.append(pairwise_ranking)
        
        # Shuffle and split ratings
        np.random.shuffle(all_rating_list)
        n_train_ratings = int(len(all_rating_list) * config.train_fraction)
        train_ratings = all_rating_list[:n_train_ratings]
        test_ratings = all_rating_list[n_train_ratings:]
        
        # Shuffle and split pairwise rankings
        np.random.shuffle(all_pairwise_ranking_list)
        n_train_pairwise_rankings = int(len(all_pairwise_ranking_list) * config.train_fraction)
        train_pairwise_rankings = all_pairwise_ranking_list[:n_train_pairwise_rankings]
        test_pairwise_rankings = all_pairwise_ranking_list[n_train_pairwise_rankings:]
        
        stats = {
            'total_possible_ratings': len(all_rating_list),
            'total_pairwise_rankings': len(all_pairwise_ranking_list),
            'train_ratings': len(train_ratings), 'test_ratings': len(test_ratings), 
            'train_pairwise_rankings': len(train_pairwise_rankings), 
            'test_pairwise_rankings': len(test_pairwise_rankings),
            'train_fraction': config.train_fraction,
            'test_fraction': config.test_fraction,
        }
        
        logger.info(f"Dataset statistics: {stats}")
        
        return ICLRAnnotationDataset(
            embeddings=embeddings, annotator_preferences=annotator_preferences,
            base_scores=base_scores, all_ratings=all_ratings,
            all_pairwise_rankings=all_pairwise_ranking_list,
            observed_ratings=train_ratings, missing_ratings=test_ratings,
            observed_pairwise_rankings=train_pairwise_rankings, 
            missing_pairwise_rankings=test_pairwise_rankings,
            stats=stats
        )
    
    def save_dataset(self, dataset: ICLRAnnotationDataset, output_dir: Path, name: str = "iclr_dataset"):
        """Save dataset to JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ground_truth = {
            'embeddings': dataset.embeddings.tolist(),
            'annotator_preferences': dataset.annotator_preferences.tolist(),
            'base_scores': dataset.base_scores.tolist()
        }
        
        with open(output_dir / f"{name}_ground_truth.json", 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        # Train data has both ratings and pairwise rankings
        train_data = {
            'ratings': dataset.observed_ratings, 
            'pairwise_rankings': dataset.observed_pairwise_rankings
        }
        with open(output_dir / f"{name}_train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        # Test data has both ratings and pairwise rankings
        test_data = {
            'ratings': dataset.missing_ratings, 
            'pairwise_rankings': dataset.missing_pairwise_rankings
        }
        with open(output_dir / f"{name}_test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        with open(output_dir / f"{name}_stats.json", 'w') as f:
            json.dump(dataset.stats, f, indent=2)
        
        logger.info(f"Dataset saved to {output_dir}/")
        logger.info(f"Files: {name}_{{ground_truth,train,test,stats}}.json")


def main():
    """Test data generation."""
    logging.basicConfig(level=logging.INFO)
    
    # Use centralized configuration
    from config import ExperimentConfig
    experiment_config = ExperimentConfig()
    
    # Convert to ICLRDatasetConfig using first instance
    instance_config = experiment_config.instances[0]
    iclr_config = ICLRDatasetConfig(
        K=instance_config.K,
        I=instance_config.I,
        J=instance_config.J,
        D=instance_config.D,
        C=instance_config.C,
        max_pairs_per_tied_group=instance_config.max_pairs_per_tied_group,
        min_group_size=instance_config.min_group_size,
        max_group_size=instance_config.max_group_size,
        train_fraction=experiment_config.train_fraction,
        test_fraction=experiment_config.test_fraction,
        sigma_annotator=instance_config.sigma_annotator,
        sigma_measurement=instance_config.sigma_measurement,
        alpha_dirichlet=instance_config.alpha_dirichlet,
        temperature=instance_config.temperature
    )
    
    generator = ICLRDataGenerator()
    dataset = generator.generate_dataset(iclr_config, seed=12345)
    
    # Save to proper subfolder structure
    output_dir = experiment_config.get_instance_data_dir(0)
    generator.save_dataset(dataset, output_dir, "iclr_complete")
    
    # Print some statistics
    print(f"\nGenerated dataset:")
    print(f"- Ratings: {len(dataset.all_ratings.flatten())}")
    print(f"- Pairwise rankings: {len(dataset.all_pairwise_rankings)}")
    print(f"- Train ratings: {len(dataset.observed_ratings)}")
    print(f"- Test ratings: {len(dataset.missing_ratings)}")
    print(f"- Train pairwise rankings: {len(dataset.observed_pairwise_rankings)}")
    print(f"- Test pairwise rankings: {len(dataset.missing_pairwise_rankings)}")
    
    # Show a few examples
    print(f"\nExample ratings:")
    for i, rating in enumerate(dataset.observed_ratings[:3]):
        print(f"  {rating}")
    
    print(f"\nExample pairwise rankings:")
    for i, ranking in enumerate(dataset.observed_pairwise_rankings[:3]):
        print(f"  {ranking}")


if __name__ == "__main__":
    main()