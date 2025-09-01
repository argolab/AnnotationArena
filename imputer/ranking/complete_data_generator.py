#!/usr/bin/env python3
"""
Complete data generator for mixed annotation types.

This module generates the full annotation space and splits it into 
observed/missing subsets based on configurable percentages.
"""

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
class DatasetConfig:
    """Configuration for dataset generation"""
    # Dimensions
    K: int = 10        # items
    I: int = 5         # attributes  
    J: int = 5         # annotators
    D: int = 32        # embedding dimension
    C: int = 5         # rating categories
    ranking_size: int = 10  # items per ranking
    rankings_per_annotator_attribute: int = 1  # number of rankings per annotator-attribute pair
    
    # Train/test split (50-50 split of all data)
    train_fraction: float = 0.80      # 50% for training
    test_fraction: float = 0.20       # 50% for testing
    
    # Model hyperparameters
    sigma_annotator: float = 0.1       # annotator variance
    sigma_measurement: float = 0.1     # measurement noise
    alpha_dirichlet: float = 1.0       # rating threshold concentration
    temperature: float = 0.5           # ranking temperature

@dataclass  
class AnnotationDataset:
    """Complete annotation dataset with observed/missing splits"""
    
    # Ground truth model
    embeddings: np.ndarray              # [K, D]
    annotator_preferences: np.ndarray   # [I*J, D] 
    base_scores: np.ndarray             # [I*J, K]
    
    # Complete annotation space
    all_ratings: np.ndarray             # [I*J, K] - all possible ratings
    all_rating_observed: np.ndarray     # [I*J, K] - 1=observed, 0=missing
    
    all_comparisons: List[Dict]         # All comparison data
    all_rankings: List[Dict]            # All ranking data
    
    # Observed/missing splits
    observed_ratings: List[Dict]
    missing_ratings: List[Dict]
    observed_comparisons: List[Dict]
    missing_comparisons: List[Dict]
    observed_rankings: List[Dict]
    missing_rankings: List[Dict]
    
    # Summary statistics
    stats: Dict[str, Any]


class CompleteDataGenerator:
    """Generates complete annotation datasets using Stan"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize generator with Stan model"""
        if not STAN_AVAILABLE:
            raise ImportError("Stan not available - cannot generate data")
            
        if model_path is None:
            model_path = Path(__file__).parent / "models" / "complete_data_generator.stan"
        
        logger.info(f"Compiling Stan model: {model_path}")
        self.model = stan.CmdStanModel(stan_file=str(model_path))
        logger.info("Stan model compiled successfully")
    
    def generate_dataset(self, config: DatasetConfig, seed: Optional[int] = None) -> AnnotationDataset:
        """Generate complete annotation dataset"""
        
        logger.info(f"Generating dataset with config: {config}")
        logger.info(f"Expected annotations - Ratings: {config.I*config.J*config.K}, "
                   f"Rankings: {config.I*config.J*config.rankings_per_annotator_attribute}")
        
        # Prepare Stan data
        stan_data = {
            'K': config.K,
            'I': config.I, 
            'J': config.J,
            'D': config.D,
            'C': config.C,
            'ranking_size': config.ranking_size,
            'rankings_per_annotator_attribute': config.rankings_per_annotator_attribute,
            'observed_rating_fraction': 1.0,  # Generate all data, split in Python
            'observed_ranking_fraction': 1.0,
            'sigma_annotator': config.sigma_annotator,
            'sigma_measurement': config.sigma_measurement,
            'alpha_dirichlet': config.alpha_dirichlet,
            'temperature': config.temperature
        }
        
        # Generate data with Stan
        logger.info("Running Stan data generation...")
        fit = self.model.sample(
            data=stan_data,
            chains=1,
            iter_sampling=1,
            iter_warmup=0,
            adapt_engaged=False,
            fixed_param=True,
            seed=seed
        )
        logger.info("Stan generation completed")
        
        # Extract results
        return self._extract_dataset(fit, config, seed)
    
    def _extract_dataset(self, fit, config: DatasetConfig, seed: Optional[int] = None) -> AnnotationDataset:
        """Extract and organize generated data"""
        
        logger.info("Extracting and organizing generated data...")
        
        # Extract ground truth model
        embeddings = fit.stan_variable('embeddings')[0]
        annotator_preferences = fit.stan_variable('annotator_preferences')[0] 
        base_scores = fit.stan_variable('base_scores')[0]
        
        # Extract complete annotation space (all generated, no splitting yet)
        all_ratings = fit.stan_variable('all_rating_values')[0]
        
        all_ranking_items = fit.stan_variable('all_ranking_items')[0]
        all_ranking_orders = fit.stan_variable('all_ranking_orders')[0]
        
        # Now do deterministic splitting in Python
        logger.info("Performing deterministic train/test split...")
        
        # Set random seed for reproducible splits
        if seed is not None:
            np.random.seed(seed + 1000)  # Offset to avoid collision with Stan seed
        
        # 1. CREATE ALL RATINGS
        all_rating_list = []
        for ij_idx in range(config.I * config.J):
            i = (ij_idx // config.J) + 1
            j = (ij_idx % config.J) + 1
            for k in range(config.K):
                rating = {
                    'attribute': i,
                    'annotator': j,
                    'item': k + 1,
                    'value': int(all_ratings[ij_idx, k])
                }
                all_rating_list.append(rating)
        
        # Shuffle and split ratings into train/test
        np.random.shuffle(all_rating_list)
        n_train_ratings = int(len(all_rating_list) * config.train_fraction)
        train_ratings = all_rating_list[:n_train_ratings]
        test_ratings = all_rating_list[n_train_ratings:]
        
        # 2. CREATE ALL RANKINGS
        all_ranking_list = []
        for ij_idx in range(config.I * config.J):
            i = (ij_idx // config.J) + 1
            j = (ij_idx % config.J) + 1
            
            # Create multiple rankings for this annotator-attribute pair
            for ranking_idx in range(config.rankings_per_annotator_attribute):
                global_ranking_idx = ij_idx * config.rankings_per_annotator_attribute + ranking_idx
                ranking = {
                    'attribute': i,
                    'annotator': j,
                    'items': [int(x) for x in all_ranking_items[global_ranking_idx]],
                    'order': [int(x) for x in all_ranking_orders[global_ranking_idx]]
                }
                all_ranking_list.append(ranking)
        
        # Shuffle and split rankings into train/test
        np.random.shuffle(all_ranking_list)
        n_train_rankings = int(len(all_ranking_list) * config.train_fraction)
        train_rankings = all_ranking_list[:n_train_rankings]
        test_rankings = all_ranking_list[n_train_rankings:]
        
        # Compute statistics
        stats = {
            'total_possible_ratings': len(all_rating_list),
            'total_possible_rankings': len(all_ranking_list),
            
            'train_ratings': len(train_ratings),
            'test_ratings': len(test_ratings), 
            'train_rankings': len(train_rankings),
            'test_rankings': len(test_rankings),
            
            'train_fraction': len(train_ratings) / len(all_rating_list),
            'test_fraction': len(test_ratings) / len(all_rating_list),
        }
        
        logger.info(f"Dataset statistics: {stats}")
        
        return AnnotationDataset(
            embeddings=embeddings,
            annotator_preferences=annotator_preferences,
            base_scores=base_scores,
            all_ratings=all_ratings,
            all_rating_observed=None,  # Not used in new approach
            all_comparisons=[],  # No longer used
            all_rankings=all_ranking_list,
            observed_ratings=train_ratings,
            missing_ratings=test_ratings,
            observed_comparisons=[],  # No longer used
            missing_comparisons=[],   # No longer used
            observed_rankings=train_rankings,
            missing_rankings=test_rankings,
            stats=stats
        )
    
    def save_dataset(self, dataset: AnnotationDataset, output_dir: Path, name: str = "dataset"):
        """Save complete dataset to files"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save ground truth model
        ground_truth = {
            'embeddings': dataset.embeddings.tolist(),
            'annotator_preferences': dataset.annotator_preferences.tolist(),
            'base_scores': dataset.base_scores.tolist()
        }
        
        with open(output_dir / f"{name}_ground_truth.json", 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        # Save train data 
        train_data = {
            'ratings': dataset.observed_ratings,
            'rankings': dataset.observed_rankings
        }
        
        with open(output_dir / f"{name}_train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        # Save test data  
        test_data = {
            'ratings': dataset.missing_ratings,
            'rankings': dataset.missing_rankings
        }
        
        with open(output_dir / f"{name}_test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Save statistics
        with open(output_dir / f"{name}_stats.json", 'w') as f:
            json.dump(dataset.stats, f, indent=2)
        
        logger.info(f"Dataset saved to {output_dir}/")
        logger.info(f"Files: {name}_{{ground_truth,train,test,stats}}.json")


def main():
    """Test the complete data generator"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration with default hyperparameters
    config = DatasetConfig()
    
    # Generate dataset
    generator = CompleteDataGenerator()
    dataset = generator.generate_dataset(config, seed=12345)
    
    # Save dataset
    output_dir = Path(__file__).parent / "generated_data"
    generator.save_dataset(dataset, output_dir, "test_complete")
    
    # Print summary
    print("Dataset Generated Successfully!")
    print(f"Train: {len(dataset.observed_ratings)} ratings, "
          f"{len(dataset.observed_rankings)} rankings")
    print(f"Test: {len(dataset.missing_ratings)} ratings, "
          f"{len(dataset.missing_rankings)} rankings")
    print(f"Split: {dataset.stats['train_fraction']:.1%} train, "
          f"{dataset.stats['test_fraction']:.1%} test")


if __name__ == "__main__":
    main()