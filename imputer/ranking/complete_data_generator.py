#!/usr/bin/env python3

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
    K: int = 10  # number of items
    I: int = 5  # number of attributes  
    J: int = 3   # number of annotators
    D: int = 16  # embedding dimension
    C: int = 5   # number of rating categories
    ranking_size: int = 5  # size of ranking sets
    rankings_per_annotator_attribute: int = 10  # rankings per (annotator, attribute) pair
    
    train_fraction: float = 0.80
    test_fraction: float = 0.20
    
    sigma_annotator: float = 0.3
    sigma_measurement: float = 0.1
    alpha_dirichlet: float = 2.0
    temperature: float = 0.5

@dataclass  
class AnnotationDataset:
    embeddings: np.ndarray
    annotator_preferences: np.ndarray
    base_scores: np.ndarray
    all_ratings: np.ndarray
    all_rating_observed: np.ndarray
    all_comparisons: List[Dict]
    all_rankings: List[Dict]
    observed_ratings: List[Dict]
    missing_ratings: List[Dict]
    observed_comparisons: List[Dict]
    missing_comparisons: List[Dict]
    observed_rankings: List[Dict]
    missing_rankings: List[Dict]
    stats: Dict[str, Any]


class CompleteDataGenerator:
    
    def __init__(self, model_path: Optional[str] = None):
        if not STAN_AVAILABLE:
            raise ImportError("Stan not available - cannot generate data")
            
        if model_path is None:
            model_path = Path(__file__).parent / "models" / "complete_data_generator.stan"
        
        logger.info(f"Compiling Stan model: {model_path}")
        self.model = stan.CmdStanModel(stan_file=str(model_path))
        logger.info("Stan model compiled successfully")
    
    def generate_dataset(self, config: DatasetConfig, seed: Optional[int] = None) -> AnnotationDataset:
        """Generate synthetic annotation dataset using Stan."""
        logger.info(f"Generating dataset with config: {config}")
        logger.info(f"Expected annotations - Ratings: {config.I*config.J*config.K}, "
                   f"Rankings: {config.I*config.J*config.rankings_per_annotator_attribute}")
        
        stan_data = {
            'K': config.K, 'I': config.I, 'J': config.J, 'D': config.D, 'C': config.C,
            'ranking_size': config.ranking_size,
            'rankings_per_annotator_attribute': config.rankings_per_annotator_attribute,
            'observed_rating_fraction': 1.0, 'observed_ranking_fraction': 1.0,
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
    
    def _extract_dataset(self, fit, config: DatasetConfig, seed: Optional[int] = None) -> AnnotationDataset:
        """Extract and organize generated data into train/test splits."""
        logger.info("Extracting and organizing generated data...")
        
        embeddings = fit.stan_variable('embeddings')[0]
        annotator_preferences = fit.stan_variable('annotator_preferences')[0] 
        base_scores = fit.stan_variable('base_scores')[0]
        
        all_ratings = fit.stan_variable('all_rating_values')[0]
        all_ranking_items = fit.stan_variable('all_ranking_items')[0]
        all_ranking_orders = fit.stan_variable('all_ranking_orders')[0]
        
        logger.info("Performing deterministic train/test split...")
        
        if seed is not None:
            np.random.seed(seed + 1000)
        
        all_rating_list = []
        for ij_idx in range(config.I * config.J):
            i = (ij_idx // config.J) + 1
            j = (ij_idx % config.J) + 1
            for k in range(config.K):
                rating = {
                    'attribute': i, 'annotator': j, 'item': k + 1,
                    'value': int(all_ratings[ij_idx, k])
                }
                all_rating_list.append(rating)
        
        np.random.shuffle(all_rating_list)
        n_train_ratings = int(len(all_rating_list) * config.train_fraction)
        train_ratings = all_rating_list[:n_train_ratings]
        test_ratings = all_rating_list[n_train_ratings:]
        
        all_ranking_list = []
        for ij_idx in range(config.I * config.J):
            i = (ij_idx // config.J) + 1
            j = (ij_idx % config.J) + 1
            
            for ranking_idx in range(config.rankings_per_annotator_attribute):
                global_ranking_idx = ij_idx * config.rankings_per_annotator_attribute + ranking_idx
                ranking = {
                    'attribute': i, 'annotator': j,
                    'items': [int(x) for x in all_ranking_items[global_ranking_idx]],
                    'order': [int(x) for x in all_ranking_orders[global_ranking_idx]]
                }
                all_ranking_list.append(ranking)
        
        np.random.shuffle(all_ranking_list)
        n_train_rankings = int(len(all_ranking_list) * config.train_fraction)
        train_rankings = all_ranking_list[:n_train_rankings]
        test_rankings = all_ranking_list[n_train_rankings:]
        
        stats = {
            'total_possible_ratings': len(all_rating_list),
            'total_possible_rankings': len(all_ranking_list),
            'train_ratings': len(train_ratings), 'test_ratings': len(test_ratings), 
            'train_rankings': len(train_rankings), 'test_rankings': len(test_rankings),
            'train_fraction': len(train_ratings) / len(all_rating_list),
            'test_fraction': len(test_ratings) / len(all_rating_list),
        }
        
        logger.info(f"Dataset statistics: {stats}")
        
        return AnnotationDataset(
            embeddings=embeddings, annotator_preferences=annotator_preferences,
            base_scores=base_scores, all_ratings=all_ratings, all_rating_observed=None,
            all_comparisons=[], all_rankings=all_ranking_list,
            observed_ratings=train_ratings, missing_ratings=test_ratings,
            observed_comparisons=[], missing_comparisons=[],
            observed_rankings=train_rankings, missing_rankings=test_rankings,
            stats=stats
        )
    
    def save_dataset(self, dataset: AnnotationDataset, output_dir: Path, name: str = "dataset"):
        """Save dataset to JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ground_truth = {
            'embeddings': dataset.embeddings.tolist(),
            'annotator_preferences': dataset.annotator_preferences.tolist(),
            'base_scores': dataset.base_scores.tolist()
        }
        
        with open(output_dir / f"{name}_ground_truth.json", 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        train_data = {'ratings': dataset.observed_ratings, 'rankings': dataset.observed_rankings}
        with open(output_dir / f"{name}_train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        test_data = {'ratings': dataset.missing_ratings, 'rankings': dataset.missing_rankings}
        with open(output_dir / f"{name}_test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        with open(output_dir / f"{name}_stats.json", 'w') as f:
            json.dump(dataset.stats, f, indent=2)
        
        logger.info(f"Dataset saved to {output_dir}/")
        logger.info(f"Files: {name}_{{ground_truth,train,test,stats}}.json")


def main():
    """Test data generation."""
    logging.basicConfig(level=logging.INFO)
    
    config = DatasetConfig()
    generator = CompleteDataGenerator()
    dataset = generator.generate_dataset(config, seed=12345)
    
    output_dir = Path(__file__).parent / "generated_data"
    generator.save_dataset(dataset, output_dir, "test_complete")
    
    print("Dataset Generated Successfully!")
    print(f"Train: {len(dataset.observed_ratings)} ratings, "
          f"{len(dataset.observed_rankings)} rankings")
    print(f"Test: {len(dataset.missing_ratings)} ratings, "
          f"{len(dataset.missing_rankings)} rankings")
    print(f"Split: {dataset.stats['train_fraction']:.1%} train, "
          f"{dataset.stats['test_fraction']:.1%} test")


if __name__ == "__main__":
    main()