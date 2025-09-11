#!/usr/bin/env python3
"""Centralized configuration for ranking annotation experiments."""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ExperimentConfig:
    """Master configuration for ranking annotation experiments."""
    
    # ICLR Pairwise Experiment Configuration (from iclr_data_generator.py)
    K: int = 30   # number of items
    I: int = 10   # number of attributes  
    J: int = 5    # number of annotators
    D: int = 64   # embedding dimension
    C: int = 5    # number of rating categories
    ranking_size: int = 2  # size of ranking sets (pairwise)
    rankings_per_annotator_attribute: int = 10  # rankings per (annotator, attribute) pair
    
    # Pairwise ranking limits (ICLR specific)
    max_pairs_per_tied_group: int = 10  # Maximum pairwise comparisons per tied group
    min_group_size: int = 2             # Minimum group size to generate pairs
    max_group_size: int = 6             # Maximum group size for all pairs
    
    train_fraction: float = 0.80
    test_fraction: float = 0.20
    
    sigma_annotator: float = 0.3    # annotator preference variance
    sigma_measurement: float = 0.1  # measurement noise variance
    alpha_dirichlet: float = 2.0    # Dirichlet concentration for rating thresholds
    temperature: float = 0.5        # temperature for ranking generation
    
    sigma_embedding_prior: float = 1.0   # embedding prior scale
    sigma_preference_prior: float = 1.0  # preference prior scale
    
    chains: int = 3           # number of MCMC chains
    iter_warmup: int = 1000   # warmup iterations
    iter_sampling: int = 1000 # sampling iterations
    adapt_delta: float = 0.8  # target acceptance rate
    max_treedepth: int = 10   # maximum tree depth
    
    budget_fractions: List[float] = None
    
    save_stan_output: bool = True  # save detailed Stan logs
    output_dir: str = "domain_results"
    
    def __post_init__(self):
        if self.budget_fractions is None:
            self.budget_fractions = [0.1, 1.0]
    
    def to_data_generation_config(self):
        """Convert to DatasetConfig for data generation."""
        from complete_data_generator import DatasetConfig
        return DatasetConfig(
            K=self.K, I=self.I, J=self.J, D=self.D, C=self.C,
            ranking_size=self.ranking_size,
            rankings_per_annotator_attribute=self.rankings_per_annotator_attribute,
            train_fraction=self.train_fraction, test_fraction=self.test_fraction,
            sigma_annotator=self.sigma_annotator, sigma_measurement=self.sigma_measurement,
            alpha_dirichlet=self.alpha_dirichlet, temperature=self.temperature
        )
    
    def to_iclr_data_generation_config(self):
        """Convert to ICLRDatasetConfig for ICLR pairwise data generation."""
        from iclr_data_generator import ICLRDatasetConfig
        return ICLRDatasetConfig(
            K=self.K, I=self.I, J=self.J, D=self.D, C=self.C,
            max_pairs_per_tied_group=self.max_pairs_per_tied_group,
            min_group_size=self.min_group_size,
            max_group_size=self.max_group_size,
            train_fraction=self.train_fraction, test_fraction=self.test_fraction,
            sigma_annotator=self.sigma_annotator, sigma_measurement=self.sigma_measurement,
            alpha_dirichlet=self.alpha_dirichlet, temperature=self.temperature
        )
    
    def to_domain_model_config(self):
        """Convert to DomainModelConfig for training."""
        from domain_model_trainer import DomainModelConfig
        return DomainModelConfig(
            chains=self.chains, iter_warmup=self.iter_warmup,
            iter_sampling=self.iter_sampling, adapt_delta=self.adapt_delta,
            max_treedepth=self.max_treedepth, sigma_annotator=self.sigma_annotator,
            sigma_measurement=self.sigma_measurement, alpha_dirichlet=self.alpha_dirichlet,
            temperature=self.temperature, sigma_embedding_prior=self.sigma_embedding_prior,
            sigma_preference_prior=self.sigma_preference_prior,
            budget_fractions=self.budget_fractions.copy()
        )

DEFAULT_CONFIG = ExperimentConfig()

def load_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """Load configuration from file or return default."""
    if config_path is None:
        return DEFAULT_CONFIG
    return DEFAULT_CONFIG