"""
New experimental configuration system for Phase 5.

Supports all 4 evaluation strategies with wall clock timing and comprehensive logging.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class DataConfig:
    """Data generation configuration."""
    # Instance parameters
    num_instances: int = 5
    train_test_split: float = 0.7  # 70% train, 30% test

    # Data dimensions per instance
    K: int = 30   # number of items
    I: int = 10   # number of attributes
    J: int = 5    # number of annotators
    C: int = 5    # number of rating categories
    D: int = 32   # embedding dimension

    # Data generation parameters
    sigma_annotator: float = 0.3
    sigma_measurement: float = 0.1
    alpha_dirichlet: float = 2.0
    temperature: float = 0.5
    sigma_embedding_prior: float = 1.0
    sigma_preference_prior: float = 1.0

    # Pairwise ranking limits
    max_pairs_per_tied_group: int = 10
    min_group_size: int = 2
    max_group_size: int = 6
    rankings_per_annotator_attribute: int = 10


@dataclass
class ModelConfig:
    """Neural model architecture configuration."""
    encoder_layers: int = 4
    attention_heads: int = 8
    embedding_dim: int = 64
    dropout: float = 0.1
    embedding_type: str = "atom"
    max_rank_size: int = 2

    # Loss weighting configuration
    masked_loss_weight: float = 1.0      # Weight for masked portion of loss
    observed_loss_weight: float = 1.0    # Weight for observed portion of loss


@dataclass
class PretrainingConfig:
    """Pretraining configuration for Sequential/Mixed MIT."""
    strategy: str = "sequential"  # "sequential" or "mixed"
    total_batches: int = 1000
    batch_size: int = 32  # Number of masked versions per batch
    learning_rate: float = 1e-3
    train_heldout_split: float = 0.8
    eval_frequency: int = 100
    masking_rates: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    device: str = field(default="cpu", init=False)  # Set from main config, not user-configurable


@dataclass
class FinetuningConfig:
    """Finetuning configuration for GeneralMIT (both strategies)."""
    finetuning_steps: int = 200
    batch_size: int = 16  # Number of masked versions per batch
    learning_rate: float = 5e-4
    train_heldout_split: float = 0.8
    eval_frequency: int = 20  # Evaluate every N steps during finetuning
    test_masking_rate: float = 0.5  # Test instance masking rate for T_O/T_M split
    masking_rates: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    device: str = field(default="cpu", init=False)  # Set from main config, not user-configurable


@dataclass
class DomainConfig:
    """Domain model (EM) configuration with multiple sample counts."""
    # MCMC parameters
    chains: int = 4
    iter_warmup: int = 500
    iter_sampling: int = 1000
    adapt_delta: float = 0.8
    max_treedepth: int = 10

    # Multiple sample counts for evaluation
    sample_counts: List[int] = field(default_factory=lambda: [100, 200, 500, 1000, 2000])


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    test_masking_rate: float = 0.5
    eval_on_test_during_finetuning: bool = True  # Enable test evaluation during finetuning
    test_eval_frequency: int = 10  # Evaluate test set every N steps during finetuning


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Sub-configurations
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    pretraining_config: PretrainingConfig = field(default_factory=PretrainingConfig)
    finetuning_config: FinetuningConfig = field(default_factory=FinetuningConfig)
    domain_config: DomainConfig = field(default_factory=DomainConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Strategy selection (can disable strategies for testing)
    enabled_strategies: List[str] = field(default_factory=lambda: [
        "Pretrained_Imputer",
        "Pretrain_Finetuned_Imputer",
        "Finetuned_Imputer",
        "Domain_Model"
    ])

    # Output configuration
    output_dir: str = "experiment_results"
    save_models: bool = True
    save_training_histories: bool = True
    log_wall_times: bool = True

    # Experiment metadata
    experiment_name: str = "ranking_imputation_experiment"
    random_seed: int = 42

    # Device configuration (centralized)
    device: str = "cpu"  # "cpu" or "cuda" - applied to all components

    def __post_init__(self):
        """Set device in sub-configurations from main device setting."""
        self.pretraining_config.device = self.device
        self.finetuning_config.device = self.device

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = self._to_dict()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def _to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'data_config': self.data_config.__dict__,
            'model_config': self.model_config.__dict__,
            'pretraining_config': self.pretraining_config.__dict__,
            'finetuning_config': self.finetuning_config.__dict__,
            'domain_config': self.domain_config.__dict__,
            'evaluation_config': self.evaluation_config.__dict__,
            'enabled_strategies': self.enabled_strategies,
            'output_dir': self.output_dir,
            'save_models': self.save_models,
            'save_training_histories': self.save_training_histories,
            'log_wall_times': self.log_wall_times,
            'experiment_name': self.experiment_name,
            'random_seed': self.random_seed,
            'device': self.device
        }

    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        return cls(
            data_config=DataConfig(**config_dict['data_config']),
            model_config=ModelConfig(**config_dict['model_config']),
            pretraining_config=PretrainingConfig(**config_dict['pretraining_config']),
            finetuning_config=FinetuningConfig(**config_dict['finetuning_config']),
            domain_config=DomainConfig(**config_dict['domain_config']),
            evaluation_config=EvaluationConfig(**config_dict['evaluation_config']),
            enabled_strategies=config_dict['enabled_strategies'],
            output_dir=config_dict['output_dir'],
            save_models=config_dict['save_models'],
            save_training_histories=config_dict['save_training_histories'],
            log_wall_times=config_dict['log_wall_times'],
            experiment_name=config_dict['experiment_name'],
            random_seed=config_dict['random_seed'],
            device=config_dict.get('device', 'cpu')  # Default to 'cpu' for backward compatibility
        )


def create_test_config() -> ExperimentConfig:
    """Create a small test configuration for development."""
    return ExperimentConfig(
        data_config=DataConfig(
            num_instances=3,
            K=10,  # Smaller for testing
            I=3,
            J=3,
            D=16,  # Smaller for testing
            train_test_split=0.6  # 2 train, 1 test
        ),
        model_config=ModelConfig(
            encoder_layers=2,  # Smaller for testing
            attention_heads=4,
            embedding_dim=32,
            masked_loss_weight=2.0,  # Weight masked losses higher for testing
            observed_loss_weight=1.0
        ),
        pretraining_config=PretrainingConfig(
            total_batches=50,  # Much smaller for testing
            batch_size=8,
            eval_frequency=10  # Evaluate every 10 steps in test
        ),
        finetuning_config=FinetuningConfig(
            finetuning_steps=20,  # Smaller for testing
            eval_frequency=5,   # Evaluate every 5 steps in test
            test_masking_rate=0.3  # Match test expectation
        ),
        domain_config=DomainConfig(
            iter_warmup=100,  # Smaller for testing
            iter_sampling=200,
            sample_counts=[50, 100]  # Smaller for testing
        ),
        experiment_name="test_experiment",
        device="cpu"  # Explicitly set to CPU for testing
    )