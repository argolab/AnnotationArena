#!/usr/bin/env python3
"""Centralized configuration for ranking annotation experiments."""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import json
from pathlib import Path

@dataclass
class InstanceConfig:
    """Configuration for a single data instance."""
    K: int = 30   # number of items
    I: int = 10   # number of attributes  
    J: int = 5    # number of annotators
    D: int = 64   # embedding dimension
    C: int = 5    # number of rating categories
    
    # Data generation parameters
    sigma_annotator: float = 0.3    # annotator preference variance
    sigma_measurement: float = 0.1  # measurement noise variance
    alpha_dirichlet: float = 2.0    # Dirichlet concentration for rating thresholds
    temperature: float = 0.5        # temperature for ranking generation
    sigma_embedding_prior: float = 1.0   # embedding prior scale
    sigma_preference_prior: float = 1.0  # preference prior scale
    
    # Pairwise ranking limits
    max_pairs_per_tied_group: int = 10
    min_group_size: int = 2
    max_group_size: int = 6
    rankings_per_annotator_attribute: int = 10

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    encoder_layers: int = 4
    attention_heads: int = 8
    embedding_dim: int = 64
    dropout: float = 0.1
    embedding_type: str = "pairwise"

@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 40
    learning_rate: float = 1e-3
    embedding_anchor_reg: float = 0.0
    masking_rate: float = 0.5
    evaluation_frequency: int = 1
    
@dataclass
class ExperimentConfig:
    """Master configuration for ranking annotation experiments."""
    
    # Experiment type
    experiment_type: str = "single_instance"  # "single_instance" or "multi_instance"
    
    # Data configuration
    train_fraction: float = 0.80
    test_fraction: float = 0.20
    
    # Instance specifications
    instances: List[InstanceConfig] = field(default_factory=lambda: [InstanceConfig()])
    train_instance_indices: List[int] = field(default_factory=lambda: [0])
    test_instance_indices: List[int] = field(default_factory=lambda: [0])
    
    # Model and training
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Output configuration
    base_output_dir: str = "OUTPUT"
    save_plots: bool = True
    device: str = "cpu"
    
    # Domain model configuration (for backwards compatibility)
    chains: int = 3
    iter_warmup: int = 1000
    iter_sampling: int = 1000
    adapt_delta: float = 0.8
    max_treedepth: int = 10
    budget_fractions: List[float] = field(default_factory=lambda: [0.1, 1.0])
    save_stan_output: bool = True
    
    @property
    def num_instances(self) -> int:
        return len(self.instances)
    
    @property
    def output_dir(self) -> Path:
        """Get experiment-specific output directory."""
        base_dir = Path(self.base_output_dir) / "IMPUTER"
        if self.experiment_type == "single_instance":
            return base_dir / "single_instance"
        else:
            return base_dir / f"multi_instance_{len(self.train_instance_indices)}train_{len(self.test_instance_indices)}test"
    
    @property
    def data_dir(self) -> Path:
        """Get experiment-specific data directory."""
        base_dir = Path("generated_data")
        if self.experiment_type == "single_instance":
            return base_dir / "single_instance"
        else:
            return base_dir / f"multi_instance_{self.num_instances}instances"
    
    def get_instance_data_dir(self, instance_idx: int) -> Path:
        """Get data directory for specific instance."""
        return self.data_dir / f"instance_{instance_idx}"
    
    @classmethod
    def create_single_instance(
        cls, 
        instance_config: Optional[InstanceConfig] = None,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        **kwargs
    ) -> 'ExperimentConfig':
        """Create configuration for single instance experiment."""
        if instance_config is None:
            instance_config = InstanceConfig()
        if model_config is None:
            model_config = ModelConfig()
        if training_config is None:
            training_config = TrainingConfig()
            
        return cls(
            experiment_type="single_instance",
            instances=[instance_config],
            train_instance_indices=[0],
            test_instance_indices=[0],
            model_config=model_config,
            training_config=training_config,
            **kwargs
        )
    
    @classmethod
    def create_multi_instance(
        cls,
        instances: List[InstanceConfig],
        train_instance_indices: List[int],
        test_instance_indices: List[int],
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        **kwargs
    ) -> 'ExperimentConfig':
        """Create configuration for multi-instance experiment."""
        if model_config is None:
            model_config = ModelConfig()
        if training_config is None:
            training_config = TrainingConfig()
            
        return cls(
            experiment_type="multi_instance",
            instances=instances,
            train_instance_indices=train_instance_indices,
            test_instance_indices=test_instance_indices,
            model_config=model_config,
            training_config=training_config,
            **kwargs
        )
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for JSON serialization
        config_dict = {
            'experiment_type': self.experiment_type,
            'train_fraction': self.train_fraction,
            'test_fraction': self.test_fraction,
            'instances': [vars(instance) for instance in self.instances],
            'train_instance_indices': self.train_instance_indices,
            'test_instance_indices': self.test_instance_indices,
            'model_config': vars(self.model_config),
            'training_config': vars(self.training_config),
            'base_output_dir': self.base_output_dir,
            'save_plots': self.save_plots,
            'device': self.device,
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct objects
        instances = [InstanceConfig(**instance_dict) for instance_dict in config_dict['instances']]
        model_config = ModelConfig(**config_dict['model_config'])
        training_config = TrainingConfig(**config_dict['training_config'])
        
        return cls(
            experiment_type=config_dict['experiment_type'],
            train_fraction=config_dict['train_fraction'],
            test_fraction=config_dict['test_fraction'],
            instances=instances,
            train_instance_indices=config_dict['train_instance_indices'],
            test_instance_indices=config_dict['test_instance_indices'],
            model_config=model_config,
            training_config=training_config,
            base_output_dir=config_dict['base_output_dir'],
            save_plots=config_dict['save_plots'],
            device=config_dict['device'],
        )
    
    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.experiment_type not in ["single_instance", "multi_instance"]:
            raise ValueError(f"Invalid experiment_type: {self.experiment_type}")
        
        if not self.instances:
            raise ValueError("At least one instance configuration required")
        
        max_idx = len(self.instances) - 1
        for idx in self.train_instance_indices + self.test_instance_indices:
            if idx > max_idx:
                raise ValueError(f"Instance index {idx} exceeds available instances (0-{max_idx})")
        
        if self.experiment_type == "single_instance":
            if len(self.instances) != 1:
                raise ValueError("Single instance experiment requires exactly 1 instance")
            if self.train_instance_indices != [0] or self.test_instance_indices != [0]:
                raise ValueError("Single instance experiment requires train and test indices to be [0]")
    
    def get_legacy_properties(self) -> dict:
        """Get legacy properties for backwards compatibility."""
        if not self.instances:
            raise ValueError("No instances configured")
        
        # Use first instance for legacy compatibility
        first_instance = self.instances[0]
        return {
            'K': first_instance.K,
            'I': first_instance.I,
            'J': first_instance.J,
            'C': first_instance.C,
            'D': first_instance.D,
            'ranking_size': 2,  # Always 2 for pairwise
            'sigma_annotator': first_instance.sigma_annotator,
            'sigma_measurement': first_instance.sigma_measurement,
            'alpha_dirichlet': first_instance.alpha_dirichlet,
            'temperature': first_instance.temperature,
        }
    
    def __post_init__(self):
        self.validate()

DEFAULT_CONFIG = ExperimentConfig()

def load_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """Load configuration from file or return default."""
    if config_path is None:
        return DEFAULT_CONFIG
    return ExperimentConfig.load_from_file(config_path)