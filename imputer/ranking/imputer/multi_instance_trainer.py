"""
Multi-Instance Training Framework

This module provides different strategies for training models across multiple instances
using a generator-based approach for clean separation of data generation and training logic.
"""

import random
from typing import List, Iterator, Tuple, Any, Dict
import torch
from .trainer import ImputerTrainer
from .eval import EvaluationEngine
from .data import DataConverter


class MultiInstanceTrainerBase:
    """Base class for multi-instance training with generator-based data generation."""
    
    def __init__(self, model, eval_engine: EvaluationEngine, config, converter):
        self.model = model
        self.eval_engine = eval_engine
        self.config = config
        self.converter = converter
        self.trainer = ImputerTrainer(model, config.learning_rate, device=config.device)
    
    def train_on_instances(self, train_instances: List[Dict], test_instances: List[Dict]):
        """Unified training loop that uses generator from subclass."""
        generator = self.create_training_generator(
            train_instances, 
            self.config.total_batches, 
            self.config.batch_size
        )
        
        for step, batch in enumerate(generator):
            print(self.trainer.train_step(batch))
            
            # if self.should_evaluate(step):
            #     self.evaluate_on_test_instances(test_instances)
    
    def create_training_generator(self, train_instances: List[Dict], total_batches: int, batch_size: int) -> Iterator[Dict]:
        """Override in subclasses - returns iterator of batches."""
        raise NotImplementedError
    
    def should_evaluate(self, step: int) -> bool:
        """Override for evaluation frequency control."""
        return step % self.config.eval_frequency == 0
    
    def create_masked_batch(self, instance_data: Dict, masking_rate: float, batch_size: int) -> Dict:
        """Create batch with specified masking rate and batch size for self-supervised learning."""
        variables = self.converter.create_variables(instance_data)

        batch = self.converter.create_masked_batch(variables, masking_rate, batch_size)
        
        return batch
    
    def create_evaluation_batch(self, instance_data: Dict, masking_rate: float) -> Dict:
        """Create evaluation batch with Test_M (masked) and Test_O (observed) split."""
        # Extract variables and data from instance (this should be test data)
        rating_variables, ranking_variables = self.converter.create_variables_from_actual_data(
            instance_data, instance_data
        )
        
        # Process test data to get rating and ranking data dictionaries
        rating_data, ranking_data = self.converter.process_training_data(instance_data)
        
        # Create evaluation batch with masking (for evaluation: M% masked, (1-M)% observed)
        batch = self.converter.create_batch(
            rating_variables=rating_variables,
            ranking_variables=ranking_variables,
            rating_data=rating_data,
            ranking_data=ranking_data,
            mode="test",  # Use test mode for evaluation
            masking_rate=masking_rate
        )
        
        return batch
    
    def evaluate_on_test_instances(self, test_instances: List[Dict]):
        """Evaluate model on all test instances."""
        for test_instance in test_instances:
            # Create evaluation batch with masking
            batch = self.create_evaluation_batch(test_instance, self.config.test_masking_rate)
            
            # Use the evaluation engine to evaluate on this test instance
            # This will be implemented when we integrate with the evaluation engine
            raise NotImplementedError("MultiInstanceTrainerBase.evaluate_on_test_instances not yet implemented")


class SequentialMIT(MultiInstanceTrainerBase):
    """Sequential Multi-Instance Trainer: exhausts each instance before moving to next."""
    
    def create_training_generator(self, train_instances: List[Dict], total_batches: int, batch_size: int) -> Iterator[Dict]:
        """Generate batches per instance sequentially."""
        batches_per_instance = total_batches // len(train_instances)
        
        for instance_data in train_instances:
            for _ in range(batches_per_instance):
                masking_rate = random.choice(self.config.masking_rates)
                batch = self.create_masked_batch(instance_data, masking_rate, batch_size)
                yield batch


class MixedMIT(MultiInstanceTrainerBase):
    """Mixed Multi-Instance Trainer: IID sampling from all instances."""
    
    def create_training_generator(self, train_instances: List[Dict], total_batches: int, batch_size: int) -> Iterator[Dict]:
        """IID sampling from all instances."""
        for _ in range(total_batches):
            instance_data = random.choice(train_instances)
            masking_rate = random.choice(self.config.masking_rates)
            batch = self.create_masked_batch(instance_data, masking_rate, batch_size)
            yield batch


class GeneralMIT(MultiInstanceTrainerBase):
    """General Multi-Instance Trainer: for finetuning on single instances."""
    
    def finetune_on_instance(self, pretrained_model, instance_data: Dict):
        """Finetune on single instance (for test instance finetuning)."""
        # Split instance into Test_O_Masked and Test_O_Observed
        # Train on observed, validate on masked
        # This will be implemented when we integrate with the evaluation engine
        raise NotImplementedError("GeneralMIT.finetune_on_instance not yet implemented")
