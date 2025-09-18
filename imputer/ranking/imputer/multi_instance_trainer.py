"""
Multi-Instance Training Framework

This module provides different strategies for training models across multiple instances
with clean separation of concerns: MIT handles masking and data coordination,
trainer handles only training with callbacks for evaluation.
"""

import random
import copy
from typing import List, Iterator, Tuple, Any, Dict
import torch
from .trainer import ImputerTrainer, EvaluationCallback
from .eval import EvaluationEngine
from .data import DataConverter, RankingData


class MultiInstanceTrainerBase:
    """Base class for multi-instance training with masking and evaluation coordination."""

    def __init__(self, model, eval_engine: EvaluationEngine, config, converter: DataConverter):
        self.model = model
        self.eval_engine = eval_engine
        self.config = config
        self.converter = converter
        self.trainer = ImputerTrainer(model, config.learning_rate, device=config.device)
        self.train_heldout_split = getattr(config, 'train_heldout_split', 0.8)  # 80/20 default

    def apply_masking(self, variables: List[RankingData], masking_rate: float) -> List[RankingData]:
        """Apply masking: M% of variables are masked, 100 - M% are observed."""
        if len(variables) == 0:
            return []

        masked_variables = []
        num_to_mask = int(len(variables) * masking_rate)
        masked_indices = random.sample(list(range(len(variables))), num_to_mask)

        for i, var in enumerate(variables):
            if i in masked_indices:
                # Create masked version
                masked_var = RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=var.is_listwise,
                    item_ids=var.item_ids,
                    is_masked=True,  # Mark as masked
                    rating_value=var.rating_value,  # Keep original value for reference
                    ranking_order=var.ranking_order  # Keep original order for reference
                )
                masked_variables.append(masked_var)
            else:
                # Keep original (observed) for conditioning
                observed_var = RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=var.is_listwise,
                    item_ids=var.item_ids,
                    is_masked=False,  # Mark as observed
                    rating_value=var.rating_value,
                    ranking_order=var.ranking_order
                )
                masked_variables.append(observed_var)

        return masked_variables

    def split_train_heldout(self, variables: List[RankingData]) -> Tuple[List[RankingData], List[RankingData]]:
        """Split variables into training and heldout sets based on train_heldout_split ratio."""
        if len(variables) == 0:
            return [], []

        num_train = int(len(variables) * self.train_heldout_split)
        shuffled_vars = copy.deepcopy(variables)
        random.shuffle(shuffled_vars)

        train_vars = shuffled_vars[:num_train]
        heldout_vars = shuffled_vars[num_train:]

        return train_vars, heldout_vars
    
    def train_on_instances(self, train_instances: List[Dict], test_instances: List[Dict]):
        """Train using batch generator with multiple masked versions per batch."""
        # Set up evaluation callback on heldout data
        heldout_variables = self.setup_heldout_evaluation_callback(train_instances)

        # Create batch generator and train
        training_results = []
        batch_generator = self.create_training_batch_generator(train_instances)

        for step, batch_of_masked_versions in enumerate(batch_generator):
            if step >= self.config.total_batches:
                break

            # Train on all masked versions in this batch
            result = self.trainer.train_step(batch_of_masked_versions)
            training_results.append(result)

            if step % max(1, self.config.total_batches // 10) == 0:
                print(f"Step {step}: {result}")

        return {
            'training_results': training_results,
            'heldout_variables': heldout_variables
        }

    def create_training_batch_generator(self, train_instances: List[Dict]) -> Iterator[List[List[RankingData]]]:
        """Create generator that yields batches of masked versions - override in subclasses."""
        raise NotImplementedError("Subclasses must implement create_training_batch_generator")


    def setup_heldout_evaluation_callback(self, train_instances: List[Dict]) -> List[RankingData]:
        """Set up evaluation callback on heldout data - override in subclasses."""
        raise NotImplementedError("Subclasses must implement setup_heldout_evaluation_callback")
            


class SequentialMIT(MultiInstanceTrainerBase):
    """Sequential Multi-Instance Trainer: process instances sequentially with train/heldout splits."""

    def __init__(self, model, eval_engine: EvaluationEngine, config, converter: DataConverter):
        super().__init__(model, eval_engine, config, converter)
        self.current_instance_idx = 0
        self.instance_train_sets = []
        self.instance_heldout_sets = []

    def setup_heldout_evaluation_callback(self, train_instances: List[Dict]) -> List[RankingData]:
        """Set up instance-wise train/heldout splits and evaluation callback."""
        all_heldout_variables = []

        for instance_data in train_instances:
            variables = self.converter.create_variables(instance_data)
            train_vars, heldout_vars = self.split_train_heldout(variables)

            self.instance_train_sets.append(train_vars)
            self.instance_heldout_sets.append(heldout_vars)
            all_heldout_variables.extend(heldout_vars)

        # Set up evaluation callback on combined heldout data
        if all_heldout_variables:
            callback = EvaluationCallback(
                eval_engine=self.eval_engine,
                test_variables=all_heldout_variables,
                test_data={},  # Not needed for direct variable evaluation
                converter=self.converter,
                masking_rate=random.choice(self.config.masking_rates),
                device=self.config.device
            )
            self.trainer.register_callback(callback)

        return all_heldout_variables

    def create_training_batch_generator(self, train_instances: List[Dict]) -> Iterator[List[List[RankingData]]]:
        """Sequential: cycle through instances, creating multiple masked versions per batch."""
        while True:  # Infinite generator
            for instance_train_vars in self.instance_train_sets:
                if len(instance_train_vars) == 0:
                    continue

                # Create batch_size different masked versions of this instance's training variables
                batch_of_masked_versions = []
                for _ in range(self.config.batch_size):
                    masking_rate = random.choice(self.config.masking_rates)
                    masked_version = self.apply_masking(instance_train_vars, masking_rate)
                    batch_of_masked_versions.append(masked_version)

                yield batch_of_masked_versions



class MixedMIT(MultiInstanceTrainerBase):
    """Mixed Multi-Instance Trainer: IID sampling from combined instance pool."""

    def __init__(self, model, eval_engine: EvaluationEngine, config, converter: DataConverter):
        super().__init__(model, eval_engine, config, converter)
        self.instance_train_sets = []
        self.instance_heldout_sets = []

    def setup_heldout_evaluation_callback(self, train_instances: List[Dict]) -> List[RankingData]:
        """Set up instance-wise train/heldout splits for mixed sampling."""
        all_heldout_variables = []

        for instance_data in train_instances:
            variables = self.converter.create_variables(instance_data)
            train_vars, heldout_vars = self.split_train_heldout(variables)

            self.instance_train_sets.append(train_vars)
            self.instance_heldout_sets.append(heldout_vars)
            all_heldout_variables.extend(heldout_vars)

        # Set up evaluation callback on combined heldout data
        if all_heldout_variables:
            callback = EvaluationCallback(
                eval_engine=self.eval_engine,
                test_variables=all_heldout_variables,
                test_data={},  # Not needed for direct variable evaluation
                converter=self.converter,
                masking_rate=random.choice(self.config.masking_rates),
                device=self.config.device
            )
            self.trainer.register_callback(callback)

        return all_heldout_variables

    def create_training_batch_generator(self, train_instances: List[Dict]) -> Iterator[List[List[RankingData]]]:
        """Mixed: randomly sample instance, create multiple masked versions per batch."""
        while True:  # Infinite generator
            # Randomly select an instance's training variables
            if not self.instance_train_sets:
                # Fallback if not properly initialized
                random_instance_data = random.choice(train_instances)
                instance_vars = self.converter.create_variables(random_instance_data)
                train_vars, _ = self.split_train_heldout(instance_vars)
            else:
                # Use pre-split training sets
                train_vars = random.choice([train_set for train_set in self.instance_train_sets if len(train_set) > 0])

            if len(train_vars) == 0:
                continue

            # Create batch_size different masked versions of this randomly selected instance
            batch_of_masked_versions = []
            for _ in range(self.config.batch_size):
                masking_rate = random.choice(self.config.masking_rates)
                masked_version = self.apply_masking(train_vars, masking_rate)
                batch_of_masked_versions.append(masked_version)

            yield batch_of_masked_versions



class GeneralMIT(MultiInstanceTrainerBase):
    """General Multi-Instance Trainer: for finetuning on single test instances."""

    def __init__(self, model, eval_engine: EvaluationEngine, config, converter: DataConverter):
        super().__init__(model, eval_engine, config, converter)
        self.test_instance_variables = []
        self.t_o_train_vars = []
        self.t_o_heldout_vars = []

    def finetune_on_instance(self, pretrained_model, test_instance: Dict):
        """
        Finetune pretrained model on test instance.

        Flow:
        1. Test Instance -> T_O (observed) + T_M (masked)
        2. T_O -> T_O_train + T_O_heldout
        3. Train on T_O_train, eval on T_O_heldout during training
        4. Final eval on full test instance (T_O + T_M)
        """
        # Step 1: Split test instance into T_O and T_M
        self.test_instance_variables = self.converter.create_variables(test_instance)
        test_masking_rate = getattr(self.config, 'test_masking_rate', 0.5)

        # Apply test-level masking to get T_O and T_M
        t_o_variables = []
        t_m_variables = []

        num_to_mask = int(len(self.test_instance_variables) * test_masking_rate)
        masked_indices = random.sample(list(range(len(self.test_instance_variables))), num_to_mask)

        for i, var in enumerate(self.test_instance_variables):
            if i in masked_indices:
                # This becomes T_M (will be masked in final evaluation)
                t_m_var = copy.deepcopy(var)
                t_m_var.is_masked = True
                t_m_variables.append(t_m_var)
            else:
                # This becomes T_O (observed for finetuning)
                t_o_var = copy.deepcopy(var)
                t_o_var.is_masked = False
                t_o_variables.append(t_o_var)

        # Step 2: Split T_O into training and heldout
        self.t_o_train_vars, self.t_o_heldout_vars = self.split_train_heldout(t_o_variables)

        # Step 3: Set up evaluation callback on T_O_heldout for training monitoring
        if self.t_o_heldout_vars:
            callback = EvaluationCallback(
                eval_engine=self.eval_engine,
                test_variables=self.t_o_heldout_vars,
                test_data={},
                converter=self.converter,
                masking_rate=random.choice(self.config.masking_rates),
                device=self.config.device
            )
            self.trainer.register_callback(callback)

        # Step 4: Finetune on T_O_train using batch generator
        finetuning_results = []
        batch_generator = self.create_training_batch_generator([])
        finetuning_steps = getattr(self.config, 'finetuning_steps', 100)

        for step, batch_of_masked_versions in enumerate(batch_generator):
            if step >= finetuning_steps:
                break

            if not batch_of_masked_versions or len(batch_of_masked_versions) == 0:
                continue

            result = self.trainer.train_step(batch_of_masked_versions)
            finetuning_results.append(result)

            if step % max(1, finetuning_steps // 10) == 0:
                print(f"Finetuning step {step}: {result}")

        # Step 5: Final evaluation on full test instance (T_O + T_M)
        # Create test variables with proper masking (T_M masked, T_O observed)
        final_test_variables = t_o_variables + t_m_variables

        final_results = self.eval_engine.evaluate_model(
            model=self.model,
            variables=final_test_variables,
            masking_rate=0.0,  # No additional masking - already have T_M marked as masked
            converter=self.converter,
            device=self.config.device
        )

        return {
            'finetuning_results': finetuning_results,
            'final_evaluation': final_results,
            't_o_variables': t_o_variables,
            't_m_variables': t_m_variables
        }

    def setup_heldout_evaluation_callback(self, train_instances: List[Dict]) -> List[RankingData]:
        """Not used in GeneralMIT - evaluation set up in finetune_on_instance."""
        return self.t_o_heldout_vars

    def create_training_batch_generator(self, train_instances: List[Dict]) -> Iterator[List[List[RankingData]]]:
        """General: create multiple masked versions of T_O training data for finetuning."""
        while True:  # Infinite generator
            if not self.t_o_train_vars or len(self.t_o_train_vars) == 0:
                # Return empty batch if no T_O training data
                yield []
                continue

            # Create batch_size different masked versions of T_O training variables
            batch_of_masked_versions = []
            for _ in range(self.config.batch_size):
                masking_rate = random.choice(self.config.masking_rates)
                masked_version = self.apply_masking(self.t_o_train_vars, masking_rate)
                batch_of_masked_versions.append(masked_version)

            yield batch_of_masked_versions

