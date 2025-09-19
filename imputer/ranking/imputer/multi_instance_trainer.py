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
import sys
from .trainer import ImputerTrainer, EvaluationCallback
from .eval import EvaluationEngine
from .data import DataConverter, RankingData


class MultiInstanceTrainerBase:
    """Base class for multi-instance training with masking and evaluation coordination."""

    def __init__(self, model, eval_engine: EvaluationEngine, config, converter: DataConverter, model_config=None):
        self.model = model
        self.eval_engine = eval_engine
        self.config = config
        self.converter = converter

        # Extract loss weights from model_config if provided
        masked_loss_weight = 1.0
        observed_loss_weight = 1.0
        if model_config is not None:
            masked_loss_weight = getattr(model_config, 'masked_loss_weight', 1.0)
            observed_loss_weight = getattr(model_config, 'observed_loss_weight', 1.0)

        self.trainer = ImputerTrainer(
            model, config.learning_rate, device=config.device,
            masked_loss_weight=masked_loss_weight,
            observed_loss_weight=observed_loss_weight
        )
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
        """Train using batch generator with enhanced progress tracking and callback collection."""
        from tqdm import tqdm

        # Set up evaluation callback on heldout data
        heldout_variables = self.setup_heldout_evaluation_callback(train_instances)

        # Create batch generator and train
        training_results = []
        callback_results = []
        batch_generator = self.create_training_batch_generator(train_instances)

        # Enhanced progress bar
        pbar = tqdm(total=self.config.total_batches, desc=f"{self.__class__.__name__} Training")

        current_instance_idx = 0

        for step, batch_of_masked_versions in enumerate(batch_generator):
            if step >= self.config.total_batches:
                break

            # Update which instance we're processing (for Sequential)
            if hasattr(self, 'instance_train_sets'):
                current_instance_idx = step % len(self.instance_train_sets)

            # Train on all masked versions in this batch
            result = self.trainer.train_step(batch_of_masked_versions)
            training_results.append(result)

            # Update progress bar with instance info
            pbar.set_postfix({
                'instance': f"{current_instance_idx}/{len(train_instances)}" if hasattr(self, 'instance_train_sets') else "Mixed",
                'loss': f"{result.get('total_loss', 0):.4f}"
            })
            pbar.update(1)

            # Evaluate at regular intervals
            eval_freq = getattr(self.config, 'eval_frequency', self.config.total_batches // 10)
            if step % max(1, eval_freq) == 0:
                # Call callbacks and collect results
                step_callback_results = self.trainer._call_epoch_end_callbacks(step)
                if step_callback_results:
                    callback_results.extend(step_callback_results)

                # Enhanced printing
                self._print_training_progress(step, result, step_callback_results, current_instance_idx, len(train_instances))

        pbar.close()

        return {
            'training_results': training_results,
            'heldout_variables': heldout_variables,
            'callback_results': callback_results
        }

    def create_training_batch_generator(self, train_instances: List[Dict]) -> Iterator[List[List[RankingData]]]:
        """Create generator that yields batches of masked versions - override in subclasses."""
        raise NotImplementedError("Subclasses must implement create_training_batch_generator")

    def _print_training_progress(self, step, train_result, callback_results, instance_idx, total_instances):
        """Enhanced structured printing."""
        print(f"\n=== Step {step+1}/{self.config.total_batches} ===")
        if hasattr(self, 'instance_train_sets'):
            print(f"Processing Instance: {instance_idx+1}/{total_instances}")

        print(f"Train - Loss: {train_result.get('total_loss', 0):.4f}, "
              f"Rating: {train_result.get('rating_loss', 0):.4f}, "
              f"Ranking: {train_result.get('ranking_loss', 0):.4f}")

        if callback_results:
            for cb_result in callback_results:
                if 'total_loss' in cb_result:
                    print(f"Heldout - Loss: {cb_result.get('total_loss', 0):.4f}, "
                          f"Rating Acc: {cb_result.get('rating_accuracy', 0):.3f}, "
                          f"Ranking Acc: {cb_result.get('ranking_accuracy', 0):.3f}")

    def setup_heldout_evaluation_callback(self, train_instances: List[Dict]) -> List[RankingData]:
        """Set up evaluation callback on heldout data - override in subclasses."""
        raise NotImplementedError("Subclasses must implement setup_heldout_evaluation_callback")
            


class SequentialMIT(MultiInstanceTrainerBase):
    """Sequential Multi-Instance Trainer: process instances sequentially with train/heldout splits."""

    def __init__(self, model, eval_engine: EvaluationEngine, config, converter: DataConverter, model_config=None):
        super().__init__(model, eval_engine, config, converter, model_config)
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

    def __init__(self, model, eval_engine: EvaluationEngine, config, converter: DataConverter, model_config=None):
        super().__init__(model, eval_engine, config, converter, model_config)
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

    def __init__(self, model, eval_engine: EvaluationEngine, config, converter: DataConverter, model_config=None):
        super().__init__(model, eval_engine, config, converter, model_config)
        self.test_instance_variables = []
        self.t_o_train_vars = []
        self.t_o_heldout_vars = []

    def finetune_on_instance(self, pretrained_model, test_instance: Dict, full_test_instances=None, eval_config=None):
        """
        Finetune pretrained model on test instance.

        Flow:
        1. Test Instance -> T_O (observed) + T_M (masked)
        2. T_O -> T_O_train + T_O_heldout
        3. Train on T_O_train, eval on T_O_heldout during training
        4. Final eval on full test instance (T_O + T_M)

        Args:
            pretrained_model: Model to finetune
            test_instance: Single test instance to finetune on
            full_test_instances: All test instances for evaluation during training (optional)
            eval_config: Evaluation configuration for test evaluation (optional)
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

        # Step 3: Set up evaluation callbacks for training monitoring
        if self.t_o_heldout_vars:
            # T_O heldout evaluation callback
            heldout_callback = EvaluationCallback(
                eval_engine=self.eval_engine,
                test_variables=self.t_o_heldout_vars,
                test_data={},
                converter=self.converter,
                masking_rate=random.choice(self.config.masking_rates),
                device=self.config.device
            )
            self.trainer.register_callback(heldout_callback)

        # Set up test instance evaluation callback if enabled and test instances provided
        if (full_test_instances is not None and eval_config is not None and
            getattr(eval_config, 'eval_on_test_during_finetuning', False)):

            # Convert all test instances to variables for evaluation
            test_eval_variables = []
            for test_inst in full_test_instances:
                test_vars = self.converter.create_variables(test_inst)
                test_eval_variables.extend(test_vars)

            if test_eval_variables:
                test_callback = EvaluationCallback(
                    eval_engine=self.eval_engine,
                    test_variables=test_eval_variables,
                    test_data={},
                    converter=self.converter,
                    masking_rate=eval_config.test_masking_rate,
                    device=self.config.device
                )
                # Add identifier to distinguish test vs heldout results
                test_callback.callback_type = "test_evaluation"
                self.trainer.register_callback(test_callback)

        # Step 4: Finetune on T_O_train using batch generator with progress tracking
        from tqdm import tqdm

        finetuning_results = []
        callback_results = []
        batch_generator = self.create_training_batch_generator([])
        finetuning_steps = getattr(self.config, 'finetuning_steps', 100)

        # Progress bar for finetuning
        pbar = tqdm(total=finetuning_steps, desc="Finetuning")

        for step, batch_of_masked_versions in enumerate(batch_generator):
            if step >= finetuning_steps:
                break

            if not batch_of_masked_versions or len(batch_of_masked_versions) == 0:
                continue

            result = self.trainer.train_step(batch_of_masked_versions)
            finetuning_results.append(result)

            # Update progress bar
            pbar.set_postfix({'loss': f"{result.get('total_loss', 0):.4f}"})
            pbar.update(1)

            # Evaluate heldout at regular intervals
            eval_freq = getattr(self.config, 'eval_frequency', finetuning_steps // 10)
            test_eval_freq = getattr(eval_config, 'test_eval_frequency', finetuning_steps // 10) if eval_config else eval_freq

            should_eval_heldout = step % max(1, eval_freq) == 0
            should_eval_test = step % max(1, test_eval_freq) == 0

            if should_eval_heldout or should_eval_test:
                # Call callbacks and collect results
                step_callback_results = self.trainer._call_epoch_end_callbacks(step)
                if step_callback_results:
                    callback_results.extend(step_callback_results)

                # Enhanced progress printing with separate test and heldout metrics
                print(f"\nFinetuning Step {step+1}/{finetuning_steps}")
                print(f"Train   - Loss: {result.get('total_loss', 0):.4f}, "
                      f"Rating: {result.get('rating_loss', 0):.4f}, "
                      f"Ranking: {result.get('ranking_loss', 0):.4f}")

                if step_callback_results:
                    # Separate heldout and test results
                    heldout_results = []
                    test_results = []

                    for cb_result in step_callback_results:
                        if 'total_loss' in cb_result:
                            callback_type = getattr(cb_result, 'callback_type', 'heldout')
                            if callback_type == 'test_evaluation':
                                test_results.append(cb_result)
                            else:
                                heldout_results.append(cb_result)

                    # Print heldout results
                    for cb_result in heldout_results:
                        print(f"Heldout - Loss: {cb_result.get('total_loss', 0):.4f}, "
                              f"Rating Acc: {cb_result.get('rating_accuracy', 0):.3f}, "
                              f"Ranking Acc: {cb_result.get('ranking_accuracy', 0):.3f}")

                    # Print test results
                    for cb_result in test_results:
                        print(f"Test    - Loss: {cb_result.get('total_loss', 0):.4f}, "
                              f"Rating Acc: {cb_result.get('rating_accuracy', 0):.3f}, "
                              f"Ranking Acc: {cb_result.get('ranking_accuracy', 0):.3f}")

        pbar.close()

        # Step 5: Final evaluation on full test instance (T_O + T_M)
        # Create test variables with explicit masking preservation
        final_test_variables = []

        # Ensure T_O variables are marked as observed
        for var in t_o_variables:
            var.is_masked = False
            final_test_variables.append(var)

        # Ensure T_M variables are marked as masked
        for var in t_m_variables:
            var.is_masked = True
            final_test_variables.append(var)

        final_results = self.eval_engine.evaluate_model(
            model=self.model,
            variables=final_test_variables,
            masking_rate=0.0,  # Use existing is_masked flags
            converter=self.converter,
            device=self.config.device
        )

        return {
            'finetuning_results': finetuning_results,
            'callback_results': callback_results,
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

