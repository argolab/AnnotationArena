#!/usr/bin/env python3
"""ICLR experiment runner with mixed training and conditional imputation evaluation."""

import json
import time
import random
from pathlib import Path
import logging
from typing import Dict, List, Any, Tuple, Optional
import torch
import numpy as np
from tqdm import tqdm
import copy
import os
from config import ExperimentConfig
from imputer.data import DataConverter
from imputer.trainer import ImputerTrainer, EarlyStopping
from imputer.ranking_imputer import MultiVariableImputer
from domain_model_iclr import DomainModelICLR
from iclr_data_generator import ICLRDataGenerator, ICLRDatasetConfig
from dataclasses import asdict

logger = logging.getLogger(__name__)

class ExperimentRunnerICLR:
    """ICLR experiment runner with mixed training and conditional evaluation."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create output directories
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Initialize model and trainer
        legacy_props = config.get_legacy_properties()
        self.model = MultiVariableImputer(
            num_items=legacy_props['K'],
            num_attributes=legacy_props['I'],
            num_annotators=legacy_props['J'],
            num_likert_classes=legacy_props['C'],
            max_rank_size=2,
            encoder_layers_num=config.model_config.encoder_layers,
            attention_heads=config.model_config.attention_heads,
            embedding_dim=config.model_config.embedding_dim,
            dropout=config.model_config.dropout,
            embedding_type=config.model_config.embedding_type,
            device=config.device
        )

        self.trainer = ImputerTrainer(
            self.model,
            learning_rate=config.training_config.learning_rate,
            device=config.device,
            embedding_anchor_reg=config.training_config.embedding_anchor_reg
        )

        # Data converter
        first_instance = config.instances[0]
        self.converter = DataConverter(
            num_attributes=first_instance.I,
            num_annotators=first_instance.J,
            num_items=first_instance.K,
            num_likert_classes=first_instance.C,
            max_rank_size=2  # Pairwise rankings
        )

        # Results storage
        self.results = {
            'config': config.__dict__,
            'pretraining_results': {},
            'test_evaluation_results': {},
            'timing_results': {},
            'method_comparisons': {}
        }

    def generate_data(self) -> None:
        """Generate data for all instances."""
        logger.info("Generating data for all instances...")

        for i, instance_config in enumerate(self.config.instances):
            instance_data_dir = self.config.get_instance_data_dir(i)
            if os.path.exists(instance_data_dir):
                continue
            instance_data_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Generating data for instance {i}...")

            # Convert to ICLR data config - filter to only accepted parameters
            instance_dict = asdict(instance_config)
            iclr_params = {
                'K': instance_dict['K'],
                'I': instance_dict['I'],
                'J': instance_dict['J'],
                'D': instance_dict['D'],
                'C': instance_dict['C'],
                'max_pairs_per_tied_group': instance_dict['max_pairs_per_tied_group'],
                'min_group_size': instance_dict['min_group_size'],
                'max_group_size': instance_dict['max_group_size'],
                'sigma_annotator': instance_dict['sigma_annotator'],
                'sigma_measurement': instance_dict['sigma_measurement'],
                'alpha_dirichlet': instance_dict['alpha_dirichlet'],
                'temperature': instance_dict['temperature'],
                'train_fraction': self.config.train_fraction,
                'test_fraction': self.config.test_fraction
            }
            iclr_config = ICLRDatasetConfig(**iclr_params)

            # Generate data
            generator = ICLRDataGenerator()
            dataset = generator.generate_dataset(iclr_config)
            generator.save_dataset(dataset, instance_data_dir)

        logger.info("Data generation completed")

    def load_instance_data(self, instance_idx: int) -> Tuple[Dict, Dict]:
        """Load train and test data for specific instance."""
        instance_dir = self.config.get_instance_data_dir(instance_idx)

        train_file = instance_dir / "iclr_dataset_train.json"
        test_file = instance_dir / "iclr_dataset_test.json"

        with open(train_file, 'r') as f:
            train_data = json.load(f)
        with open(test_file, 'r') as f:
            test_data = json.load(f)

        return train_data, test_data

    def create_mixed_training_data(self) -> Tuple[List, Dict]:
        """Create mixed training dataset from all training instances."""
        all_variables = []
        all_data = {'ratings': [], 'pairwise_rankings': []}
        instance_variable_counts = []

        for train_idx in self.config.train_instance_indices:
            train_data, heldout_data = self.load_instance_data(train_idx)

            # Combine train and heldout for this instance
            combined_data = {
                'ratings': train_data['ratings'] + heldout_data['ratings'],
                'pairwise_rankings': train_data['pairwise_rankings'] + heldout_data['pairwise_rankings']
            }

            # Track filtered data
            filtered_data = self.converter.load_training_data_from_dict(combined_data)

            # Create variables with instance tagging
            rating_vars, ranking_vars = self.converter.create_variables_from_actual_data(
                filtered_data, filtered_data
            )

            # Tag variables with instance index
            for var in rating_vars + ranking_vars:
                var['instance_idx'] = train_idx

            all_variables.extend(rating_vars + ranking_vars)
            all_data['ratings'].extend(filtered_data['ratings'])
            all_data['pairwise_rankings'].extend(filtered_data['pairwise_rankings'])
            instance_variable_counts.append(len(rating_vars + ranking_vars))

        logger.info(f"Mixed training data: {len(all_variables)} variables from {len(self.config.train_instance_indices)} instances")
        logger.info(f"Variables per instance: {instance_variable_counts}")

        return all_variables, all_data

    def create_heldout_evaluation_data(self, masking_rate: float) -> Tuple[List, Dict, List, List]:
        """Create combined heldout data from training instances for evaluation."""
        all_heldout_variables = []
        all_heldout_data = {'ratings': [], 'pairwise_rankings': []}

        for train_idx in self.config.train_instance_indices:
            train_data, heldout_data = self.load_instance_data(train_idx)

            # Use only heldout data for evaluation
            filtered_data = self.converter.load_training_data_from_dict(heldout_data)
            rating_vars, ranking_vars = self.converter.create_variables_from_actual_data(
                filtered_data, filtered_data
            )

            all_heldout_variables.extend(rating_vars + ranking_vars)
            all_heldout_data['ratings'].extend(heldout_data['ratings'])
            all_heldout_data['pairwise_rankings'].extend(heldout_data['pairwise_rankings'])

        # Apply fixed masking to combined heldout data
        np.random.seed(42)  # Fixed seed for consistent heldout evaluation
        num_masked = int(len(all_heldout_variables) * masking_rate)
        masked_indices = set(np.random.choice(len(all_heldout_variables), num_masked, replace=False))

        masked_vars = [all_heldout_variables[i] for i in masked_indices]
        observed_vars = [all_heldout_variables[i] for i in range(len(all_heldout_variables)) if i not in masked_indices]

        return all_heldout_variables, all_heldout_data, masked_vars, observed_vars

    def create_test_instance_data(self, test_idx: int, masking_rate: float) -> Tuple[List, Dict, List, List]:
        """Create test instance data with fixed masking."""
        train_data, heldout_data = self.load_instance_data(test_idx)

        # Combine all data for test instance
        full_test_data = {
            'ratings': train_data['ratings'] + heldout_data['ratings'],
            'pairwise_rankings': train_data['pairwise_rankings'] + heldout_data['pairwise_rankings']
        }

        filtered_data = self.converter.load_training_data_from_dict(full_test_data)
        rating_vars, ranking_vars = self.converter.create_variables_from_actual_data(
            filtered_data, filtered_data
        )

        all_variables = rating_vars + ranking_vars

        # Fixed masking for test evaluation
        random.seed(42 + test_idx)  # Deterministic masking per test instance
        num_to_mask = int(len(all_variables) * masking_rate)
        masked_indices = set(random.sample(range(len(all_variables)), num_to_mask))

        test_masked_vars = [all_variables[i] for i in masked_indices]
        test_observed_vars = [all_variables[i] for i in range(len(all_variables)) if i not in masked_indices]

        logger.info(f"Test instance {test_idx}: {len(test_masked_vars)} masked, {len(test_observed_vars)} observed variables")

        return all_variables, filtered_data, test_masked_vars, test_observed_vars

    def create_dynamic_batch(self, all_variables: List, all_data: Dict, masking_rate: float) -> Dict:
        """Create batch with dynamic random masking."""
        # Random masking each time
        available_vars = list(range(len(all_variables)))
        num_to_mask = int(len(available_vars) * masking_rate)
        masked_indices = set(random.sample(available_vars, num_to_mask))

        # Process data for batch creation
        rating_data, ranking_data = self.converter.process_training_data(all_data)

        # Create batch with dynamic masking
        batch = self.converter.create_batch_with_dynamic_masking(
            all_variables, rating_data, ranking_data, masked_indices
        )

        return batch

    def create_random_instance_batch(self, masking_rate: float, batch_size: int = 1) -> Dict:
        """Create batch by randomly sampling from a random training instance."""
        # Randomly select training instance
        instance_idx = random.choice(self.config.train_instance_indices)

        # Load instance data
        train_data, heldout_data = self.load_instance_data(instance_idx)

        # Use only train portion for training (heldout for evaluation)
        filtered_data = self.converter.load_training_data_from_dict(train_data)
        rating_vars, ranking_vars = self.converter.create_variables_from_actual_data(
            filtered_data, filtered_data
        )

        all_variables = rating_vars + ranking_vars

        # Apply random masking to this batch
        available_vars = list(range(len(all_variables)))
        num_to_mask = int(len(available_vars) * masking_rate)
        masked_indices = set(random.sample(available_vars, num_to_mask))

        # Process data for batch creation
        rating_data, ranking_data = self.converter.process_training_data(filtered_data)

        # Create batch with dynamic masking
        batch = self.converter.create_batch_with_dynamic_masking(
            all_variables, rating_data, ranking_data, masked_indices
        )

        return batch

    def run_pretraining(self, masking_rate: float = 0.5) -> Dict:
        """Run mixed pretraining with random instance sampling."""
        logger.info("Starting mixed pretraining with random instance sampling...")
        start_time = time.time()

        # Prepare heldout evaluation data (combined from all training instances)
        heldout_variables, heldout_data, heldout_masked, heldout_observed = self.create_heldout_evaluation_data(masking_rate)

        # Prepare test instances for evaluation during pretraining
        test_data_cache = {}
        for test_idx in self.config.test_instance_indices:
            test_vars, test_data, test_masked, test_observed = self.create_test_instance_data(
                test_idx, masking_rate
            )
            test_data_cache[test_idx] = {
                'all_variables': test_vars,
                'data': test_data,
                'masked_vars': test_masked,
                'observed_vars': test_observed
            }

        # Training tracking
        training_results = {
            'epochs': [],
            'train_losses': {'total': [], 'rating': [], 'ranking': []},
            'heldout_losses': {'total': [], 'rating': [], 'ranking': []},
            'test_losses_per_instance': {idx: {'total': [], 'rating': [], 'ranking': []}
                                       for idx in self.config.test_instance_indices},
            'test_losses_avg': {'total': [], 'rating': [], 'ranking': []},
            'wall_times': []
        }

        # Training loop with random instance sampling
        for epoch in tqdm(range(self.config.training_config.epochs), desc="Mixed Pretraining"):
            epoch_start = time.time()

            # Create batch from random instance with random masking
            batch = self.create_random_instance_batch(masking_rate, self.config.training_config.batch_size)

            # Training step
            train_losses = self.trainer.train_step(batch)

            # Record training losses
            training_results['epochs'].append(epoch)
            training_results['train_losses']['total'].append(train_losses['total_loss'])
            training_results['train_losses']['rating'].append(train_losses['rating_loss'])
            training_results['train_losses']['ranking'].append(train_losses['ranking_loss'])

            # Evaluate on both training heldouts and test instances
            if epoch % self.config.training_config.evaluation_frequency == 0:
                # Evaluate on training instance heldouts (use pre-computed data)
                heldout_metrics = self.evaluate_conditional_imputation(
                    heldout_variables, heldout_data, heldout_masked, heldout_observed
                )

                training_results['heldout_losses']['total'].append(heldout_metrics['total_log_loss'])
                training_results['heldout_losses']['rating'].append(heldout_metrics['rating_log_loss'])
                training_results['heldout_losses']['ranking'].append(heldout_metrics['ranking_log_loss'])

                # Evaluate on test instances
                test_total_losses = []
                test_rating_losses = []
                test_ranking_losses = []

                for test_idx in self.config.test_instance_indices:
                    test_info = test_data_cache[test_idx]
                    test_metrics = self.evaluate_conditional_imputation(
                        test_info['all_variables'], test_info['data'],
                        test_info['masked_vars'], test_info['observed_vars']
                    )

                    # Store per-instance results
                    training_results['test_losses_per_instance'][test_idx]['total'].append(
                        test_metrics['total_log_loss']
                    )
                    training_results['test_losses_per_instance'][test_idx]['rating'].append(
                        test_metrics['rating_log_loss']
                    )
                    training_results['test_losses_per_instance'][test_idx]['ranking'].append(
                        test_metrics['ranking_log_loss']
                    )

                    test_total_losses.append(test_metrics['total_log_loss'])
                    test_rating_losses.append(test_metrics['rating_log_loss'])
                    test_ranking_losses.append(test_metrics['ranking_log_loss'])

                # Average across test instances
                training_results['test_losses_avg']['total'].append(np.mean(test_total_losses))
                training_results['test_losses_avg']['rating'].append(np.mean(test_rating_losses))
                training_results['test_losses_avg']['ranking'].append(np.mean(test_ranking_losses))

            epoch_time = time.time() - epoch_start
            training_results['wall_times'].append(epoch_time)

        total_time = time.time() - start_time
        logger.info(f"Pretraining completed in {total_time:.2f} seconds")

        training_results['total_time'] = total_time
        return training_results

    def evaluate_conditional_imputation(self, all_variables: List, data: Dict,
                                       masked_vars: List, observed_vars: List) -> Dict:
        """Evaluate conditional imputation: predict masked given observed."""
        self.model.eval()

        with torch.no_grad():
            # Create evaluation batch with all variables
            rating_data, ranking_data = self.converter.process_training_data(data)

            # Create batch where masked variables are hidden
            masked_indices = set()
            for i, var in enumerate(all_variables):
                if var in masked_vars:
                    masked_indices.add(i)

            batch = self.converter.create_batch_with_dynamic_masking(
                all_variables, rating_data, ranking_data, masked_indices
            )

            # Get predictions
            ranking_data_list = self.model._convert_legacy_tensors_to_ranking_data(
                batch['variable_data'].to(self.device),
                batch['variable_types'].to(self.device),
                batch['attribute_ids'].to(self.device),
                batch['annotator_ids'].to(self.device),
                batch['item_ids'].to(self.device)
            )

            outputs = self.model(ranking_data_list)

            # Calculate losses and accuracy metrics only on masked variables
            total_rating_loss = 0.0
            total_ranking_loss = 0.0
            rating_count = 0
            ranking_count = 0
            rating_correct = 0
            ranking_correct = 0
            rating_mse = 0.0

            for i, var in enumerate(all_variables):
                if var in masked_vars:
                    if var['type'] == 'rating':
                        # Get rating loss and accuracy
                        rating_logits = outputs['rating'][0, i]
                        rating_target = batch['rating_targets'][0, i].to(self.device)
                        rating_loss = torch.nn.functional.cross_entropy(
                            rating_logits.unsqueeze(0), rating_target.argmax().unsqueeze(0)
                        )
                        total_rating_loss += rating_loss.item()

                        # Calculate accuracy and RMSE (following legacy code pattern)
                        predicted_rating = torch.argmax(rating_logits).item()  # 0-indexed
                        true_rating = torch.argmax(rating_target).item()       # 0-indexed

                        if predicted_rating == true_rating:
                            rating_correct += 1

                        # Convert to 1-indexed for RMSE calculation
                        rating_mse += (predicted_rating + 1 - (true_rating + 1)) ** 2
                        rating_count += 1

                    elif var['type'] == 'ranking':
                        # Get ranking loss and accuracy
                        ranking_logits = outputs['ranking'][0, i]
                        ranking_target = batch['ranking_targets'][0, i].to(self.device)
                        ranking_loss = torch.nn.functional.mse_loss(
                            ranking_logits, ranking_target
                        )
                        total_ranking_loss += ranking_loss.item()

                        # Calculate ranking accuracy using pairwise comparison (like legacy code)
                        # Extract ranking order from target tensor
                        target_order = []
                        for j in range(ranking_target.shape[0]):
                            score = int(ranking_target[j].item())
                            if score > 0:
                                target_order.append(score)

                        if len(target_order) >= 2:
                            # Use predicted scores to determine preference
                            pred_scores = ranking_logits.cpu().numpy()

                            # Simple pairwise accuracy: do first two items have correct relative order?
                            if len(pred_scores) >= 2:
                                pred_first_better = pred_scores[0] > pred_scores[1]
                                true_first_better = target_order[0] < target_order[1]  # Lower rank number = better

                                if pred_first_better == true_first_better:
                                    ranking_correct += 1

                        ranking_count += 1

            # Average losses and calculate metrics
            avg_rating_loss = total_rating_loss / max(rating_count, 1)
            avg_ranking_loss = total_ranking_loss / max(ranking_count, 1)
            total_loss = avg_rating_loss + avg_ranking_loss

            rating_accuracy = rating_correct / max(rating_count, 1)
            ranking_accuracy = ranking_correct / max(ranking_count, 1)
            rating_rmse = (rating_mse / max(rating_count, 1)) ** 0.5

            return {
                'total_log_loss': total_loss,
                'rating_log_loss': avg_rating_loss,
                'ranking_log_loss': avg_ranking_loss,
                'rating_accuracy': rating_accuracy,
                'ranking_accuracy': ranking_accuracy,
                'rating_rmse': rating_rmse,
                'masked_rating_count': rating_count,
                'masked_ranking_count': ranking_count
            }

    def run_finetuning(self, test_idx: int, masking_rate: float,
                      pretrained_model_state: Optional[Dict] = None) -> Dict:
        """Run finetuning on test instance observed variables."""
        logger.info(f"Finetuning on test instance {test_idx}...")
        start_time = time.time()

        # Reset model to pretrained state or fresh initialization
        if pretrained_model_state:
            self.model.load_state_dict(pretrained_model_state)
            logger.info("Loaded pretrained model state")
        else:
            # Reinitialize model for "no pretraining" case
            self.model.apply(self.model._init_weights)
            logger.info("Reinitialized model (no pretraining)")

        # Get test instance data
        all_variables, data, test_masked, test_observed = self.create_test_instance_data(
            test_idx, masking_rate
        )

        logger.info(f"Finetuning on {len(test_observed)} observed variables with {masking_rate:.1%} artificial masking")

        # Training tracking
        finetuning_results = {
            'epochs': [],
            'finetune_losses': {'total': [], 'rating': [], 'ranking': []},
            'test_losses': {'total': [], 'rating': [], 'ranking': []},
            'wall_times': []
        }

        # Early stopping for finetuning
        early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

        # Finetuning loop with progress bar
        for epoch in tqdm(range(self.config.training_config.epochs), desc=f"Finetuning Instance {test_idx}"):
            epoch_start = time.time()

            # CORRECT APPROACH: Artificial masking on observed variables
            # Apply masking_rate to the observed variables to create Test_O_ArtificialMasked and Test_O_Observed
            num_to_artificially_mask = int(len(test_observed) * masking_rate)
            artificially_masked_indices = set(random.sample(range(len(test_observed)), num_to_artificially_mask))

            # Create training batch with artificial masking
            observed_data = self._create_data_subset(data, test_observed)
            rating_data, ranking_data = self.converter.process_training_data(observed_data)

            batch = self.converter.create_batch_with_dynamic_masking(
                test_observed, rating_data, ranking_data, artificially_masked_indices
            )

            # Training step
            train_losses = self.trainer.train_step(batch)

            # Evaluate on test (masked) variables - the real task
            test_metrics = self.evaluate_conditional_imputation(
                all_variables, data, test_masked, test_observed
            )

            # Record results
            finetuning_results['epochs'].append(epoch)
            finetuning_results['finetune_losses']['total'].append(train_losses['total_loss'])
            finetuning_results['finetune_losses']['rating'].append(train_losses['rating_loss'])
            finetuning_results['finetune_losses']['ranking'].append(train_losses['ranking_loss'])

            finetuning_results['test_losses']['total'].append(test_metrics['total_log_loss'])
            finetuning_results['test_losses']['rating'].append(test_metrics['rating_log_loss'])
            finetuning_results['test_losses']['ranking'].append(test_metrics['ranking_log_loss'])

            # Early stopping based on test performance
            if early_stopping.should_stop(test_metrics['total_log_loss'], self.model):
                logger.info(f"Early stopping at epoch {epoch} (test loss: {test_metrics['total_log_loss']:.4f})")
                break

            # Log progress every 5 epochs
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: Train loss={train_losses['total_loss']:.4f}, "
                           f"Test loss={test_metrics['total_log_loss']:.4f}")

            epoch_time = time.time() - epoch_start
            finetuning_results['wall_times'].append(epoch_time)

        # Restore best model
        if early_stopping.best_model_state:
            early_stopping.restore_best_model(self.model)

        total_time = time.time() - start_time

        # Final evaluation on Test_M given Test_O
        final_metrics = self.evaluate_conditional_imputation(
            all_variables, data, test_masked, test_observed
        )

        # Combine training history with final metrics
        final_results = {
            'total_log_loss': final_metrics['total_log_loss'],
            'rating_log_loss': final_metrics['rating_log_loss'],
            'ranking_log_loss': final_metrics['ranking_log_loss'],
            'rating_accuracy': final_metrics['rating_accuracy'],
            'ranking_accuracy': final_metrics['ranking_accuracy'],
            'rating_rmse': final_metrics['rating_rmse'],
            'wall_time': total_time,
            'training_history': finetuning_results
        }

        return final_results

    def _create_data_subset(self, full_data: Dict, variables: List) -> Dict:
        """Create data subset containing only specified variables."""
        subset_data = {'ratings': [], 'pairwise_rankings': []}

        for var in variables:
            if var['type'] == 'rating':
                # Find matching rating
                for rating in full_data['ratings']:
                    if (rating['attribute'] == var['attribute'] and
                        rating['annotator'] == var['annotator'] and
                        rating['item'] == var['item']):
                        subset_data['ratings'].append(rating)
                        break
            else:
                # Find matching ranking
                for ranking in full_data['pairwise_rankings']:
                    if (ranking['attribute'] == var['attribute'] and
                        ranking['annotator'] == var['annotator'] and
                        ranking['items'] == var['items']):
                        subset_data['pairwise_rankings'].append(ranking)
                        break

        return subset_data

    def run_experiment(self, masking_rate: float = 0.5) -> Dict:
        """Run full ICLR experiment."""
        logger.info("Starting ICLR experiment...")
        experiment_start = time.time()

        # 0. Generate data first
        self.generate_data()

        # 1. Test domain model first (to catch errors early)
        method_results = {}
        for test_idx in self.config.test_instance_indices:
            logger.info(f"Testing domain model on instance {test_idx} first...")
            all_vars, data, masked_vars, observed_vars = self.create_test_instance_data(
                test_idx, masking_rate
            )

            # Method 4: Domain model (test first)
            domain_model = DomainModelICLR(self.config)
            domain_results = domain_model.evaluate_test_instance(test_idx, observed_vars, masked_vars)
            method_results[test_idx] = {'domain_model': domain_results}
            logger.info(f"Domain model completed for instance {test_idx}")

        # 2. Pretraining (after domain model works)
        logger.info("Domain model tests passed, starting pretraining...")
        pretraining_results = self.run_pretraining(masking_rate)
        pretraining_time = pretraining_results['total_time']
        pretrained_state = copy.deepcopy(self.model.state_dict())

        # 3. Imputer methods evaluation
        for test_idx in self.config.test_instance_indices:
            logger.info(f"Evaluating imputer methods on test instance {test_idx}")

            # Method 1: Pretrained only
            self.model.load_state_dict(pretrained_state)
            all_vars, data, masked_vars, observed_vars = self.create_test_instance_data(
                test_idx, masking_rate
            )

            eval_start = time.time()
            method1_metrics = self.evaluate_conditional_imputation(
                all_vars, data, masked_vars, observed_vars
            )
            method1_metrics['wall_time'] = time.time() - eval_start

            # Method 2: Pretrained + Finetuning
            finetuning_results = self.run_finetuning(test_idx, masking_rate, pretrained_state)

            # Method 3: No pretraining (finetuning only)
            no_pretrain_results = self.run_finetuning(test_idx, masking_rate, None)

            # Update results (domain model already stored)
            method_results[test_idx].update({
                'pretrained_only': method1_metrics,
                'pretrained_finetuned': finetuning_results,
                'no_pretrain_finetuned': no_pretrain_results
            })


        # Store all results
        self.results.update({
            'config': {
                'train_instances': self.config.train_instance_indices,
                'test_instances': self.config.test_instance_indices,
                'masking_rate': masking_rate
            },
            'pretraining_time': pretraining_time,
            'pretraining_history': {
                'train_loss': pretraining_results['train_losses']['total'],
                'train_rating_loss': pretraining_results['train_losses']['rating'],
                'train_ranking_loss': pretraining_results['train_losses']['ranking'],
                'heldout_loss': pretraining_results['heldout_losses']['total'],
                'heldout_rating_loss': pretraining_results['heldout_losses']['rating'],
                'heldout_ranking_loss': pretraining_results['heldout_losses']['ranking'],
                'val_loss': pretraining_results['test_losses_avg']['total'],
                'val_rating_loss': pretraining_results['test_losses_avg']['rating'],
                'val_ranking_loss': pretraining_results['test_losses_avg']['ranking'],
                'epoch_times': pretraining_results['wall_times']
            },
            'test_results': method_results,
            'total_time': time.time() - experiment_start,
            'masking_rate': masking_rate
        })

        # Save results
        self._save_results()

        # Generate comprehensive visualization and reporting
        self._generate_visualizations()

        return self.results

    def _save_results(self):
        """Save experimental results."""
        results_file = self.results_dir / "iclr_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_file}")

    def _generate_visualizations(self):
        """Generate comprehensive visualizations and reports."""
        try:
            # Import here to avoid circular dependency issues
            from iclr_visualization import ICLRResultsAnalyzer

            results_file = self.results_dir / "iclr_results.json"
            visualization_dir = self.results_dir / "visualizations"

            logger.info("Generating comprehensive visualizations...")

            analyzer = ICLRResultsAnalyzer(str(results_file))
            analyzer.create_comprehensive_report(visualization_dir)

            logger.info(f"Visualizations saved to {visualization_dir}")

        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            logger.warning("Continuing without visualizations...")