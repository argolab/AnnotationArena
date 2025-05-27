"""
Code to run active learning experiments with noisy variables.
"""

import os
import argparse
import torch
import json
import random
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import copy
import math

from annotationArena import *
from utils import AnnotationDataset, DataManager, compute_metrics, resample_validation_dataset
from visualizations import *
from imputer import Imputer
from selection import (
    SelectionFactory, 
    VOISelectionStrategy, 
    FastVOISelectionStrategy,
    GradientSelectionStrategy,
    EntropyExampleSelectionStrategy,
    EntropyFeatureSelectionStrategy,
    BADGESelectionStrategy,
    ArgmaxVOISelectionStrategy
) 

class NoisyDataManager(DataManager):
    """Extended DataManager that creates noisy copies of annotations."""
    
    def __init__(self, base_path):
        super().__init__(base_path)
        
    def add_noise_to_llm(self, prob_dist, sigma=0.5):
        """Add Gaussian noise to LLM probability distributions in log space."""
        prob_dist = np.array(prob_dist)
        log_probs = np.log(prob_dist + 1e-10)
        noisy_log_probs = log_probs + np.random.normal(0, sigma, log_probs.shape)
        noisy_probs = np.exp(noisy_log_probs)
        noisy_probs = noisy_probs / np.sum(noisy_probs)
        return noisy_probs.tolist()
    
    def add_noise_to_human(self, one_hot, noise_level=0.3):
        """Add uniform noise to human one-hot annotations and convert back to categorical."""
        one_hot = np.array(one_hot)
        noise = np.random.uniform(-noise_level, noise_level, one_hot.shape)
        noisy = one_hot + noise
        
        noisy_categorical = np.zeros_like(one_hot)
        max_idx = np.argmax(noisy)
        noisy_categorical[max_idx] = 1.0
        
        return noisy_categorical.tolist()
    
    def prepare_data(self, num_partition=1200, known_human_questions_val=0, initial_train_ratio=0.0, 
                    dataset="hanna", cold_start=False, llm_sigma=0.5, human_noise=0.3):
        """Prepare data with noisy copies of all annotations."""
        
        if dataset == "gaussian":
            pass
        try:
            with open(os.path.join(self.base_path, "gpt-3.5-turbo-data-new.json"), "r") as f:
                llm_data = json.load(f)
            with open(os.path.join(self.base_path, "human-data-new.json"), "r") as f:
                human_data = json.load(f)
        except FileNotFoundError:
            return False
        
        text_ids = list(human_data.keys())
        if dataset == "hanna":
            question_list = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
        elif dataset == "llm_rubric":
            question_list = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']
        question_indices = {q: i for i, q in enumerate(question_list)}
        
        random.seed(42)
        random.shuffle(text_ids)
        
        initial_train_size = int(num_partition * initial_train_ratio)
        validation_size = int(num_partition * 0.3)
        test_size = int(num_partition * 0.2)
        active_pool_size = num_partition - initial_train_size - validation_size - test_size

        initial_train_texts = text_ids[:initial_train_size]
        validation_texts = text_ids[initial_train_size:initial_train_size + validation_size]
        test_texts = text_ids[initial_train_size + validation_size:initial_train_size + validation_size + test_size]
        active_pool_texts = text_ids[initial_train_size + validation_size + test_size:
                                initial_train_size + validation_size + test_size + active_pool_size]
        
        initial_train_data = []
        validation_data = []
        test_data = []
        active_pool_data = []
        
        self._prepare_entries_with_noise(initial_train_texts, initial_train_data, 'train', llm_data, human_data, 
                                       question_list, question_indices, known_human_questions_val, dataset, 
                                       cold_start, llm_sigma, human_noise)
        self._prepare_entries_with_noise(validation_texts, validation_data, 'validation', llm_data, human_data, 
                                       question_list, question_indices, known_human_questions_val, dataset, 
                                       cold_start, llm_sigma, human_noise)
        self._prepare_entries_with_noise(test_texts, test_data, 'test', llm_data, human_data, 
                                       question_list, question_indices, known_human_questions_val, dataset, 
                                       cold_start, llm_sigma, human_noise)
        self._prepare_entries_with_noise(active_pool_texts, active_pool_data, 'active_pool', llm_data, human_data, 
                                       question_list, question_indices, known_human_questions_val, dataset, 
                                       cold_start, llm_sigma, human_noise)
        
        for key, data in zip(['train', 'validation', 'test', 'active_pool', 'original_train', 'original_validation', 'original_test', 'original_active_pool'],
                             [initial_train_data, validation_data, test_data, active_pool_data, initial_train_data, validation_data, test_data, active_pool_data]):
            with open(self.paths[key], "w") as f:
                json.dump(data, f)
        
        return True
    
    def _prepare_entries_with_noise(self, texts, data_list, split_type, llm_data, human_data, question_list, 
                              question_indices, known_human_questions_val, dataset, cold_start, 
                              llm_sigma, human_noise):
        """Prepare entries with original + noisy copies."""
        
        if dataset == "hanna":
            for text_id in texts:
                if text_id not in llm_data:
                    continue
                
                entry = {
                    "known_questions": [], 
                    "input": [], 
                    "answers": [], 
                    "annotators": [],
                    "questions": [],
                    "orig_split": split_type,
                    "observation_history": [],
                    "is_noisy": []  # Track which positions are noisy
                }
                
                annotators = list(human_data[text_id].keys())
                
                # Process original LLM questions
                for q_idx, question in enumerate(question_list):
                    true_prob = llm_data[text_id][question]
                    
                    if cold_start and split_type == 'active_pool':
                        mask_bit = 1
                        combined_input = [mask_bit] + [0.0] * 5
                        entry["known_questions"].append(0)
                    else:
                        mask_bit = 0
                        combined_input = [mask_bit] + true_prob
                        entry["known_questions"].append(1)
                    
                    entry["input"].append(combined_input)
                    entry["answers"].append(true_prob)
                    entry["annotators"].append(-1)
                    entry["questions"].append(question_indices[question])
                    entry["is_noisy"].append(False)
                
                # Process noisy LLM questions
                for q_idx, question in enumerate(question_list):
                    true_prob = llm_data[text_id][question]
                    noisy_prob = self.add_noise_to_llm(true_prob, llm_sigma)
                    
                    if cold_start and split_type == 'active_pool':
                        mask_bit = 1
                        combined_input = [mask_bit] + [0.0] * 5
                        entry["known_questions"].append(0)
                    else:
                        mask_bit = 0
                        combined_input = [mask_bit] + noisy_prob
                        entry["known_questions"].append(1)
                    
                    entry["input"].append(combined_input)
                    entry["answers"].append(noisy_prob)
                    entry["annotators"].append(-1)
                    entry["questions"].append(question_indices[question])
                    entry["is_noisy"].append(True)

                # Process original human questions
                for judge_id in annotators:
                    for q_idx, question in enumerate(question_list):
                        true_score = human_data[text_id][judge_id][question]
                        true_prob = [0.0] * 5
                        
                        if isinstance(true_score, (int, float)):
                            if true_score % 1 != 0:  
                                rounded_score = math.ceil(true_score)
                                rounded_score = max(min(rounded_score, 5), 1)
                                index = rounded_score - 1
                                true_prob[index] = 1.0
                            else:
                                true_score = max(min(int(true_score), 5), 1)
                                index = true_score - 1
                                true_prob[index] = 1.0
                        
                        self._add_human_annotation(entry, true_prob, split_type, known_human_questions_val, 
                                                q_idx, judge_id, question_indices[question], False)
                
                # Process noisy human questions ONLY if human_noise > 0
                if human_noise > 0:
                    for judge_id in annotators:
                        for q_idx, question in enumerate(question_list):
                            true_score = human_data[text_id][judge_id][question]
                            true_prob = [0.0] * 5
                            
                            if isinstance(true_score, (int, float)):
                                if true_score % 1 != 0:  
                                    rounded_score = math.ceil(true_score)
                                    rounded_score = max(min(rounded_score, 5), 1)
                                    index = rounded_score - 1
                                    true_prob[index] = 1.0
                                else:
                                    true_score = max(min(int(true_score), 5), 1)
                                    index = true_score - 1
                                    true_prob[index] = 1.0
                            
                            noisy_prob = self.add_noise_to_human(true_prob, human_noise)
                            self._add_human_annotation(entry, noisy_prob, split_type, known_human_questions_val,
                                                    q_idx, judge_id, question_indices[question], True)
                
                data_list.append(entry)

        elif dataset == "llm_rubric":
            # Similar implementation for llm_rubric with 4 classes instead of 5
            for text_id in texts:
                annotators = list(human_data[text_id].keys())
                for annotator in annotators:
                    entry = {
                        "known_questions": [], 
                        "input": [], 
                        "answers": [], 
                        "annotators": [],
                        "questions": [],
                        "orig_split": split_type,
                        "observation_history": [],
                        "is_noisy": []
                    }
                    
                    # Original + Noisy LLM questions (4 classes)
                    for q_idx, question in enumerate(question_list):
                        true_prob = llm_data[text_id][question]
                        
                        # Original
                        if cold_start and split_type in ['active_pool', 'validation']:
                            mask_bit = 1
                            combined_input = [mask_bit] + [0.0] * 4
                            entry["known_questions"].append(0)
                        else:
                            mask_bit = 0
                            combined_input = [mask_bit] + true_prob
                            entry["known_questions"].append(1)
                        
                        entry["input"].append(combined_input)
                        entry["answers"].append(true_prob)
                        entry["annotators"].append(-1)
                        entry["questions"].append(question_indices[question])
                        entry["is_noisy"].append(False)
                        
                        # Noisy
                        noisy_prob = self.add_noise_to_llm(true_prob, llm_sigma)
                        if cold_start and split_type in ['active_pool', 'validation']:
                            mask_bit = 1
                            combined_input = [mask_bit] + [0.0] * 4
                            entry["known_questions"].append(0)
                        else:
                            mask_bit = 0
                            combined_input = [mask_bit] + noisy_prob
                            entry["known_questions"].append(1)
                        
                        entry["input"].append(combined_input)
                        entry["answers"].append(noisy_prob)
                        entry["annotators"].append(-1)
                        entry["questions"].append(question_indices[question])
                        entry["is_noisy"].append(True)

                    # Original Human questions (4 classes)
                    for q_idx, question in enumerate(question_list):
                        true_score = human_data[text_id][annotator][question]
                        true_prob = [0.0] * 4
                        
                        if isinstance(true_score, (int, float)):
                            if true_score % 1 != 0:  
                                rounded_score = math.ceil(true_score)
                                rounded_score = max(min(rounded_score, 4), 1)
                                index = rounded_score - 1
                                true_prob[index] = 1.0
                            else:
                                true_score = max(min(int(true_score), 4), 1)
                                index = true_score - 1
                                true_prob[index] = 1.0
                        
                        # Original
                        self._add_human_annotation_4class(entry, true_prob, split_type, known_human_questions_val,
                                                        q_idx, annotator, question_indices[question], False)
                    
                    # Noisy Human questions ONLY if human_noise > 0
                    if human_noise > 0:
                        for q_idx, question in enumerate(question_list):
                            true_score = human_data[text_id][annotator][question]
                            true_prob = [0.0] * 4
                            
                            if isinstance(true_score, (int, float)):
                                if true_score % 1 != 0:  
                                    rounded_score = math.ceil(true_score)
                                    rounded_score = max(min(rounded_score, 4), 1)
                                    index = rounded_score - 1
                                    true_prob[index] = 1.0
                                else:
                                    true_score = max(min(int(true_score), 4), 1)
                                    index = true_score - 1
                                    true_prob[index] = 1.0
                            
                            # Noisy
                            noisy_prob = self.add_noise_to_human(true_prob, human_noise)
                            self._add_human_annotation_4class(entry, noisy_prob, split_type, known_human_questions_val,
                                                            q_idx, annotator, question_indices[question], True)
                    
                    data_list.append(entry)
    
    def _add_human_annotation(self, entry, prob, split_type, known_human_questions_val, q_idx, judge_id, question_idx, is_noisy):
        """Helper for adding human annotations (5 classes)."""
        if split_type == 'active_pool':
            mask_bit = 1
            combined_input = [mask_bit] + [0.0] * 5
            entry["known_questions"].append(0)
        elif split_type == 'train':
            mask_bit = 0
            combined_input = [mask_bit] + prob
            entry["known_questions"].append(1)
        elif split_type == 'validation':
            if q_idx < known_human_questions_val:
                mask_bit = 0
                combined_input = [mask_bit] + prob
                entry["known_questions"].append(1)
            else:
                mask_bit = 1
                combined_input = [mask_bit] + [0.0] * 5
                entry["known_questions"].append(0)
        elif split_type == 'test':
            if random.random() < 0.5:
                mask_bit = 1
                combined_input = [mask_bit] + [0.0] * 5
                entry["known_questions"].append(0)
            else:
                mask_bit = 0
                combined_input = [mask_bit] + prob
                entry["known_questions"].append(1)
        
        entry["input"].append(combined_input)
        entry["answers"].append(prob)   
        entry["annotators"].append(int(judge_id))
        entry["questions"].append(question_idx)
        entry["is_noisy"].append(is_noisy)
    
    def _add_human_annotation_4class(self, entry, prob, split_type, known_human_questions_val, q_idx, annotator, question_idx, is_noisy):
        """Helper for adding human annotations (4 classes)."""
        if split_type == 'active_pool':
            mask_bit = 1
            combined_input = [mask_bit] + [0.0] * 4
            entry["known_questions"].append(0)
        elif split_type == 'train':
            mask_bit = 0
            combined_input = [mask_bit] + prob
            entry["known_questions"].append(1)
        elif split_type == 'validation':
            if q_idx < known_human_questions_val:
                mask_bit = 0
                combined_input = [mask_bit] + prob
                entry["known_questions"].append(1)
            else:
                mask_bit = 1
                combined_input = [mask_bit] + [0.0] * 4
                entry["known_questions"].append(0)
        elif split_type == 'test':
            if random.random() < 0.5:
                mask_bit = 1
                combined_input = [mask_bit] + [0.0] * 4
                entry["known_questions"].append(0)
            else:
                mask_bit = 0
                combined_input = [mask_bit] + prob
                entry["known_questions"].append(1)
        
        entry["input"].append(combined_input)
        entry["answers"].append(prob)   
        entry["annotators"].append(int(annotator))
        entry["questions"].append(question_idx)
        entry["is_noisy"].append(is_noisy)

class NoisyAnnotationDataset(AnnotationDataset):
    """Extended dataset that tracks noisy vs original selections."""
    
    def __init__(self, data_path_or_list):
        super().__init__(data_path_or_list)
        
    def is_position_noisy(self, idx, position):
        """Check if a position corresponds to a noisy variable."""
        item = self.data[idx]
        if "is_noisy" in item and position < len(item["is_noisy"]):
            return item["is_noisy"][position]
        return False

def run_experiment_with_noise(
    dataset_train, dataset_val, dataset_test, 
    example_strategy, model,
    feature_strategy=None,
    cycles=5, 
    examples_per_cycle=10, 
    features_per_example=None,
    observe_all_features=False,
    epochs_per_cycle=3, 
    batch_size=8, 
    lr=1e-4,
    device=None, 
    resample_validation=False, 
    loss_type="cross_entropy",
    run_until_exhausted=False,
    gradient_top_only=False,
    cold_start=False,
):
    """Enhanced experiment runner that tracks noisy vs original selections."""
    
    arena = AnnotationArena(model, device)
    arena.set_dataset(dataset_train)
    
    metrics = {
        'training_losses': [],
        'val_losses': [],
        'examples_annotated': [],
        'features_annotated': [],
        'val_metrics': [],
        'test_expected_losses': [],
        'test_annotated_losses': [],
        'benefit_cost_ratios': [],
        'observation_costs': [],
        'remaining_pool_size': [],
        # New noise-specific metrics
        'original_selections_per_cycle': [],
        'noisy_selections_per_cycle': [],
        'selection_ratios_per_cycle': [],
        'cumulative_original_selections': [],
        'cumulative_noisy_selections': []
    }
    
    active_pool = list(range(len(dataset_train)))
    annotated_examples = []
    test_overlap_annotations = {}
    cycle_count = 0
    total_original_selections = 0
    total_noisy_selections = 0
    
    # Initial evaluation
    arena.set_dataset(dataset_val)
    val_metrics = arena.evaluate(list(range(len(dataset_val))))
    metrics['val_metrics'].append(val_metrics)
    metrics['val_losses'].append(val_metrics["avg_expected_loss"])
    
    arena.set_dataset(dataset_test)
    test_metrics = arena.evaluate(list(range(len(dataset_test))))
    metrics['test_expected_losses'].append(test_metrics["avg_expected_loss"])
    metrics['test_annotated_losses'].append(test_metrics["avg_expected_loss"])
    
    max_cycles = float('inf') if run_until_exhausted else cycles
    
    while cycle_count < max_cycles:
        if not active_pool:
            print(f"Active pool exhausted after {cycle_count} cycles")
            break
            
        print(f"=== Cycle {cycle_count+1}/{cycles if not run_until_exhausted else 'until exhausted'} ===")
        print(f"Active pool size: {len(active_pool)}")
        arena.set_dataset(dataset_train)

        valid_active_pool = []
        for idx in active_pool:
            if dataset_train.get_masked_positions(idx):
                valid_active_pool.append(idx)
        
        if len(valid_active_pool) < len(active_pool):
            print(f"Filtered active pool from {len(active_pool)} to {len(valid_active_pool)} examples")
            active_pool = valid_active_pool
        
        if not active_pool:
            print("No examples with masked positions left in active pool")
            break
        
        if resample_validation and cycle_count > 0:
            dataset_val, active_pool, val_indices = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, list(set(annotated_examples)), 
                strategy="add_selected", update_percentage=20
            ) 
        
        # Select examples (same as original)
        if example_strategy == "random":
            selected_examples = random.sample(active_pool, min(examples_per_cycle, len(active_pool)))
        elif example_strategy in ["gradient", "entropy", "badge"]:
            strategy_class = SelectionFactory.create_example_strategy(example_strategy, model, device, gradient_top_only=gradient_top_only)
            active_pool_examples = [dataset_train.get_data_entry(idx) for idx in active_pool]
            active_pool_subset = NoisyAnnotationDataset(active_pool_examples)
            
            selected_indices, _ = strategy_class.select_examples(
                active_pool_subset, num_to_select=min(examples_per_cycle, len(active_pool)),
                val_dataset=dataset_val, num_samples=3, batch_size=batch_size
            )
            
            selected_examples = [active_pool[idx] for idx in selected_indices]
        else:
            raise ValueError(f"Unknown example strategy: {example_strategy}")
        
        print(f"Selected {len(selected_examples)} examples")
        
        active_pool = [idx for idx in active_pool if idx not in selected_examples]
        
        for example in selected_examples:
            if example not in annotated_examples:
                annotated_examples.append(example)
        
        if resample_validation:
            if cold_start:
                dataset_val, active_pool_after_resample, val_examples = resample_validation_dataset(
                    dataset_train, dataset_val, active_pool, annotated_examples, 
                    strategy="add_selected_partial", selected_examples=selected_examples
                )
            else:
                dataset_val, active_pool_after_resample, val_examples = resample_validation_dataset(
                    dataset_train, dataset_val, active_pool, annotated_examples, 
                    strategy="add_selected", selected_examples=selected_examples
                )
            if cold_start:
                for idx in selected_examples:
                    if idx not in val_examples:
                        active_pool.append(idx)
            else:
                active_pool = active_pool_after_resample
        
        metrics['remaining_pool_size'].append(len(active_pool))
        
        total_features_annotated = 0
        cycle_benefit_cost_ratios = []
        cycle_observation_costs = []
        cycle_original_selections = 0
        cycle_noisy_selections = 0
        
        # Annotate selected examples with noise tracking
        for example_idx in selected_examples:
            arena.register_example(example_idx, add_all_positions=False)
            
            masked_positions = dataset_train.get_masked_positions(example_idx)
            if not masked_positions:
                continue
                
            if observe_all_features:
                positions_to_annotate = masked_positions
                position_benefit_costs = [(pos, 1.0, 1.0, 1.0) for pos in positions_to_annotate]
            else:
                if features_per_example is None:
                    features_per_example = 5
                
                features_to_annotate = min(features_per_example, len(masked_positions))
                
                candidate_variables = [
                    f"example_{example_idx}_position_{pos}" 
                    for pos in masked_positions
                ]
                
                if feature_strategy == "random":
                    selected_variables = random.sample(candidate_variables, features_to_annotate)
                    position_benefit_costs = []
                    for var in selected_variables:
                        _, pos = arena._parse_variable_id(var)
                        position_benefit_costs.append((pos, 1.0, 1.0, 1.0))
                elif feature_strategy == "sequential":
                    positions_to_annotate = masked_positions[:features_to_annotate]
                    position_benefit_costs = [(pos, 1.0, 1.0, 1.0) for pos in positions_to_annotate]
                elif feature_strategy in ["voi", "fast_voi", "entropy", "voi_argmax"]:
                    feature_suggestions = arena.suggest(
                        candidate_variables=candidate_variables,
                        strategy=feature_strategy, loss_type=loss_type
                    )
                    if not feature_suggestions:
                        continue
                    
                    position_benefit_costs = []
                    for var, benefit, cost, ratio, *extra in feature_suggestions[:features_to_annotate]:
                        _, pos = arena._parse_variable_id(var)
                        position_benefit_costs.append((pos, benefit, cost, ratio))
                else:
                    raise ValueError(f"Unknown feature strategy: {feature_strategy}")
            
            test_example_idx = example_idx % len(dataset_test)
            if test_example_idx not in test_overlap_annotations:
                test_overlap_annotations[test_example_idx] = []
            
            # Track noisy vs original selections
            for pos_data in position_benefit_costs:
                position = pos_data[0]
                
                # Check if this position is noisy
                is_noisy = dataset_train.is_position_noisy(example_idx, position)
                if is_noisy:
                    cycle_noisy_selections += 1
                else:
                    cycle_original_selections += 1
                
                annotation_success = arena.observe_position(example_idx, position)
                
                if annotation_success:
                    total_features_annotated += 1
                
                if len(pos_data) >= 4:
                    _, benefit, cost, ratio = pos_data[:4]
                    cycle_benefit_cost_ratios.append(ratio)
                    cycle_observation_costs.append(cost)
                else:
                    cycle_benefit_cost_ratios.append(1.0)
                    cycle_observation_costs.append(1.0)
                
                test_positions = dataset_test.get_masked_positions(test_example_idx)
                if position in test_positions:
                    test_overlap_annotations[test_example_idx].append(position)
                
                variable_id = f"example_{example_idx}_position_{position}"
                arena.predict(variable_id, train=True)
        
        # Update noise tracking metrics
        total_original_selections += cycle_original_selections
        total_noisy_selections += cycle_noisy_selections
        
        metrics['original_selections_per_cycle'].append(cycle_original_selections)
        metrics['noisy_selections_per_cycle'].append(cycle_noisy_selections)
        metrics['cumulative_original_selections'].append(total_original_selections)
        metrics['cumulative_noisy_selections'].append(total_noisy_selections)
        
        total_selections = cycle_original_selections + cycle_noisy_selections
        if total_selections > 0:
            selection_ratio = cycle_noisy_selections / total_selections
        else:
            selection_ratio = 0.0
        metrics['selection_ratios_per_cycle'].append(selection_ratio)
        
        print(f"Cycle {cycle_count+1}: Original={cycle_original_selections}, Noisy={cycle_noisy_selections}, Ratio={selection_ratio:.3f}")
        print(f"Total features annotated: {total_features_annotated}")
        
        # Training and evaluation (same as original)
        if total_features_annotated > 0:
            training_metrics = arena.train(
                epochs=epochs_per_cycle, batch_size=batch_size, lr=lr, 
                revisit_examples=True
            )
            metrics['training_losses'].append(training_metrics["avg_loss"])
        else:
            metrics['training_losses'].append(0.0)
        
        arena.set_dataset(dataset_val)
        val_metrics = arena.evaluate(list(range(len(dataset_val))))
        metrics['val_metrics'].append(val_metrics)
        metrics['val_losses'].append(val_metrics["avg_expected_loss"])
        
        arena.set_dataset(dataset_test)
        test_metrics = arena.evaluate(list(range(len(dataset_test))))
        metrics['test_expected_losses'].append(test_metrics["avg_expected_loss"])
        
        annotated_test_dataset = copy.deepcopy(dataset_test)
        annotations_applied = 0
        
        test_arena = AnnotationArena(model, device)
        test_arena.set_dataset(annotated_test_dataset)
        
        for test_idx, positions in test_overlap_annotations.items():
            for pos in positions:
                if test_arena.observe_position(test_idx, pos):
                    annotations_applied += 1

        if annotations_applied > 0:
            test_arena.set_dataset(annotated_test_dataset)
            annotated_test_metrics = test_arena.evaluate(list(range(len(annotated_test_dataset))))
            metrics['test_annotated_losses'].append(annotated_test_metrics["avg_expected_loss"])
        else:
            metrics['test_annotated_losses'].append(test_metrics["avg_expected_loss"])
        
        metrics['examples_annotated'].append(len(selected_examples))
        metrics['features_annotated'].append(total_features_annotated)
        metrics['benefit_cost_ratios'].append(np.mean(cycle_benefit_cost_ratios) if cycle_benefit_cost_ratios else 0.0)
        metrics['observation_costs'].append(np.sum(cycle_observation_costs) if cycle_observation_costs else 0.0)
        
        cycle_count += 1
        
    metrics['test_metrics'] = test_metrics
    arena_metrics = arena.get_metrics_history()
    metrics['arena_training_losses'] = arena_metrics["training_losses"]
    metrics['observation_history'] = arena_metrics["observation_history"]
    metrics['prediction_history'] = arena_metrics["prediction_history"]
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run Active Learning Experiments with Noisy Variables.")
    parser.add_argument("--cycles", type=int, default=5, help="Number of active learning cycles")
    parser.add_argument("--examples_per_cycle", type=int, default=20, help="Number of examples to select per cycle")
    parser.add_argument("--features_per_example", type=int, default=5, help="Number of features to select per example")
    parser.add_argument("--epochs_per_cycle", type=int, default=3, help="Number of training epochs per cycle")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--experiment", type=str, default="all", 
                      help="Experiment to run (all, random_all, gradient_voi, gradient_fast_voi, gradient_sequential)")
    parser.add_argument("--resample_validation", action="store_true", help="Resample validation set on each cycle")
    parser.add_argument("--loss_type", type=str, default="cross_entropy", help="Type of loss to use (cross_entropy, l2)")
    parser.add_argument("--run_until_exhausted", action="store_true", help="Run until annotation pool is exhausted")
    parser.add_argument("--dataset", type=str, default="hanna", help="Dataset to run the experiment")
    parser.add_argument("--runner", type=str, default="prabhav", help="Pass name to change directory paths")
    parser.add_argument("--cold_start", type=bool, default=False, help="Start with no annotation")
    parser.add_argument("--llm_sigma", type=float, default=0.5, help="Gaussian noise sigma for LLM annotations")
    parser.add_argument("--human_noise", type=float, default=0.3, help="Uniform noise level for human annotations")
    args = parser.parse_args()
    
    if args.runner == 'prabhav':
        base_path = '/export/fs06/psingh54/ActiveRubric-Internal/outputs'
    else:
        base_path = "outputs"

    dataset = args.dataset
    models_path = os.path.join(base_path, "models")
    results_path = os.path.join(base_path, f"results_noisy_{dataset}")
    os.makedirs(results_path, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"LLM noise sigma: {args.llm_sigma}, Human noise level: {args.human_noise}")
    
    # Initialize model with doubled input dimensions
    if dataset == "hanna":
        model = Imputer(
            question_num=7, max_choices=5, encoder_layers_num=6,
            attention_heads=4, hidden_dim=64, num_annotator=18, 
            annotator_embedding_dim=19, dropout=0.1
        ).to(device)
    elif dataset == "llm_rubric":
        model = Imputer(
            question_num=9, max_choices=4, encoder_layers_num=6,
            attention_heads=4, hidden_dim=64, num_annotator=24, 
            annotator_embedding_dim=24, dropout=0.1
        ).to(device)
    
    experiment_results = {}
    
    experiments_to_run = []
    if args.experiment == "all":
        # experiments_to_run = ["gradient_voi", "gradient_random", "random_random", "entropy_7"]
        experiments_to_run = ["gradient_voi", "random_random"]
        
    else:
        experiments_to_run = [args.experiment]
    
    for experiment in experiments_to_run:
        print(f"\n=== Running Noisy {experiment} Experiment ===")
        
        if args.runner == "prabhav":
            data_manager = NoisyDataManager(base_path + '/data/')
        else:
            data_manager = NoisyDataManager(base_path + f'/data_{dataset}/')

        if dataset == "hanna":
            data_manager.prepare_data(num_partition=1200, initial_train_ratio=0.0, dataset=dataset, 
                                    cold_start=args.cold_start, llm_sigma=args.llm_sigma, human_noise=args.human_noise)
        elif dataset == "llm_rubric":
            data_manager.prepare_data(num_partition=225, initial_train_ratio=0.0, dataset=dataset, 
                                    cold_start=args.cold_start, llm_sigma=args.llm_sigma, human_noise=args.human_noise)
        
        model_copy = copy.deepcopy(model)

        train_dataset = NoisyAnnotationDataset(data_manager.paths['train'])
        val_dataset = NoisyAnnotationDataset(data_manager.paths['validation'])
        test_dataset = NoisyAnnotationDataset(data_manager.paths['test'])
        active_pool_dataset = NoisyAnnotationDataset(data_manager.paths['active_pool'])
        
        print(f"Loaded datasets: Train={len(train_dataset)}, Val={len(val_dataset)}, "
            f"Test={len(test_dataset)}, Active Pool={len(active_pool_dataset)}")

        if experiment == "gradient_voi":
            results = run_experiment_with_noise(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="voi", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted
            )

        elif experiment == "gradient_random":
            results = run_experiment_with_noise(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="random", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted
            )
        
        elif experiment == "random_random":
            results = run_experiment_with_noise(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="random", feature_strategy="random", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted
            )

        elif experiment == "entropy_7":
            results = run_experiment_with_noise(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="entropy", feature_strategy="entropy", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted
            )

        else:
            print(f"Unknown experiment: {experiment}, skipping")
            continue
        
        experiment_results[experiment] = results
        
        torch.save(model_copy.state_dict(), os.path.join(models_path, f"noisy_{experiment}.pth"))
        with open(os.path.join(results_path, f"noisy_{experiment}.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        # Print noise selection summary
        print(f"\n=== Noise Selection Summary for {experiment} ===")
        total_orig = sum(results['original_selections_per_cycle'])
        total_noisy = sum(results['noisy_selections_per_cycle'])
        if total_orig + total_noisy > 0:
            final_ratio = total_noisy / (total_orig + total_noisy)
            print(f"Total Original: {total_orig}, Total Noisy: {total_noisy}")
            print(f"Final Noise Selection Ratio: {final_ratio:.3f}")
    
    if experiment_results:
        with open(os.path.join(results_path, "combined_noisy_results.json"), "w") as f:
            json.dump(experiment_results, f, indent=4)
            
        print(f"Noisy experiment results saved to {results_path}")

if __name__ == "__main__":
    main()