"""
Code to run active learning experiments with multi-level noisy variables.
Supports low, medium, and heavy noise levels for both LLM and human annotations.
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
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import pairwise_distances

from annotationArena import *
from utils import AnnotationDataset, DataManager, compute_metrics, resample_validation_dataset
from visualizations import *
from imputer import Imputer
from imputer_embedding import ImputerEmbedding
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

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_embeddings_features(dataset_entries, model_name='all-MiniLM-L6-v2'):
    """Extract sentence transformer embeddings for K-centers algorithm."""
    embedding_model = SentenceTransformer(model_name)
    features = []
    
    for entry in dataset_entries:
        if 'text_embedding' in entry and entry['text_embedding']:
            embedding = np.array(entry['text_embedding'][0])
        else:
            inputs = np.array(entry['input'])
            answer_dists = inputs[:, 1:] 
            mean_dist = np.mean(answer_dists, axis=0)
            std_dist = np.std(answer_dists, axis=0)
            entropy_per_pos = []
            
            for dist in answer_dists:
                if np.sum(dist) > 0:
                    normalized = dist / np.sum(dist)
                    entropy = -np.sum(normalized * np.log(normalized + 1e-10))
                    entropy_per_pos.append(entropy)
                else:
                    entropy_per_pos.append(0.0)

            embedding = np.concatenate([
                mean_dist, 
                std_dist, 
                [np.mean(entropy_per_pos), np.std(entropy_per_pos)]
            ])
        
        features.append(embedding)
    
    return np.array(features)

def extract_model_embeddings(dataset, example_indices, model, device):
    """Extract embeddings using the current imputer model state."""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for idx in example_indices:
            known_questions, inputs, answers, annotators, questions, model_embeddings = dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)
            annotators = annotators.unsqueeze(0).to(device)
            questions = questions.unsqueeze(0).to(device)
            if model_embeddings is not None:
                model_embeddings = model_embeddings.unsqueeze(0).to(device)
            
            # Get model's internal representation
            if hasattr(model, 'encoder'):
                if model_embeddings is not None:
                    feature_x, param_x = model.encoder.position_encoder(inputs, annotators, questions, model_embeddings)
                else:
                    feature_x, param_x = model.encoder.position_encoder(inputs, annotators, questions)
                
                # Use the feature representation as embedding
                embedding = feature_x[0].mean(dim=0).cpu().numpy()  # Average across positions
            else:
                # Fallback to statistical features
                entry = dataset.get_data_entry(idx)
                inputs_array = np.array(entry['input'])
                answer_dists = inputs_array[:, 1:]
                embedding = np.concatenate([
                    np.mean(answer_dists, axis=0),
                    np.std(answer_dists, axis=0),
                    [np.mean(answer_dists), np.std(answer_dists)]
                ])
            
            embeddings.append(embedding)
    
    return np.array(embeddings)

def greedy_k_centers(features, k, random_seed=42):
    """Greedy K-centers algorithm for diverse subset selection."""
    np.random.seed(random_seed)
    n = len(features)
    
    if k >= n:
        return list(range(n))
    
    selected = [np.random.randint(n)]
    
    for _ in range(k - 1):
        distances = []
        for i in range(n):
            if i in selected:
                distances.append(0)
            else:
                min_dist = min(np.linalg.norm(features[i] - features[j]) for j in selected)
                distances.append(min_dist)
                
        next_center = np.argmax(distances)
        selected.append(next_center)
    
    return selected

class NoisyDataManager(DataManager):
    """Extended DataManager that creates multi-level noisy copies of annotations."""
    
    def __init__(self, base_path):
        super().__init__(base_path)
        self.noise_levels = ['low', 'medium', 'heavy']
        
    def add_noise_to_llm_low(self, prob_dist, confidence_reduction=0.3):
        """Low noise: preserve argmax, reduce confidence."""
        original = np.array(prob_dist)
        argmax_idx = np.argmax(original)
        
        new_max_prob = max(0.5, original[argmax_idx] - confidence_reduction)
        remaining_prob = 1.0 - new_max_prob
        other_prob = remaining_prob / (len(original) - 1) if len(original) > 1 else 0.0
        
        new_dist = np.full_like(original, other_prob)
        new_dist[argmax_idx] = new_max_prob
        return new_dist.tolist()
    
    def add_noise_to_llm_medium(self, prob_dist, flip_prob=0.9, noise_strength=0.6):
        """Medium noise: flip argmax ~70% of time, make distribution flatter."""
        original = np.array(prob_dist)
        argmax_idx = np.argmax(original)
        
        if np.random.random() < flip_prob:
            # Force argmax flip
            other_indices = [i for i in range(len(original)) if i != argmax_idx]
            new_argmax = np.random.choice(other_indices)
            
            # Create flatter distribution with new argmax
            new_dist = np.random.dirichlet([1.5] * len(original))
            new_dist[new_argmax] = max(new_dist[new_argmax], 0.4)
            new_dist = new_dist / np.sum(new_dist)
        else:
            # Keep same argmax but flatten distribution
            new_max_prob = max(0.4, original[argmax_idx] - noise_strength)
            remaining_prob = 1.0 - new_max_prob
            other_prob = remaining_prob / (len(original) - 1) if len(original) > 1 else 0.0
            
            new_dist = np.full_like(original, other_prob)
            new_dist[argmax_idx] = new_max_prob
        
        return new_dist.tolist()
    
    def add_noise_to_llm_heavy(self, prob_dist, uniformity=0.8):
        """Heavy noise: always flip argmax, make very flat distribution."""
        original = np.array(prob_dist)
        argmax_idx = np.argmax(original)
        num_classes = len(original)
        
        # Always flip argmax
        other_indices = [i for i in range(num_classes) if i != argmax_idx]
        new_argmax = np.random.choice(other_indices)
        
        if np.random.random() < uniformity:
            # Very flat/uniform distribution
            noise = np.random.normal(0, 0.05, num_classes)
            uniform_base = 1.0 / num_classes
            new_dist = np.full(num_classes, uniform_base) + noise
            new_dist = np.abs(new_dist)
            
            # Ensure new argmax is slightly higher
            new_dist[new_argmax] += 0.1
            new_dist = new_dist / np.sum(new_dist)
        else:
            # Random but ensure argmax flip
            new_dist = np.random.dirichlet([0.8] * num_classes)
            new_dist[new_argmax] = max(new_dist[new_argmax], 0.3)
            new_dist = new_dist / np.sum(new_dist)
        
        return new_dist.tolist()

    def add_noise_to_llm(self, prob_dist, alpha_multiplier=1.0, noise_level='medium'):
        """Add noise to LLM probability distributions using improved noise functions."""
        if noise_level == 'low':
            return self.add_noise_to_llm_low(prob_dist)
        elif noise_level == 'medium':
            return self.add_noise_to_llm_medium(prob_dist)
        elif noise_level == 'heavy':
            return self.add_noise_to_llm_heavy(prob_dist)
        else:
            return prob_dist

    def add_noise_to_human(self, one_hot, flip_prob=0.8):
        """Single level human noise: flip argmax with given probability."""
        one_hot = np.array(one_hot)
        original_category = np.argmax(one_hot)
        
        if np.random.random() < flip_prob:
            num_categories = len(one_hot)
            other_categories = [i for i in range(num_categories) if i != original_category]
            new_category = np.random.choice(other_categories)
            noisy_one_hot = np.zeros_like(one_hot)
            noisy_one_hot[new_category] = 1.0
            return noisy_one_hot.tolist()
        else:
            return one_hot.tolist()
    
    def prepare_data(self, num_partition=1200, known_human_questions_val=0, initial_train_ratio=0.0, 
            dataset="hanna", cold_start=False, llm_alpha_multiplier=1.0, human_flip_prob=0.3, 
            use_embedding=False, validation_set_size=50, active_set_size=100):
        """Prepare data with multi-level noisy copies. K-centers applied dynamically in experiment loop."""
        
        print(f"Preparing data with improved multi-level noise:")
        print(f"   - LLM noise levels: {self.noise_levels} (improved functions)")
        print(f"   - Human noise: single level (flip={human_flip_prob})")
        print(f"   - Text embeddings: {use_embedding}")
        print(f"   - Validation set size: {validation_set_size}")
        print(f"   - Active set size: {active_set_size} (applied dynamically)")

        if use_embedding and not dataset == "hanna":
            raise ValueError("Text embeddings only supported for HANNA dataset")
        if dataset == "gaussian":
            pass
        try:
            with open(os.path.join(self.base_path, "gpt-3.5-turbo-data-new.json"), "r") as f:
                llm_data = json.load(f)
            with open(os.path.join(self.base_path, "human-data-new.json"), "r") as f:
                human_data = json.load(f)
        except FileNotFoundError:
            return False

        if use_embedding and not os.path.exists(os.path.join(self.base_path, "text_embeddings.json")):
            print("Preparing text embeddings with SentenceBERT...")
            self.prepare_text_embeddings(num_partition)
            print("Text embeddings ready\n")
        
        text_ids = list(human_data.keys())
        if dataset == "hanna":
            question_list = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
        elif dataset == "llm_rubric":
            question_list = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']
        question_indices = {q: i for i, q in enumerate(question_list)}
        
        random.seed(42)
        random.shuffle(text_ids)
        
        # New data split structure: N-T-E-V-A (no K-centers here, done dynamically)
        initial_train_size = int(num_partition * initial_train_ratio)
        test_size = int(num_partition * 0.2)
        
        initial_train_texts = text_ids[:initial_train_size]
        test_texts = text_ids[initial_train_size:initial_train_size + test_size]
        
        remaining_pool = text_ids[initial_train_size + test_size:initial_train_size + test_size + (num_partition - initial_train_size - test_size)]
        
        validation_texts = remaining_pool[:validation_set_size]
        
        # Keep full active pool (no K-centers reduction here)
        active_pool_texts = remaining_pool[validation_set_size:]
        
        initial_train_data = []
        validation_data = []
        test_data = []
        active_pool_data = []
        
        print('Creating multi-level noisy annotations...')
        print('   - Processing Train split...')
        self._prepare_entries_with_multilevel_noise(initial_train_texts, initial_train_data, 'train', llm_data, human_data, 
                                    question_list, question_indices, known_human_questions_val, dataset, 
                                    cold_start, llm_alpha_multiplier, human_flip_prob, use_embedding)
        print('    - Processing Validation split...')
        self._prepare_entries_with_multilevel_noise(validation_texts, validation_data, 'validation', llm_data, human_data, 
                                    question_list, question_indices, known_human_questions_val, dataset, 
                                    cold_start, llm_alpha_multiplier, human_flip_prob, use_embedding)
        print('    - Processing Test split...')
        self._prepare_entries_with_multilevel_noise(test_texts, test_data, 'test', llm_data, human_data, 
                                    question_list, question_indices, known_human_questions_val, dataset, 
                                    cold_start, llm_alpha_multiplier, human_flip_prob, use_embedding)
        print('    - Processing Active Pool (full size, K-centers applied dynamically)...')
        self._prepare_entries_with_multilevel_noise(active_pool_texts, active_pool_data, 'active_pool', llm_data, human_data, 
                                    question_list, question_indices, known_human_questions_val, dataset, 
                                    cold_start, llm_alpha_multiplier, human_flip_prob, use_embedding)
        
        print('Saving datasets...')
        for key, data in tqdm(zip(['train', 'validation', 'test', 'active_pool', 'original_train', 'original_validation', 'original_test', 'original_active_pool'],
                            [initial_train_data, validation_data, test_data, active_pool_data, initial_train_data, validation_data, test_data, active_pool_data])):
            with open(self.paths[key], "w") as f:
                json.dump(data, f)

        print('Multi-level noisy data creation complete!')
        print(f'Final dataset sizes: Train={len(initial_train_data)}, Val={len(validation_data)}, Test={len(test_data)}, Active={len(active_pool_data)}')
        
        if len(active_pool_data) > 0:
            sample_entry = active_pool_data[0]
            annotations_per_example = len(sample_entry['input'])
            print(f'DEBUG: Annotations per example: {annotations_per_example}')

        return True
    
    def _prepare_entries_with_multilevel_noise(self, texts, data_list, split_type, llm_data, human_data, question_list, 
                          question_indices, known_human_questions_val, dataset, cold_start, 
                          llm_alpha_multiplier, human_flip_prob, use_embedding):
        """Prepare entries with original + 3 noise level copies (low, medium, heavy)."""
        
        if use_embedding:
            with open(os.path.join(self.base_path, "text_embeddings.json"), "r") as file:
                text_embeddings = json.load(file)
            with open(os.path.join(self.base_path, "propmt_embeddings.json"), "r") as file:
               question_embeddings = json.load(file)
        
        if dataset == "hanna":
            for text_id in tqdm(texts, desc=f"{split_type.capitalize()} processing"):
                if text_id not in llm_data:
                    continue
                
                entry = {
                    "known_questions": [], 
                    "input": [], 
                    "answers": [], 
                    "true_answers": [],
                    "annotators": [],
                    "questions": [],
                    "orig_split": split_type,
                    "observation_history": [],
                    "noise_info": []
                }
                
                annotators = list(human_data[text_id].keys())

                self._add_llm_annotations(entry, llm_data[text_id], question_list, question_indices, 
                                        split_type, cold_start, llm_alpha_multiplier)
                
                self._add_human_annotations(entry, human_data[text_id], annotators, question_list, 
                                          question_indices, split_type, known_human_questions_val, 
                                          cold_start, human_flip_prob)
                
                if use_embedding:
                    self._add_text_embeddings(entry, text_id, text_embeddings, question_embeddings, 
                                            question_list, annotators)
                
                data_list.append(entry)

        elif dataset == "llm_rubric":
            for text_id in texts:
                annotators = list(human_data[text_id].keys())
                for annotator in annotators:
                    entry = {
                        "known_questions": [], 
                        "input": [], 
                        "answers": [], 
                        "true_answers": [],
                        "annotators": [],
                        "questions": [],
                        "orig_split": split_type,
                        "observation_history": [],
                        "noise_info": []
                    }
                    
                    self._add_llm_annotations_4class(entry, llm_data[text_id], question_list, question_indices,
                                                   split_type, cold_start, llm_alpha_multiplier)
                    
                    self._add_human_annotations_4class(entry, human_data[text_id][annotator], annotator,
                                                     question_list, question_indices, split_type, 
                                                     known_human_questions_val, cold_start, human_flip_prob)
                    
                    data_list.append(entry)
    
    def _add_llm_annotations(self, entry, llm_text_data, question_list, question_indices, 
                       split_type, cold_start, llm_alpha_multiplier):
        """Add original LLM annotations plus 3 noise levels."""
        for q_idx, question in enumerate(question_list):
            true_prob = llm_text_data[question]
            
            self._add_single_llm_annotation(entry, true_prob, question_indices[question], 
                                        split_type, cold_start, noise_type='original')
            
            for noise_level in self.noise_levels:
                noisy_prob = self.add_noise_to_llm(true_prob, llm_alpha_multiplier, noise_level)
                self._add_single_llm_annotation(entry, noisy_prob, question_indices[question], 
                                            split_type, cold_start, noise_type=f'llm_{noise_level}', 
                                            true_answer=true_prob)
    
    def _add_human_annotations(self, entry, human_text_data, annotators, question_list, 
                         question_indices, split_type, known_human_questions_val, 
                         cold_start, human_flip_prob):
        """Add original human annotations plus single noise level for each annotator."""
        for judge_id in annotators:
            for q_idx, question in enumerate(question_list):
                true_score = human_text_data[judge_id][question]
                true_prob = self._score_to_prob(true_score, 5)
                
                self._add_single_human_annotation(entry, true_prob, int(judge_id), 
                                                question_indices[question], split_type, 
                                                known_human_questions_val, q_idx, 
                                                noise_type='original')
                
                if human_flip_prob > 0:
                    noisy_prob = self.add_noise_to_human(true_prob, human_flip_prob)
                    self._add_single_human_annotation(entry, noisy_prob, int(judge_id),
                                                    question_indices[question], split_type,
                                                    known_human_questions_val, q_idx,
                                                    noise_type='human_noisy',
                                                    true_answer=true_prob)

    def _add_single_llm_annotation(self, entry, prob, question_idx, split_type, cold_start, 
                                 noise_type='original', true_answer=None):
        """Add a single LLM annotation with proper masking logic."""
        if cold_start and split_type in ['active_pool', 'validation']:
            mask_bit = 1
            combined_input = [mask_bit] + [0.0] * 5
            entry["known_questions"].append(0)
        else:
            mask_bit = 0
            combined_input = [mask_bit] + prob
            entry["known_questions"].append(1)
        
        entry["input"].append(combined_input)
        entry["answers"].append(prob)
        entry["true_answers"].append(true_answer if true_answer is not None else prob)
        entry["annotators"].append(-1)
        entry["questions"].append(question_idx)
        entry["noise_info"].append(noise_type)

    def _add_single_human_annotation(self, entry, prob, judge_id, question_idx, split_type, 
                                   known_human_questions_val, q_idx, noise_type='original', 
                                   true_answer=None):
        """Add a single human annotation with proper masking logic."""
        if split_type == 'active_pool':
            mask_bit = 1
            combined_input = [mask_bit] + [0.0] * 5
            entry["known_questions"].append(0)
        elif split_type == 'validation':
            if q_idx < known_human_questions_val:
                mask_bit = 0
                combined_input = [mask_bit] + prob
                entry["known_questions"].append(1)
            else:
                mask_bit = 1
                combined_input = [mask_bit] + [0.0] * 5
                entry["known_questions"].append(0)
        elif split_type == 'train':
            if noise_type == 'original':
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
        entry["true_answers"].append(true_answer if true_answer is not None else prob)   
        entry["annotators"].append(judge_id)
        entry["questions"].append(question_idx)
        entry["noise_info"].append(noise_type)

    def _add_llm_annotations_4class(self, entry, llm_text_data, question_list, question_indices, 
                                  split_type, cold_start, llm_alpha_multiplier):
        """Add LLM annotations for 4-class problems (LLM_RUBRIC)."""
        for q_idx, question in enumerate(question_list):
            true_prob = llm_text_data[question]
            
            self._add_single_llm_annotation_4class(entry, true_prob, question_indices[question], 
                                                 split_type, cold_start, noise_type='original')
            
            for noise_level in self.noise_levels:
                noisy_prob = self.add_noise_to_llm(true_prob, llm_alpha_multiplier, noise_level)
                self._add_single_llm_annotation_4class(entry, noisy_prob, question_indices[question], 
                                                     split_type, cold_start, noise_type=f'llm_{noise_level}',
                                                     true_answer=true_prob)

    def _add_single_llm_annotation_4class(self, entry, prob, question_idx, split_type, cold_start, 
                                        noise_type='original', true_answer=None):
        """Add single LLM annotation for 4-class problems."""
        if cold_start and split_type in ['active_pool', 'validation']:
            mask_bit = 1
            combined_input = [mask_bit] + [0.0] * 4
            entry["known_questions"].append(0)
        else:
            mask_bit = 0
            combined_input = [mask_bit] + prob
            entry["known_questions"].append(1)
        
        entry["input"].append(combined_input)
        entry["answers"].append(prob)
        entry["true_answers"].append(true_answer if true_answer is not None else prob)
        entry["annotators"].append(-1)
        entry["questions"].append(question_idx)
        entry["noise_info"].append(noise_type)

    def _add_human_annotations_4class(self, entry, human_annotator_data, annotator, question_list, 
                                    question_indices, split_type, known_human_questions_val, 
                                    cold_start, human_flip_prob):
        """Add human annotations for 4-class problems."""
        for q_idx, question in enumerate(question_list):
            true_score = human_annotator_data[question]
            true_prob = self._score_to_prob(true_score, 4)
            
            self._add_single_human_annotation_4class(entry, true_prob, int(annotator),
                                                   question_indices[question], split_type,
                                                   known_human_questions_val, q_idx, 
                                                   noise_type='original')
            
            if human_flip_prob > 0:
                noisy_prob = self.add_noise_to_human(true_prob, human_flip_prob)
                self._add_single_human_annotation_4class(entry, noisy_prob, int(annotator),
                                                       question_indices[question], split_type,
                                                       known_human_questions_val, q_idx,
                                                       noise_type='human_noisy',
                                                       true_answer=true_prob)

    def _add_single_human_annotation_4class(self, entry, prob, annotator, question_idx, split_type, 
                                          known_human_questions_val, q_idx, noise_type='original', 
                                          true_answer=None):
        """Add single human annotation for 4-class problems."""
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
        entry["true_answers"].append(true_answer if true_answer is not None else prob)   
        entry["annotators"].append(annotator)
        entry["questions"].append(question_idx)
        entry["noise_info"].append(noise_type)

    def _add_text_embeddings(self, entry, text_id, text_embeddings, question_embeddings, 
                       question_list, annotators):
        """Add text embeddings for all annotation positions."""
        
        total_positions = len(entry["input"])
        entry["text_embedding"] = [[] for _ in range(total_positions)]
        
        # Simply assign embeddings based on the question for each position
        for pos_idx in range(total_positions):
            question_idx = entry['questions'][pos_idx]
            question = question_list[question_idx]
            
            text_embedding = text_embeddings[int(text_id)]
            question_embedding = question_embeddings[question]
            final_embedding = (torch.tensor(text_embedding) + torch.tensor(question_embedding)).tolist()
            
            entry["text_embedding"][pos_idx] = final_embedding

    def _score_to_prob(self, score, num_classes):
        """Convert score to one-hot probability distribution."""
        prob = [0.0] * num_classes
        if isinstance(score, (int, float)):
            if score % 1 != 0:  
                rounded_score = math.ceil(score)
                rounded_score = max(min(rounded_score, num_classes), 1)
                index = rounded_score - 1
                prob[index] = 1.0
            else:
                score = max(min(int(score), num_classes), 1)
                index = score - 1
                prob[index] = 1.0
        return prob


class MultiLevelNoisyAnnotationDataset(AnnotationDataset):
    """Extended dataset that tracks multi-level noise selections."""
    
    def __init__(self, data_path_or_list):
        super().__init__(data_path_or_list)
        
    def get_noise_type(self, idx, position):
        """Get the noise type for a specific position."""
        item = self.data[idx]
        if "noise_info" in item and position < len(item["noise_info"]):
            return item["noise_info"][position]
        return 'unknown'
    
    def is_position_noisy(self, idx, position):
        """Check if a position corresponds to any noisy variable."""
        noise_type = self.get_noise_type(idx, position)
        return noise_type != 'original'


def run_experiment_with_multilevel_noise(
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
    human_cost=1.0,
    llm_cost=1.0,
    validation_set_size=50,
    active_set_size=100,
    initial_train_dataset=None
):
    """Enhanced experiment runner with dynamic K-centers active set selection."""

    if initial_train_dataset is not None and len(initial_train_dataset) > 0:
        arena = AnnotationArena(model, device)
        print(f"Initial training on {len(initial_train_dataset)} clean examples...")
        arena.set_dataset(initial_train_dataset)
        
        for idx in range(len(initial_train_dataset)):
            arena.register_example(idx, add_all_positions=False)
            known_positions = initial_train_dataset.get_known_positions(idx)
            for pos in known_positions:
                arena.observe_position(idx, pos)
                variable_id = f"example_{idx}_position_{pos}"
                arena.predict(variable_id, train=True)
        
        arena.train(epochs=10, batch_size=batch_size, lr=lr)
        print("Initial training completed!")
    
    arena = AnnotationArena(model, device)
    arena.set_dataset(dataset_train)
    
    noise_categories = ['original_llm', 'llm_low', 'llm_medium', 'llm_heavy',
                       'original_human', 'human_noisy']
    
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
        'active_subset_size': []
    }
    
    for category in noise_categories:
        metrics[f'{category}_selections_per_cycle'] = []
        metrics[f'cumulative_{category}_selections'] = []
    
    metrics['selection_breakdown_per_cycle'] = []
    
    # Start with full active pool
    active_pool = list(range(len(dataset_train)))
    annotated_examples = []
    test_overlap_annotations = {}
    cycle_count = 0
    
    cumulative_selections = {category: 0 for category in noise_categories}
    validation_example_indices = list(range(len(dataset_val)))
    
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
            
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle_count+1}/{cycles if not run_until_exhausted else '∞'}")
        print(f"Full active pool: {len(active_pool)} examples")
        print(f"{'='*60}")
        
        arena.set_dataset(dataset_train)

        # Filter valid active pool
        valid_active_pool = []
        for idx in active_pool:
            if dataset_train.get_masked_positions(idx):
                valid_active_pool.append(idx)
        
        if len(valid_active_pool) < len(active_pool):
            print(f"Filtered pool: {len(active_pool)} → {len(valid_active_pool)} (removed examples with no masked positions)")
            active_pool = valid_active_pool
        
        if not active_pool:
            print("No examples with masked positions remaining")
            break
        
        # Resample validation if needed
        if resample_validation and cycle_count > 0:
            current_val_indices = list(range(len(dataset_val)))  # Get current validation indices
            active_pool.extend(current_val_indices)  # Add them back to active pool
            
            dataset_val, active_pool, validation_example_indices = resample_validation_dataset(
                dataset_train, dataset_val, active_pool, annotated_examples, 
                strategy="balanced_fixed_size", 
                selected_examples=annotated_examples[-examples_per_cycle:] if annotated_examples else [],
                validation_set_size=validation_set_size
            )
        
        # DYNAMIC K-CENTERS: Apply K-centers to current active pool using model embeddings
        print(f"Applying dynamic K-centers to select {min(active_set_size, len(active_pool))} from {len(active_pool)} examples...")
        if len(active_pool) > active_set_size:
            # Extract model embeddings for current pool
            model_embeddings = extract_model_embeddings(dataset_train, active_pool, model, device)
            selected_subset_indices = greedy_k_centers(model_embeddings, active_set_size, random_seed=42 + cycle_count)
            active_subset = [active_pool[i] for i in selected_subset_indices]
            print(f"K-centers selected {len(active_subset)} diverse examples from pool")
        else:
            active_subset = active_pool.copy()
            print(f"Pool smaller than target, using all {len(active_subset)} examples")
        
        metrics['active_subset_size'].append(len(active_subset))
        
        # Debug logging
        print(f"DEBUG: Active subset size: {len(active_subset)}")
        if len(active_subset) > 0:
            sample_idx = active_subset[0]
            sample_entry = dataset_train.get_data_entry(sample_idx)
            sample_features = len(sample_entry['input'])
        
        # Example selection from active subset
        if example_strategy == "random":
            selected_examples = random.sample(active_subset, min(examples_per_cycle, len(active_subset)))
        elif example_strategy in ["gradient", "entropy", "badge"]:
            strategy_class = SelectionFactory.create_example_strategy(example_strategy, model, device, gradient_top_only=gradient_top_only)
            active_subset_examples = [dataset_train.get_data_entry(idx) for idx in active_subset]
            active_subset_dataset = MultiLevelNoisyAnnotationDataset(active_subset_examples)
            
            selected_indices, _ = strategy_class.select_examples(
                active_subset_dataset, num_to_select=min(examples_per_cycle, len(active_subset)),
                val_dataset=dataset_val, num_samples=3, batch_size=batch_size
            )
            
            selected_examples = [active_subset[idx] for idx in selected_indices]
        else:
            raise ValueError(f"Unknown example strategy: {example_strategy}")
        
        print(f"Selected {len(selected_examples)} examples from active subset")

        # Remove selected examples from full active pool
        active_pool = [idx for idx in active_pool if idx not in selected_examples]

        # Put back the unselected examples from active_subset
        if len(active_subset) < len(active_pool) + len(selected_examples): 
            unselected_from_subset = [idx for idx in active_subset if idx not in selected_examples]
            active_pool.extend(unselected_from_subset)
        
        for example in selected_examples:
            if example not in annotated_examples:
                annotated_examples.append(example)
        
        metrics['remaining_pool_size'].append(len(active_pool))
        
        total_features_annotated = 0
        cycle_benefit_cost_ratios = []
        cycle_observation_costs = []
        cycle_selections = {category: 0 for category in noise_categories}
        
        for example_idx in selected_examples:
            arena.register_example(example_idx, add_all_positions=False)
            
            masked_positions = dataset_train.get_masked_positions(example_idx)
            if not masked_positions:
                continue
                                
            if observe_all_features:
                positions_to_annotate = masked_positions
                position_benefit_costs = []
                for pos in positions_to_annotate:
                    entry = dataset_train.get_data_entry(example_idx)
                    is_llm = entry['annotators'][pos] == -1
                    cost = llm_cost if is_llm else human_cost
                    position_benefit_costs.append((pos, 1.0, cost, 1.0/cost))
            else:
                if features_per_example is None:
                    features_per_example = 5
                
                features_to_annotate = min(features_per_example, len(masked_positions))
                
                candidate_variables = [
                    f"example_{example_idx}_position_{pos}" 
                    for pos in masked_positions
                ]
                
                example_costs = {}
                entry = dataset_train.get_data_entry(example_idx)
                for pos in masked_positions:
                    is_llm = entry['annotators'][pos] == -1
                    example_costs[pos] = llm_cost if is_llm else human_cost
                
                if feature_strategy == "random":
                    selected_variables = random.sample(candidate_variables, features_to_annotate)
                    position_benefit_costs = []
                    for var in selected_variables:
                        _, pos = arena._parse_variable_id(var)
                        cost = example_costs[pos]
                        position_benefit_costs.append((pos, 1.0, cost, 1.0/cost))
                elif feature_strategy == "sequential":
                    positions_to_annotate = masked_positions[:features_to_annotate]
                    position_benefit_costs = []
                    for pos in positions_to_annotate:
                        cost = example_costs[pos]
                        position_benefit_costs.append((pos, 1.0, cost, 1.0/cost))
                elif feature_strategy in ["voi", "fast_voi", "entropy", "voi_argmax"]:
                    feature_suggestions = arena.suggest(
                        candidate_variables=candidate_variables,
                        strategy=feature_strategy, loss_type=loss_type
                    )
                    if not feature_suggestions:
                        continue
                    
                    position_benefit_costs = []
                    for var, benefit, orig_cost, orig_ratio, *extra in feature_suggestions[:features_to_annotate]:
                        _, pos = arena._parse_variable_id(var)
                        actual_cost = example_costs[pos]
                        actual_ratio = benefit / actual_cost
                        position_benefit_costs.append((pos, benefit, actual_cost, actual_ratio))
                else:
                    raise ValueError(f"Unknown feature strategy: {feature_strategy}")
                        
            test_example_idx = example_idx % len(dataset_test)
            if test_example_idx not in test_overlap_annotations:
                test_overlap_annotations[test_example_idx] = []
            
            for pos_data in position_benefit_costs:
                position = pos_data[0]
                
                noise_type = dataset_train.get_noise_type(example_idx, position)
                is_llm = dataset_train.get_data_entry(example_idx)['annotators'][position] == -1
                
                if is_llm:
                    if noise_type == 'original':
                        category = 'original_llm'
                    elif 'low' in noise_type:
                        category = 'llm_low'
                    elif 'medium' in noise_type:
                        category = 'llm_medium'
                    elif 'heavy' in noise_type:
                        category = 'llm_heavy'
                    else:
                        category = 'original_llm'
                else:
                    if noise_type == 'original':
                        category = 'original_human'
                    elif 'noisy' in noise_type:
                        category = 'human_noisy'
                    else:
                        category = 'original_human'
                
                cycle_selections[category] += 1
                cumulative_selections[category] += 1
                
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
        
        for category in noise_categories:
            metrics[f'{category}_selections_per_cycle'].append(cycle_selections[category])
            metrics[f'cumulative_{category}_selections'].append(cumulative_selections[category])
        
        metrics['selection_breakdown_per_cycle'].append(cycle_selections.copy())
        
        total_cycle_selections = sum(cycle_selections.values())
        print(f"\nSELECTION BREAKDOWN (Cycle {cycle_count+1}):")
        print(f"   Total selected: {total_cycle_selections}")
        if total_cycle_selections > 0:
            print(f"   LLM Original: {cycle_selections['original_llm']} ({cycle_selections['original_llm']/total_cycle_selections*100:.1f}%)")
            print(f"   LLM Low:      {cycle_selections['llm_low']} ({cycle_selections['llm_low']/total_cycle_selections*100:.1f}%)")
            print(f"   LLM Medium:   {cycle_selections['llm_medium']} ({cycle_selections['llm_medium']/total_cycle_selections*100:.1f}%)")
            print(f"   LLM Heavy:    {cycle_selections['llm_heavy']} ({cycle_selections['llm_heavy']/total_cycle_selections*100:.1f}%)")
            print(f"   Human Original: {cycle_selections['original_human']} ({cycle_selections['original_human']/total_cycle_selections*100:.1f}%)")
            print(f"   Human Noisy:    {cycle_selections['human_noisy']} ({cycle_selections['human_noisy']/total_cycle_selections*100:.1f}%)")
        
        print(f"Total features annotated: {total_features_annotated}")
        
        if total_features_annotated > 0:
            print(f"Training model ({epochs_per_cycle} epochs)...")
            training_metrics = arena.train(
                epochs=epochs_per_cycle, batch_size=batch_size, lr=lr, 
                revisit_examples=True
            )
            metrics['training_losses'].append(training_metrics["avg_loss"])
            print(f"   Training loss: {training_metrics['avg_loss']:.4f}")
        else:
            metrics['training_losses'].append(0.0)
        
        arena.set_dataset(dataset_val)
        val_metrics = arena.evaluate(list(range(len(dataset_val))))
        metrics['val_metrics'].append(val_metrics)
        metrics['val_losses'].append(val_metrics["avg_expected_loss"])
        print(f"Validation loss: {val_metrics['avg_expected_loss']:.4f}")
        
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
        
        print(f"Test loss: {test_metrics['avg_expected_loss']:.4f}")
        
        metrics['examples_annotated'].append(len(selected_examples))
        metrics['features_annotated'].append(total_features_annotated)
        metrics['benefit_cost_ratios'].append(np.mean(cycle_benefit_cost_ratios) if cycle_benefit_cost_ratios else 0.0)
        metrics['observation_costs'].append(np.sum(cycle_observation_costs) if cycle_observation_costs else 0.0)
        
        cycle_count += 1
        
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    
    total_cumulative = sum(cumulative_selections.values())
    if total_cumulative > 0:
        print(f"FINAL SELECTION SUMMARY:")
        print(f"   Total annotations: {total_cumulative}")
        for category in noise_categories:
            count = cumulative_selections[category]
            percentage = count / total_cumulative * 100
            print(f"   {category:15s}: {count:4d} ({percentage:5.1f}%)")
    
    metrics['test_metrics'] = test_metrics
    arena_metrics = arena.get_metrics_history()
    metrics['arena_training_losses'] = arena_metrics["training_losses"]
    metrics['observation_history'] = arena_metrics["observation_history"]
    metrics['prediction_history'] = arena_metrics["prediction_history"]
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run Active Learning Experiments with Multi-Level Noisy Variables.")
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
    parser.add_argument("--llm_alpha_multiplier", type=float, default=1.0, help="Base Dirichlet concentration multiplier for LLM annotations")
    parser.add_argument("--human_flip_prob", type=float, default=0.3, help="Base probability of flipping human annotations")
    parser.add_argument("--use_embedding", type=bool, default=False, help="Use embeddings for texts")
    parser.add_argument("--human_cost", type=float, default=1.0, help="Cost of human annotations")
    parser.add_argument("--llm_cost", type=float, default=1.0, help="Cost of LLM annotations")
    parser.add_argument("--validation_set_size", type=int, default=50, help="Fixed size for validation set")
    parser.add_argument("--active_set_size", type=int, default=100, help="Size of active subset selected by K-centers each cycle")
    args = parser.parse_args()
    
    if args.runner == 'prabhav':
        base_path = '/export/fs06/psingh54/ActiveRubric-Internal/outputs'
    else:
        base_path = "outputs"

    dataset = args.dataset
    models_path = os.path.join(base_path, "models")
    results_path = os.path.join(base_path, f"results_multilevel_noisy_{dataset}/experiment_both")
    os.makedirs(results_path, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Base noise parameters: LLM α={args.llm_alpha_multiplier}, Human flip={args.human_flip_prob}")
    print(f"Annotation costs: Human={args.human_cost}, LLM={args.llm_cost}")
    print(f"Dynamic K-centers active subset: {args.active_set_size}, Validation set: {args.validation_set_size}")

    if args.use_embedding:
        ModelClass = ImputerEmbedding
    else:
        ModelClass = Imputer
    
    if dataset == "hanna":
        model = ModelClass(
            question_num=7, max_choices=5, encoder_layers_num=6,
            attention_heads=4, hidden_dim=64, num_annotator=18, 
            annotator_embedding_dim=19, dropout=0.1
        ).to(device)
    elif dataset == "llm_rubric":
        model = ModelClass(
            question_num=9, max_choices=4, encoder_layers_num=6,
            attention_heads=4, hidden_dim=64, num_annotator=24, 
            annotator_embedding_dim=24, dropout=0.1
        ).to(device)
    
    experiment_results = {}
    
    experiments_to_run = []
    if args.experiment == "all":
        experiments_to_run = ["random_random", "gradient_voi"]
    else:
        experiments_to_run = [args.experiment]
    
    for experiment in experiments_to_run:
        print(f"\n{'='*80}")
        print(f"RUNNING MULTI-LEVEL NOISY {experiment.upper()} EXPERIMENT")
        print(f"{'='*80}")
        
        if args.runner == "prabhav":
            data_manager = NoisyDataManager(base_path + '/data/')
        else:
            data_manager = NoisyDataManager(base_path + f'/data_{dataset}/')

        if dataset == "hanna":
            data_manager.prepare_data(num_partition=1200, initial_train_ratio=0.0, dataset=dataset, 
                        cold_start=args.cold_start, llm_alpha_multiplier=args.llm_alpha_multiplier, 
                        human_flip_prob=args.human_flip_prob, use_embedding=args.use_embedding, 
                        validation_set_size=args.validation_set_size, active_set_size=args.active_set_size)
        elif dataset == "llm_rubric":
            data_manager.prepare_data(num_partition=225, initial_train_ratio=0.1, dataset=dataset, 
                                    cold_start=args.cold_start, llm_alpha_multiplier=args.llm_alpha_multiplier, 
                                    human_flip_prob=args.human_flip_prob, use_embedding=args.use_embedding,
                                    validation_set_size=args.validation_set_size, active_set_size=args.active_set_size)
        
        model_copy = copy.deepcopy(model)

        train_dataset = MultiLevelNoisyAnnotationDataset(data_manager.paths['train'])
        val_dataset = MultiLevelNoisyAnnotationDataset(data_manager.paths['validation'])
        test_dataset = MultiLevelNoisyAnnotationDataset(data_manager.paths['test'])
        active_pool_dataset = MultiLevelNoisyAnnotationDataset(data_manager.paths['active_pool'])
        
        print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}, Active={len(active_pool_dataset)}")
        
        initial_train_dataset = train_dataset if len(train_dataset) > 0 else None
        if initial_train_dataset:
            print('Will perform initial training on clean annotations!')

        if experiment == "gradient_voi":
            results = run_experiment_with_multilevel_noise(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="gradient", feature_strategy="voi_argmax", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                loss_type=args.loss_type, run_until_exhausted=args.run_until_exhausted,
                human_cost=args.human_cost, llm_cost=args.llm_cost, validation_set_size=args.validation_set_size,
                active_set_size=args.active_set_size, initial_train_dataset=initial_train_dataset
            )
        
        elif experiment == "random_random":
            results = run_experiment_with_multilevel_noise(
                active_pool_dataset, val_dataset, test_dataset,
                example_strategy="random", feature_strategy="random", model=model_copy,
                observe_all_features=False, features_per_example=args.features_per_example,
                cycles=args.cycles, examples_per_cycle=args.examples_per_cycle,
                epochs_per_cycle=args.epochs_per_cycle, batch_size=args.batch_size, lr=args.lr,
                device=device, resample_validation=args.resample_validation,
                run_until_exhausted=args.run_until_exhausted,
                human_cost=args.human_cost, llm_cost=args.llm_cost, validation_set_size=args.validation_set_size,
                active_set_size=args.active_set_size, initial_train_dataset=initial_train_dataset
            )

        else:
            print(f"Unknown experiment: {experiment}, skipping")
            continue
        
        experiment_results[experiment] = results
        
        torch.save(model_copy.state_dict(), os.path.join(models_path, f"multilevel_noisy_{experiment}.pth"))
        
        file_name = f"multilevel_noisy_{experiment}"
        if args.use_embedding:
            file_name += "_with_embedding"
        
        with open(os.path.join(results_path, f"{file_name}.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {experiment.upper()} SUMMARY")
        print(f"{'='*60}")
        
        noise_categories = ['original_llm', 'llm_low', 'llm_medium', 'llm_heavy',
                           'original_human', 'human_noisy']
        
        total_final = 0
        for category in noise_categories:
            if f'cumulative_{category}_selections' in results:
                final_count = results[f'cumulative_{category}_selections'][-1] if results[f'cumulative_{category}_selections'] else 0
                total_final += final_count
        
        if total_final > 0:
            print(f"Final selection distribution:")
            for category in noise_categories:
                if f'cumulative_{category}_selections' in results:
                    final_count = results[f'cumulative_{category}_selections'][-1] if results[f'cumulative_{category}_selections'] else 0
                    percentage = (final_count / total_final) * 100
                    print(f"   {category:15s}: {final_count:4d} ({percentage:5.1f}%)")
    
    if experiment_results:
        combined_file_name = "combined_multilevel_noisy_results"
        if args.use_embedding:
            combined_file_name += "_with_embedding"
        
        with open(os.path.join(results_path, f"{combined_file_name}.json"), "w") as f:
            json.dump(experiment_results, f, indent=4)
            
        print(f"\nMulti-level noisy experiment results saved to {results_path}")


if __name__ == "__main__":
    main()