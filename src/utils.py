"""
Utility and Dataset Creation + Management for Active Learner framework.

Author: Prabhav Singh / Haojun Shi
"""

import os
import json
import random
import math
import logging
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm.auto import tqdm
import copy
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

random.seed(90)
torch.manual_seed(90)
np.random.seed(90)

class DataManager:
    """Manages data preparation and handling for annotation experiments."""
    
    def __init__(self, config):
        """Initialize data manager with config."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.paths = config.get_data_paths()
        self.fixed_paths = config.get_fixed_paths()
        logger.info(f"DataManager initialized with config paths")
        logger.debug(f"Data paths: {self.paths}")
        logger.debug(f"Fixed paths: {self.fixed_paths}")
    
    def prepare_data(self, num_partition=1200, known_human_questions_val=0, initial_train_ratio=0.0, dataset="hanna", cold_start=False, use_embedding=False):
        """Prepare data splits for active learning experiments."""
        logger.info(f"Preparing data: num_partition={num_partition}, dataset={dataset}, cold_start={cold_start}, use_embedding={use_embedding}")
        print(f"Use embedding: {use_embedding}")
        print(self.config.INPUT_DATA_DIR)
        
        if os.path.exists(self.paths['active_pool']):
            logger.info("Data already exists, skipping preparation")
            return

        if use_embedding and not dataset == "hanna":
            raise ValueError("Not yet support other datasets with text embedding")
        if dataset == "gaussian":
            pass
        try:
            logger.info(f"Loading LLM data from {self.fixed_paths['gpt_data']}")
            with open(self.fixed_paths['gpt_data'], "r") as f:
                llm_data = json.load(f)
            logger.info(f"Loading human data from {self.fixed_paths['human_data']}")
            with open(self.fixed_paths['human_data'], "r") as f:
                human_data = json.load(f)
            logger.info(f"Loaded LLM data: {len(llm_data)} entries, Human data: {len(human_data)} entries")
        except FileNotFoundError as e:
            logger.error(f"Data files not found: {e}")
            return False

        if use_embedding and not os.path.exists(os.path.join(self.config.INPUT_DATA_DIR, "text_embeddings.json")):
            logger.info("Preparing text embeddings with sentence bert")
            print("Preparing all text embeddings with sentence bert")
            self.prepare_text_embeddings(num_partition)
            print("Done\n")
            logger.info("Text embeddings preparation completed")
        
        text_ids = list(human_data.keys())
        if dataset == "hanna":
            question_list = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
        elif dataset == "llm_rubric":
            question_list = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']
        question_indices = {q: i for i, q in enumerate(question_list)}
        logger.info(f"Question list: {question_list}")
        
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
        
        logger.info(f"Data splits - Train: {len(initial_train_texts)}, Val: {len(validation_texts)}, Test: {len(test_texts)}, Active: {len(active_pool_texts)}")
        
        initial_train_data = []
        validation_data = []
        test_data = []
        active_pool_data = []

        logger.info("Creating annotation data for train split")
        print('-- Creating Annotation for Train --')
        self._prepare_entries(initial_train_texts, initial_train_data, 'train', llm_data, human_data, question_list, question_indices, known_human_questions_val, dataset=dataset, cold_start=cold_start, use_embedding=use_embedding)
        logger.info("Creating annotation data for validation split")
        print('-- Creating Annotation for Validation --')
        self._prepare_entries(validation_texts, validation_data, 'validation', llm_data, human_data, question_list, question_indices, known_human_questions_val, dataset=dataset, cold_start=cold_start, use_embedding=use_embedding)
        logger.info("Creating annotation data for test split")
        print('-- Creating Annotation for Test --')
        self._prepare_entries(test_texts, test_data, 'test', llm_data, human_data, question_list, question_indices, known_human_questions_val, dataset=dataset, cold_start=cold_start, use_embedding=use_embedding)
        logger.info("Creating annotation data for active pool split")
        print('-- Creating Annotation for Active Pool --')
        self._prepare_entries(active_pool_texts, active_pool_data, 'active_pool', llm_data, human_data, question_list, question_indices, known_human_questions_val, dataset=dataset, cold_start=cold_start, use_embedding=use_embedding)
        
        logger.info("Saving all data splits")
        print('Saving Data')
        for key, data in tqdm(zip(['train', 'validation', 'test', 'active_pool', 'original_train', 'original_validation', 'original_test', 'original_active_pool'],
                             [initial_train_data, validation_data, test_data, active_pool_data, initial_train_data, validation_data, test_data, active_pool_data])):
            with open(self.paths[key], "w") as f:
                print(self.paths[key])
                json.dump(data, f)
            logger.debug(f"Saved {key} with {len(data)} entries to {self.paths[key]}")

        logger.info("Data preparation completed successfully")
        print('ALL DATA CREATED!')
        return True
    
    def prepare_text_embeddings(self, num_partition):
        """Prepare text embeddings for HANNA dataset."""
        logger.info(f"Loading HANNA stories from {self.fixed_paths['hanna_stories']}")
        df = pd.read_csv(self.fixed_paths['hanna_stories'])
        texts = df['TEXT'].head(num_partition)
        logger.info(f"Processing {len(texts)} text entries for embeddings")

        def split_text(entry):
            prompt = ""
            story = ""
            if isinstance(entry, str):
                if "Prompt:" in entry and "Story:" in entry:
                    parts = entry.split("Story:", 1)
                    prompt = parts[0].replace("Prompt:", "").strip()
                    story = parts[1].strip()
            return pd.Series([prompt, story])

        df_split = texts.apply(split_text)
        df_split.columns = ['Prompt', 'Story']
        data_list = df_split.to_dict(orient='records')

        prompts_stories_path = os.path.join(self.config.INPUT_DATA_DIR, "prompts_and_stories.json")
        with open(prompts_stories_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved prompts and stories to {prompts_stories_path}")

        logger.info("Generating embeddings using SentenceTransformer")
        all_embeddings = []
        for entry in tqdm(data_list):
            all_embeddings.append((model.encode([entry["Story"]])[0, :] + model.encode([entry["Prompt"]])[0, :]).tolist())
        
        embeddings_path = os.path.join(self.config.INPUT_DATA_DIR, "text_embeddings.json")
        with open(embeddings_path, 'w') as f:
            json.dump(all_embeddings, f, indent=2)
        logger.info(f"Saved {len(all_embeddings)} embeddings to {embeddings_path}")

    def _prepare_entries(self, texts, data_list, split_type, llm_data, human_data, question_list, question_indices, known_human_questions_val, dataset, cold_start=False, use_embedding=False):
        """Prepare data entries for a specific split."""
        logger.info(f"Preparing {len(texts)} entries for {split_type} split, dataset={dataset}")
        
        if dataset == "hanna":
            
            if use_embedding:
                logger.debug("Loading question data and text data for embeddings")
                with open(self.fixed_paths['questions'], "r") as file:
                    question_data = json.load(file)
                with open(os.path.join(self.config.INPUT_DATA_DIR, "prompts_and_stories.json"), "r", encoding="utf-8") as file:
                    text_data = json.load(file)

            for text_id in tqdm(texts):
                if text_id not in llm_data:
                    logger.debug(f"Skipping text_id {text_id} - not found in LLM data")
                    continue
                
                entry = {
                    "known_questions": [], 
                    "input": [], 
                    "answers": [], 
                    "annotators": [],
                    "questions": [],
                    "orig_split": split_type,
                    "observation_history": [],
                    "text_embedding": []
                }
                
                annotators = list(human_data[text_id].keys())
                logger.debug(f"Processing text_id {text_id} with {len(annotators)} annotators")
                
                # Process LLM questions
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

                    if use_embedding:
                        sentence = text_data[int(text_id)]["Prompt"] + text_data[int(text_id)]["Story"] + question_data[question]
                        embedding = model.encode([sentence])[0, :]
                        entry["text_embedding"].append(embedding.tolist())

                # Process human questions
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
                        else:
                            raise ValueError(f"Unexpected score type: {true_score}")
                        
                        if split_type == 'active_pool':
                            mask_bit = 1
                            combined_input = [mask_bit] + [0.0] * 5
                            entry["known_questions"].append(0)
                            entry["input"].append(combined_input)

                        elif split_type == 'train':
                            mask_bit = 0
                            combined_input = [mask_bit] + true_prob
                            entry["known_questions"].append(1)
                            entry["input"].append(combined_input)

                        elif split_type == 'validation':
                            if q_idx < known_human_questions_val:
                                mask_bit = 0
                                combined_input = [mask_bit] + true_prob
                                entry["known_questions"].append(1)
                            else:
                                mask_bit = 1
                                combined_input = [mask_bit] + [0.0] * 5
                                entry["known_questions"].append(0)
                            entry["input"].append(combined_input)
                            
                        elif split_type == 'test':
                            if random.random() < 0.5:
                                mask_bit = 1
                                combined_input = [mask_bit] + [0.0] * 5
                                entry["known_questions"].append(0)
                            else:
                                mask_bit = 0
                                combined_input = [mask_bit] + true_prob
                                entry["known_questions"].append(1)
                            entry["input"].append(combined_input)
                            
                        entry["answers"].append(true_prob)   
                        entry["annotators"].append(int(judge_id))
                        entry["questions"].append(question_indices[question])

                        if use_embedding:
                            sentence = text_data[int(text_id)]["Prompt"] + text_data[int(text_id)]["Story"] + question_data[question]
                            embedding = model.encode([sentence])[0, :]
                            entry["text_embedding"].append(embedding.tolist())

                data_list.append(entry)
                logger.debug(f"Created entry for text_id {text_id} with {len(entry['input'])} annotations")

        elif dataset == "llm_rubric":
            logger.info("Processing LLM Rubric dataset entries")

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
                        "observation_history": []
                    }
                
                    # Process LLM questions
                    for q_idx, question in enumerate(question_list):
                        true_prob = llm_data[text_id][question]
                        
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

                    # Process human questions
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
                        else:
                            raise ValueError(f"Unexpected score type: {true_score}")
                        
                        if split_type == 'active_pool':
                            mask_bit = 1
                            combined_input = [mask_bit] + [0.0] * 4
                            entry["known_questions"].append(0)
                            entry["input"].append(combined_input)

                        elif split_type == 'train':
                            mask_bit = 0
                            combined_input = [mask_bit] + true_prob
                            entry["known_questions"].append(1)
                            entry["input"].append(combined_input)

                        elif split_type == 'validation':
                            if q_idx < known_human_questions_val:
                                mask_bit = 0
                                combined_input = [mask_bit] + true_prob
                                entry["known_questions"].append(1)
                            else:
                                mask_bit = 1
                                combined_input = [mask_bit] + [0.0] * 4
                                entry["known_questions"].append(0)
                            entry["input"].append(combined_input)

                        elif split_type == 'test':
                            if random.random() < 0.5:
                                mask_bit = 1
                                combined_input = [mask_bit] + [0.0] * 4
                                entry["known_questions"].append(0)
                            else:
                                mask_bit = 0
                                combined_input = [mask_bit] + true_prob
                                entry["known_questions"].append(1)
                            entry["input"].append(combined_input)
                            
                        entry["answers"].append(true_prob)   
                        entry["annotators"].append(int(annotator))
                        entry["questions"].append(question_indices[question])
                    
                    data_list.append(entry)
                    logger.debug(f"Created LLM rubric entry for text_id {text_id}, annotator {annotator}")

        logger.info(f"Completed preparing {len(data_list)} entries for {split_type} split")

class AnnotationDataset(Dataset):
    """Dataset class for handling annotated data."""
    
    def __init__(self, data_path_or_list):
        """Initialize dataset from path or list."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(data_path_or_list, list):
            self.data = data_path_or_list
            logger.info(f"Created dataset from list with {len(self.data)} entries")
        else:
            with open(data_path_or_list, 'r') as f:
                self.data = json.load(f)
            logger.info(f"Loaded dataset from {data_path_or_list} with {len(self.data)} entries")
                
        for entry in self.data:
            if "observation_history" not in entry:
                entry["observation_history"] = []
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a dataset item by index."""
        item = self.data[idx]
        known_questions = torch.tensor(item['known_questions'], dtype=torch.int64)
        inputs = torch.tensor(item['input'], dtype=torch.float32)
        answers = torch.tensor(item['answers'], dtype=torch.float32)
        annotators = torch.tensor(item['annotators'], dtype=torch.int64)
        questions = torch.tensor(item['questions'], dtype=torch.int64)
        if "text_embedding" in item.keys():
            embedding = torch.tensor(item['text_embedding'], dtype=torch.float32)
            return known_questions, inputs, answers, annotators, questions, embedding
        
        return known_questions, inputs, answers, annotators, questions, None
    
    def get_data_entry(self, idx):
        """Get the raw data entry for an index."""
        return self.data[idx]
    
    def get_masked_positions(self, idx):
        """Get positions of masked annotations."""
        item = self.data[idx]
        masked_positions = []
        
        for i in range(len(item['input'])):
            if item['input'][i][0] == 1:
                masked_positions.append(i)
        
        logger.debug(f"Example {idx} has {len(masked_positions)} masked positions")
        return masked_positions
    
    def get_known_positions(self, idx):
        """Get positions of known annotations."""
        item = self.data[idx]
        known_positions = []
        
        for i in range(len(item['input'])):
            if item['input'][i][0] == 0:
                known_positions.append(i)
        
        logger.debug(f"Example {idx} has {len(known_positions)} known positions")
        return known_positions
    
    def get_human_positions(self, idx):
        """Get positions of all human annotations."""
        item = self.data[idx]
        human_positions = []
        
        for i in range(len(item['annotators'])):
            if item['annotators'][i] >= 0:
                human_positions.append(i)
        
        return human_positions
    
    def get_llm_positions(self, idx):
        """Get positions of all LLM annotations."""
        item = self.data[idx]
        llm_positions = []
        
        for i in range(len(item['annotators'])):
            if item['annotators'][i] == -1:
                llm_positions.append(i)
        
        return llm_positions

    def observe_position(self, idx, position):
        """Mark a position as observed and update the input tensor."""
        item = self.data[idx]
        
        if item['input'][position][0] == 0:
            logger.debug(f"Position {position} in example {idx} already observed")
            return False
        
        num_class = len(item["answers"][position])
        item['input'][position][0] = 0
        
        if 'true_answers' in item and item['true_answers']:
            training_target = item['true_answers'][position]
        else:
            training_target = item['answers'][position]
        
        for i in range(num_class):
            try:
                item['input'][position][i+1] = training_target[i]
            except IndexError:
                continue
        
        item['known_questions'][position] = 1
        
        item['observation_history'].append({
            'position': position,
            'timestamp': len(item['observation_history']),
            'annotator': item['annotators'][position],
            'question': item['questions'][position],
            'answer': item['answers'][position] 
        })
        
        logger.debug(f"Observed position {position} in example {idx}, annotator {item['annotators'][position]}, question {item['questions'][position]}")
        return True
    
    def save(self, path):
        """Save dataset to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.data, f)
        logger.info(f"Saved dataset to {path}")
    
    def update_data_entry(self, idx, entry):
        """Update a data entry with new values."""
        self.data[idx] = entry
        logger.debug(f"Updated data entry {idx}")

def compute_metrics(preds, true):
    """Compute evaluation metrics for predictions."""
    logger.debug(f"Computing metrics for {len(preds)} predictions")
    
    rmse = np.sqrt(np.mean((preds - true) ** 2))
    
    try:
        pearson_val, _ = pearsonr(preds, true)
    except:
        pearson_val = 0.0
        
    try:
        spearman_val, _ = spearmanr(preds, true)
    except:
        spearman_val = 0.0
        
    try:
        kendall_val, _ = kendalltau(preds, true)
    except:
        kendall_val = 0.0
    
    mae = np.mean(np.abs(preds - true))
    
    if preds.ndim > 1 and preds.shape[1] > 1:
        pred_class = np.argmax(preds, axis=1)
        true_class = np.argmax(true, axis=1)
        accuracy = np.mean(pred_class == true_class)
    else:
        accuracy = np.mean(np.abs(preds - true) <= 0.5)
    
    metrics = {
        "rmse": rmse,
        "mae": mae, 
        "pearson": pearson_val, 
        "spearman": spearman_val, 
        "kendall": kendall_val,
        "accuracy": accuracy
    }
    
    logger.debug(f"Computed metrics: RMSE={rmse:.4f}, Pearson={pearson_val:.4f}")
    return metrics

def minimum_bayes_risk_l2(distribution):
    """Compute the minimum Bayes risk decision for L2 loss."""
    if hasattr(distribution, 'mean'):
        return distribution.mean.item()
    
    if isinstance(distribution, torch.Tensor):
        values = torch.arange(1, 6, device=distribution.device)
        return torch.sum(distribution * values).item()
    
    values = np.arange(1, 6)
    return np.sum(distribution * values)

def minimum_bayes_risk_ce(distribution):
    """Compute the minimum Bayes risk decision for cross-entropy loss."""
    if isinstance(distribution, torch.Tensor):
        return torch.argmax(distribution).item()
    return np.argmax(distribution)

def resample_validation_dataset(dataset_train, dataset_val, active_pool, annotated_examples, 
                               strategy="balanced", update_percentage=25, selected_examples=None, 
                               validation_set_size=50, current_val_indices=None):
    """Resample validation dataset using various strategies."""
    current_val_size = len(dataset_val)
    validation_example_indices = []
    
    logger.info(f"Resampling validation dataset - Strategy: {strategy}, Current size: {current_val_size}")
    
    if strategy == "balanced":
        num_to_update = max(1, int(current_val_size * update_percentage / 100))
        new_val_indices = []
        
        if annotated_examples:
            num_from_annotated = min(len(annotated_examples), num_to_update // 2)
            if num_from_annotated > 0:
                annotated_sample = random.sample(annotated_examples, num_from_annotated)
                new_val_indices.extend(annotated_sample)
        
        remaining_needed = num_to_update - len(new_val_indices)
        if remaining_needed > 0 and active_pool:
            remaining_active = [idx for idx in active_pool if idx not in annotated_examples]
            num_from_pool = min(len(remaining_active), remaining_needed)
            if num_from_pool > 0:
                pool_sample = random.sample(remaining_active, num_from_pool)
                new_val_indices.extend(pool_sample)
        
        if new_val_indices:
            keep_size = current_val_size - len(new_val_indices)
            
            new_val_data = []
            kept_val_indices = []
            if keep_size > 0:
                for i in range(min(keep_size, current_val_size)):
                    new_val_data.append(dataset_val.get_data_entry(i))
                    kept_val_indices.append(validation_example_indices[i])
            
            for idx in new_val_indices:
                new_val_data.append(dataset_train.get_data_entry(idx))
            
            validation_example_indices = kept_val_indices + new_val_indices
            
            new_dataset_val = AnnotationDataset(new_val_data)
            updated_active_pool = [idx for idx in active_pool if idx not in new_val_indices]
            
            logger.info(f"Resampled validation set: {len(new_dataset_val)} examples ({len(new_val_indices)} new)")
            print(f"Resampled validation set: {len(new_dataset_val)} examples ({len(new_val_indices)} new)")
            return new_dataset_val, updated_active_pool, validation_example_indices
        
    elif strategy == "add_selected" and selected_examples:
        new_val_data = []
        
        for i in range(current_val_size):
            new_val_data.append(dataset_val.get_data_entry(i))
        
        examples_added = 0
        for idx in selected_examples:
            new_val_data.append(dataset_train.get_data_entry(idx))
            examples_added += 1
            if idx not in validation_example_indices:
                validation_example_indices.append(idx)
        
        new_dataset_val = AnnotationDataset(new_val_data)
        
        logger.info(f"Added {examples_added} selected examples to validation set")
        print(f"Added {examples_added} selected examples to validation set (now {len(new_dataset_val)} examples)")
        return new_dataset_val, active_pool, validation_example_indices

    elif strategy == "add_selected_partial" and selected_examples:
        new_val_data = []
        
        for i in range(current_val_size):
            new_val_data.append(dataset_val.get_data_entry(i))
        
        examples_added = 0
        for idx in selected_examples:
            if idx not in validation_example_indices and random.random() > 0.5:
                new_val_data.append(dataset_train.get_data_entry(idx))
                validation_example_indices.append(idx)
                examples_added += 1
        
        new_dataset_val = AnnotationDataset(new_val_data)
        
        logger.info(f"Added {examples_added} selected examples to validation set (partial)")
        print(f"Added {examples_added} selected examples to validation set (now {len(new_dataset_val)} examples)")
        return new_dataset_val, active_pool, validation_example_indices
    
    elif strategy == "fixed_size_resample":
        if current_val_indices is None:
            current_val_indices = list(range(len(dataset_val)))
        
        combined_pool = current_val_indices + active_pool
        
        if len(combined_pool) >= validation_set_size:
            new_val_indices = random.sample(combined_pool, validation_set_size)
        else:
            new_val_indices = combined_pool
        
        new_val_data = []
        for idx in new_val_indices:
            new_val_data.append(dataset_train.get_data_entry(idx))
        
        updated_active_pool = [idx for idx in combined_pool if idx not in new_val_indices]
        
        new_dataset_val = AnnotationDataset(new_val_data)
        validation_example_indices = new_val_indices
        
        logger.info(f"Fixed size resampled validation set: {len(new_dataset_val)} examples")
        print(f"Fixed size resampled validation set: {len(new_dataset_val)} examples")
        print(f"Updated active pool size: {len(updated_active_pool)}")
        
        return new_dataset_val, updated_active_pool, validation_example_indices

    elif strategy == "balanced_fixed_size":
        if not selected_examples:
            return dataset_val, active_pool, validation_example_indices
        
        half_size = validation_set_size // 2
        selected_count = min(half_size, len(selected_examples))
        unselected_count = validation_set_size - selected_count
        
        unselected_pool = [idx for idx in active_pool if idx not in selected_examples]
        unselected_count = min(unselected_count, len(unselected_pool))
        
        if selected_count > 0:
            selected_sample = random.sample(selected_examples, selected_count)
        else:
            selected_sample = []
            
        if unselected_count > 0:
            unselected_sample = random.sample(unselected_pool, unselected_count)
        else:
            unselected_sample = []
        
        new_val_indices = selected_sample + unselected_sample
        
        new_val_data = []
        for idx in new_val_indices:
            new_val_data.append(dataset_train.get_data_entry(idx))
        
        updated_active_pool = [idx for idx in active_pool if idx not in new_val_indices]
        
        new_dataset_val = AnnotationDataset(new_val_data)
        validation_example_indices = new_val_indices
        
        logger.info(f"Balanced fixed size resampled validation set: {len(new_dataset_val)} examples")
        logger.info(f"Selected examples: {len(selected_sample)}, Unselected examples: {len(unselected_sample)}")
        print(f"Balanced fixed size resampled validation set: {len(new_dataset_val)} examples")
        print(f"  Selected examples: {len(selected_sample)}, Unselected examples: {len(unselected_sample)}")
        print(f"Updated active pool size: {len(updated_active_pool)}")
        
        return new_dataset_val, updated_active_pool, validation_example_indices
    
    return dataset_val, active_pool, validation_example_indices

if __name__ == "__main__":
    from config import Config
    config = Config("prabhav")  # or "local"
    data_manager = DataManager(config)
    data_manager.prepare_data(1200, cold_start=True, use_embedding=True)