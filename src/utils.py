"""
Utility and Dataset Management for SummEval Active Learner framework.
Uses existing JSONL data with expert_annotations and turker_annotations.

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

# Initialize sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

random.seed(90)
torch.manual_seed(90)
np.random.seed(90)

class DataManager:
    """Manages SummEval data preparation using existing JSONL file."""
    
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.paths = config.get_data_paths()
        
        # SummEval configuration
        self.dimensions = ['coherence', 'consistency', 'fluency', 'relevance']
        self.dimension_indices = {dim: i for i, dim in enumerate(self.dimensions)}
        
        logger.info(f"SummEval DataManager initialized")
    
    def load_summeval_jsonl(self, jsonl_path):
        """Load SummEval data from JSONL file."""
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"SummEval JSONL not found at {jsonl_path}")
        
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                # Validate required fields
                if ('expert_annotations' in entry and 'turker_annotations' in entry 
                    and 'decoded' in entry and 'id' in entry):
                    data.append(entry)
        
        logger.info(f"Loaded {len(data)} entries from {jsonl_path}")
        return data
    
    def prepare_text_embeddings(self, summeval_data):
        """Generate embeddings for summary + dimension combinations."""
        embeddings_path = os.path.join(self.config.INPUT_DATA_DIR, "summeval_embeddings.json")
        
        if os.path.exists(embeddings_path):
            logger.info("Loading existing SummEval embeddings")
            with open(embeddings_path, 'r') as f:
                return json.load(f)
        
        logger.info("Generating SummEval embeddings")
        embeddings_data = {}
        
        # Dimension descriptions for context
        dim_descriptions = {
            'coherence': 'Coherence: The collective quality of all sentences. Well-structured and organized.',
            'consistency': 'Consistency: The factual alignment between summary and source.',
            'fluency': 'Fluency: The quality of individual sentences. No formatting or grammar errors.',
            'relevance': 'Relevance: The selection of important content from the source.'
        }
        
        for entry in tqdm(summeval_data, desc="Generating embeddings"):
            entry_id = entry['id']
            summary_text = entry['decoded']
            
            # Create embedding for each dimension
            dimension_embeddings = []
            for dim in self.dimensions:
                text_to_embed = f"{dim_descriptions[dim]} Summary: {summary_text}"
                embedding = model.encode([text_to_embed], show_progress_bar=False)[0].tolist()
                dimension_embeddings.append(embedding)
            
            embeddings_data[entry_id] = dimension_embeddings
        
        # Save embeddings
        os.makedirs(self.config.INPUT_DATA_DIR, exist_ok=True)
        with open(embeddings_path, 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        logger.info(f"Generated embeddings for {len(embeddings_data)} summaries")
        return embeddings_data
    
    def prepare_data(self, jsonl_path, num_partition=1600, known_human_questions_val=0, 
                    initial_train_ratio=0.0, dataset="summeval", cold_start=True, use_embedding=True):
        """Prepare SummEval data splits for active learning."""
        logger.info(f"Preparing SummEval data from {jsonl_path}")
        logger.info(f"Using all {num_partition} data points with cold start")
        
        # Check if data already exists
        if os.path.exists(self.paths['active_pool']):
            logger.info("SummEval data already exists, skipping preparation")
            return True
        
        # Load JSONL data
        summeval_data = self.load_summeval_jsonl(jsonl_path)
        
        # Use all 1600 data points
        if len(summeval_data) != num_partition:
            logger.warning(f"Expected {num_partition} entries, found {len(summeval_data)}")
        
        # Generate embeddings if using embeddings
        if use_embedding:
            embeddings_data = self.prepare_text_embeddings(summeval_data)
        else:
            embeddings_data = None
        
        # Shuffle data for splits
        random.seed(42)
        random.shuffle(summeval_data)
        
        # Create data splits: Test=20%, Active=60%, Validation=20%, Train=0% (cold start)
        test_size = int(num_partition * 0.2)  # 20%
        active_size = int(num_partition * 0.6)  # 60%
        validation_size = num_partition - test_size - active_size  # 20% (remaining)
        train_size = 0  # Cold start - empty initial train
        
        test_data = summeval_data[:test_size]
        active_data = summeval_data[test_size:test_size + active_size]
        val_data = summeval_data[test_size + active_size:test_size + active_size + validation_size]
        train_data = []  # Empty for cold start
        
        logger.info(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, "
                   f"Test: {len(test_data)}, Active: {len(active_data)}")
        
        # Convert to internal format
        train_entries = []
        val_entries = []
        test_entries = []
        active_entries = []
        
        logger.info("Creating annotation data for train split")
        self._prepare_entries(train_data, train_entries, 'train', embeddings_data, cold_start, use_embedding)
        
        logger.info("Creating annotation data for validation split")
        self._prepare_entries(val_data, val_entries, 'validation', embeddings_data, cold_start, use_embedding)
        
        logger.info("Creating annotation data for test split")
        self._prepare_entries(test_data, test_entries, 'test', embeddings_data, cold_start, use_embedding)
        
        logger.info("Creating annotation data for active pool split")
        self._prepare_entries(active_data, active_entries, 'active_pool', embeddings_data, cold_start, use_embedding)
        
        # Save data splits
        os.makedirs(os.path.dirname(self.paths['train']), exist_ok=True)
        
        splits = {
            'train': train_entries,
            'validation': val_entries, 
            'test': test_entries,
            'active_pool': active_entries
        }
        
        for split_name, entries in splits.items():
            with open(self.paths[split_name], 'w') as f:
                json.dump(entries, f, indent=2)
            with open(self.paths[f'original_{split_name}'], 'w') as f:
                json.dump(entries, f, indent=2)
            logger.info(f"Saved {split_name}: {len(entries)} entries")
        
        logger.info("SummEval data preparation completed")
        return True
    
    def _prepare_entries(self, summeval_data, output_list, split_type, embeddings_data, cold_start, use_embedding):
        """Convert SummEval JSONL format to internal format."""
        for entry in tqdm(summeval_data, desc=f"Processing {split_type}"):
            entry_id = entry['id']
            summary_text = entry['decoded']
            
            # Get annotations
            expert_annotations = entry['expert_annotations']
            turker_annotations = entry['turker_annotations']
            
            # Ensure we have 3 experts and 5 turkers
            if len(expert_annotations) != 3 or len(turker_annotations) != 5:
                logger.warning(f"Skipping {entry_id} - incomplete annotations: "
                             f"{len(expert_annotations)} experts, {len(turker_annotations)} turkers")
                continue
            
            internal_entry = {
                "known_questions": [],
                "input": [],
                "answers": [],
                "annotators": [],
                "questions": [],
                "orig_split": split_type,
                "observation_history": [],
                "text_embedding": [],
                "summary_id": entry_id,
                "summary_text": summary_text,
                "model": entry.get('model_id', 'unknown')
            }
            
            # Process 3 experts (annotator IDs 0-2)
            for expert_idx, expert_ann in enumerate(expert_annotations):
                for dim_idx, dimension in enumerate(self.dimensions):
                    score = expert_ann[dimension]
                    if use_embedding and embeddings_data:
                        embedding = embeddings_data[entry_id][dim_idx]
                    else:
                        embedding = [0.0] * 384  # Default embedding size
                    
                    self._add_position(internal_entry, expert_idx, dim_idx, score, 
                                     split_type, cold_start, embedding)
            
            # Process 5 turkers (annotator IDs 3-7)
            for turker_idx, turker_ann in enumerate(turker_annotations):
                annotator_id = turker_idx + 3  # Map to IDs 3-7
                for dim_idx, dimension in enumerate(self.dimensions):
                    score = turker_ann[dimension]
                    if use_embedding and embeddings_data:
                        embedding = embeddings_data[entry_id][dim_idx]
                    else:
                        embedding = [0.0] * 384  # Default embedding size
                    
                    self._add_position(internal_entry, annotator_id, dim_idx, score,
                                     split_type, cold_start, embedding)
            
            # Verify we have 32 positions (8 annotators × 4 dimensions)
            if len(internal_entry["input"]) == 32:
                output_list.append(internal_entry)
            else:
                logger.warning(f"Skipping {entry_id} - has {len(internal_entry['input'])} positions instead of 32")
    
    def _add_position(self, entry, annotator_id, dim_idx, score, split_type, cold_start, embedding):
        """Add a single position to the entry."""
        # Convert score to one-hot (1-5 -> 0-4)
        true_prob = [0.0] * 5
        if isinstance(score, (int, float)) and 1 <= score <= 5:
            score_idx = int(score) - 1
            true_prob[score_idx] = 1.0
        else:
            true_prob[2] = 1.0  # Default to middle score
            logger.warning(f"Invalid score {score}, using default")
        
        # All splits are masked for cold start
        if cold_start:
            mask_bit = 1
            combined_input = [mask_bit] + [0.0] * 5
            entry["known_questions"].append(0)
        else:
            # Non-cold start logic (not used in this setup)
            if split_type == 'train':
                mask_bit = 0
                combined_input = [mask_bit] + true_prob
                entry["known_questions"].append(1)
            else:
                mask_bit = 1
                combined_input = [mask_bit] + [0.0] * 5
                entry["known_questions"].append(0)
        
        entry["input"].append(combined_input)
        entry["answers"].append(true_prob)
        entry["annotators"].append(annotator_id)
        entry["questions"].append(dim_idx)
        entry["text_embedding"].append(embedding)


class AnnotationDataset(Dataset):
    """Dataset class for handling SummEval annotated data."""
    
    def __init__(self, data_path_or_list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(data_path_or_list, list):
            self.data = data_path_or_list
            logger.info(f"Created SummEval dataset from list with {len(self.data)} entries")
        else:
            with open(data_path_or_list, 'r') as f:
                self.data = json.load(f)
            logger.info(f"Loaded SummEval dataset from {data_path_or_list} with {len(self.data)} entries")
                
        for entry in self.data:
            if "observation_history" not in entry:
                entry["observation_history"] = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        known_questions = torch.tensor(item['known_questions'], dtype=torch.int64)
        inputs = torch.tensor(item['input'], dtype=torch.float32)
        answers = torch.tensor(item['answers'], dtype=torch.float32)
        annotators = torch.tensor(item['annotators'], dtype=torch.int64)
        questions = torch.tensor(item['questions'], dtype=torch.int64)
        
        if "text_embedding" in item and item["text_embedding"]:
            embeddings = torch.tensor(item['text_embedding'], dtype=torch.float32)
        else:
            # Default embeddings if not available
            embeddings = torch.zeros(len(item['input']), 384, dtype=torch.float32)
        
        return known_questions, inputs, answers, annotators, questions, embeddings
    
    def get_data_entry(self, idx):
        return self.data[idx]
    
    def get_masked_positions(self, idx):
        item = self.data[idx]
        masked_positions = []
        
        for i in range(len(item['input'])):
            if item['input'][i][0] == 1:  # mask_bit == 1
                masked_positions.append(i)
        
        logger.debug(f"Example {idx} has {len(masked_positions)} masked positions")
        return masked_positions
    
    def get_known_positions(self, idx):
        item = self.data[idx]
        known_positions = []
        
        for i in range(len(item['input'])):
            if item['input'][i][0] == 0:  # mask_bit == 0
                known_positions.append(i)
        
        logger.debug(f"Example {idx} has {len(known_positions)} known positions")
        return known_positions
    
    def get_expert_positions(self, idx):
        """Get positions of expert annotations (annotator IDs 0-2)."""
        item = self.data[idx]
        expert_positions = []
        
        for i in range(len(item['annotators'])):
            if item['annotators'][i] < 3:  # Expert annotators
                expert_positions.append(i)
        
        return expert_positions
    
    def get_crowdworker_positions(self, idx):
        """Get positions of crowdworker annotations (annotator IDs 3-7)."""
        item = self.data[idx]
        crowdworker_positions = []
        
        for i in range(len(item['annotators'])):
            if item['annotators'][i] >= 3:  # Crowdworker annotators
                crowdworker_positions.append(i)
        
        return crowdworker_positions
    
    def observe_position(self, idx, position):
        """Mark a position as observed and update the input tensor."""
        item = self.data[idx]
        
        if item['input'][position][0] == 0:
            logger.debug(f"Position {position} in example {idx} already observed")
            return False
        
        # Update input with true answer
        item['input'][position][0] = 0  # Unmask
        true_answer = item['answers'][position]
        for i in range(5):  # 5 choices for SummEval
            item['input'][position][i+1] = true_answer[i]
        
        item['known_questions'][position] = 1
        
        # Add to observation history
        item['observation_history'].append({
            'position': position,
            'timestamp': len(item['observation_history']),
            'annotator': item['annotators'][position],
            'question': item['questions'][position],
            'answer': item['answers'][position]
        })
        
        logger.debug(f"Observed position {position} in example {idx}, "
                    f"annotator {item['annotators'][position]}, dimension {item['questions'][position]}")
        return True
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2)
        logger.info(f"Saved SummEval dataset to {path}")
    
    def update_data_entry(self, idx, entry):
        self.data[idx] = entry
        logger.debug(f"Updated SummEval data entry {idx}")


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


def get_experiment_config(experiment_name):
    """Get experiment-specific configuration for SummEval evaluation."""
    
    config_map = {
        "random_5": {
            "feature_selection_strategy": "random", 
            "target_questions": [0, 1, 2, 3]  # All 4 dimensions
        },
        "gradient_voi_all_questions": {
            "feature_selection_strategy": "voi", 
            "target_questions": [0, 1, 2, 3]  # All 4 dimensions
        },
        "variable_gradient_comparison": {
            "feature_selection_strategy": "voi",
            "target_questions": [0, 1, 2, 3]  # All 4 dimensions
        }
    }
    
    return config_map.get(experiment_name, {
        "feature_selection_strategy": "voi",
        "target_questions": [0, 1, 2, 3]  # Default to all dimensions
    })


if __name__ == "__main__":
    from config import Config
    config = Config("local")
    data_manager = DataManager(config)
    data_manager.prepare_data("/export/fs06/psingh54/ActiveRubric-Internal/src/input/fixed/model_annotations.aligned.jsonl", cold_start=True, use_embedding=True)