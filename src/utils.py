import os
import json
import random
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm.auto import tqdm
import copy

# Setting seeds for reproducibility
random.seed(90)
torch.manual_seed(90)
np.random.seed(90)

# Base paths
BASE_PATH = "/export/fs06/psingh54/ActiveRubric-Internal/outputs"
DATA_PATH = os.path.join(BASE_PATH, "data")
MODELS_PATH = os.path.join(BASE_PATH, "models")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

# Create directories
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

class DataManager:
    """
    Manages data preparation and handling for annotation experiments.
    """
    
    def __init__(self, base_path=DATA_PATH):
        """Initialize data manager with paths for data storage."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_path = base_path
        self.paths = {
            'train': os.path.join(base_path, "initial_train.json"),
            'validation': os.path.join(base_path, "validation.json"),
            'test': os.path.join(base_path, "test.json"),
            'active_pool': os.path.join(base_path, "active_pool.json"),
            'original_train': os.path.join(base_path, "original_initial_train.json"),
            'original_validation': os.path.join(base_path, "original_validation.json"),
            'original_test': os.path.join(base_path, "original_test.json"),
            'original_active_pool': os.path.join(base_path, "original_active_pool.json"),
            'gradient_alignment': os.path.join(base_path, "gradient_alignment"),
            'random': os.path.join(base_path, "random")
        }
        os.makedirs(self.paths['gradient_alignment'], exist_ok=True)
        os.makedirs(self.paths['random'], exist_ok=True)
    
    def prepare_data(self, num_partition=1200, known_human_questions_val=0, initial_train_ratio=0.0):
        """
        Prepare data splits for active learning experiments.
        
        Args:
            num_partition: Total number of examples to use
            known_human_questions_val: Number of human questions to keep observed in validation set
            initial_train_ratio: Ratio of data to use for initial training (cold start = 0.0)
            
        Returns:
            bool: Success status
        """
        try:
            with open(os.path.join(self.base_path, "gpt-3.5-turbo-data-new.json"), "r") as f:
                llm_data = json.load(f)
            with open(os.path.join(self.base_path, "human-data-new.json"), "r") as f:
                human_data = json.load(f)
        except FileNotFoundError:
            return False
        
        text_ids = list(human_data.keys())
        question_list = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
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
        
        self._prepare_entries(initial_train_texts, initial_train_data, 'train', llm_data, human_data, question_list, question_indices, known_human_questions_val)
        self._prepare_entries(validation_texts, validation_data, 'validation', llm_data, human_data, question_list, question_indices, known_human_questions_val)
        self._prepare_entries(test_texts, test_data, 'test', llm_data, human_data, question_list, question_indices, known_human_questions_val)
        self._prepare_entries(active_pool_texts, active_pool_data, 'active_pool', llm_data, human_data, question_list, question_indices, known_human_questions_val)
        
        for key, data in zip(['train', 'validation', 'test', 'active_pool', 'original_train', 'original_validation', 'original_test', 'original_active_pool'],
                             [initial_train_data, validation_data, test_data, active_pool_data, initial_train_data, validation_data, test_data, active_pool_data]):
            with open(self.paths[key], "w") as f:
                print(self.paths[key])
                json.dump(data, f)
        
        return True
    
    def _prepare_entries(self, texts, data_list, split_type, llm_data, human_data, question_list, question_indices, known_human_questions_val):
        """Prepare data entries for a specific split."""
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
                "observation_history": []  # Track history of observations for this entry
            }
            
            annotators = list(human_data[text_id].keys())
            
            for q_idx, question in enumerate(question_list):
                true_prob = llm_data[text_id][question]
                
                mask_bit = 0 
                combined_input = [mask_bit] + true_prob
                
                entry["known_questions"].append(1)
                entry["input"].append(combined_input)
                entry["answers"].append(true_prob)
                entry["annotators"].append(-1)
                entry["questions"].append(question_indices[question])

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
                        mask_bit = 1  # Masked
                        combined_input = [mask_bit] + [0.0] * 5
                        entry["known_questions"].append(0)
                        entry["input"].append(combined_input)
                        
                    elif split_type == 'train':
                        mask_bit = 0  # Observed
                        combined_input = [mask_bit] + true_prob
                        entry["known_questions"].append(1)
                        entry["input"].append(combined_input)

                    elif split_type == 'validation':
                        if q_idx < known_human_questions_val:
                            mask_bit = 0  # Observed
                            combined_input = [mask_bit] + true_prob
                            entry["known_questions"].append(1)
                        else:
                            mask_bit = 1  # Masked
                            combined_input = [mask_bit] + [0.0] * 5
                            entry["known_questions"].append(0)
                        entry["input"].append(combined_input)

                    elif split_type == 'test':
                        if random.random() < 0.5:
                            mask_bit = 1  # Masked
                            combined_input = [mask_bit] + [0.0] * 5
                            entry["known_questions"].append(0)
                        else:
                            mask_bit = 0  # Observed
                            combined_input = [mask_bit] + true_prob
                            entry["known_questions"].append(1)
                        entry["input"].append(combined_input)
                        
                    entry["answers"].append(true_prob)   
                    entry["annotators"].append(int(judge_id))
                    entry["questions"].append(question_indices[question])
            
            data_list.append(entry)


class AnnotationDataset(Dataset):
    """
    Dataset class for handling annotated data.
    """
    
    def __init__(self, data_path_or_list):
        """
        Initialize dataset from path or list.
        
        Args:
            data_path_or_list: Either a file path to a JSON file or a list of data entries
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(data_path_or_list, list):
            self.data = data_path_or_list
        else:
            with open(data_path_or_list, 'r') as f:
                self.data = json.load(f)
                
        # Add observation history if not present
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
        
        return known_questions, inputs, answers, annotators, questions
    
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
                
        return masked_positions
    
    def get_known_positions(self, idx):
        """Get positions of known annotations."""
        item = self.data[idx]
        known_positions = []
        
        for i in range(len(item['input'])):
            if item['input'][i][0] == 0:
                known_positions.append(i)
                
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
        """
        Mark a position as observed and update the input tensor.
        
        Args:
            idx: Example index
            position: Position to observe
            
        Returns:
            bool: Success status
        """
        item = self.data[idx]
        
        # Check if already observed
        if item['input'][position][0] == 0:
            return False
        
        # Update input tensor
        item['input'][position][0] = 0  # Mark as observed
        for i in range(5):  # Assuming 5 classes
            item['input'][position][i+1] = item['answers'][position][i]
        
        # Update known_questions
        item['known_questions'][position] = 1
        
        # Add to observation history
        item['observation_history'].append({
            'position': position,
            'timestamp': len(item['observation_history']),
            'annotator': item['annotators'][position],
            'question': item['questions'][position],
            'answer': item['answers'][position]
        })
        
        return True
    
    def save(self, path):
        """Save dataset to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.data, f)
    
    def update_data_entry(self, idx, entry):
        """Update a data entry with new values."""
        self.data[idx] = entry


def compute_metrics(preds, true):
    """
    Compute evaluation metrics for predictions.
    
    Args:
        preds: Numpy array of predictions
        true: Numpy array of ground truth
        
    Returns:
        dict: Dictionary of metrics
    """
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
    
    # Calculate additional error metrics
    mae = np.mean(np.abs(preds - true))
    
    # Calculate calibration error if we have probability distributions
    if preds.ndim > 1 and preds.shape[1] > 1:
        # This is a simplified approach for categorical predictions
        pred_class = np.argmax(preds, axis=1)
        true_class = np.argmax(true, axis=1)
        accuracy = np.mean(pred_class == true_class)
    else:
        # For scalar predictions, use a simple accuracy within 0.5
        accuracy = np.mean(np.abs(preds - true) <= 0.5)
    
    return {
        "rmse": rmse,
        "mae": mae, 
        "pearson": pearson_val, 
        "spearman": spearman_val, 
        "kendall": kendall_val,
        "accuracy": accuracy
    }


def minimum_bayes_risk_l2(distribution):
    """
    Compute the minimum Bayes risk decision for L2 loss.
    For L2 loss, this is the mean of the distribution.
    
    Args:
        distribution: PyTorch distribution or tensor of probabilities
        
    Returns:
        float: Minimum Bayes risk decision
    """
    if hasattr(distribution, 'mean'):
        return distribution.mean.item()
    
    # For categorical distribution represented as probabilities
    if isinstance(distribution, torch.Tensor):
        values = torch.arange(1, 6, device=distribution.device)
        return torch.sum(distribution * values).item()
    
    # Numpy fallback
    values = np.arange(1, 6)
    return np.sum(distribution * values)


def minimum_bayes_risk_ce(distribution):
    """
    Compute the minimum Bayes risk decision for cross-entropy loss.
    For cross-entropy, this is the most probable class.
    
    Args:
        distribution: Tensor or numpy array of probabilities
        
    Returns:
        int: Most probable class index
    """
    if isinstance(distribution, torch.Tensor):
        return torch.argmax(distribution).item()
    return np.argmax(distribution)

def resample_validation_dataset(dataset_train, dataset_val, active_pool, annotated_examples, 
                               strategy="balanced", update_percentage=25, selected_examples = None):
    """
    Resample the validation dataset using annotated examples and active pool.
    
    Args:
        dataset_train: Training dataset (source of data)
        dataset_val: Current validation dataset
        active_pool: List of indices available in the active pool
        annotated_examples: List of indices of examples that have been annotated
        strategy: Strategy for resampling ("balanced", "add_only", "replace_all")
        update_percentage: Percentage of validation set to update (default: 25%)
        
    Returns:
        tuple: (new_validation_dataset, updated_active_pool)
    """
    current_val_size = len(dataset_val)
    
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
            
            # Create new validation dataset
            new_val_data = []
            if keep_size > 0:
                for i in range(min(keep_size, current_val_size)):
                    new_val_data.append(dataset_val.get_data_entry(i))
            
            # Add new examples
            for idx in new_val_indices:
                new_val_data.append(dataset_train.get_data_entry(idx))
            
            # Create new dataset
            new_dataset_val = AnnotationDataset(new_val_data)
            updated_active_pool = [idx for idx in active_pool if idx not in new_val_indices]
            
            print(f"Resampled validation set: {len(new_dataset_val)} examples ({len(new_val_indices)} new)")
            return new_dataset_val, updated_active_pool
    
    elif strategy == "add_only":

        max_to_add = max(1, int(current_val_size * update_percentage / 100))
        num_to_add = min(len(annotated_examples), max_to_add)
        
        if num_to_add > 0:
            # Get examples to add
            examples_to_add = random.sample(annotated_examples, num_to_add)
            
            # Create new validation dataset
            new_val_data = []
            
            # Keep all existing validation examples
            for i in range(current_val_size):
                new_val_data.append(dataset_val.get_data_entry(i))
            
            # Add new examples
            for idx in examples_to_add:
                new_val_data.append(dataset_train.get_data_entry(idx))
            
            new_dataset_val = AnnotationDataset(new_val_data)
            
            print(f"Added {num_to_add} annotated examples to validation set (now {len(new_dataset_val)} examples)")
            return new_dataset_val, active_pool
    
    if strategy == "replace_all":
        val_size = min(current_val_size, len(active_pool))
        if val_size > 0:

            new_val_indices = random.sample(active_pool, val_size)
            
            current_val_indices = [i for i in range(len(dataset_val))]
            
            new_val_data = []
            for idx in new_val_indices:
                new_val_data.append(dataset_train.get_data_entry(idx))
            
            new_dataset_val = AnnotationDataset(new_val_data)
            
            updated_active_pool = [idx for idx in active_pool if idx not in new_val_indices]
            
            for old_val_idx in current_val_indices:
                if old_val_idx not in annotated_examples and old_val_idx not in updated_active_pool:
                    updated_active_pool.append(old_val_idx)
            
            print(f"Completely replaced validation set with {len(new_dataset_val)} new examples")
            print(f"Active pool size: {len(updated_active_pool)}")
            return new_dataset_val, updated_active_pool
        
    elif strategy == "add_selected" and selected_examples:

        new_val_data = []
        
        for i in range(current_val_size):
            new_val_data.append(dataset_val.get_data_entry(i))
        
        examples_added = 0
        for idx in selected_examples:
            new_val_data.append(dataset_train.get_data_entry(idx))
            examples_added += 1
        
        new_dataset_val = AnnotationDataset(new_val_data)
        
        print(f"Added {examples_added} selected examples to validation set (now {len(new_dataset_val)} examples)")
        return new_dataset_val, active_pool
    
    return dataset_val, active_pool