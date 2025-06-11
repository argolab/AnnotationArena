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
import pandas as pd
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Setting seeds for reproducibility
random.seed(90)
torch.manual_seed(90)
np.random.seed(90)

class DataManager:
    """
    Manages data preparation and handling for annotation experiments.
    """
    
    def __init__(self, base_path):
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
    
    def prepare_data(self, num_partition=1200, known_human_questions_val=0, initial_train_ratio=0.0, dataset="hanna", cold_start=False, use_embedding=False):
        """
        Prepare data splits for active learning experiments.
        
        Args:
            num_partition: Total number of examples to use
            known_human_questions_val: Number of human questions to keep observed in validation set
            initial_train_ratio: Ratio of data to use for initial training (cold start = 0.0)
            dataset: Dataset to use ("hanna" or "llm_rubric")
            cold_start: If True, both LLM and human questions will be unknown in active pool
            
        Returns:
            bool: Success status
        """
        print(f"Use embedding: {use_embedding}")
        '''if os.path.exists(os.path.join(self.base_path, "initial_train.json")):
            print("Data was created before!")
            return'''
        if use_embedding and not dataset == "hanna":
            raise ValueError("Not yet support other datasets with text embedding")
        if dataset == "gaussian":
            pass
        try:
            with open(os.path.join(self.base_path, "gpt-3.5-turbo-data-new.json"), "r") as f:
                llm_data = json.load(f)
            with open(os.path.join(self.base_path, "human-data-new.json"), "r") as f:
                human_data = json.load(f)
        except FileNotFoundError:
            return False

        '''if use_embedding and not os.path.exists(os.path.join(self.base_path, "text_embeddings.json")):
            print("Preparing all text embeddings with sentence bert")
            self.prepare_text_embeddings(num_partition)
            print("Done\n")'''
        
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

        print('-- Creating Annotation for Train --')
        self._prepare_entries(initial_train_texts, initial_train_data, 'train', llm_data, human_data, question_list, question_indices, known_human_questions_val, dataset=dataset, cold_start=cold_start, use_embedding=use_embedding)
        print('-- Creating Annotation for Validation --')
        self._prepare_entries(validation_texts, validation_data, 'validation', llm_data, human_data, question_list, question_indices, known_human_questions_val, dataset=dataset, cold_start=cold_start, use_embedding=use_embedding)
        print('-- Creating Annotation for Test --')
        self._prepare_entries(test_texts, test_data, 'test', llm_data, human_data, question_list, question_indices, known_human_questions_val, dataset=dataset, cold_start=cold_start, use_embedding=use_embedding)
        print('-- Creating Annotation for Active Pool --')
        self._prepare_entries(active_pool_texts, active_pool_data, 'active_pool', llm_data, human_data, question_list, question_indices, known_human_questions_val, dataset=dataset, cold_start=cold_start, use_embedding=use_embedding)
        
        print('Saving Data')
        for key, data in tqdm(zip(['train', 'validation', 'test', 'active_pool', 'original_train', 'original_validation', 'original_test', 'original_active_pool'],
                             [initial_train_data, validation_data, test_data, active_pool_data, initial_train_data, validation_data, test_data, active_pool_data])):
            with open(self.paths[key], "w") as f:
                print(self.paths[key])
                json.dump(data, f)

        print('ALL DATA CREATED!')
        
        return True
    
    def prepare_text_embeddings(self, num_partition):

        df = pd.read_csv(os.path.join(self.base_path, "hanna_stories_annotations_updated.csv"))
        texts = df['TEXT'].head(num_partition)

        # Define a function to extract prompt and story
        def split_text(entry):
            prompt = ""
            story = ""
            if isinstance(entry, str):
                if "Prompt:" in entry and "Story:" in entry:
                    parts = entry.split("Story:", 1)
                    prompt = parts[0].replace("Prompt:", "").strip()
                    story = parts[1].strip()
            return pd.Series([prompt, story])

        # Apply to all 1200 entries
        df_split = texts.apply(split_text)
        df_split.columns = ['Prompt', 'Story']

        # Save as JSON
        data_list = df_split.to_dict(orient='records')

        with open(os.path.join(self.base_path, "prompts_and_stories.json"), 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)

        all_embeddings = []
        for entry in tqdm(data_list):
            all_embeddings.append((model.encode([entry["Story"]])[0, :] + model.encode([entry["Prompt"]])[0, :]).tolist())
        with open(os.path.join(self.base_path, "text_embeddings.json"), 'w') as f:
            json.dump(all_embeddings, f, indent=2)

    
    def _prepare_entries(self, texts, data_list, split_type, llm_data, human_data, question_list, question_indices, known_human_questions_val, dataset, cold_start=False, use_embedding=False):
        """
        Prepare data entries for a specific split.
        
        Args:
            texts: List of text IDs to process
            data_list: List to append prepared entries to
            split_type: Type of data split ('train', 'validation', 'test', 'active_pool')
            llm_data: Dictionary of LLM-generated answers
            human_data: Dictionary of human-provided answers
            question_list: List of question IDs
            question_indices: Dictionary mapping question IDs to indices
            known_human_questions_val: Number of human questions to keep observed in validation set
            dataset: Dataset to use ("hanna" or "llm_rubric")
            cold_start: If True, both LLM and human questions will be unknown in active pool
        """
        if dataset == "hanna":
            
            if use_embedding:
                with open(os.path.join(self.base_path, "questions.json"), "r") as file:
                    question_data = json.load(file)

                with open(os.path.join(self.base_path, "prompts_and_stories.json"), "r", encoding="utf-8") as file:
                    text_data = json.load(file)

            for text_id in tqdm(texts):
                if text_id not in llm_data:
                    continue
                
                entry = {
                    "known_questions": [], 
                    "input": [], 
                    "answers": [], 
                    "annotators": [],
                    "questions": [],
                    "orig_split": split_type,
                    "observation_history": [],  # Track history of observations for this entry
                    "text_embedding": []
                }
                
                annotators = list(human_data[text_id].keys())
                
                # Process LLM questions
                for q_idx, question in enumerate(question_list):
                    true_prob = llm_data[text_id][question]
                    
                    # Determine if this LLM question should be masked based on cold_start and split_type
                    if cold_start and split_type == 'active_pool':
                        # In cold start mode, mask LLM questions in active pool
                        mask_bit = 1  # Masked
                        combined_input = [mask_bit] + [0.0] * 5
                        entry["known_questions"].append(0)
                    else:
                        # Default behavior: LLM questions are known
                        mask_bit = 0  # Observed
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

                        if use_embedding:
                            sentence = text_data[int(text_id)]["Prompt"] + text_data[int(text_id)]["Story"] + question_data[question]
                            embedding = model.encode([sentence])[0, :]
                            entry["text_embedding"].append(embedding.tolist())
                
                data_list.append(entry)

        elif dataset == "llm_rubric":

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
                        "observation_history": []  # Track history of observations for this entry
                    }
                
                    # Process LLM questions
                    for q_idx, question in enumerate(question_list):
                        true_prob = llm_data[text_id][question]
                        
                        # Determine if this LLM question should be masked based on cold_start and split_type
                        if cold_start and split_type in ['active_pool', 'validation']:
                            # In cold start mode, mask LLM questions in active pool
                            mask_bit = 1  # Masked
                            combined_input = [mask_bit] + [0.0] * 4  # Note: llm_rubric uses 4 values
                            entry["known_questions"].append(0)
                        else:
                            # Default behavior: LLM questions are known
                            mask_bit = 0  # Observed
                            combined_input = [mask_bit] + true_prob
                            entry["known_questions"].append(1)
                        
                        entry["input"].append(combined_input)
                        entry["answers"].append(true_prob)
                        entry["annotators"].append(-1)
                        entry["questions"].append(question_indices[question])

                    # Process human questions
                    for q_idx, question in enumerate(question_list):
                        true_score = human_data[text_id][annotator][question]
                        true_prob = [0.0] * 4  # Note: llm_rubric uses 4 values
                        
                        if isinstance(true_score, (int, float)):
                            if true_score % 1 != 0:  
                                rounded_score = math.ceil(true_score)
                                rounded_score = max(min(rounded_score, 4), 1)  # Adjusted for 4 values
                                index = rounded_score - 1
                                true_prob[index] = 1.0
                            else:
                                true_score = max(min(int(true_score), 4), 1)  # Adjusted for 4 values
                                index = true_score - 1
                                true_prob[index] = 1.0
                        else:
                            raise ValueError(f"Unexpected score type: {true_score}")
                        
                        if split_type == 'active_pool':
                            mask_bit = 1  # Masked
                            combined_input = [mask_bit] + [0.0] * 4
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
                                combined_input = [mask_bit] + [0.0] * 4
                                entry["known_questions"].append(0)
                            entry["input"].append(combined_input)

                        elif split_type == 'test':
                            if random.random() < 0.5:
                                mask_bit = 1  # Masked
                                combined_input = [mask_bit] + [0.0] * 4
                                entry["known_questions"].append(0)
                            else:
                                mask_bit = 0  # Observed
                                combined_input = [mask_bit] + true_prob
                                entry["known_questions"].append(1)
                            entry["input"].append(combined_input)
                            
                        entry["answers"].append(true_prob)   
                        entry["annotators"].append(int(annotator))
                        entry["questions"].append(question_indices[question])
                    
                    data_list.append(entry)

            return

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
    
    # def observe_position(self, idx, position):
    #     """
    #     Mark a position as observed and update the input tensor.
        
    #     Args:
    #         idx: Example index
    #         position: Position to observe
            
    #     Returns:
    #         bool: Success status
    #     """
    #     item = self.data[idx]
        
    #     # Check if already observed
    #     if item['input'][position][0] == 0:
    #         return False
    #     num_class = len(item["answers"][position])
    #     # Update input tensor
    #     item['input'][position][0] = 0  # Mark as observed
    #     for i in range(num_class):  # Assuming 5 classes
    #         try:
    #             item['input'][position][i+1] = item['answers'][position][i]
    #         except IndexError:
    #             continue
        
    #     # Update known_questions
    #     item['known_questions'][position] = 1
        
    #     # Add to observation history
    #     item['observation_history'].append({
    #         'position': position,
    #         'timestamp': len(item['observation_history']),
    #         'annotator': item['annotators'][position],
    #         'question': item['questions'][position],
    #         'answer': item['answers'][position]
    #     })
        
    #     return True

    def observe_position(self, idx, position):

        item = self.data[idx]
        

        if item['input'][position][0] == 0:
            return False
        
        num_class = len(item["answers"][position])
        item['input'][position][0] = 0  # Mark as observed
        
        if 'true_answers' in item and item['true_answers']:
            training_target = item['true_answers'][position]
        else:
            training_target = item['answers'][position]
        
        for i in range(num_class):
            try:
                item['input'][position][i+1] = training_target[i]
            except IndexError:
                continue
        
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
                               strategy="balanced", update_percentage=25, selected_examples=None, 
                               validation_set_size=50, current_val_indices=None):
    
    current_val_size = len(dataset_val)
    validation_example_indices = []
    
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
        
        print(f"Added {examples_added} selected examples to validation set (now {len(new_dataset_val)} examples)")
        return new_dataset_val, active_pool, validation_example_indices

    elif strategy == "add_selected_partial" and selected_examples:
        new_val_data = []
        examples_added = 0
        
        added_example = random.sample(annotated_examples, min(len(dataset_val), len(annotated_examples) // 2))
        remaining_size = len(dataset_val) - len(added_example)
        remaining_pool = []
        combined_pool = list(range(len(dataset_train)))
        for idx in combined_pool:
            if idx not in added_example:
                remaining_pool.append(idx)
        extra_example = random.sample(remaining_pool, remaining_size)
        validation_pool = added_example + extra_example
        for idx in validation_pool:
            new_val_data.append(dataset_train.get_data_entry(idx))
            validation_example_indices.append(idx)
            examples_added += 1

        new_dataset_val = AnnotationDataset(new_val_data)  
        new_active_pool = []
        for i in combined_pool:
            if i not in validation_example_indices:
                new_active_pool.append(i)
        
        print(f"Added {examples_added} selected examples to validation set (now {len(new_dataset_val)} examples)")
        return new_dataset_val, new_active_pool, validation_example_indices
    
    elif strategy == "fixed_size_resample":
        combined_pool = list(range(len(dataset_train)))
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
        
        print(f"Balanced fixed size resampled validation set: {len(new_dataset_val)} examples")
        print(f"  Selected examples: {len(selected_sample)}, Unselected examples: {len(unselected_sample)}")
        print(f"Updated active pool size: {len(updated_active_pool)}")
        
        return new_dataset_val, updated_active_pool, validation_example_indices
    
    return dataset_val, active_pool, validation_example_indices

if __name__ == "__main__":
    data_manager = DataManager("../outputs/data_hanna")
    # data_manager = DataManager("/export/fs06/psingh54/ActiveRubric-Internal/outputs/data")
    data_manager.prepare_data(1200, cold_start=True, use_embedding=True)
