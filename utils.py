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


def plot_metrics_over_time(metrics_dict, save_path=None):
    """
    Plot metrics over time with confidence intervals.
    
    Args:
        metrics_dict: Dictionary of metrics over time
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for metric_name, values in metrics_dict.items():
        timestamps = list(range(len(values)))
        mean_values = [v['mean'] for v in values]
        
        # Plot mean line
        ax.plot(timestamps, mean_values, label=metric_name, linewidth=2, marker='o')
        
        # Plot confidence interval if available
        if 'lower' in values[0] and 'upper' in values[0]:
            lower_values = [v['lower'] for v in values]
            upper_values = [v['upper'] for v in values]
            ax.fill_between(timestamps, lower_values, upper_values, alpha=0.2)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Metrics Over Time', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return fig, ax


def plot_loss_and_metrics(results_dict, save_path=None):
    """
    Plot training losses and evaluation metrics over time for all strategies.
    
    Args:
        results_dict: Dictionary with strategy names as keys and experiment results as values
        save_path: Base path to save the plots
    """
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Define colors and markers
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    markers = ['o', 's', '^', 'D', 'X']
    
    # 1. Training Losses
    ax1 = axes[0, 0]
    for i, (strategy, results) in enumerate(results_dict.items()):
        if 'training_losses' in results:
            losses = results['training_losses']
            cycles = list(range(len(losses)))
            
            ax1.plot(cycles, losses, 
                    linestyle='-', 
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], 
                    label=strategy,
                    linewidth=2,
                    markersize=8)
    
    ax1.set_title('Training Loss Over Cycles', fontsize=14)
    ax1.set_xlabel('Cycle', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    
    # 2. Validation Losses
    ax2 = axes[0, 1]
    for i, (strategy, results) in enumerate(results_dict.items()):
        if 'val_losses' in results:
            losses = results['val_losses']
            cycles = list(range(len(losses)))
            
            ax2.plot(cycles, losses, 
                    linestyle='-', 
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], 
                    label=strategy,
                    linewidth=2,
                    markersize=8)
    
    ax2.set_title('Validation Loss Over Cycles', fontsize=14)
    ax2.set_xlabel('Cycle', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')
    
    # 3. RMSE Evolution
    ax3 = axes[1, 0]
    for i, (strategy, results) in enumerate(results_dict.items()):
        if 'val_metrics' in results:
            rmse_values = [metrics['rmse'] for metrics in results['val_metrics']]
            cycles = list(range(len(rmse_values)))
            
            ax3.plot(cycles, rmse_values, 
                    linestyle='-', 
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], 
                    label=strategy,
                    linewidth=2,
                    markersize=8)
    
    ax3.set_title('RMSE Over Cycles', fontsize=14)
    ax3.set_xlabel('Cycle', fontsize=12)
    ax3.set_ylabel('RMSE', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper right')
    
    # 4. Correlation Evolution
    ax4 = axes[1, 1]
    for i, (strategy, results) in enumerate(results_dict.items()):
        if 'val_metrics' in results:
            pearson_values = [metrics['pearson'] for metrics in results['val_metrics']]
            cycles = list(range(len(pearson_values)))
            
            ax4.plot(cycles, pearson_values, 
                    linestyle='-', 
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], 
                    label=strategy,
                    linewidth=2,
                    markersize=8)
    
    ax4.set_title('Pearson Correlation Over Cycles', fontsize=14)
    ax4.set_xlabel('Cycle', fontsize=12)
    ax4.set_ylabel('Pearson Correlation', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(loc='lower right')
    
    # Save the combined plot
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_combined.png", dpi=300, bbox_inches='tight')
    
    # Create additional plot for benefit/cost analysis
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 8))
    
    # 5. Benefit/Cost Ratios
    ax5 = axes2[0]
    for i, (strategy, results) in enumerate(results_dict.items()):
        if 'benefit_cost_ratios' in results:
            bc_ratios = results['benefit_cost_ratios']
            cycles = list(range(len(bc_ratios)))
            
            ax5.plot(cycles, bc_ratios, 
                    linestyle='-', 
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], 
                    label=strategy,
                    linewidth=2,
                    markersize=8)
    
    ax5.set_title('Benefit/Cost Ratio Over Cycles', fontsize=14)
    ax5.set_xlabel('Cycle', fontsize=12)
    ax5.set_ylabel('Benefit/Cost Ratio', fontsize=12)
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.legend(loc='upper right')
    
    # 6. Cumulative Annotation Cost
    ax6 = axes2[1]
    for i, (strategy, results) in enumerate(results_dict.items()):
        if 'observation_costs' in results:
            costs = results['observation_costs']
            cumulative_costs = np.cumsum(costs)
            cycles = list(range(len(costs)))
            
            ax6.plot(cycles, cumulative_costs, 
                    linestyle='-', 
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], 
                    label=strategy,
                    linewidth=2,
                    markersize=8)
    
    ax6.set_title('Cumulative Annotation Cost', fontsize=14)
    ax6.set_xlabel('Cycle', fontsize=12)
    ax6.set_ylabel('Cumulative Cost', fontsize=12)
    ax6.grid(True, linestyle='--', alpha=0.7)
    ax6.legend(loc='upper left')
    
    # Save the benefit/cost plot
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_benefit_cost.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return True


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