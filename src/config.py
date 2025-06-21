"""
Configuration management for Active Learner framework.

Author: Prabhav Singh / Haojun Shi
"""

import os
from datetime import datetime

class Config:
    """Configuration management for Active Learner framework."""
    
    def __init__(self, runner="local"):
        self.runner = runner
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if runner == "prabhav":
            self.BASE_PATH = "/export/fs06/psingh54/ActiveRubric-Internal/src"
        else:
            self.BASE_PATH = "."
    
    @property
    def INPUT_DATA_DIR(self):
        return os.path.join(self.BASE_PATH, "input", "data")
    
    @property
    def INPUT_FIXED_DIR(self):
        return os.path.join(self.BASE_PATH, "input", "fixed")
    
    @property
    def LOGS_DIR(self):
        return os.path.join(self.BASE_PATH, "logs")
    
    @property
    def OUTPUT_RESULTS_DIR(self):
        return os.path.join(self.BASE_PATH, "output", "results")
    
    @property
    def OUTPUT_PLOTS_DIR(self):
        return os.path.join(self.BASE_PATH, "output", "plots")
    
    @property
    def MODELS_DIR(self):
        return os.path.join(self.BASE_PATH, "output", "models")
    
    def get_data_paths(self, dataset="hanna"):
        """Get data file paths for a specific dataset."""
        data_dir = self.INPUT_DATA_DIR
        return {
            'train': os.path.join(data_dir, "initial_train.json"),
            'validation': os.path.join(data_dir, "validation.json"),
            'test': os.path.join(data_dir, "test.json"),
            'active_pool': os.path.join(data_dir, "active_pool.json"),
            'original_train': os.path.join(data_dir, "original_initial_train.json"),
            'original_validation': os.path.join(data_dir, "original_validation.json"),
            'original_test': os.path.join(data_dir, "original_test.json"),
            'original_active_pool': os.path.join(data_dir, "original_active_pool.json"),
        }
    
    def get_fixed_paths(self):
        """Get paths to fixed data files."""
        fixed_dir = self.INPUT_FIXED_DIR
        return {
            'questions': os.path.join(fixed_dir, "questions.json"),
            'human_data': os.path.join(fixed_dir, "human-data-new.json"),
            'gpt_data': os.path.join(fixed_dir, "gpt-3.5-turbo-data-new.json"),
            'hanna_stories': os.path.join(fixed_dir, "hanna_stories_annotations_updated.csv")
        }
    
    def get_experiment_paths(self, experiment_name):
        """Get paths for a specific experiment."""
        exp_dir = os.path.join(self.OUTPUT_RESULTS_DIR, f"{experiment_name}_{self.timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        
        log_file = os.path.join(self.LOGS_DIR, f"experiment_log_{experiment_name}_{self.timestamp}.log")
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        
        model_file = os.path.join(self.MODELS_DIR, f"{experiment_name}_{self.timestamp}.pth")
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        
        return {
            'results_dir': exp_dir,
            'log_file': log_file,
            'model_file': model_file
        }
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        dirs = [
            self.INPUT_DATA_DIR,
            self.INPUT_FIXED_DIR,
            self.LOGS_DIR,
            self.OUTPUT_RESULTS_DIR,
            self.OUTPUT_PLOTS_DIR,
            self.MODELS_DIR
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

class ModelConfig:
    """Model architecture and training configurations."""
    
    HANNA = {
        'question_num': 7,
        'max_choices': 5,
        'encoder_layers_num': 6,
        'attention_heads': 4,
        'hidden_dim': 64,
        'num_annotator': 18,
        'annotator_embedding_dim': 19,
        'dropout': 0.1
    }
    
    LLM_RUBRIC = {
        'question_num': 9,
        'max_choices': 4,
        'encoder_layers_num': 6,
        'attention_heads': 4,
        'hidden_dim': 64,
        'num_annotator': 24,
        'annotator_embedding_dim': 24,
        'dropout': 0.1
    }
    
    @classmethod
    def get_config(cls, dataset):
        """Get model configuration for dataset."""
        if dataset == "hanna":
            return cls.HANNA
        elif dataset == "llm_rubric":
            return cls.LLM_RUBRIC
        else:
            return cls.HANNA

class DefaultHyperparams:
    """Default hyperparameters for experiments."""
    
    LR = 1e-4
    EPOCHS_PER_CYCLE = 5
    BATCH_SIZE = 16
    CYCLES = 10
    EXAMPLES_PER_CYCLE = 50
    FEATURES_PER_EXAMPLE = 5
    ACTIVE_SET_SIZE = 100
    VALIDATION_SET_SIZE = 50
    NUM_PATTERNS_PER_EXAMPLE = 3
    VISIBLE_RATIO = 0.5
    GRADIENT_TOP_ONLY = True
    LOSS_TYPE = "cross_entropy"
    TRAIN_OPTION = "dynamic_masking"
    USE_EMBEDDING = True
    COLD_START = True
    RESAMPLE_VALIDATION = True
    RUN_UNTIL_EXHAUSTED = False