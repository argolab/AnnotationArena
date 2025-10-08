"""
Ground Truth Extractor for STAN Experiments

Extracts ground truth parameters from generated data for oracle experiments.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any


class GroundTruthExtractor:
    """Extract ground truth parameters from generated data."""
    
    def extract_true_embeddings(self, data_path: Path) -> np.ndarray:
        """Extract true item embeddings (K embeddings) from ground truth."""
        with open(data_path / "iclr_complete_ground_truth.json", 'r') as f:
            ground_truth = json.load(f)
        return np.array(ground_truth['embeddings'])  # Shape: [K, D]
    
    def extract_true_parameters(self, data_path: Path) -> Dict[str, Any]:
        """Extract all ground truth parameters."""
        with open(data_path / "iclr_complete_ground_truth.json", 'r') as f:
            ground_truth = json.load(f)
        
        return {
            'embeddings': ground_truth['embeddings'],  # K item embeddings
            'mean_preferences': ground_truth['mean_preferences'],
            'annotator_preferences': ground_truth['annotator_preferences'],
            'rating_thresholds': ground_truth['rating_thresholds'],
            'K': len(ground_truth['embeddings']),  # Number of items
            'D': len(ground_truth['embeddings'][0])  # Embedding dimension
        }
    
    def extract_true_preferences(self, data_path: Path) -> Dict[str, np.ndarray]:
        """Extract preference parameters from ground truth."""
        with open(data_path / "iclr_complete_ground_truth.json", 'r') as f:
            ground_truth = json.load(f)
        
        return {
            'mean_preferences': np.array(ground_truth['mean_preferences']),
            'annotator_preferences': np.array(ground_truth['annotator_preferences'])
        }
    
    def extract_true_thresholds(self, data_path: Path) -> np.ndarray:
        """Extract rating thresholds from ground truth."""
        with open(data_path / "iclr_complete_ground_truth.json", 'r') as f:
            ground_truth = json.load(f)
        
        return np.array(ground_truth['rating_thresholds'])
    
    def get_data_dimensions(self, data_path: Path) -> Dict[str, int]:
        """Get data dimensions from ground truth."""
        with open(data_path / "iclr_complete_ground_truth.json", 'r') as f:
            ground_truth = json.load(f)
        
        return {
            'K': len(ground_truth['embeddings']),
            'D': len(ground_truth['embeddings'][0]),
            'I': len(ground_truth['mean_preferences']),
            'J': len(ground_truth['annotator_preferences']) // len(ground_truth['mean_preferences']),
            'C': len(ground_truth['rating_thresholds'][0])
        }
