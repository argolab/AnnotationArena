"""
Model saving utilities for attention analysis.

Saves trained models and associated metadata to enable post-hoc attention pattern analysis
without requiring full retraining.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_model_for_attention_analysis(model: torch.nn.Module,
                                    bn_structure: Any,  # pyagrum.BayesNet
                                    adjacency_matrix: np.ndarray,
                                    test_dataset: Any,
                                    experiment_config: Dict[str, Any],
                                    output_dir: str,
                                    model_name: str) -> str:
    """
    Save model and associated data for attention analysis.
    
    Args:
        model: Trained GraphImputer model
        bn_structure: Original pyagrum BayesNet structure
        adjacency_matrix: Adjacency matrix representation
        test_dataset: Test dataset with samples and missing patterns
        experiment_config: Configuration used for training
        output_dir: Directory to save model artifacts
        model_name: Name for saved model files
        
    Returns:
        Path to saved model directory
    """
    save_dir = Path(output_dir) / "saved_models" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    model_path = save_dir / "model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_nodes': model.n_nodes,
            'n_states': model.n_states,
            'architecture': str(model.__class__.__name__),
            # Add other relevant model parameters from model creation
        }
    }, model_path)
    
    # Save BN structure (pickle for pyagrum object)
    bn_path = save_dir / "bn_structure.pkl"
    with open(bn_path, 'wb') as f:
        pickle.dump(bn_structure, f)
    
    # Save adjacency matrix
    adj_path = save_dir / "adjacency_matrix.npy"
    np.save(adj_path, adjacency_matrix)
    
    # Save test dataset samples (all samples for analysis)
    test_samples = []
    max_samples = len(test_dataset)
    
    for i in range(max_samples):
        sample = test_dataset[i]
        # Convert tensors to numpy for storage
        if isinstance(sample, tuple):
            test_samples.append({
                'inputs': sample[0].cpu().numpy(),
                'structure_info': sample[1].cpu().numpy(),
                'dimensions': sample[2].cpu().numpy(),
                'mask': sample[3].cpu().numpy(),
                'targets': sample[4].cpu().numpy(),
                'cpt_info': sample[5].cpu().numpy()
            })
    
    test_path = save_dir / "test_samples.pkl"
    with open(test_path, 'wb') as f:
        pickle.dump(test_samples, f)
    
    # Save experiment configuration
    config_path = save_dir / "experiment_config.pkl"
    with open(config_path, 'wb') as f:
        pickle.dump(experiment_config, f)
    
    # Create manifest file
    manifest = {
        'model_name': model_name,
        'n_nodes': adjacency_matrix.shape[0],
        'n_test_samples': len(test_samples),
        'experiment_config': experiment_config,
        'files': {
            'model': 'model.pt',
            'bn_structure': 'bn_structure.pkl',
            'adjacency_matrix': 'adjacency_matrix.npy',
            'test_samples': 'test_samples.pkl',
            'experiment_config': 'experiment_config.pkl'
        }
    }
    
    manifest_path = save_dir / "manifest.json"
    import json
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    logger.info(f"Model saved for attention analysis: {save_dir}")
    return str(save_dir)


def load_model_for_attention_analysis(model_dir: str) -> Dict[str, Any]:
    """
    Load saved model and associated data for attention analysis.
    
    Args:
        model_dir: Directory containing saved model artifacts
        
    Returns:
        Dictionary containing loaded model and associated data
    """
    model_dir = Path(model_dir)
    
    # Load manifest
    manifest_path = model_dir / "manifest.json"
    import json
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Load model state dict
    model_path = model_dir / manifest['files']['model']
    model_checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load BN structure
    bn_path = model_dir / manifest['files']['bn_structure']
    with open(bn_path, 'rb') as f:
        bn_structure = pickle.load(f)
    
    # Load adjacency matrix
    adj_path = model_dir / manifest['files']['adjacency_matrix']
    adjacency_matrix = np.load(adj_path)
    
    # Load test samples
    test_path = model_dir / manifest['files']['test_samples']
    with open(test_path, 'rb') as f:
        test_samples = pickle.load(f)
    
    # Load experiment configuration
    config_path = model_dir / manifest['files']['experiment_config']
    with open(config_path, 'rb') as f:
        experiment_config = pickle.load(f)
    
    loaded_data = {
        'model_checkpoint': model_checkpoint,
        'bn_structure': bn_structure,
        'adjacency_matrix': adjacency_matrix,
        'test_samples': test_samples,
        'experiment_config': experiment_config,
        'manifest': manifest
    }
    
    logger.info(f"Loaded model for attention analysis from {model_dir}")
    return loaded_data


def reconstruct_model_from_saved(saved_data: Dict[str, Any]) -> torch.nn.Module:
    """
    Reconstruct GraphImputer model from saved data.
    
    Args:
        saved_data: Data loaded from load_model_for_attention_analysis
        
    Returns:
        Reconstructed model with loaded weights
    """
    from imputer.architecture import create_model
    
    checkpoint = saved_data['model_checkpoint']
    config = saved_data['experiment_config']
    n_nodes = saved_data['adjacency_matrix'].shape[0]
    
    # Reconstruct model using same parameters as training
    # For binary nodes: input_dim=3 (mask, state_0, state_1), structure_dim=n_nodes
    # CPT dimension is computed from adjacency matrix
    adj_matrix = saved_data['adjacency_matrix']
    max_cpt_size = 2 ** (adj_matrix.sum(axis=0).max() + 1)  # 2^(max_parents + 1)
    
    model = create_model(
        n_nodes=n_nodes,
        input_dim=3,  # [mask, state_0, state_1] 
        structure_dim=n_nodes,  # adjacency matrix dimension
        cpt_dim=int(max_cpt_size),
        model_size=config.get('imputer_size', 'Large')
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Reconstructed {config.get('imputer_size', 'Large')} model with {n_nodes} nodes")
    return model