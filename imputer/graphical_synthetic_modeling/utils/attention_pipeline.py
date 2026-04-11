"""
Complete attention analysis pipeline for progressive imputation experiments.

Orchestrates model loading, attention capture, pattern analysis, and visualization
to provide comprehensive insights into transformer attention patterns.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json

from utils.attention_analysis import AttentionCapture, analyze_attention_patterns, create_attention_heatmaps, save_attention_analysis
from utils.model_saving import load_model_for_attention_analysis, reconstruct_model_from_saved
from imputer.architecture import create_model, DEVICE

logger = logging.getLogger(__name__)


def run_attention_analysis_on_saved_model(model_dir: str, 
                                         output_dir: str = "attention_analysis",
                                         n_samples: int = 10) -> None:
    """
    Run complete attention analysis on a saved model with averaged patterns.
    
    Args:
        model_dir: Directory containing saved model artifacts
        output_dir: Directory to save attention analysis results
        n_samples: Number of test samples to analyze
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting attention analysis on {model_dir}")
    
    # Load saved model and data
    saved_data = load_model_for_attention_analysis(model_dir)
    model = reconstruct_model_from_saved(saved_data)
    model = model.to(DEVICE)
    
    adjacency_matrix = saved_data['adjacency_matrix']
    test_samples = saved_data['test_samples']
    experiment_config = saved_data['experiment_config']
    
    n_nodes = adjacency_matrix.shape[0]
    n_analyze = min(n_samples, len(test_samples))
    
    logger.info(f"Analyzing {n_analyze} samples from {n_nodes}-node graph")
    
    # Collect attention patterns across all samples
    all_attention_weights = {}
    all_analyses = []
    sample_observed_masks = []
    
    for sample_idx in range(n_analyze):
        sample = test_samples[sample_idx]
        
        # Convert sample data back to tensors
        inputs = torch.FloatTensor(sample['inputs']).unsqueeze(0).to(DEVICE)
        structure_info = torch.FloatTensor(sample['structure_info']).unsqueeze(0).to(DEVICE)
        dimensions = torch.LongTensor(sample['dimensions']).unsqueeze(0).to(DEVICE)
        mask = torch.FloatTensor(sample['mask']).unsqueeze(0)
        
        # Fix CPT info: extract proper CPTs for observed nodes
        from imputer.architecture import extract_cpts_for_nodes
        observed_nodes = [i for i in range(n_nodes) if sample['mask'][i] == 0]  # mask=0 means observed
        max_cpt_size = 2 ** (adjacency_matrix.sum(axis=0).max() + 1)
        cpt_info_array = extract_cpts_for_nodes(saved_data['bn_structure'], observed_nodes, n_nodes, int(max_cpt_size))
        cpt_info = torch.FloatTensor(cpt_info_array).unsqueeze(0).to(DEVICE)
        
        # Create observed mask (1=observed, 0=unobserved)
        observed_mask = 1.0 - mask  # Invert mask bit
        sample_observed_masks.append(observed_mask)
        
        if sample_idx % 50 == 0:
            logger.info(f"Processing sample {sample_idx}: {observed_mask.sum().int().item()}/{n_nodes} nodes observed")
        
        # Capture attention patterns
        with AttentionCapture(model) as attention_capture:
            # Forward pass to trigger attention capture
            _ = model(inputs, structure_info, cpt_info, dimensions)
            
            # Get averaged attention weights
            attention_weights = attention_capture.get_averaged_attention()
        
        if not attention_weights:
            logger.warning(f"No attention weights captured for sample {sample_idx}")
            continue
        
        # Accumulate attention weights for averaging
        if not all_attention_weights:
            # Initialize with first sample
            for layer_name, weights in attention_weights.items():
                all_attention_weights[layer_name] = weights.clone()
        else:
            # Add to running sum
            for layer_name, weights in attention_weights.items():
                if layer_name in all_attention_weights:
                    all_attention_weights[layer_name] += weights
        
        # Analyze attention patterns for this sample
        analysis = analyze_attention_patterns(
            attention_weights=attention_weights,
            adjacency_matrix=adjacency_matrix,
            observed_mask=observed_mask,
            sample_idx=0  # Only one sample in batch
        )
        all_analyses.append(analysis)
    
    # Average attention weights across all samples
    n_valid_samples = len(all_analyses)
    logger.info(f"Computing average attention patterns across {n_valid_samples} samples")
    for layer_name in all_attention_weights:
        all_attention_weights[layer_name] /= n_valid_samples
        logger.debug(f"Averaged {layer_name}: mean={all_attention_weights[layer_name].mean().item():.4f}")
    
    # Use average observation pattern for visualization
    avg_observed_mask = torch.stack(sample_observed_masks).mean(dim=0)
    
    # Create separate Bayesian Network structure plot
    from utils.graph_visualization import create_bayesian_network_plot
    create_bayesian_network_plot(adjacency_matrix, output_dir)
    
    # Create visualization with averaged attention patterns
    create_attention_heatmaps(
        attention_weights=all_attention_weights,
        adjacency_matrix=adjacency_matrix,
        observed_mask=avg_observed_mask,
        sample_idx=0,
        output_dir=output_dir
    )
    
    # Save averaged analysis
    averaged_analysis = {
        'n_samples_analyzed': len(all_analyses),
        'averaged_attention': True,
        'sample_analyses': all_analyses
    }
    save_attention_analysis(averaged_analysis, output_dir, 0)
    
    # Create summary analysis across all samples
    create_attention_summary(output_dir, n_analyze, experiment_config)
    
    logger.info(f"Attention analysis completed. Results saved to {output_dir}")


def create_attention_summary(output_dir: str, n_samples: int, 
                           experiment_config: Dict[str, Any]) -> None:
    """
    Create summary analysis across all analyzed samples.
    
    Args:
        output_dir: Directory containing individual sample analyses
        n_samples: Number of samples analyzed
        experiment_config: Experiment configuration for context
    """
    output_dir = Path(output_dir)
    
    # Aggregate statistics across samples
    layer_stats = {}  # layer_name -> list of stats across samples
    
    for sample_idx in range(n_samples):
        analysis_file = output_dir / f"attention_analysis_sample_{sample_idx}.json"
        
        if not analysis_file.exists():
            continue
            
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        # Aggregate layer-wise statistics
        for layer_name, layer_data in analysis['layer_patterns'].items():
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {
                    'attention_entropy': [],
                    'attention_focus': [],
                    'parent_attention_ratio': [],
                    'privacy_compliance_score': []
                }
            
            for metric in layer_stats[layer_name]:
                if metric in layer_data:
                    layer_stats[layer_name][metric].append(layer_data[metric])
    
    # Compute aggregate statistics
    summary = {
        'experiment_config': experiment_config,
        'n_samples_analyzed': n_samples,
        'layer_summary': {}
    }
    
    for layer_name, stats in layer_stats.items():
        layer_summary = {}
        
        for metric, values in stats.items():
            if values:
                layer_summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        summary['layer_summary'][layer_name] = layer_summary
    
    # Save summary
    summary_file = output_dir / "attention_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Create summary visualization
    create_summary_plots(summary, str(output_dir))
    
    logger.info(f"Attention summary saved to {summary_file}")


def create_summary_plots(summary: Dict[str, Any], output_dir: str) -> None:
    """
    Create summary visualizations of attention patterns across layers.
    
    Args:
        summary: Summary statistics across all samples
        output_dir: Directory to save summary plots
    """
    import matplotlib.pyplot as plt
    
    layer_names = sorted(summary['layer_summary'].keys())
    n_layers = len(layer_names)
    
    if n_layers == 0:
        logger.warning("No layer data for summary plots")
        return
    
    # Create 2x2 subplot for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['attention_entropy', 'attention_focus', 'parent_attention_ratio', 'privacy_compliance_score']
    titles = ['Attention Entropy', 'Attention Focus', 'Parent Attention Ratio', 'Privacy Compliance']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i // 2, i % 2]
        
        means = []
        stds = []
        layer_indices = []
        
        for j, layer_name in enumerate(layer_names):
            layer_data = summary['layer_summary'][layer_name]
            if metric in layer_data:
                means.append(layer_data[metric]['mean'])
                stds.append(layer_data[metric]['std'])
                layer_indices.append(j + 1)
        
        if means:
            ax.errorbar(layer_indices, means, yerr=stds, 
                       marker='o', capsize=5, capthick=2, linewidth=2)
            ax.set_xlabel('Layer')
            ax.set_ylabel(title)
            ax.set_title(f'{title} Across Layers')
            ax.set_xticks(layer_indices)
            ax.set_xticklabels([f'L{i}' for i in layer_indices])
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No {metric} data', transform=ax.transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/attention_summary_plots.png", dpi=300, bbox_inches='tight')
    logger.info(f"Summary plots saved to {output_dir}/attention_summary_plots.png")
    plt.close()


def save_model_after_training(experiment: Any, 
                            output_dir: str,
                            imputer_size: str,
                            save_models: bool = True) -> Optional[str]:
    """
    Save model after training for later attention analysis.
    
    Args:
        experiment: ProgressiveExperiment instance
        output_dir: Directory to save models
        imputer_size: Size of the imputer model
        save_models: Whether to actually save models
        
    Returns:
        Path to saved model directory, or None if not saved
    """
    if not save_models:
        return None
        
    try:
        from utils.model_saving import save_model_for_attention_analysis
        
        # Get the best performing model from the experiment
        model = experiment.neural_models.get(imputer_size)
        if model is None:
            logger.warning(f"No {imputer_size} model found in experiment")
            return None
        
        # Create model name
        model_name = f"{imputer_size.lower()}_imputer_{experiment.n_nodes}nodes_final"
        
        # Prepare experiment config
        config = {
            'n_nodes': experiment.n_nodes,
            'target_parents': experiment.target_parents,
            'missing_rate': experiment.missing_rate,
            'imputer_size': imputer_size,
            'cpt_generation': getattr(experiment, 'cpt_generation', 'default'),
            'logistic_std': getattr(experiment, 'logistic_std', 1.0)
        }
        
        # Save model for attention analysis
        saved_path = save_model_for_attention_analysis(
            model=model,
            bn_structure=experiment.bn,
            adjacency_matrix=experiment.adj_matrix,
            test_dataset=experiment.test_dataset,
            experiment_config=config,
            output_dir=output_dir,
            model_name=model_name
        )
        
        logger.info(f"Saved {imputer_size} model for attention analysis: {saved_path}")
        return saved_path
        
    except Exception as e:
        logger.error(f"Failed to save model for attention analysis: {e}")
        return None


def create_attention_analysis_report(results: Dict[str, Any], output_dir: str = "plots",
                                   missing_rate: Optional[float] = None) -> None:
    """
    Main entry point for attention analysis reporting.
    
    Looks for saved models and runs attention analysis on them.
    
    Args:
        results: Results dictionary from experiment_runner  
        output_dir: Directory containing saved models and for analysis output
        missing_rate: Missing rate for filename suffixes
    """
    output_dir = Path(output_dir)
    # Navigate to the main outputs directory from the plots subdirectory
    # output_dir is something like: outputs/plots/experiment_.../missing_rate_0.5/
    # We need to go to: outputs/saved_models/
    outputs_root = output_dir
    while outputs_root.name != "outputs" and outputs_root.parent != outputs_root:
        outputs_root = outputs_root.parent
    
    if outputs_root.name == "outputs":
        models_dir = outputs_root / "saved_models"
    else:
        # Fallback if we can't find outputs directory
        models_dir = output_dir.parent.parent / "saved_models"
    
    if not models_dir.exists():
        logger.warning(f"No saved models found in {models_dir}")
        return
    
    # Find saved models
    saved_models = list(models_dir.glob("*/manifest.json"))
    
    if not saved_models:
        logger.warning(f"No model manifests found in {models_dir}")
        return
    
    logger.info(f"Found {len(saved_models)} saved models for attention analysis")
    
    # Analyze each saved model
    for manifest_path in saved_models:
        model_dir = manifest_path.parent
        model_name = model_dir.name
        
        # Create analysis output directory
        analysis_dir = output_dir / f"attention_analysis_{model_name}"
        
        try:
            run_attention_analysis_on_saved_model(
                model_dir=str(model_dir),
                output_dir=str(analysis_dir),
                n_samples=500  # Analyze all test samples
            )
            logger.info(f"Completed attention analysis for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed attention analysis for {model_name}: {e}")
    
    logger.info("Attention analysis report completed")