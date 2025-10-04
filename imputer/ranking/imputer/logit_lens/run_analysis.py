#!/usr/bin/env python3
"""Main script for running logit lens analysis on trained imputer models."""

import argparse
from imputer.logit_lens.analyzer import LogitLensResults
import torch
import json
from pathlib import Path
from typing import List, Dict, Any

from imputer.data import DataConverter, RankingData
from imputer.ranking_imputer import MultiVariableImputer
from imputer.logit_lens import LogitLensAnalyzer, LogitLensVisualizer
from stan.pipeline.bundle import GroundTruthBundle


def load_model_and_data(model_path: str, data_path: str, device: str = 'cpu') -> tuple:
    """Load a trained model and data bundle."""
    
    # Load model checkpoint onto the target device
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Debug: print checkpoint keys and config
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Extract model configuration - should always be present in properly saved checkpoints
    if 'model_config' not in checkpoint:
        raise ValueError(f"No model_config found in checkpoint {model_path}. "
                        "This checkpoint was saved with an older version. "
                        "Please retrain the model with the updated trainer.")
    
    config = checkpoint['model_config']
    print(f"Loaded model_config: {config}")
    
    # Debug: print some state_dict keys to see what we're trying to load
    state_dict = checkpoint['state_dict']
    print(f"State dict keys (first 10): {list(state_dict.keys())[:10]}")
    print(f"Number of blocks in checkpoint: {len([k for k in state_dict.keys() if k.startswith('blocks.') and 'Q.weight' in k])}")
    
    max_rank_size = config['max_rank_size']
    
    # Create model
    print(f"Creating model with config: {config}")
    model = MultiVariableImputer(
        num_attributes=config['num_attributes'],
        num_annotators=config['num_annotators'],
        num_items=config['num_items'],
        num_likert_classes=config['num_likert_classes'],
        max_rank_size=max_rank_size,
        encoder_layers_num=config['encoder_layers_num'],
        attention_heads=config['attention_heads'],
        embedding_dim=config['embedding_dim'],
        dropout=config['dropout'],
        embedding_type=config['embedding_type'],
        device=device
    )
    
    print(f"Created model with {len(model.blocks)} blocks, embedding_dim={model.embedding_dim}")
    
    # Load state dict
    print("Loading state dict...")
    model.load_state_dict(checkpoint['state_dict'])
    
    converter = DataConverter(
        num_attributes=config['num_attributes'],
        num_annotators=config['num_annotators'],
        num_items=config['num_items'],
        num_likert_classes=config['num_likert_classes'],
        max_rank_size=max_rank_size
    )
    
    # Load data bundle
    with open(data_path, 'r') as f:
        bundle_dict = json.load(f)
    
    # Create GroundTruthBundle from JSON data
    data_bundle = GroundTruthBundle.from_dict(bundle_dict)
    
    # Create variables
    train_observed = converter.create_variables_from_bundle(
        data_bundle, partition='train', status='observed'
    )
    train_missing = converter.create_variables_from_bundle(
        data_bundle, partition='train', status='missing'
    )
    test_observed = converter.create_variables_from_bundle(
        data_bundle, partition='test', status='observed'
    )
    test_missing = converter.create_variables_from_bundle(
        data_bundle, partition='test', status='missing'
    )
    
    # Combine for analysis
    train_variables = train_observed + train_missing
    test_variables = test_observed + test_missing
    
    return model, converter, train_variables, test_variables


def run_logit_lens_analysis(model: MultiVariableImputer, 
                          converter: DataConverter,
                          train_variables: List[RankingData],
                          test_variables: List[RankingData],
                          device: str = 'cuda') -> LogitLensResults:
    """Run logit lens analysis."""
    
    print("Running Logit Lens Analysis...")
    
    analyzer = LogitLensAnalyzer(model, converter, device)
    results = analyzer.analyze_all_layers(train_variables, test_variables)
    
    print(f"Analysis complete. Analyzed {len(results.all_variables[0].layer_analyses)} layers.")
    
    return results


def create_visualizations(results: LogitLensResults, output_dir: Path) -> None:
    """Create and save visualizations."""
    
    print("Creating visualizations...")
    
    visualizer = LogitLensVisualizer(results)
    
    # Create the three main plots
    visualizer.plot_train_performance(save_path=str(output_dir / "train_performance.png"))
    visualizer.plot_test_performance(save_path=str(output_dir / "test_performance.png"))
    visualizer.plot_all_performance(save_path=str(output_dir / "all_performance.png"))
    
    # Save results
    results_path = output_dir / "logit_lens_results.json"
    visualizer.save_results(str(results_path))
    
    # Print summary
    visualizer.print_summary()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run logit lens analysis on imputer models')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data bundle JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results and visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run analysis on')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    print("Loading model and data...")
    model, converter, train_variables, test_variables = load_model_and_data(
        args.model_path, args.data_path, args.device
    )
    
    print(f"Loaded model with {len(model.blocks)} transformer layers")
    print(f"Train variables: {len(train_variables)}")
    print(f"Test variables: {len(test_variables)}")
    
    # Run analysis
    logit_results = run_logit_lens_analysis(
        model, converter, train_variables, test_variables, args.device
    )
    create_visualizations(logit_results, output_dir)
    
    print("Analysis complete!")


if __name__ == '__main__':
    main()