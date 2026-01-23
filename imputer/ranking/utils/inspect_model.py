#!/usr/bin/env python3
"""
Model Parameter Inspector
Easy-to-use tool for inspecting trained model checkpoints (.pt files)

Usage:
    python utils/inspect_model.py <model_path> [options]
    
Examples:
    # Basic inspection
    python utils/inspect_model.py OUTPUT/IMPUTER/run_name_marformer/model.pt
    
    # Show only parameter names and shapes
    python utils/inspect_model.py model.pt --summary
    
    # Filter by parameter name pattern
    python utils/inspect_model.py model.pt --filter "embedding"
    
    # Show statistics for specific parameters
    python utils/inspect_model.py model.pt --stats "blocks.0"
    
    # Export to JSON
    python utils/inspect_model.py model.pt --export params.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import numpy as np


def load_checkpoint(model_path: str) -> Dict[str, Any]:
    """Load a PyTorch checkpoint file."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    return checkpoint


def get_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Extract state_dict from checkpoint (handles different formats)."""
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    else:
        # Assume the checkpoint itself is the state_dict
        return checkpoint


def format_size(num_params: int) -> str:
    """Format parameter count in human-readable format."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def get_tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """Get statistics for a tensor."""
    tensor_np = tensor.detach().cpu().numpy()
    return {
        'min': float(np.min(tensor_np)),
        'max': float(np.max(tensor_np)),
        'mean': float(np.mean(tensor_np)),
        'std': float(np.std(tensor_np)),
        'shape': list(tensor.shape),
        'numel': int(tensor.numel()),
        'dtype': str(tensor.dtype)
    }


def print_model_config(checkpoint: Dict[str, Any]):
    """Print model configuration if available."""
    if 'model_config' in checkpoint:
        print("\n" + "="*70)
        print("MODEL CONFIGURATION")
        print("="*70)
        config = checkpoint['model_config']
        for key, value in config.items():
            print(f"  {key}: {value}")
    elif 'config' in checkpoint:
        print("\n" + "="*70)
        print("MODEL CONFIGURATION")
        print("="*70)
        config = checkpoint['config']
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print("\n(No model configuration found in checkpoint)")


def print_checkpoint_info(checkpoint: Dict[str, Any]):
    """Print general checkpoint information."""
    print("\n" + "="*70)
    print("CHECKPOINT INFORMATION")
    print("="*70)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'loss_dict' in checkpoint:
        print(f"Loss dict keys: {list(checkpoint['loss_dict'].keys())}")
    if 'optimizer_state_dict' in checkpoint:
        print("Optimizer state dict: Present")


def print_parameter_summary(state_dict: Dict[str, torch.Tensor], filter_pattern: Optional[str] = None):
    """Print summary of all parameters (names and shapes only)."""
    print("\n" + "="*70)
    print("PARAMETER SUMMARY")
    print("="*70)
    
    total_params = 0
    filtered_params = 0
    
    # Filter parameters if pattern provided
    params_to_show = state_dict.items()
    if filter_pattern:
        params_to_show = [(k, v) for k, v in params_to_show if filter_pattern.lower() in k.lower()]
        filtered_params = len(params_to_show)
    
    for name, param in params_to_show:
        num_params = param.numel()
        total_params += num_params
        print(f"  {name:60s} {str(param.shape):20s} {format_size(num_params):>10s}")
    
    print(f"\nTotal parameters: {format_size(total_params)}")
    if filter_pattern:
        print(f"Filtered parameters: {filtered_params} (matching '{filter_pattern}')")


def print_parameter_details(state_dict: Dict[str, torch.Tensor], 
                           filter_pattern: Optional[str] = None,
                           stats_pattern: Optional[str] = None):
    """Print detailed information about parameters."""
    print("\n" + "="*70)
    print("PARAMETER DETAILS")
    print("="*70)
    
    total_params = 0
    trainable_params = 0
    
    # Determine which parameters to show stats for
    params_to_show = state_dict.items()
    if filter_pattern:
        params_to_show = [(k, v) for k, v in params_to_show if filter_pattern.lower() in k.lower()]
    if stats_pattern:
        params_to_show = [(k, v) for k, v in params_to_show if stats_pattern.lower() in k.lower()]
    
    for name, param in params_to_show:
        num_params = param.numel()
        total_params += num_params
        trainable_params += num_params if param.requires_grad else 0
        
        stats = get_tensor_stats(param)
        print(f"\n{name}")
        print(f"  Shape: {stats['shape']}")
        print(f"  Parameters: {format_size(num_params)} ({num_params:,})")
        print(f"  Dtype: {stats['dtype']}")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
    
    print(f"\n{'='*70}")
    print(f"Total parameters: {format_size(total_params)} ({total_params:,})")
    print(f"Trainable parameters: {format_size(trainable_params)} ({trainable_params:,})")


def export_to_json(state_dict: Dict[str, torch.Tensor], 
                  checkpoint: Dict[str, Any],
                  output_path: str,
                  filter_pattern: Optional[str] = None):
    """Export parameter information to JSON."""
    output = {
        'model_config': checkpoint.get('model_config', {}),
        'checkpoint_info': {
            'epoch': checkpoint.get('epoch', None),
            'keys': list(checkpoint.keys())
        },
        'parameters': {}
    }
    
    params_to_export = state_dict.items()
    if filter_pattern:
        params_to_export = [(k, v) for k, v in params_to_export if filter_pattern.lower() in k.lower()]
    
    for name, param in params_to_export:
        stats = get_tensor_stats(param)
        output['parameters'][name] = stats
    
    # Calculate totals
    total_params = sum(p.numel() for p in state_dict.values())
    output['totals'] = {
        'total_parameters': total_params,
        'total_parameters_formatted': format_size(total_params),
        'num_layers': len([k for k in state_dict.keys() if 'blocks.' in k or 'layer' in k.lower()])
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nExported parameter information to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect trained model parameters from .pt checkpoint files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to model checkpoint file (.pt)'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show only parameter names and shapes (summary mode)'
    )
    
    parser.add_argument(
        '--filter',
        type=str,
        default=None,
        help='Filter parameters by name pattern (case-insensitive)'
    )
    
    parser.add_argument(
        '--stats',
        type=str,
        default=None,
        help='Show detailed statistics for parameters matching pattern'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        default=None,
        help='Export parameter information to JSON file'
    )
    
    parser.add_argument(
        '--no-config',
        action='store_true',
        help='Skip printing model configuration'
    )
    
    parser.add_argument(
        '--no-checkpoint-info',
        action='store_true',
        help='Skip printing checkpoint information'
    )
    
    args = parser.parse_args()
    
    try:
        # Load checkpoint
        checkpoint = load_checkpoint(args.model_path)
        state_dict = get_state_dict(checkpoint)
        
        # Print checkpoint info
        if not args.no_checkpoint_info:
            print_checkpoint_info(checkpoint)
        
        # Print model config
        if not args.no_config:
            print_model_config(checkpoint)
        
        # Print parameter information
        if args.summary:
            print_parameter_summary(state_dict, filter_pattern=args.filter)
        elif args.stats:
            print_parameter_details(state_dict, filter_pattern=args.filter, stats_pattern=args.stats)
        else:
            # Default: show summary and details
            print_parameter_summary(state_dict, filter_pattern=args.filter)
            if args.filter or args.stats:
                print_parameter_details(state_dict, filter_pattern=args.filter, stats_pattern=args.stats)
        
        # Export if requested
        if args.export:
            export_to_json(state_dict, checkpoint, args.export, filter_pattern=args.filter)
        
        print("\n" + "="*70)
        print("Inspection complete!")
        print("="*70)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

