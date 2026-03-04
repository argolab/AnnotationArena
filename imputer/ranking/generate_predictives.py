#!/usr/bin/env python3
"""
Generate predictives from a trained model checkpoint without retraining.

This script loads a trained model and generates train_predictives.json and 
test_predictives.json files needed for JK diagnostics.

Usage:
    python generate_predictives.py \
        --model-path OUTPUT/IMPUTER/easy_axis_JK_D8_sa05_sm01_kp10_uc0_hI1_hJ0_hK0_smar/model.pt \
        --data-dir OUTPUT/generated_data/easy_axis_JK_D8_sa05_sm01_kp10_uc0_hI1_hJ0_hK0_smar \
        [--output-dir OUTPUT/IMPUTER/easy_axis_JK_D8_sa05_sm01_kp10_uc0_hI1_hJ0_hK0_smar] \
        [--device cuda]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from imputer.data import DataConverter, RankingData
from imputer.ranking_imputer import MultiVariableImputer
from stan.pipeline.bundle import GroundTruthBundle
from stan.pipeline.io import save_predictives
from imputer.utils import build_predictives


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictives from trained model checkpoint"
    )
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--data-dir", required=True, help="Directory containing data_bundle.json")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as model directory)")
    parser.add_argument("--device", default="cuda", help="Device to use (default: cuda)")
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    data_dir = Path(args.data_dir)
    bundle_path = data_dir / "data_bundle.json"
    
    if not bundle_path.exists():
        raise FileNotFoundError(f"Data bundle not found: {bundle_path}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use the directory containing the model file
        output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from: {model_path}")
    print(f"Loading data from: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load bundle
    with open(bundle_path, 'r') as f:
        bundle_data = json.load(f)
    bundle = GroundTruthBundle.from_dict(bundle_data)
    
    # Try to load train_config.json from model directory (more reliable than checkpoint)
    train_config_path = model_path.parent / "train_config.json"
    train_config_data = None
    if train_config_path.exists():
        with open(train_config_path, 'r') as f:
            train_config_data = json.load(f)
        print(f"Found train_config.json at {train_config_path}")
    
    # Load model checkpoint (load to CPU first, then move to target device)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract model config - prefer train_config.json over checkpoint
    if train_config_data and "model" in train_config_data:
        model_config = train_config_data["model"].copy()
        print(f"Model config from train_config.json: {model_config}")
    elif "model_config" in checkpoint:
        model_config = checkpoint["model_config"].copy()
        print(f"Model config from checkpoint: {model_config}")
    else:
        raise ValueError(f"No model_config found in checkpoint {model_path} or train_config.json")
    
    model_config["device"] = args.device
    
    # Create model with all parameters
    model = MultiVariableImputer(
        num_attributes=model_config["num_attributes"],
        num_annotators=model_config["num_annotators"],
        num_items=model_config["num_items"],
        num_likert_classes=model_config["num_likert_classes"],
        max_rank_size=model_config.get("max_rank_size", 2),
        encoder_layers_num=model_config.get("encoder_layers_num", 6),
        attention_heads=model_config.get("attention_heads", 8),
        embedding_dim=model_config.get("embedding_dim", 128),
        dropout=model_config.get("dropout", 0.1),
        device=args.device,
        use_gelu_after_attention=model_config.get("use_gelu_after_attention", False),
        use_final_norm=model_config.get("use_final_norm", True),
        normalize_parameter=model_config.get("normalize_parameter", True),
        num_ffn_layers=model_config.get("num_ffn_layers", 4),
        temperature=model_config.get("temperature", 1.0),
        use_concat_embedding=model_config.get("use_concat_embedding", False),
        batch_size=1,
        enable_pointer_mechanism=True,
    )
    
    # Load state dict
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint must contain 'state_dict'")
    
    # Use load_state_dict_with_warnings if available, otherwise use regular load_state_dict
    if hasattr(model, 'load_state_dict_with_warnings'):
        missing_keys, unexpected_keys = model.load_state_dict_with_warnings(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: {len(missing_keys)} missing keys (not loaded)")
        if unexpected_keys:
            print(f"Warning: {len(unexpected_keys)} unexpected keys (ignored)")
    else:
        model.load_state_dict(state_dict, strict=False)
    
    model.to(args.device)
    print("Model loaded successfully")
    
    # Create converter
    converter = DataConverter(
        num_attributes=model_config["num_attributes"],
        num_annotators=model_config["num_annotators"],
        num_items=model_config["num_items"],
        num_likert_classes=model_config["num_likert_classes"],
        max_rank_size=model_config.get("max_rank_size", 2),
    )
    
    # Create variable sets
    train_observed = converter.create_variables_from_bundle(bundle, partition="train", status="observed")
    train_missing = converter.create_variables_from_bundle(bundle, partition="train", status="missing")
    test_observed = converter.create_variables_from_bundle(bundle, partition="test", status="observed")
    test_missing = converter.create_variables_from_bundle(bundle, partition="test", status="missing")
    
    train_all = train_observed + train_missing
    test_all = test_observed + test_missing
    
    print(f"Loaded data: {len(train_missing)} train_missing, {len(test_all)} test_all")
    
    # Generate predictives
    print("Generating train_predictives.json (train_missing)...")
    train_predictives = build_predictives(model, train_missing)
    save_predictives(output_dir, train_predictives, "train_predictives.json")
    print(f"Saved train_predictives.json to {output_dir}")
    
    print("Generating test_predictives.json (test_all)...")
    test_predictives = build_predictives(model, test_all)
    save_predictives(output_dir, test_predictives, "test_predictives.json")
    print(f"Saved test_predictives.json to {output_dir}")
    
    print("\n✓ Predictives generated successfully!")
    print(f"  - train_predictives.json: {len(train_predictives['predictions'])} predictions")
    print(f"  - test_predictives.json: {len(test_predictives['predictions'])} predictions")
    print(f"\nYou can now run JK diagnostics:")
    print(f"  python utils/jk_diagnostics.py \\")
    print(f"    --data-bundle {bundle_path} \\")
    print(f"    --imputer-predictives {output_dir} \\")
    print(f"    --slice train_missing")


if __name__ == "__main__":
    main()
