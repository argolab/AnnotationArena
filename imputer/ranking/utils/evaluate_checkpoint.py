#!/usr/bin/env python3
"""
Evaluate a Marformer checkpoint and generate visualization files.

Usage:
    python utils/evaluate_checkpoint.py \
        --model-path OUTPUT/IMPUTER/run_name_marformer/model_epoch_0065.pt \
        --data-dir OUTPUT/generated_data/run_name \
        [--output-dir OUTPUT/IMPUTER/run_name_marformer] \
        [--device cuda]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from imputer.data import DataConverter, RankingData
from imputer.ranking_imputer import MultiVariableImputer
from imputer.eval import EvaluationEngine
from stan.pipeline.bundle import GroundTruthBundle
from stan.pipeline.io import save_test_metrics, save_predictives


def _sizes_from_configs(configs: dict) -> dict:
    """Extract sizes from configs.json (datagen section)."""
    if "datagen" not in configs:
        raise ValueError("configs.json missing 'datagen' section")
    dg = configs["datagen"]
    required = ["K_train", "K_test", "I", "J", "C"]
    missing = [k for k in required if k not in dg]
    if missing:
        raise ValueError(f"configs.datagen missing keys: {missing}")
    return {
        "num_items": int(dg["K_train"]) + int(dg["K_test"]),
        "num_attributes": int(dg["I"]),
        "num_annotators": int(dg["J"]),
        "num_likert_classes": int(dg["C"]),
    }


def _build_predictives(model: MultiVariableImputer, variables: list[RankingData]) -> dict:
    """Create a serializable predictions dict for the given variables with probability distributions."""
    model.eval()
    with torch.no_grad():
        out = model(variables)
        rating_logits = out["rating"]
        ranking_logits = out["ranking"]

        # Compute probability distributions
        rating_probs = torch.softmax(rating_logits[0], dim=-1).cpu().numpy()  # [N, num_classes]

        preds: list[dict] = []
        for i, var in enumerate(variables):
            entry: dict = {
                "attribute": var.attribute_id,
                "annotator": var.annotator_id,
                "items": var.item_ids,
                "is_listwise": var.is_listwise,
                "status": var.status,
                "instance": var.instance,
            }
            if not var.is_listwise:
                # Add argmax prediction
                entry["predicted_rating_class"] = int(torch.argmax(rating_logits[0, i]).item())
                # Add full probability distribution
                entry["rating_probabilities"] = rating_probs[i].tolist()
                if var.rating_value is not None:
                    entry["true_rating_class"] = int(var.rating_value)
            else:
                scores = ranking_logits[0, i].cpu().numpy()
                # Add raw logits for ranking
                entry["ranking_logits"] = scores.tolist()
                if (var.ranking_order or []) and len(var.ranking_order) == 2:
                    import numpy as np
                    probs = np.exp(scores[:2]) / np.exp(scores[:2]).sum()
                    pred_first_wins = probs[0] > probs[1]
                    pred_ranking = [1, 2] if pred_first_wins else [2, 1]
                    # Add pairwise probabilities
                    entry["ranking_probabilities"] = probs.tolist()
                else:
                    pred_ranking = var.ranking_order
                entry["predicted_ranking"] = pred_ranking
                if var.ranking_order is not None:
                    entry["true_ranking"] = var.ranking_order
            preds.append(entry)
    model.train()
    return {"predictions": preds}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Marformer checkpoint and generate visualization files")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--data-dir", required=True, help="Directory containing data_bundle.json and configs.json")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as model directory)")
    parser.add_argument("--device", default="cuda", help="Device to use (default: cuda)")
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    data_dir = Path(args.data_dir)
    bundle_path = data_dir / "data_bundle.json"
    configs_path = data_dir / "configs.json"
    
    if not bundle_path.exists():
        raise FileNotFoundError(f"Data bundle not found: {bundle_path}")
    if not configs_path.exists():
        raise FileNotFoundError(f"Configs file not found: {configs_path}")
    
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
    
    # Try to load train_config.json from model directory (more reliable than checkpoint)
    train_config_path = model_path.parent / "train_config.json"
    train_config_data = None
    if train_config_path.exists():
        with open(train_config_path, 'r') as f:
            train_config_data = json.load(f)
        print(f"Found train_config.json at {train_config_path}")
    
    # Load configs
    with open(configs_path, 'r') as f:
        configs = json.load(f)
    sizes = _sizes_from_configs(configs)
    
    # Load bundle
    with open(bundle_path, 'r') as f:
        bundle_data = json.load(f)
    bundle = GroundTruthBundle.from_dict(bundle_data)
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=args.device)
    
    # Extract model config - prefer train_config.json over checkpoint
    if train_config_data and "model" in train_config_data:
        model_config = train_config_data["model"].copy()
        print(f"Model config from train_config.json: {model_config}")
    elif "model_config" in checkpoint:
        model_config = checkpoint["model_config"].copy()
        print(f"Model config from checkpoint: {model_config}")
    else:
        # Fallback: use sizes from configs
        model_config = {
            "num_attributes": sizes["num_attributes"],
            "num_annotators": sizes["num_annotators"],
            "num_items": sizes["num_items"],
            "num_likert_classes": sizes["num_likert_classes"],
            "max_rank_size": 2,  # Default
            "embedding_dim": 64,  # Default, may need adjustment
            "encoder_layers_num": 6,  # Default
            "attention_heads": 8,  # Default
            "dropout": 0.1,  # Default
            "embedding_type": "atom",
            "device": args.device
        }
        print(f"Warning: Using default model config: {model_config}")
    
    # Ensure device is set
    model_config["device"] = args.device
    
    # Infer architecture from state_dict if available
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint must contain 'state_dict' or 'model_state_dict'")
    
    # Infer architecture parameters from state_dict (for older checkpoints)
    # Check if param_scale exists (indicates normalize_parameter=False in current codebase)
    has_param_scale = any("param_scale" in k for k in state_dict.keys())
    # Check if proj_in/proj_out exist (may indicate older codebase version)
    has_proj = any("proj_in" in k for k in state_dict.keys())
    
    # Infer model_dim from Q/K/V weights
    if "blocks.0.Q.weight" in state_dict:
        checkpoint_model_dim = state_dict["blocks.0.Q.weight"].shape[0]
        print(f"Inferred model_dim from checkpoint: {checkpoint_model_dim}")
    else:
        checkpoint_model_dim = None
    
    # Use train_config values as primary source, infer only if missing
    if "num_ffn_layers" not in model_config:
        # Infer num_ffn_layers from FFN structure
        ffn_keys = [k for k in state_dict.keys() if "ff.net" in k]
        if ffn_keys:
            # Count unique layer indices in ff.net.X
            layer_indices = set()
            for k in ffn_keys:
                parts = k.split(".")
                for i, part in enumerate(parts):
                    if part == "net" and i + 1 < len(parts):
                        try:
                            layer_indices.add(int(parts[i + 1]))
                        except ValueError:
                            pass
            inferred_num_ffn_layers = max(layer_indices) // 3 + 1 if layer_indices else 4
            print(f"Inferred num_ffn_layers: {inferred_num_ffn_layers}")
            model_config["num_ffn_layers"] = inferred_num_ffn_layers
        else:
            model_config["num_ffn_layers"] = 4
    
    # Set defaults for missing keys
    model_config.setdefault("normalize_parameter", True)
    model_config.setdefault("use_final_norm", True)
    model_config.setdefault("use_gelu_after_attention", False)
    model_config.setdefault("temperature", 1.0)
    model_config.setdefault("use_concat_embedding", False)
    
    print(f"Final model config: {model_config}")
    
    # Create model with inferred config
    model = MultiVariableImputer(
        num_attributes=model_config["num_attributes"],
        num_annotators=model_config["num_annotators"],
        num_items=model_config["num_items"],
        embedding_dim=model_config["embedding_dim"],
        num_likert_classes=model_config["num_likert_classes"],
        max_rank_size=model_config.get("max_rank_size", 2),
        encoder_layers_num=model_config.get("encoder_layers_num", 6),
        attention_heads=model_config.get("attention_heads", 8),
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
    
    # Load model state with strict=False to handle architecture mismatches
    print("Loading model state_dict (strict=False to handle architecture differences)...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys (not loaded). First few: {missing_keys[:5]}")
        if len(missing_keys) > 5:
            print(f"  ... and {len(missing_keys) - 5} more")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys (ignored). First few: {unexpected_keys[:5]}")
        if len(unexpected_keys) > 5:
            print(f"  ... and {len(unexpected_keys) - 5} more")
    
    if missing_keys or unexpected_keys:
        print("\nNote: Architecture mismatches are expected if checkpoint was saved with a different codebase version.")
        print("The model will use loaded weights where possible and random initialization for missing parameters.")
    
    model.to(args.device)
    print("Model loaded successfully")
    
    # Create converter with proper parameters (matching run_imputer.py)
    converter = DataConverter(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=model_config.get("max_rank_size", 2),
    )
    
    # Optional validation
    errors = converter.validate_bundle(bundle)
    if errors:
        print(f"Warning: Bundle validation errors: {errors}")
    
    # Create different variable sets - must specify exact partition and status
    train_observed = converter.create_variables_from_bundle(bundle, partition="train", status="observed")
    test_observed = converter.create_variables_from_bundle(bundle, partition="test", status="observed")
    train_missing = converter.create_variables_from_bundle(bundle, partition="train", status="missing")
    test_missing = converter.create_variables_from_bundle(bundle, partition="test", status="missing")
    
    train_all = train_observed + train_missing
    test_all = test_observed + test_missing
    
    print(f"Loaded data: {len(train_all)} train variables, {len(test_all)} test variables")
    
    # Evaluate model
    print("Evaluating model...")
    eval_engine = EvaluationEngine()
    results = eval_engine.evaluate_model(
        model=model,
        variables=test_all,
        converter=converter,
        device=args.device
    )
    
    # Save metrics
    metrics_obj = {
        "total_loss": results.total_loss,
        "rating_loss": results.rating_loss,
        "ranking_loss": results.ranking_loss,
        "num_rating_evaluations": results.num_rating_evaluations,
        "num_ranking_evaluations": results.num_ranking_evaluations,
        "observed_metrics": results.observed_metrics,
        "missing_metrics": results.missing_metrics,
        "masked_metrics": results.masked_metrics,
    }
    save_test_metrics(output_dir, metrics_obj)
    print(f"Saved test metrics to {output_dir / 'test_metrics.json'}")
    
    # Save predictives
    print("Generating predictives...")
    predictives = _build_predictives(model, test_all)
    save_predictives(output_dir, predictives)
    print(f"Saved predictives to {output_dir / 'predictives.json'}")
    
    # Print summary
    print("\n=== EVALUATION RESULTS ===")
    print(f"Total Loss: {results.total_loss:.4f}")
    print(f"Rating Loss: {results.rating_loss:.4f}")
    print(f"Rating Accuracy: {results.rating_accuracy:.4f}")
    print(f"Rating RMSE: {results.rating_rmse:.4f}")
    print(f"\nMissing Metrics:")
    print(f"  Rating Loss: {results.missing_metrics.get('rating_loss', 'N/A'):.4f}")
    print(f"  Rating Accuracy: {results.missing_metrics.get('rating_accuracy', 'N/A'):.4f}")
    print(f"  Rating RMSE: {results.missing_metrics.get('rating_rmse', 'N/A'):.4f}")
    print(f"\nObserved Metrics:")
    print(f"  Rating Loss: {results.observed_metrics.get('rating_loss', 'N/A'):.4f}")
    print(f"  Rating Accuracy: {results.observed_metrics.get('rating_accuracy', 'N/A'):.4f}")
    print(f"  Rating RMSE: {results.observed_metrics.get('rating_rmse', 'N/A'):.4f}")
    
    print(f"\nFiles saved to: {output_dir}")
    print("You can now run visualization with:")
    print(f"  python3 utils/visualize.py --run-dir {output_dir}")


if __name__ == "__main__":
    main()

