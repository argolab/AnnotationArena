"""Utility functions for imputer operations."""

from typing import Dict, Any, List
import torch
import numpy as np
from imputer.data import RankingData
from imputer.ranking_imputer import MultiVariableImputer


def convert_to_json_serializable(obj: Any) -> Any:
    """Convert PyTorch tensors and other non-serializable objects to native Python types."""
    if isinstance(obj, torch.Tensor):
        # Convert tensor to Python scalar or list
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif obj is None:
        return None
    else:
        # Try to convert to native type if possible
        try:
            if isinstance(obj, (int, float, str, bool)):
                return obj
            # Try to get item() if it has it (like numpy scalars)
            if hasattr(obj, 'item'):
                return obj.item()
        except:
            pass
        return obj


def sizes_from_configs(configs: Dict[str, Any]) -> Dict[str, int]:
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


def build_predictives(model: MultiVariableImputer, variables: List[RankingData]) -> Dict[str, Any]:
    """Create a serializable predictions dict for the given variables with probability distributions."""
    model.eval()
    with torch.no_grad():
        out = model(variables)
        rating_logits = out["rating"]
        ranking_logits = out["ranking"]

        # Compute probability distributions
        rating_probs = torch.softmax(rating_logits[0], dim=-1).cpu().numpy()  # [N, num_classes]

        preds: List[Dict[str, Any]] = []
        for i, var in enumerate(variables):
            entry: Dict[str, Any] = {
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


def calculate_rmse(predictions: List[int], targets: List[int]) -> float:
    """Calculate RMSE for rating predictions (on 1-5 scale)."""
    if len(predictions) == 0:
        return 0.0

    # Convert from 0-4 internal representation to 1-5 rating scale
    pred_ratings = [p + 1 for p in predictions]
    true_ratings = [t + 1 for t in targets]

    mse = np.mean([(p - t)**2 for p, t in zip(pred_ratings, true_ratings)])
    return np.sqrt(mse)
