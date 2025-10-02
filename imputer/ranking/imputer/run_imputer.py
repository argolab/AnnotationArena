import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
import time

from data import DataConverter, RankingData
from ranking_imputer import MultiVariableImputer
from trainer import ImputerTrainer, EvaluationCallback
from eval import EvaluationEngine
import sys
sys.path.insert(0, "..")
from stan.pipeline.bundle import GroundTruthBundle
from stan.pipeline.io import new_run_dir, save_metrics, save_predictives


def _sizes_from_configs(configs: Dict[str, Any]) -> Dict[str, int]:
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


def _build_predictives(model: MultiVariableImputer, variables: List[RankingData]) -> Dict[str, Any]:
    """Create a serializable predictions dict for the given variables."""
    model.eval()
    with torch.no_grad():
        out = model(variables)
        rating_logits = out["rating"]
        ranking_logits = out["ranking"]

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
                entry["predicted_rating_class"] = int(torch.argmax(rating_logits[0, i]).item())
                if var.rating_value is not None:
                    entry["true_rating_class"] = int(var.rating_value)
            else:
                scores = ranking_logits[0, i].cpu().numpy()
                if (var.ranking_order or []) and len(var.ranking_order) == 2:
                    import numpy as np
                    probs = np.exp(scores[:2]) / np.exp(scores[:2]).sum()
                    pred_first_wins = probs[0] > probs[1]
                    pred_ranking = [1, 2] if pred_first_wins else [2, 1]
                else:
                    pred_ranking = var.ranking_order
                entry["predicted_ranking"] = pred_ranking
                if var.ranking_order is not None:
                    entry["true_ranking"] = var.ranking_order
            preds.append(entry)
    model.train()
    return {"predictions": preds}


def main():
    parser = argparse.ArgumentParser(description="Run imputer training/evaluation on a data bundle")
    parser.add_argument("--data-dir", required=True, help="Directory containing data_bundle.json and configs.json")
    parser.add_argument("--output-root", default="OUTPUT/IMPUTER", help="Root output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--masking-rate", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rank-size", type=int, default=2)
    parser.add_argument("--transductive_learning", action="store_true")
    parser.add_argument("--full_random", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save model checkpoints during training")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    bundle_path = data_dir / "data_bundle.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"data_bundle.json not found in {data_dir}")
    configs_path = data_dir / "configs.json"
    if not configs_path.exists():
        raise FileNotFoundError(f"configs.json not found in {data_dir}")

    # Load bundle and configs
    with open(bundle_path, "r") as f:
        bundle_dict = json.load(f)
    bundle = GroundTruthBundle.from_dict(bundle_dict)
    with open(configs_path, "r") as f:
        configs = json.load(f)

    # Sizes from configs and build converter
    sizes = _sizes_from_configs(configs)
    converter = DataConverter(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=args.max_rank_size,
    )

    # Optional validation
    errors = converter.validate_bundle(bundle)
    if errors:
        raise ValueError(f"Bundle validation errors: {errors}")

    # Prepare variables
    train_observed = converter.create_variables_from_bundle(bundle, partition="train", status="observed")
    train_missing = converter.create_variables_from_bundle(bundle, partition="train", status="missing")
    test_observed = converter.create_variables_from_bundle(bundle, partition="test", status="observed")
    test_missing = converter.create_variables_from_bundle(bundle, partition="test", status="missing")
    train_all: List[RankingData] = train_observed + train_missing
    test_all: List[RankingData] = test_observed + test_missing
    if args.full_random:
        random = True
    else:
        random = False
    # Build model
    model = MultiVariableImputer(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=args.max_rank_size,
        device=args.device,
        encoder_layers_num=6,
        attention_heads=8,
        embedding_dim=128,
        randomness=random
    )

    # Trainer
    trainer = ImputerTrainer(
        model=model,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=None,  # Will be set later if checkpoints are enabled
        save_checkpoints=False,  # Will be set later if checkpoints are enabled
    )

    # Register evaluation callback on test set (runs each epoch)
    eval_engine = EvaluationEngine()
    trainer.register_callback(
        EvaluationCallback(
            eval_engine=eval_engine,
            test_variables=test_all,
            converter=converter,
            device=args.device,
            name="test_all_evaluation",
        )
    )
    trainer.register_callback(
        EvaluationCallback(
            eval_engine=eval_engine,
            test_variables=train_all,
            converter=converter,
            device=args.device,
            name="train_all_evaluation",
        )
    )
    train_vars = train_observed
    if args.transductive_learning:
        print("Using transductive learning")
        train_vars += test_observed
    
    # Set up checkpoint directory if saving is enabled (before training)
    if args.save_checkpoints:
        # Create a temporary run directory to get the checkpoint path
        temp_run_dir = new_run_dir(Path(args.output_root))
        checkpoint_dir = str(temp_run_dir / "checkpoints")
        trainer.checkpoint_dir = checkpoint_dir
        trainer.save_checkpoints = True
        print(f"Checkpoint saving enabled. Checkpoints will be saved to: {checkpoint_dir}")
    
    start_time = time.time()
    # Train
    trainer.train(
        train_observed_vars=train_vars,
        train_missing_vars=train_missing,
        masking_rate=args.masking_rate,
        epochs=args.epochs,
        call_callbacks_every=1,
        save_checkpoints_every=args.checkpoint_every,
        verbose=True,
    )

    running_time = time.time() - start_time
    print(running_time)

    # Evaluate
    results = eval_engine.evaluate_model(model=model, variables=test_all, converter=converter, device=args.device)

    # Output
    run_dir = new_run_dir(Path(args.output_root))

    # Save model
    model_path = run_dir / "model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "sizes": sizes,
        "max_rank_size": args.max_rank_size,
    }, model_path)

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
    save_metrics(run_dir, metrics_obj)

    # Save predictives on test
    predictives = _build_predictives(model, test_all)
    save_predictives(run_dir, predictives)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics and predictives to {run_dir}")


if __name__ == "__main__":
    main()


