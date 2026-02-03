import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

import torch
import time

from imputer.data import DataConverter, RankingData
from imputer.ranking_imputer import MultiVariableImputer
from imputer.trainer import ImputerTrainer, EvaluationCallback, EarlyStopping
from imputer.eval import EvaluationEngine
import sys
sys.path.insert(0, "..")
from stan.pipeline.bundle import GroundTruthBundle
from stan.pipeline.io import new_run_dir, save_test_metrics, save_predictives


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
    parser = argparse.ArgumentParser(description="Run imputer training/evaluation on a data bundle")
    parser.add_argument("--data-dir", required=True, help="Directory containing data_bundle.json and configs.json")
    parser.add_argument("--output-root", default="OUTPUT/IMPUTER", help="Root output directory")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name (default: auto-generated timestamp)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--masking-rate", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rank-size", type=int, default=2)
    parser.add_argument("--transductive_learning", action="store_true")
    parser.add_argument("--full_random", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save model checkpoints during training")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--train-all-observed", action="store_true",
                        help="Convert all training missing to observed for artificial masking (more training data)")
    parser.add_argument("--test-only-training", action="store_true",
                        help="Train only on test_observed, evaluate on test_missing (ignore training set completely)")

    # Model architecture arguments
    parser.add_argument("--encoder-layers", type=int, default=6, help="Number of transformer encoder layers (default: 6)")
    parser.add_argument("--attention-heads", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension (default: 128)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--num_ffn_layers", type=int, default=4, help="FFN Layers")

    # Loss weighting arguments
    parser.add_argument("--masked-loss-weight", type=float, default=8.0, help="Weight for masked entry loss (default: 8.0)")
    parser.add_argument("--observed-loss-weight", type=float, default=1.0, help="Weight for observed entry loss (default: 1.0)")
    parser.add_argument("--decay-observed-weight", action="store_true", help="Enable linear decay of observed loss weight to 0")
    parser.add_argument("--decay-observed-epochs", type=int, default=20, help="Number of epochs to decay observed weight over (default: 20)")

    # Architectural improvements
    parser.add_argument("--use-gelu-after-attention", action="store_true", help="Apply GeLU activation after attention (before residual)")
    parser.add_argument("--use-final-norm", action="store_true", default=True, help="Apply final LayerNorm after all transformer blocks (default: True, recommended for Pre-LN)")
    parser.add_argument("--no-final-norm", dest="use_final_norm", action="store_false", help="Disable final LayerNorm (not recommended)")
    parser.add_argument("--mask-augmentations", type=int, default=1, help="Number of different masking patterns per epoch (default: 1, no augmentation)")
    parser.add_argument("--normalize-parameter", action="store_true", default=False, help="Whether to apply norm to parameter")
    parser.add_argument(
        "--use-concat-embedding",
        action="store_true",
        default=False,
        help="Use concatenation-based AtomCompositional embedding instead of projection mixing",
    )

    # Temperature scaling for calibration
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for scaling logits (T > 1 softens predictions, default: 1.0)")

    # Early stopping arguments
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping based on test missing metrics")
    parser.add_argument("--early-stopping-metric", type=str, default="loss", choices=["loss", "accuracy"], help="Metric to monitor for early stopping: 'loss' (rating_loss) or 'accuracy' (rating_accuracy) (default: loss)")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Number of epochs with no improvement before stopping (default: 10)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4, help="Minimum change to qualify as improvement (default: 1e-4)")
    parser.add_argument("--overwrite-existing-data", action="store_true", help="Overwrite existing output directory if it exists")

    parser.add_argument("--include-train-observed", type=bool, default=False, help="Whether to include train observed in the marformer when evaluating")

    args = parser.parse_args()

    # Check for mutually exclusive flags
    if args.test_only_training and args.transductive_learning:
        raise ValueError("--test-only-training and --transductive-learning are mutually exclusive")

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

    def print_counts(name, data):
        total = len(data)
        rating_count = sum(1 for var in data if not var.is_listwise)
        ranking_count = sum(1 for var in data if var.is_listwise)
        print(f"{name}: total={total} (ratings={rating_count}, rankings={ranking_count})")

    print_counts("train_observed", train_observed)
    print_counts("train_missing", train_missing)
    print_counts("test_observed", test_observed)
    print_counts("test_missing", test_missing)

    # Test-only training mode: use test set for training, ignore training set
    if args.test_only_training:
        print("Test-only training mode: Training on test_observed, evaluating on test_missing")
        print("Training set will be completely ignored")
        train_vars_for_training = test_observed
        train_missing_for_trainer = test_missing
        train_all = None  # Not used in test-only mode
    else:
        # EXPERIMENTAL: Optionally convert all training missing to observed for fully observed training
        # This allows us to artificially mask more of the training data
        if args.train_all_observed:
            print("\033[91mConverting all training missing to observed (train-all-observed mode)\033[0m")
            train_missing_as_observed = []
            for var in train_missing:
                # Create a copy with status=2 (observed) instead of status=0 (missing)
                train_missing_as_observed.append(RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=var.is_listwise,
                    item_ids=var.item_ids,
                    status=2,  # observed instead of missing
                    instance=var.instance,
                    rating_value=var.rating_value,
                    ranking_order=var.ranking_order,
                ))
            train_observed_full = train_observed + train_missing_as_observed
            train_missing_for_trainer = []  # Empty since we converted them to observed
        else:
            print("Using standard training (only originally observed data for masking)")
            train_observed_full = train_observed
            train_missing_for_trainer = train_missing

        train_vars_for_training = train_observed_full
        train_all = train_observed + train_missing

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
        encoder_layers_num=args.encoder_layers,
        attention_heads=args.attention_heads,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        randomness=random,
        use_gelu_after_attention=args.use_gelu_after_attention,
        use_final_norm=args.use_final_norm,
        normalize_parameter=args.normalize_parameter,
        num_ffn_layers=args.num_ffn_layers,
        temperature=args.temperature,
        use_concat_embedding=bool(args.use_concat_embedding),
    )

    # Print temperature scaling status
    if args.temperature != 1.0:
        print(f"Temperature scaling enabled: T = {args.temperature:.2f} (T > 1 softens predictions, T < 1 sharpens)")
    else:
        print("Temperature scaling disabled (T = 1.0)")

    # Trainer
    trainer = ImputerTrainer(
        model=model,
        learning_rate=args.lr,
        device=args.device,
        masked_loss_weight=args.masked_loss_weight,
        observed_loss_weight=args.observed_loss_weight,
        checkpoint_dir=None,  # Will be set later if checkpoints are enabled
        save_checkpoints=False,  # Will be set later if checkpoints are enabled
    )

    # Register evaluation callback on test set (runs each epoch)
    eval_engine = EvaluationEngine()
    if args.include_train_observed:
        trainer.register_callback(
            EvaluationCallback(
                eval_engine=eval_engine,
                test_variables=train_observed+test_all,
                converter=converter,
                device=args.device,
                name="test_all_evaluation",
                train_indices=[i for i in range(len(train_observed))]
            )
        )
    else:
        trainer.register_callback(
            EvaluationCallback(
                eval_engine=eval_engine,
                test_variables=test_all,
                converter=converter,
                device=args.device,
                name="test_all_evaluation",
                train_indices=None
            )
        )

    # Only register train_all evaluation if not in test-only mode
    if not args.test_only_training:
        trainer.register_callback(
            EvaluationCallback(
                eval_engine=eval_engine,
                test_variables=train_all,
                converter=converter,
                device=args.device,
                name="train_all_evaluation",
            )
        )

    # Set up training variables
    if args.test_only_training:
        # Already set above: train_vars_for_training = test_observed
        train_vars = train_vars_for_training
    else:
        train_vars = train_vars_for_training
        if args.transductive_learning:
            print("Using transductive learning")
            train_vars += test_observed
    
    # Create the main run directory first (before training)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Handle --overwrite-existing-data flag: remove existing directory if it exists
    if args.run_name:
        potential_run_dir = output_root / args.run_name
        if potential_run_dir.exists() and args.overwrite_existing_data:
            print("\033[91mWARNING: Overwriting existing output directory: {}\033[0m".format(potential_run_dir))
            shutil.rmtree(potential_run_dir)
    
    run_dir = new_run_dir(output_root, run_name=args.run_name)

    # Save train configuration snapshot next to outputs
    train_config = {
        "data": {
            "data_dir": str(data_dir),
            "bundle_path": str(bundle_path),
            "configs_path": str(configs_path),
        },
        "resolved_sizes": sizes,
        "model": {
            "num_attributes": sizes["num_attributes"],
            "num_annotators": sizes["num_annotators"],
            "num_items": sizes["num_items"],
            "num_likert_classes": sizes["num_likert_classes"],
            "max_rank_size": args.max_rank_size,
            "encoder_layers_num": len(model.blocks),
            "attention_heads": model.blocks[0].attention_heads,
            "embedding_dim": model.embedding_dim,
            "dropout": args.dropout,
            "embedding_type": "atom",
            "device": args.device,
            "include_sign_bit_in_params": True,
            "use_gelu_after_attention": args.use_gelu_after_attention,
            "use_final_norm": args.use_final_norm,
            "temperature": args.temperature
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "masking_rate": args.masking_rate,
            "train_all_observed": bool(args.train_all_observed),
            "test_only_training": bool(args.test_only_training),
            "masked_loss_weight": args.masked_loss_weight,
            "observed_loss_weight": args.observed_loss_weight,
            "decay_observed_weight": bool(args.decay_observed_weight),
            "decay_observed_epochs": args.decay_observed_epochs if args.decay_observed_weight else None,
            "mask_augmentations": args.mask_augmentations,
            "transductive_learning": bool(args.transductive_learning),
            "full_random": bool(args.full_random),
            "save_checkpoints": bool(args.save_checkpoints),
            "checkpoint_every": args.checkpoint_every,
            "early_stopping": bool(args.early_stopping),
            "early_stopping_metric": args.early_stopping_metric if args.early_stopping else None,
            "early_stopping_patience": args.early_stopping_patience if args.early_stopping else None,
            "early_stopping_min_delta": args.early_stopping_min_delta if args.early_stopping else None,
        },
        "run": {
            "run_dir": str(run_dir)
        }
    }
    with open(run_dir / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=2)
    
    # Set up checkpoint directory if saving is enabled (using separate folder with _checkpoints suffix)
    if args.save_checkpoints:
        checkpoint_run_dir = Path(str(run_dir) + "_checkpoints")
        checkpoint_run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = str(checkpoint_run_dir)
        trainer.checkpoint_dir = checkpoint_dir
        trainer.save_checkpoints = True
        print(f"Checkpoint saving enabled. Checkpoints will be saved to: {checkpoint_dir}")

    # Set up early stopping if enabled
    early_stopping = None
    if args.early_stopping:
        mode = "min" if args.early_stopping_metric == "loss" else "max"
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            mode=mode
        )
        print(f"Early stopping enabled:")
        print(f"  Metric: test_missing_{args.early_stopping_metric}")
        print(f"  Mode: {mode} (patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta})")

    start_time = time.time()

    ###################################################################################################
    # Train
    training_results = trainer.train(
        train_observed_vars=train_vars,
        train_missing_vars=train_missing_for_trainer,
        masking_rate=args.masking_rate,
        epochs=args.epochs,
        call_callbacks_every=1,
        save_checkpoints_every=args.checkpoint_every,
        verbose=True,
        mask_augmentations=args.mask_augmentations,
        early_stopping=early_stopping,
        early_stopping_metric=args.early_stopping_metric,
        decay_observed_weight=args.decay_observed_weight,
        decay_observed_epochs=args.decay_observed_epochs,
    )
    ###################################################################################################

    running_time = time.time() - start_time
    print(running_time)

    # Save training history - separate test and train metrics
    callback_history = training_results.get('callback_history', [])
    test_history = [entry for entry in callback_history if entry.get('name') == 'test_all_evaluation']

    with open(run_dir / "test_training_history.json", "w") as f:
        json.dump(test_history, f, indent=2)

    # Only save train_training_history if not in test-only mode
    if not args.test_only_training:
        train_history = [entry for entry in callback_history if entry.get('name') == 'train_all_evaluation']
        with open(run_dir / "train_training_history.json", "w") as f:
            json.dump(train_history, f, indent=2)

    print(f"Saved training history to {run_dir}")

    # Evaluate
    if args.include_train_observed:
        results = eval_engine.evaluate_model(model=model, variables=train_vars+test_all, converter=converter, device=args.device, train_indices=[i for i in range(len(train_vars))])
    else:
        results = eval_engine.evaluate_model(model=model, variables=test_all, converter=converter, device=args.device)
    # Output (using the same run_dir created earlier)

    # Save model - extract config directly from the trained model
    model_path = run_dir / "model.pt"
    model_config = {
        'num_attributes': model.num_attributes,
        'num_annotators': model.num_annotators,
        'num_items': model.num_items,
        'num_likert_classes': model.num_likert_classes,
        'max_rank_size': model.max_rank_size,
        'encoder_layers_num': len(model.blocks),
        'attention_heads': model.blocks[0].attention_heads,
        'embedding_dim': model.embedding_dim,
        'dropout': model.blocks[0].dropout_1.p,
        'embedding_type': model.embedding_type,
        'device': args.device
    }
    print(f"Saving model with config: {model_config}")
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": model_config
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
    save_test_metrics(run_dir, metrics_obj)

    # Save predictives on test
    predictives = _build_predictives(model, test_all)
    save_predictives(run_dir, predictives)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics and predictives to {run_dir}")


if __name__ == "__main__":
    main()


