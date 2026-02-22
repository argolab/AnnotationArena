import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Set

import torch
import numpy as np
import time

from imputer.data import DataConverter, RankingData
from imputer.ranking_imputer import MultiVariableImputer
from imputer.unified_entity_imputer import UnifiedEntityImputer
from imputer.callbacks import EvaluationCallback
from imputer.eval import EvaluationEngine
import sys
sys.path.insert(0, "..")
from stan.pipeline.bundle import GroundTruthBundle
from stan.pipeline.io import new_run_dir, save_test_metrics, save_predictives

from imputer.lightning_trainer import ImputerLightningModule, create_lightning_trainer
from imputer.utils import convert_to_json_serializable, sizes_from_configs, build_predictives


def main():
    parser = argparse.ArgumentParser(description="Run imputer training/evaluation on a data bundle")
    parser.add_argument("--data-dir", required=True, help="Directory containing data_bundle.json and configs.json")
    parser.add_argument("--output-root", default="OUTPUT/IMPUTER", help="Root output directory")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name (default: auto-generated timestamp)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--masking-rate", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay (L2 regularization, default: 0.01)")
    parser.add_argument("--device", default="cuda", help="Device: 'cuda' (uses all available GPUs automatically) or 'cpu'")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices to use (None = auto-detect all available, 1 = single device, default: None)")
    parser.add_argument("--max-rank-size", type=int, default=2)
    parser.add_argument("--transductive_learning", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save model checkpoints during training")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save checkpoint every N epochs (default: 5)")
    parser.add_argument("--save-model-every", type=int, default=None, help="Save model.pt every N epochs (default: None, only at end)")
    parser.add_argument("--save-best-model", action="store_true", help="Save best model when early stopping is enabled")
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
    parser.add_argument("--d-ff", type=int, default=512, help="FFN hidden dimension (default: 512)")

    # Unified Entity architecture
    parser.add_argument("--use-unified-entity", action="store_true", help="Use unified entity architecture")
    parser.add_argument("--use-prediction-head", action="store_true", help="Use separate MLP prediction head")
    parser.add_argument("--logit-high", type=float, default=20.0, help="Value for one-hot logit spike in param stream (default: 20.0)")
    parser.add_argument("--use-annotator-deviation", action="store_true", default=True, help="Per-annotator deviation embeddings (default: True)")
    parser.add_argument("--no-annotator-deviation", dest="use_annotator_deviation", action="store_false", help="Disable per-annotator deviation embeddings")
    parser.add_argument("--annotator-dropout", type=float, default=0.5, help="Dropout for annotator deviation embeddings (default: 0.5)")

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
    
    # Gradient clipping
    parser.add_argument("--gradient-clip-val", type=float, default=0.0, help="Gradient clipping value (0 = no clipping, default: 0)")
    
    # Learning rate scheduler
    parser.add_argument("--use-cosine-schedule", action="store_true", help="Use warmup + cosine annealing scheduler")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Number of warmup steps for scheduler (default: 100)")
    
    # General batching
    parser.add_argument("--batch-size", type=int, default=1, 
                        help="Batch size for training (1=no batching, >1=batch different masking patterns, default: 1)")
    parser.add_argument("--selective-masking", action="store_true",
                        help="Whether to mask all vars with (j, k) pairs in the synthetic masking examples")
    parser.add_argument("--max-item", type=int, default=1,
                        help="Max number of items in a single imputer forward pass")
    parser.add_argument("--llm-annotator-id", type=int, default=None,
                        help="0-indexed annotator ID of the LLM. When set, masks ALL human annotations and keeps only LLM as observed during training.")
    parser.add_argument("--human-observed-rate", type=float, default=0.0,
                        help="Fraction of human annotations to keep as observed (not masked) when --llm-annotator-id is set. Default 0.0 = mask all humans.")

    # Text embedding arguments (SentenceBERT item initialization)
    parser.add_argument("--use-text-embeddings", action="store_true",
                        help="Initialise item embeddings with precomputed text embeddings (e.g. SentenceBERT). "
                             "Only affects MultiVariableImputer. Disabled by default.")
    parser.add_argument("--text-embeddings-file", type=str, default=None,
                        help="Path to .pt file containing item text embeddings [K, text_dim]. "
                             "Defaults to {data_dir}/item_text_embeddings.pt when --use-text-embeddings is set.")

    args = parser.parse_args()

    # PyTorch Lightning DDP re-runs this script in each worker process.
    # Gate filesystem side-effects (mkdir/rmtree/writes/most prints) to rank 0 to avoid races and spam.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_rank0 = (local_rank == 0)

    if args.selective_masking:
        print("Use selective mask for mask examples, with JK pairs")
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
    sizes = sizes_from_configs(configs)

    # Load optional text embeddings (SentenceBERT or similar)
    item_text_embeddings = None
    if args.use_text_embeddings:
        text_emb_path = args.text_embeddings_file or str(data_dir / "item_text_embeddings.pt")
        print(f"Loading item text embeddings from {text_emb_path}")
        item_text_embeddings = torch.load(text_emb_path, map_location="cpu")
        assert item_text_embeddings.shape[0] == sizes["num_items"], (
            f"Text embeddings shape[0]={item_text_embeddings.shape[0]} "
            f"!= num_items={sizes['num_items']}"
        )
        text_embedding_dim = item_text_embeddings.shape[1]
        print(f"  Loaded text embeddings: {item_text_embeddings.shape}  (dim={text_embedding_dim})")

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

    # Build model
    if args.use_unified_entity:
        model = UnifiedEntityImputer(
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
            use_gelu_after_attention=args.use_gelu_after_attention,
            use_final_norm=args.use_final_norm,
            normalize_parameter=args.normalize_parameter,
            num_ffn_layers=args.num_ffn_layers,
            d_ff=args.d_ff,
            temperature=args.temperature,
            batch_size=args.batch_size,
            use_prediction_head=args.use_prediction_head,
            logit_high=args.logit_high,
            use_annotator_deviation=args.use_annotator_deviation,
            annotator_dropout=args.annotator_dropout,
        )
        print(f"Using UnifiedEntityImputer (logit_high={args.logit_high}, "
              f"prediction_head={args.use_prediction_head}, "
              f"annotator_deviation={args.use_annotator_deviation})")
    else:
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
            use_gelu_after_attention=args.use_gelu_after_attention,
            use_final_norm=args.use_final_norm,
            normalize_parameter=args.normalize_parameter,
            num_ffn_layers=args.num_ffn_layers,
            temperature=args.temperature,
            use_concat_embedding=bool(args.use_concat_embedding),
            batch_size=args.batch_size,
            enable_pointer_mechanism=True,
            item_text_embeddings=item_text_embeddings,
            text_embedding_dim=text_embedding_dim if item_text_embeddings is not None else 768,
        )

    # Print temperature scaling status
    if args.temperature != 1.0:
        print(f"Temperature scaling enabled: T = {args.temperature:.2f} (T > 1 softens predictions, T < 1 sharpens)")
    else:
        print("Temperature scaling disabled (T = 1.0)")
    
    # Print gradient clipping status
    if args.gradient_clip_val > 0:
        print(f"Gradient clipping enabled: clip_val = {args.gradient_clip_val}")
    else:
        print("Gradient clipping disabled")
    
    # Print scheduler status
    if args.use_cosine_schedule:
        print(f"Warmup + Cosine scheduler enabled: warmup_steps = {args.warmup_steps}")
    else:
        print("Learning rate scheduler disabled (constant learning rate)")
    
    # Print batching status
    if args.batch_size > 1:
        print(f"Batching enabled: batch_size = {args.batch_size}")
    else:
        print("Batching disabled (batch_size = 1)")

    print("=" * 60)
    print("Using PyTorch Lightning for training (with built-in TensorBoard)")
    print("=" * 60)
    
    # Trainer setup - will be configured after run_dir is created
    lightning_module = None
    lightning_trainer = None

    # Set up training variables
    if args.test_only_training:
        # Already set above: train_vars_for_training = test_observed
        train_vars = train_vars_for_training
    else:
        train_vars = train_vars_for_training
        if args.transductive_learning:
            raise ValueError("Transductive learning is not supported yet")
    
    # Create the main run directory first (before training)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Determine a single shared run_name across DDP ranks.
    # - If user passes --run-name, use it.
    # - Otherwise, only rank 0 generates a timestamped name and writes it for other ranks to read.
    run_name_path = output_root / ".imputer_current_run_name.txt"
    if is_rank0:
        run_name_final = args.run_name
        if run_name_final is None:
            run_name_final = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        run_name_path.write_text(run_name_final)
    else:
        # Wait for rank 0 to publish the chosen run name
        wait_s = 0.0
        while not run_name_path.exists():
            time.sleep(0.05)
            wait_s += 0.05
            if wait_s > 30.0:
                raise RuntimeError(
                    f"Timed out waiting for run name from rank 0 at {run_name_path}. "
                    f"Is the rank-0 process stuck?"
                )
        run_name_final = run_name_path.read_text().strip()

    # Handle --overwrite-existing-data flag and directory creation (rank 0 only)
    run_dir = output_root / run_name_final
    if is_rank0:
        if run_dir.exists() and args.overwrite_existing_data:
            print("\033[91mWARNING: Overwriting existing output directory: {}\033[0m".format(run_dir))
            shutil.rmtree(run_dir)
        # Create directory once; other ranks will wait for it to exist
        run_dir = new_run_dir(output_root, run_name=run_name_final)
    else:
        # Wait for run_dir to be created by rank 0
        wait_s = 0.0
        while not run_dir.exists():
            time.sleep(0.05)
            wait_s += 0.05
            if wait_s > 30.0:
                raise RuntimeError(
                    f"Timed out waiting for run directory from rank 0 at {run_dir}. "
                    f"Is the rank-0 process stuck?"
                )

    # Save train configuration snapshot immediately after creating run_dir
    # This ensures config is saved even if training fails early
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
            "embedding_type": getattr(model, 'embedding_type', 'atom'),
            "device": args.device,
            "include_sign_bit_in_params": True,
            "use_gelu_after_attention": args.use_gelu_after_attention,
            "use_final_norm": args.use_final_norm,
            "normalize_parameter": args.normalize_parameter,
            "num_ffn_layers": args.num_ffn_layers,
            "d_ff": args.d_ff,
            "use_concat_embedding": bool(args.use_concat_embedding),
            "temperature": args.temperature,
            "use_unified_entity": bool(args.use_unified_entity),
            "use_prediction_head": bool(args.use_prediction_head),
            "logit_high": args.logit_high,
            "use_annotator_deviation": bool(args.use_annotator_deviation),
            "annotator_dropout": args.annotator_dropout
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "masking_rate": args.masking_rate,
            "train_all_observed": bool(args.train_all_observed),
            "test_only_training": bool(args.test_only_training),
            "masked_loss_weight": args.masked_loss_weight,
            "observed_loss_weight": args.observed_loss_weight,
            "decay_observed_weight": bool(args.decay_observed_weight),
            "decay_observed_epochs": args.decay_observed_epochs if args.decay_observed_weight else None,
            "mask_augmentations": args.mask_augmentations,
            "transductive_learning": bool(args.transductive_learning),
            "save_checkpoints": bool(args.save_checkpoints),
            "checkpoint_every": args.checkpoint_every,
            "early_stopping": bool(args.early_stopping),
            "early_stopping_metric": args.early_stopping_metric if args.early_stopping else None,
            "early_stopping_patience": args.early_stopping_patience if args.early_stopping else None,
            "early_stopping_min_delta": args.early_stopping_min_delta if args.early_stopping else None,
            "gradient_clip_val": args.gradient_clip_val,
            "use_cosine_schedule": bool(args.use_cosine_schedule),
            "warmup_steps": args.warmup_steps if args.use_cosine_schedule else None,
            "use_lightning": True,
            "batch_size": args.batch_size,
        },
        "run": {
            "run_dir": str(run_dir)
        }
    }
    if is_rank0:
        with open(run_dir / "train_config.json", "w") as f:
            json.dump(train_config, f, indent=2)
        print(f"Saved train_config.json to {run_dir / 'train_config.json'}")
    
    # Initialize evaluation engine
    eval_engine = EvaluationEngine()
    
    # Set up Lightning trainer
    test_instance_callback = EvaluationCallback(
        eval_engine=eval_engine,
        test_variables=test_all,
        converter=converter,
        device=args.device,
        name="test_instance",
        instance_name="test_instance",
        max_item=args.max_item
    )
    train_instance_callback = None
    if train_all is not None:
        train_instance_callback = EvaluationCallback(
            eval_engine=eval_engine,
            test_variables=train_all,
            converter=converter,
            device=args.device,
            name="train_instance",
            instance_name="train_instance",
            max_item=args.max_item
        )
    lightning_module = ImputerLightningModule(
        model=model,
        train_observed_vars=train_vars,
        train_missing_vars=train_missing_for_trainer,
        test_variables=test_all,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        masking_rate=args.masking_rate,
        masked_loss_weight=args.masked_loss_weight,
        observed_loss_weight=args.observed_loss_weight,
        mask_augmentations=args.mask_augmentations,
        decay_observed_weight=args.decay_observed_weight,
        decay_observed_epochs=args.decay_observed_epochs,
        eval_engine=eval_engine,
        converter=converter,
        build_predictives_fn=build_predictives,
        run_dir=str(run_dir),
        use_cosine_schedule=args.use_cosine_schedule,
        warmup_steps=args.warmup_steps,
        max_epochs=args.epochs,
        train_instance_callback=train_instance_callback,
        test_instance_callback=test_instance_callback,
        selective_masking=args.selective_masking,
        max_item=args.max_item,
        llm_annotator_id=args.llm_annotator_id,
        human_observed_rate=args.human_observed_rate,
    )
    # Device / strategy configuration
    trainer_kwargs: Dict[str, Any] = {}
    if args.device == "cuda":
        accelerator = "gpu"
        # If --devices is specified, use it; otherwise let Lightning auto-detect
        if args.devices is not None:
            trainer_kwargs["devices"] = args.devices
    else:
        accelerator = "cpu"
        # CPU always uses 1 device
        trainer_kwargs["devices"] = 1

    lightning_trainer = create_lightning_trainer(
        run_dir=str(run_dir),
        max_epochs=args.epochs,
        checkpoint_every=args.checkpoint_every if args.save_checkpoints else None,
        accelerator=accelerator,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None,
        **trainer_kwargs
    )

    start_time = time.time()

    ###################################################################################################
    # Train
    lightning_trainer.fit(lightning_module)
    training_results = {
        'training_history': lightning_module.training_history,
        'callback_history': lightning_module.callback_history
    }
    # Get the trained model from the Lightning module
    model = lightning_module.model
    ###################################################################################################

    running_time = time.time() - start_time
    print(running_time)

    # Training history is saved via Lightning's _save_epoch_artifacts
    print(f"Training history saved to {run_dir}")

    # Evaluate (use same max_item as training so item-context matches)
    results = eval_engine.evaluate_model(model=model, variables=test_all, converter=converter, device=args.device, max_item=args.max_item)

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
        'embedding_type': getattr(model, 'embedding_type', 'atom'),
        'device': args.device,
        'use_unified_entity': bool(args.use_unified_entity),
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

    # Save predictives: train_predictives.json (train_missing) and test_predictives.json (test_all)
    # Use same max_item as training so item-context matches
    if not args.test_only_training and train_missing:
        train_predictives = build_predictives(model, train_missing, max_item=args.max_item)
        save_predictives(run_dir, train_predictives, "train_predictives.json")
    test_predictives = build_predictives(model, test_all, max_item=args.max_item)
    save_predictives(run_dir, test_predictives, "test_predictives.json")

    print(f"Saved model to {model_path}")
    print(f"Saved metrics and predictives (train_predictives.json, test_predictives.json) to {run_dir}")


if __name__ == "__main__":
    main()


