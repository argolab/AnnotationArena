"""
Recurrent Entity Marformer training.

Usage:
  python -m imputer.entity_mf.recurrent.train --data-dir ... --prelude-depth 0 \\
      --num-core-layers 4 --num-recurrence 2 --coda-depth 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import LightningEnvironment

from imputer.entity_mf.types import build_default_domain3_types
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.lightning_module import EntityMarformerLightningModule
from imputer.entity_mf.train_utils import (
    add_common_training_args,
    apply_common_model_config,
    load_bundle_and_converter,
    new_run_dir,
    save_json,
    shared_model_config_dict,
    shared_training_config_dict,
)

from .config import RecurrentMarformerConfig
from .model import RecurrentEntityMarformer


def add_recurrent_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prelude-depth",
        type=int,
        default=None,
        help="Unique transformer blocks before the recurrent core.",
    )
    parser.add_argument(
        "--num-core-layers",
        type=int,
        default=None,
        help="Number of distinct parameter layers in the recurrent core.",
    )
    parser.add_argument(
        "--num-recurrence",
        type=int,
        default=None,
        help="Number of times the core stack is applied (unroll steps).",
    )
    parser.add_argument(
        "--coda-depth",
        type=int,
        default=None,
        help="Unique transformer blocks after the recurrent core.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training in an existing run directory from a checkpoint.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="last",
        help="Checkpoint for --resume: 'last', 'best', or a filename under checkpoints/.",
    )


def _resolve_resume_checkpoint(run_dir: Path, which: str) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if which == "last":
        path = ckpt_dir / "last.ckpt"
        if not path.exists():
            raise FileNotFoundError(f"last.ckpt not found in {ckpt_dir}")
        return path
    if which == "best":
        candidates = sorted(ckpt_dir.glob("best-*.ckpt"))
        if not candidates:
            raise FileNotFoundError(f"No best-*.ckpt found in {ckpt_dir}")
        return candidates[0]
    path = ckpt_dir / which
    if not path.exists():
        raise FileNotFoundError(f"{which} not found in {ckpt_dir}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Trainer for Recurrent Entity Marformer (entity_mf.recurrent)."
    )
    add_common_training_args(parser)
    add_recurrent_training_args(parser)
    parser.set_defaults(output_root="RESULTS/RECURRENT_MARFORMER")
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    data_dir = Path(args.data_dir)
    bundle, converter, sizes = load_bundle_and_converter(data_dir)

    config = RecurrentMarformerConfig()
    apply_common_model_config(config, args)
    if args.prelude_depth is not None:
        config.prelude_depth = args.prelude_depth
    if args.num_core_layers is not None:
        config.num_core_layers = args.num_core_layers
    if args.num_recurrence is not None:
        config.num_recurrence = args.num_recurrence
    if args.coda_depth is not None:
        config.coda_depth = args.coda_depth
    config.validate()

    types = build_default_domain3_types(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=sizes.get("max_rank_size", converter.max_rank_size),
        logit_high=config.logit_high,
        annotator_reg_weight=args.annotator_reg_weight,
        item_reg_weight=args.item_reg_weight,
        attribute_reg_weight=args.attribute_reg_weight,
        llm_input_dist=args.llm_input_dist,
        item_dropout_rate=args.item_dropout_rate,
        annotator_dropout_rate=args.annotator_dropout_rate,
    )

    train_observed_tmp = converter.create_variables_from_bundle(
        bundle, partition="train", status="observed"
    )
    train_missing_tmp = converter.create_variables_from_bundle(
        bundle, partition="train", status="missing"
    )
    graph0 = variable_list_to_entity_graph(train_observed_tmp + train_missing_tmp, types)
    model = RecurrentEntityMarformer(
        config=config,
        types=types,
        num_relationships=graph0.num_relationships,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    resume_ckpt: Path | None = None
    prior_history: list = []

    if args.resume:
        if not args.run_name:
            raise ValueError("--run-name is required when using --resume")
        run_dir = output_root / args.run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory not found: {run_dir}")
        resume_ckpt = _resolve_resume_checkpoint(run_dir, args.resume_checkpoint)
        history_path = run_dir / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                prior_history = json.load(f)
        print(f"Resuming {run_dir.name} from {resume_ckpt.name} ({len(prior_history)} history rows)")
    elif args.overwrite_existing_data and args.run_name:
        run_dir = output_root / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        try:
            run_dir = new_run_dir(output_root, run_name=args.run_name)
        except FileExistsError as e:
            raise RuntimeError(
                f"Run directory already exists for Recurrent Marformer: {e}"
            ) from e

    model_cfg = shared_model_config_dict(config, model.global_param_dim)
    model_cfg.update(
        {
            "prelude_depth": config.prelude_depth,
            "num_core_layers": config.num_core_layers,
            "num_recurrence": config.num_recurrence,
            "coda_depth": config.coda_depth,
            "effective_depth": config.effective_depth,
        }
    )
    train_config = {
        "model_type": "recurrent_entity_marformer",
        "data": {"data_dir": str(data_dir)},
        "resolved_sizes": sizes,
        "model": model_cfg,
        "training": shared_training_config_dict(args),
        "run": {"run_dir": str(run_dir), "seed": args.seed},
    }
    save_json(train_config, run_dir / "train_config.json")

    lightning_module = EntityMarformerLightningModule(
        model=model,
        bundle=bundle,
        converter=converter,
        types=types,
        masking_rate=args.masking_rate,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        llm_annotator_id=args.llm_annotator_id,
        human_observed_rate=args.human_observed_rate,
        always_observed_ids=args.always_observed_ids,
        max_item=args.max_item,
        run_dir=run_dir,
        transductive=bool(args.transductive_learning),
        transductive_valtest_mask=bool(args.transductive_valtest_mask),
        mask_augmentations=args.mask_augmentations,
        masked_loss_weight=args.masked_loss_weight,
        observed_loss_weight=args.observed_loss_weight,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
        lr_step_epoch=args.lr_step_epoch,
        random_item_chunks=bool(args.random_item_chunks),
        log_prefix="RecurrentMarformer",
    )
    if prior_history:
        lightning_module.training_history = list(prior_history)
        lightning_module._save_training_history_each_epoch = True

    accelerator = "gpu" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    logger = TensorBoardLogger(save_dir=str(run_dir), name="lightning_logs")
    checkpoint_best = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="best-{epoch:04d}",
        monitor="val/missing_ce",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    checkpoint_periodic = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="periodic-{epoch:04d}",
        every_n_epochs=25,
        save_top_k=-1,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_best, checkpoint_periodic],
        plugins=[LightningEnvironment()],
    )
    if resume_ckpt is not None:
        trainer.fit(lightning_module, ckpt_path=str(resume_ckpt))
    else:
        trainer.fit(lightning_module)


if __name__ == "__main__":
    main()
