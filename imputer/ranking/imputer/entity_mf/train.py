"""
Entity Marformer training module.

Usage:
python -m imputer.entity_mf.train   --data-dir path/to/your/bundle_dir   --epochs 50   --lr 1e-4   --weight-decay 0.01   --device cuda   --masking-rate 0.15
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Any

import torch
import pytorch_lightning as pl

from imputer.data import DataConverter, RankingData
from stan.pipeline.bundle import GroundTruthBundle
from imputer.utils import sizes_from_configs

from .config import EntityMarformerConfig
from .types import build_default_domain3_types
from .data import bundle_to_entity_graph
from .model import EntityMarformer
from .eval import LossStat, compute_loss_stat
from .masking import MaskingStrategy, build_default_masking_strategy


class EntityMarformerLightningModule(pl.LightningModule):
    """
    Minimal Lightning wrapper around EntityMarformer.

    Each step we apply random training masking to observed vars, build the graph,
    run forward, and compute per-type loss + deviation reg.
    """

    def __init__(
        self,
        model: EntityMarformer,
        train_observed: List[RankingData],
        train_missing: List[RankingData],
        bundle: GroundTruthBundle,
        types: Dict[str, Any],
        masking_rate: float = 0.15,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.train_observed = train_observed
        self.train_missing = train_missing
        self.bundle = bundle
        self.types = types
        self.masking_rate = masking_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.masking_strategy: MaskingStrategy = build_default_masking_strategy(masking_rate)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def train_dataloader(self):
        """Dummy dataloader; data lives in train_observed + train_missing, graph built each step."""
        ds = torch.utils.data.TensorDataset(torch.zeros(1))
        return torch.utils.data.DataLoader(ds, batch_size=1)

    def training_step(self, batch, batch_idx):
        """
        Single training step: apply random masking, build graph (full instance), forward,
        then loss breakdown: only masked loss for backprop; observed/missing for metrics.
        """
        device = self.device

        # Apply training mask
        masked_or_observed = self.masking_strategy.mask(self.train_observed)
        train_vars = masked_or_observed + self.train_missing  # full instance: observed + masked + missing
        graph = bundle_to_entity_graph(self.bundle, train_vars, self.types)
        # Full instance: all variable tokens (observed, masked, missing) + entity tokens are in the graph
        expected_var_tokens = len(train_vars)
        num_var_tokens = sum(1 for t in graph.tokens if t.type_name in ("rating", "ranking_pairwise"))
        assert num_var_tokens == expected_var_tokens, (
            f"Graph must contain all variable tokens: got {num_var_tokens}, expected {expected_var_tokens}"
        )

        params = self.model(graph, device=device)  # [1, L, P]

        loss_stat = compute_loss_stat(
            params, graph, self.model.types, self.model.global_param_dim, device
        ) # gather loss over entities

        # Deviation regularization (per-entity deviations)
        reg_loss = torch.zeros((), device=device)
        for type_name, t in self.model.types.items():
            if not t.variation.enabled or t.variation.reg_weight <= 0.0:
                continue
            table = self.model.deviation_tables.get(type_name, None)
            if table is None:
                continue
            reg_loss = reg_loss + t.variation.reg_weight * table.pow(2).sum()

        loss = loss_stat.trainable_loss + reg_loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # train/loss = trainable_loss (masked-only, sum over types) + reg_loss; breakdown below is mean loss per token by status (monitoring only, not backprop)
        self.log("train/trainable_loss", loss_stat.trainable_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/loss_masked", loss_stat.loss_masked, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/loss_observed", loss_stat.loss_observed, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/loss_missing", loss_stat.loss_missing, prog_bar=True, on_step=False, on_epoch=True)
        if reg_loss.requires_grad and reg_loss.item() != 0:
            self.log("train/reg_loss", reg_loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss


def load_bundle_and_converter(data_dir: Path) -> tuple[GroundTruthBundle, DataConverter, Dict[str, int]]:
    bundle_path = data_dir / "data_bundle.json"
    configs_path = data_dir / "configs.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"data_bundle.json not found in {data_dir}")
    if not configs_path.exists():
        raise FileNotFoundError(f"configs.json not found in {data_dir}")

    import json

    with open(bundle_path, "r") as f:
        bundle_dict = json.load(f)
    bundle = GroundTruthBundle.from_dict(bundle_dict)

    with open(configs_path, "r") as f:
        configs = json.load(f)

    sizes = sizes_from_configs(configs)

    # max_rank_size: use model_config if present (e.g. from run_imputer); else default 2 (pairwise)
    model_cfg = configs.get("model_config") or {}
    max_rank_size = int(model_cfg.get("max_rank_size", 2))

    converter = DataConverter(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=max_rank_size,
    )

    sizes["max_rank_size"] = max_rank_size
    return bundle, converter, sizes


def build_entity_marformer_from_bundle(
    bundle: GroundTruthBundle,
    converter: DataConverter,
    sizes: Dict[str, int],
    config: EntityMarformerConfig,
) -> tuple[EntityMarformer, Any]:
    # Reuse DataConverter to create RankingData variables for train partition.
    train_observed: List[RankingData] = converter.create_variables_from_bundle(
        bundle, partition="train", status="observed"
    )
    train_missing: List[RankingData] = converter.create_variables_from_bundle(
        bundle, partition="train", status="missing"
    )

    # Combine for now; masking behavior can be refined later.
    train_all = train_observed + train_missing

    types = build_default_domain3_types(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=converter.max_rank_size,
        logit_high=config.logit_high,
    )

    # Global param dimension: 1 + max(C, R), matching existing design.
    global_param_dim = 1 + max(sizes["num_likert_classes"], converter.max_rank_size)

    graph = bundle_to_entity_graph(bundle, train_all, types)
    model = EntityMarformer(
        config=config,
        types=types,
        global_param_dim=global_param_dim,
        num_relationships=graph.num_relationships,
    )
    return model, graph


def main():
    parser = argparse.ArgumentParser(description="Minimal trainer for Entity Marformer (entity_mf).")
    parser.add_argument("--data-dir", required=True, help="Directory containing data_bundle.json and configs.json")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--masking-rate", type=float, default=0.15, help="Fraction of observed vars to mask each step (like imputer)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    bundle, converter, sizes = load_bundle_and_converter(data_dir)

    train_observed = converter.create_variables_from_bundle(bundle, partition="train", status="observed")
    train_missing = converter.create_variables_from_bundle(bundle, partition="train", status="missing")

    config = EntityMarformerConfig()
    types = build_default_domain3_types(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=sizes.get("max_rank_size", converter.max_rank_size),
        logit_high=config.logit_high,
    )
    global_param_dim = 1 + max(sizes["num_likert_classes"], converter.max_rank_size)
    # Build one graph to get num_relationships and init model
    train_all = train_observed + train_missing
    graph0 = bundle_to_entity_graph(bundle, train_all, types)
    model = EntityMarformer(
        config=config,
        types=types,
        global_param_dim=global_param_dim,
        num_relationships=graph0.num_relationships,
    )

    lightning_module = EntityMarformerLightningModule(
        model=model,
        train_observed=train_observed,
        train_missing=train_missing,
        bundle=bundle,
        types=types,
        masking_rate=args.masking_rate,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    accelerator = "gpu" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator=accelerator, devices=1)
    trainer.fit(lightning_module)


if __name__ == "__main__":
    main()

