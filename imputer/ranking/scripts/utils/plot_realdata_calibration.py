#!/usr/bin/env python3
"""
Reliability diagrams for real-data ranking results.

Outputs:
  - PLOTS/TALK/LLMRubric/ece_reliability_llm_rubric_size175.png
  - PLOTS/TALK/SummEval/ece_reliability_summeval_size1280.png
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import relplot as rp
rp.config.use_tex_fonts = False

from imputer.data import DataConverter, RankingData
from imputer.entity_mf.config import EntityMarformerConfig
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.model import EntityMarformer
from imputer.entity_mf.types import build_default_domain3_types

PLOTS_ROOT = ROOT / "PLOTS/TALK"
PROB_COL_TEMPLATE = "prob_cat_{idx}"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    pretty_name: str
    sizes: list[int]
    num_classes: int
    data_root: Path
    marformer_root: Path
    stan_root: Path
    baseline_roots: dict[str, Path]
    marformer_run: Callable[[int], str]
    stan_eval_run: Callable[[int, str], str]
    baseline_run: Callable[[int], str]
    out_dir: Path


def _read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _test_missing_indices_and_labels(bundle: dict) -> tuple[list[int], np.ndarray]:
    missing = bundle.get("missing_ratings", [])
    idxs = [i for i, row in enumerate(missing) if row.get("instance") == "test"]
    labels = np.asarray([missing[i]["value"] - 1 for i in idxs], dtype=np.int64)
    return idxs, labels


def _all_class_calibration(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.arange(probs.shape[1], dtype=np.int64)
    conf = probs.flatten()
    acc = (labels[:, None] == classes[None, :]).astype(np.float32).flatten()
    return conf, acc


def _draw_empty(ax: plt.Axes, title: str) -> None:
    ax.plot([0, 1], [0, 1], color="0.75", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.text(0.5, 0.5, "no data", ha="center", va="center", color="0.6", transform=ax.transAxes)
    ax.set_title(title)


def _plot_ece(ax: plt.Axes, probs: np.ndarray, labels: np.ndarray, title: str, color: str) -> None:
    conf, acc = _all_class_calibration(probs, labels)
    diag = rp.prepare_rel_diagram(
        conf,
        acc,
        num_bootstrap=500,
        report_CE=True,
        report_CE_std=True,
    )
    ce = diag.get("ce", float("nan"))
    rp.plot_rel_diagram(
        diag,
        fig=ax.get_figure(),
        ax=ax,
        color=color,
        use_default_style=True,
        plot_density_ticks=True,
        plot_labels=True,
        legend=False,
    )
    for txt in ax.texts:
        txt.remove()
    ci_w = diag.get("ce_ci_width", 0.0)
    ax.set_title(f"{title}\nsmECE = {ce:.3f} ± {ci_w:.3f}", fontsize=15, fontweight="bold", pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)


def _marformer_best_json(spec: DatasetSpec, size: int) -> Path | None:
    run_dir = spec.marformer_root / spec.marformer_run(size) / "TEST_RESULTS"
    preferred = run_dir / "best.json"
    if preferred.exists():
        return preferred
    candidates = sorted(run_dir.glob("best*.json"))
    return candidates[0] if candidates else None


def _load_marformer_model(run_dir: Path, ckpt_path: Path) -> tuple[EntityMarformer, DataConverter, dict]:
    cfg = _read_json(run_dir / "train_config.json")
    sizes = cfg["resolved_sizes"]
    model_cfg = cfg["model"]
    training = cfg["training"]

    converter = DataConverter(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=sizes["max_rank_size"],
    )

    config = EntityMarformerConfig(
        embedding_dim=model_cfg["embedding_dim"],
        num_layers=model_cfg["num_layers"],
        attention_heads=model_cfg["attention_heads"],
        d_ff=model_cfg["d_ff"],
        num_ffn_layers=model_cfg["num_ffn_layers"],
        dropout=model_cfg.get("dropout", 0.1),
        use_per_head_rel=model_cfg.get("use_per_head_rel", False),
        use_pointer=model_cfg.get("use_pointer", True),
        use_rel_value=model_cfg.get("use_rel_value", False),
        use_addone_attn=model_cfg.get("use_addone_attn", False),
        type_embedding_init=model_cfg.get("type_embedding_init", "kaiming"),
        use_deviation_norm=model_cfg.get("use_deviation_norm", False),
        scale_shared_rel=model_cfg.get("scale_shared_rel", True),
        use_graph_mask=model_cfg.get("use_graph_mask", False),
        logit_high=model_cfg.get("logit_high", 20.0),
    )

    types = build_default_domain3_types(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=sizes["max_rank_size"],
        logit_high=config.logit_high,
        llm_input_dist=training.get("llm_input_dist", False),
        item_dropout_rate=training.get("item_dropout_rate", 1.0),
        annotator_dropout_rate=training.get("annotator_dropout_rate", 0.0),
    )

    dummy_var = RankingData(
        annotator_id=0,
        attribute_id=0,
        is_listwise=False,
        item_ids=[0],
        status=2,
        instance="train",
        rating_value=0,
    )
    dummy_graph = variable_list_to_entity_graph([dummy_var], types)
    model = EntityMarformer(config=config, types=types, num_relationships=dummy_graph.num_relationships)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {key[len("model."):]: value for key, value in ckpt["state_dict"].items() if key.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, converter, cfg


def _load_marformer_probs(spec: DatasetSpec, size: int) -> tuple[np.ndarray, np.ndarray] | None:
    best_json = _marformer_best_json(spec, size)
    if best_json is None:
        return None

    run_dir = spec.marformer_root / spec.marformer_run(size)
    best = _read_json(best_json)
    ckpt_name = best.get("checkpoint")
    if ckpt_name is None:
        return None
    ckpt_path = run_dir / "checkpoints" / ckpt_name
    if not ckpt_path.exists():
        return None

    model, converter, cfg = _load_marformer_model(run_dir, ckpt_path)
    data_dir = ROOT / cfg["data"]["data_dir"]
    bundle = converter.load_bundle_data(str(data_dir / "data_bundle.json"))
    test_observed = converter.create_variables_from_bundle(bundle, partition="test", status="observed")
    test_missing = converter.create_variables_from_bundle(bundle, partition="test", status="missing")
    test_all = test_observed + test_missing
    if not test_all:
        return None

    num_classes = cfg["resolved_sizes"]["num_likert_classes"]
    max_item = int(cfg["training"].get("max_item", 10))
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []
    all_item_ids = sorted({item_id for var in test_all for item_id in var.item_ids})

    with torch.no_grad():
        for start in range(0, len(all_item_ids), max_item):
            item_set = set(all_item_ids[start:start + max_item])
            chunk_vars = [var for var in test_all if all(item_id in item_set for item_id in var.item_ids)]
            if not chunk_vars:
                continue
            graph = variable_list_to_entity_graph(chunk_vars, model.types)
            params = model(graph, device=torch.device("cpu"))

            for idx, _var in enumerate(chunk_vars):
                tok = graph.tokens[idx]
                if tok.type_name != "rating" or tok.status != 0:
                    continue
                rating_value = (tok.raw_data or {}).get("rating_value")
                if rating_value is None:
                    continue
                logits = params[0, idx, 1:1 + num_classes]
                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
                all_labels.append(int(rating_value))

    if not all_probs:
        return None
    probs = np.asarray(all_probs, dtype=np.float32)
    labels = np.asarray(all_labels, dtype=np.int64)
    return probs, labels


def _load_stan_probs(spec: DatasetSpec, size: int, variant: str) -> tuple[np.ndarray, np.ndarray] | None:
    probs_path = spec.stan_root / spec.stan_eval_run(size, variant) / "rating_probabilities.csv"
    bundle_path = spec.data_root / spec.baseline_run(size) / "data_bundle.json"
    if not probs_path.exists() or not bundle_path.exists():
        return None

    bundle = _read_json(bundle_path)
    test_idxs, labels = _test_missing_indices_and_labels(bundle)
    if not test_idxs:
        return None

    prob_cols = [PROB_COL_TEMPLATE.format(idx=i) for i in range(1, spec.num_classes + 1)]
    df = pd.read_csv(probs_path)
    grouped = (
        df[df["missing_rating_idx"].isin(test_idxs)]
        .groupby("missing_rating_idx")[prob_cols]
        .mean()
        .reindex(test_idxs)
    )
    if grouped.isnull().any().any():
        return None
    probs = grouped.to_numpy(dtype=np.float32)
    return probs, labels


def _load_baseline_probs(spec: DatasetSpec, method: str, size: int) -> tuple[np.ndarray, np.ndarray] | None:
    pred_path = spec.baseline_roots[method] / spec.baseline_run(size) / "test_predictions.json"
    if not pred_path.exists():
        return None
    rows = _read_json(pred_path)
    if not rows:
        return None
    probs = np.asarray([row["probs"] for row in rows], dtype=np.float32)
    labels = np.asarray([row["true_label"] for row in rows], dtype=np.int64)
    return probs, labels


def _plot_panels(spec: DatasetSpec, size: int, stem: str) -> None:
    panels = [
        ("Marformer", _load_marformer_probs(spec, size), "#1f6fba"),
        ("Stan Factor", _load_stan_probs(spec, size, "Factor"), "#27ae60"),
        ("Stan Normal", _load_stan_probs(spec, size, "Normal"), "#e67e22"),
        ("REMASKER", _load_baseline_probs(spec, "REMASKER", size), "#8e44ad"),
        ("MIWAE", _load_baseline_probs(spec, "MIWAE", size), "#c0392b"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.2))
    for ax, (panel_title, payload, color) in zip(axes.flat, panels + [("", None, "0.5")]):
        if not panel_title:
            ax.axis("off")
            continue
        if payload is None:
            _draw_empty(ax, panel_title)
            continue
        probs, labels = payload
        _plot_ece(ax, probs, labels, panel_title, color)

    fig.suptitle(f"{spec.pretty_name}: Reliability at Size {size}", fontsize=20, y=0.98)
    fig.subplots_adjust(top=0.89, hspace=0.34, wspace=0.30)
    out_path = spec.out_dir / stem
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


LLM_RUBRIC = DatasetSpec(
    name="LLMRubric",
    pretty_name="LLM Rubric",
    sizes=[10, 20, 30, 40, 50, 75, 100, 125, 150, 175],
    num_classes=4,
    data_root=ROOT / "DATA/LLM_RUBRIC",
    marformer_root=ROOT / "RESULTS/MARFORMER/LLM_RUBRIC",
    stan_root=ROOT / "RESULTS/STAN/LLM_RUBRIC_T",
    baseline_roots={
        "REMASKER": ROOT / "RESULTS/BASELINES/REMASKER/LLMRUBRIC",
        "MIWAE": ROOT / "RESULTS/BASELINES/MIWAE/LLMRUBRIC",
    },
    marformer_run=lambda size: f"LLMRubric_225_25_9_{size}",
    stan_eval_run=lambda size, variant: f"LLMRubric_225_25_9_{size}_nt_{variant}_eval",
    baseline_run=lambda size: f"LLMRubric_225_25_9_{size}",
    out_dir=PLOTS_ROOT / "LLMRubric",
)

SUMMEVAL = DatasetSpec(
    name="SummEval",
    pretty_name="SummEval",
    sizes=[50, 100, 500, 750, 1000, 1280],
    num_classes=5,
    data_root=ROOT / "DATA/SUMMEVAL",
    marformer_root=ROOT / "RESULTS/MARFORMER/SUMMEVAL",
    stan_root=ROOT / "RESULTS/STAN/SUMMEVAL_T",
    baseline_roots={
        "REMASKER": ROOT / "RESULTS/BASELINES/REMASKER/SUMMEVAL",
        "MIWAE": ROOT / "RESULTS/BASELINES/MIWAE/SUMMEVAL",
    },
    marformer_run=lambda size: f"SummEval_1600_8_4_{size}",
    stan_eval_run=lambda size, variant: f"SummEval_1600_8_4_{size}_nt_{variant}_eval",
    baseline_run=lambda size: f"SummEval_1600_8_4_{size}",
    out_dir=PLOTS_ROOT / "SummEval",
)


def main() -> None:
    _plot_panels(LLM_RUBRIC, 175, "ece_reliability_llm_rubric_size175.png")
    _plot_panels(SUMMEVAL, 1280, "ece_reliability_summeval_size1280.png")


if __name__ == "__main__":
    main()
