#!/usr/bin/env python3
"""
Plot DOMAIN3-SIGMOID generalization figures.

Item generalization:
  - Marformer exact sizes: 100..1000
  - Stan Oracle / misspec: exact through 500, extended dotted beyond
  - No x-axis break (max is 1000)

Annotator generalization:
  - Marformer exact sizes: 10..50
  - Stan / misspec: plot exact existing sizes

Outputs:
  - PLOTS/DOMAIN3-SIGMOID/ITEM_GENERALIZATION/*
  - PLOTS/DOMAIN3-SIGMOID/ANNOT_GENERALIZATION/*

Flags:
  --part item|annot|all
  --cache             load Marformer probs from npz if present, save if not
  --cache-baselines   load baseline probs from npz if present, save if not
  --calibration       also produce calibration reliability plots
  --device cuda|cpu   override torch device
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from BASELINES.structured_baselines import calibration_probs_labels, load_and_fit
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.test import _load_checkpoint, _reconstruct
from scripts.utils import plot_domain3_generalization as shared


ITEM_DISPLAY_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
ITEM_STAN_CUTOFF = 500
SPLIT_PREFIX = "DOMAIN3-SIGMOID"
PROB_PREFIX = "prob_cat_"

SERIES_ORDER = [
    "Marformer",
    "Marformer NT",
    "Stan Oracle",
    "Stan Both",
    "Stan Bin",
    "Stan Proj",
    "Unigram Pool ij",
    "Naive Bayes IJK",
    "Structured NB",
    "Structured Log-Linear",
]


@dataclass(frozen=True)
class DatasetConfig:
    slug: str
    title_prefix: str
    x_label: str
    display_sizes: list[int]
    data_root: Path
    marformer_root: Path
    marformer_run_stem: str
    out_dir: Path
    summary_path: Path
    cache_path: Path
    baseline_cache_path: Path
    marformer_nt_root: Path | None = None
    marformer_nt_run_stem: str | None = None
    nt_data_root: Path | None = None
    stan_oracle_root: Path | None = None
    stan_oracle_eval_suffix: str | None = None
    misspec_root: Path | None = None
    stan_cutoff: int | None = None
    baseline_cutoff: int | None = None


ITEM_CFG = DatasetConfig(
    slug="item",
    title_prefix="Item Generalization",
    x_label="Training Items",
    display_sizes=ITEM_DISPLAY_SIZES,
    data_root=ROOT / "DATA/STAN/DOMAIN3-SIGMOID/ItemSplits/Transductive",
    marformer_root=ROOT / "RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM",
    marformer_run_stem="DOMAIN3-SIGMOID_Item_T_{size}_MARFORMER",
    marformer_nt_root=ROOT / "RESULTS/MARFORMER-NT/DOMAIN3-SIGMOID/ITEM",
    marformer_nt_run_stem="DOMAIN3-SIGMOID_Item_NT_{size}_MARFORMER",
    nt_data_root=ROOT / "DATA/STAN/DOMAIN3-SIGMOID/ItemSplits/NonTransductive",
    out_dir=ROOT / "PLOTS/DOMAIN3-SIGMOID/ITEM_GENERALIZATION",
    summary_path=ROOT / "PLOTS/DOMAIN3-SIGMOID/ITEM_GENERALIZATION/item_generalization_summary.json",
    cache_path=ROOT / "PLOTS/DOMAIN3-SIGMOID/ITEM_GENERALIZATION/item_marformer_cache.npz",
    baseline_cache_path=ROOT / "PLOTS/DOMAIN3-SIGMOID/ITEM_GENERALIZATION/item_baseline_cache.npz",
    stan_oracle_root=ROOT / "RESULTS/STAN/DOMAIN3-SIGMOID/ORACLE/ITEM",
    stan_oracle_eval_suffix="TENSOR-SIG_eval",
    misspec_root=ROOT / "RESULTS/STAN/DOMAIN3-SIGMOID/MISSPEC/ITEM",
    stan_cutoff=ITEM_STAN_CUTOFF,
    baseline_cutoff=1000,
)

ANNOT_CFG = DatasetConfig(
    slug="annot",
    title_prefix="Annotator Generalization",
    x_label="Training Annotators",
    display_sizes=[10, 15, 20, 25, 30, 35, 40, 45, 50],
    data_root=ROOT / "DATA/STAN/DOMAIN3-SIGMOID/AnnotSplits/Transductive",
    marformer_root=ROOT / "RESULTS/MARFORMER/DOMAIN3-SIGMOID/ANNOT",
    marformer_run_stem="DOMAIN3-SIGMOID_Annot_T_{size}_MARFORMER",
    out_dir=ROOT / "PLOTS/DOMAIN3-SIGMOID/ANNOT_GENERALIZATION",
    summary_path=ROOT / "PLOTS/DOMAIN3-SIGMOID/ANNOT_GENERALIZATION/annot_generalization_summary.json",
    cache_path=ROOT / "PLOTS/DOMAIN3-SIGMOID/ANNOT_GENERALIZATION/annot_marformer_cache.npz",
    baseline_cache_path=ROOT / "PLOTS/DOMAIN3-SIGMOID/ANNOT_GENERALIZATION/annot_baseline_cache.npz",
    stan_oracle_root=ROOT / "RESULTS/STAN/DOMAIN3-SIGMOID/ORACLE/ANNOT",
    stan_oracle_eval_suffix="TENSOR-SIG_eval",
    misspec_root=ROOT / "RESULTS/STAN/DOMAIN3-SIGMOID/MISSPEC/ANNOT",
    baseline_cutoff=None,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DOMAIN3-SIGMOID generalization figures.")
    parser.add_argument("--part", choices=["all", "item", "annot"], default="all")
    parser.add_argument("--cache", action="store_true",
                        help="Load Marformer probs from npz cache if present; save on fresh run.")
    parser.add_argument("--cache-baselines", action="store_true",
                        help="Load baseline probs from npz cache if present; save on fresh run.")
    parser.add_argument("--calibration", action="store_true",
                        help="Also produce calibration reliability plots.")
    parser.add_argument("--device", default=None,
                        help="Torch device for Marformer eval; default: cuda > mps > cpu.")
    return parser.parse_args()


# ── Utilities ─────────────────────────────────────────────────────────────────

def _choose_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _safe_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _runtime_from_launcher(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    value = _read_json(path).get(key)
    return None if value is None else float(value)


def _parse_best_epoch(checkpoint_name: str | None, total_epochs: int) -> int:
    if not checkpoint_name or checkpoint_name == "last.ckpt":
        return total_epochs
    match = re.search(r"epoch=(\d+)", checkpoint_name)
    return int(match.group(1)) + 1 if match else total_epochs


def _source_size_exact_or_extend(display_size: int, available: set[int], cutoff: int) -> int | None:
    if display_size <= cutoff:
        return display_size if display_size in available else None
    candidates = [s for s in available if s <= cutoff]
    return max(candidates) if candidates else None


def _source_size_for_cfg(cfg: DatasetConfig, display_size: int, available: set[int]) -> int | None:
    if cfg.slug == "annot" or cfg.stan_cutoff is None:
        return display_size if display_size in available else None
    return _source_size_exact_or_extend(display_size, available, cfg.stan_cutoff)


def _source_size_for_baseline(cfg: DatasetConfig, display_size: int, available: set[int]) -> int | None:
    if cfg.slug == "annot" or cfg.baseline_cutoff is None:
        return display_size if display_size in available else None
    return _source_size_exact_or_extend(display_size, available, cfg.baseline_cutoff)


def _marformer_run_dir(cfg: DatasetConfig, size: int) -> Path:
    return cfg.marformer_root / cfg.marformer_run_stem.format(size=size)


def _marformer_nt_run_dir(cfg: DatasetConfig, size: int) -> Path | None:
    if cfg.marformer_nt_root is None or cfg.marformer_nt_run_stem is None:
        return None
    return cfg.marformer_nt_root / cfg.marformer_nt_run_stem.format(size=size)


def _bundle_path(cfg: DatasetConfig, size: int) -> Path:
    axis = "Item" if cfg.slug == "item" else "Annot"
    return cfg.data_root / f"{SPLIT_PREFIX}_{axis}_T_{size}" / "data_bundle.json"


def _nt_bundle_path(cfg: DatasetConfig, size: int) -> Path | None:
    if cfg.nt_data_root is None:
        return None
    axis = "Item" if cfg.slug == "item" else "Annot"
    return cfg.nt_data_root / f"{SPLIT_PREFIX}_{axis}_NT_{size}" / "data_bundle.json"


def _test_missing_indices_and_labels(bundle: dict) -> tuple[list[int], np.ndarray]:
    idxs = list(bundle.get("missing_ratings_indexes_in_test_instance", []))
    labels = np.asarray([bundle["missing_ratings"][idx]["value"] - 1 for idx in idxs], dtype=np.int64)
    return idxs, labels


def _per_example_nll(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0))


def _per_example_mse(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    expected = probs @ classes
    return (expected - (labels.astype(np.float64) + 1.0)) ** 2


# ── Available size discovery ──────────────────────────────────────────────────

def _available_marformer_sizes(cfg: DatasetConfig) -> set[int]:
    return {s for s in cfg.display_sizes if _marformer_run_dir(cfg, s).exists()}


def _available_marformer_nt_sizes(cfg: DatasetConfig) -> set[int]:
    if cfg.marformer_nt_root is None or cfg.marformer_nt_run_stem is None or cfg.nt_data_root is None:
        return set()
    available: set[int] = set()
    for size in cfg.display_sizes:
        run_dir = _marformer_nt_run_dir(cfg, size)
        bundle = _nt_bundle_path(cfg, size)
        if run_dir is not None and run_dir.exists() and bundle is not None and bundle.exists():
            available.add(size)
    return available


def _available_oracle_sizes(cfg: DatasetConfig) -> set[int]:
    if cfg.stan_oracle_root is None or cfg.stan_oracle_eval_suffix is None:
        return set()
    axis = "Item" if cfg.slug == "item" else "Annot"
    regex = re.compile(
        rf"{re.escape(SPLIT_PREFIX)}_{axis}_T_(\d+)_{re.escape(cfg.stan_oracle_eval_suffix)}$"
    )
    return {
        int(m.group(1))
        for p in cfg.stan_oracle_root.glob("*")
        if p.is_dir() and (m := regex.match(p.name))
    }


def _available_misspec_sizes(cfg: DatasetConfig, suffix: str) -> set[int]:
    if cfg.misspec_root is None:
        return set()
    axis = "Item" if cfg.slug == "item" else "Annot"
    regex = re.compile(rf"{re.escape(SPLIT_PREFIX)}_{axis}_T_(\d+)_{re.escape(suffix)}_eval$")
    return {
        int(m.group(1))
        for p in cfg.misspec_root.glob("*")
        if p.is_dir() and (m := regex.match(p.name))
    }


# ── Marformer evaluation ──────────────────────────────────────────────────────

def _write_best_json_from_test_results(run_dir: Path) -> None:
    test_dir = run_dir / "TEST_RESULTS"
    candidates: list[tuple[float, float, str, dict]] = []
    for path in sorted(test_dir.glob("*.json")):
        if path.name in {"best.json", "launcher_metadata.json"}:
            continue
        data = _read_json(path)
        missing = data.get("missing", {})
        log_loss = missing.get("log_loss")
        if log_loss is None or missing.get("n", 0) == 0:
            continue
        rmse = float(missing.get("rmse") or math.inf)
        candidates.append((float(log_loss), rmse, path.name, data))
    if not candidates:
        raise RuntimeError(f"No valid checkpoint evaluation JSON in {test_dir}")
    candidates.sort(key=lambda r: (r[0], r[1], r[2]))
    best_data = candidates[0][3]
    best_data["selected_from"] = candidates[0][2]
    best_data["selection_metric"] = "missing.log_loss"
    _write_json(test_dir / "best.json", best_data)
    print(f"  best checkpoint: {candidates[0][2]}  log_loss={candidates[0][0]:.6f}")


def _ensure_marformer_best_json(run_dir: Path, device_name: str) -> Path:
    best_json = run_dir / "TEST_RESULTS" / "best.json"
    if best_json.exists():
        return best_json
    print(f"Running test.py checkpoint sweep for {run_dir.name}")
    (run_dir / "TEST_RESULTS").mkdir(parents=True, exist_ok=True)
    for mode in ("all", "last"):
        subprocess.run(
            [sys.executable, "-m", "imputer.entity_mf.test",
             "--run-dir", str(run_dir), "--checkpoint", mode, "--device", device_name],
            cwd=str(ROOT), check=True,
        )
    _write_best_json_from_test_results(run_dir)
    return best_json


def _chunk_variables_by_item_limit(variables: list, max_item: int) -> list[list]:
    item_ids = sorted({iid for var in variables for iid in var.item_ids})
    if len(item_ids) <= max_item:
        return [variables]
    chunks: list[list] = []
    for start in range(0, len(item_ids), max_item):
        item_set = set(item_ids[start: start + max_item])
        chunk = [v for v in variables if all(iid in item_set for iid in v.item_ids)]
        if chunk:
            chunks.append(chunk)
    return chunks


def _marformer_eval_max_item(cfg: DatasetConfig, size: int) -> int | None:
    if cfg.slug == "item":
        return None if size <= 500 else 500
    return None


@torch.no_grad()
def _marformer_probs_all_missing(
    run_dir: Path, checkpoint_name: str, max_item: int | None, device_name: str
) -> tuple[np.ndarray, np.ndarray]:
    model, eval_vars, _train_cfg = _reconstruct(run_dir)
    device = torch.device(device_name)
    _load_checkpoint(model, run_dir / "checkpoints" / checkpoint_name, device)
    model.to(device)
    model.eval()
    num_classes = model.types["rating"].num_classes
    prob_rows: list[np.ndarray] = []
    labels: list[int] = []
    chunks = [eval_vars] if max_item is None else _chunk_variables_by_item_limit(eval_vars, max_item)
    for chunk_vars in chunks:
        graph = variable_list_to_entity_graph(chunk_vars, model.types)
        params = model(graph, device=device)
        for idx, var in enumerate(chunk_vars):
            if not getattr(var, "is_missing", False):
                continue
            logits = params[0, idx, 1: 1 + num_classes]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float64)
            prob_rows.append(probs)
            labels.append(int(var.rating_value))
    if not prob_rows:
        raise RuntimeError(f"No missing-token probs for {run_dir.name} @ {checkpoint_name}")
    return np.vstack(prob_rows), np.asarray(labels, dtype=np.int64)


def _collect_marformer_payloads(
    cfg: DatasetConfig, device_name: str
) -> tuple[dict[int, tuple[str, np.ndarray, np.ndarray]], dict[int, float], dict[int, float]]:
    payloads: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    runtimes: dict[int, float] = {}
    logloss_override: dict[int, float] = {}
    for size in cfg.display_sizes:
        run_dir = _marformer_run_dir(cfg, size)
        if not run_dir.exists():
            continue
        best_json = _ensure_marformer_best_json(run_dir, device_name)
        best = _read_json(best_json)
        ckpt = best.get("checkpoint")
        if not ckpt:
            continue
        probs, labels = _marformer_probs_all_missing(
            run_dir, ckpt, _marformer_eval_max_item(cfg, size), device_name
        )
        payloads[size] = (ckpt, probs, labels)
        if best.get("missing", {}).get("log_loss") is not None:
            logloss_override[size] = float(best["missing"]["log_loss"])
        total_s = _runtime_from_launcher(run_dir / "launcher_metadata.json", "elapsed_seconds")
        total_epochs = int(_read_json(run_dir / "train_config.json").get("training", {}).get("epochs", 300))
        if total_s is not None:
            runtime_epochs = _parse_best_epoch(ckpt, total_epochs)
            runtimes[size] = float(total_s) * (runtime_epochs / total_epochs) / 60.0
    return payloads, runtimes, logloss_override


def _collect_marformer_nt_payloads(
    cfg: DatasetConfig, device_name: str
) -> tuple[dict[int, tuple[str, np.ndarray, np.ndarray]], dict[int, float], dict[int, float]]:
    payloads: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    runtimes: dict[int, float] = {}
    logloss_override: dict[int, float] = {}
    for size in sorted(_available_marformer_nt_sizes(cfg)):
        run_dir = _marformer_nt_run_dir(cfg, size)
        if run_dir is None:
            continue
        best_json = _ensure_marformer_best_json(run_dir, device_name)
        best = _read_json(best_json)
        ckpt = best.get("checkpoint")
        if not ckpt:
            continue
        probs, labels = _marformer_probs_all_missing(
            run_dir, ckpt, _marformer_eval_max_item(cfg, size), device_name
        )
        payloads[size] = (ckpt, probs, labels)
        if best.get("missing", {}).get("log_loss") is not None:
            logloss_override[size] = float(best["missing"]["log_loss"])
        total_s = _runtime_from_launcher(run_dir / "launcher_metadata.json", "elapsed_seconds")
        total_epochs = int(_read_json(run_dir / "train_config.json").get("training", {}).get("epochs", 300))
        if total_s is not None:
            runtime_epochs = _parse_best_epoch(ckpt, total_epochs)
            runtimes[size] = float(total_s) * (runtime_epochs / total_epochs) / 60.0
    return payloads, runtimes, logloss_override


# ── Stan payload collection ───────────────────────────────────────────────────

def _stan_probs_from_eval_dir(bundle_path: Path, eval_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    probs_path = eval_dir / "rating_probabilities.csv"
    if not probs_path.exists():
        return None
    bundle = _read_json(bundle_path)
    test_idxs, labels = _test_missing_indices_and_labels(bundle)
    if not test_idxs:
        return None
    df = pd.read_csv(probs_path)
    prob_cols = sorted(c for c in df.columns if c.startswith(PROB_PREFIX))
    grouped = (
        df[df["missing_rating_idx"].isin(test_idxs)]
        .groupby("missing_rating_idx")[prob_cols]
        .mean()
        .reindex(test_idxs)
    )
    if grouped.isnull().any().any():
        return None
    return grouped.to_numpy(dtype=np.float64), labels


def _collect_oracle_payloads(
    cfg: DatasetConfig,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, float]]:
    payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    runtimes: dict[int, float] = {}
    if cfg.stan_oracle_root is None or cfg.stan_oracle_eval_suffix is None:
        return payloads, runtimes
    axis = "Item" if cfg.slug == "item" else "Annot"
    for size in sorted(_available_oracle_sizes(cfg)):
        eval_dir = cfg.stan_oracle_root / f"{SPLIT_PREFIX}_{axis}_T_{size}_{cfg.stan_oracle_eval_suffix}"
        print(f"[Stan Oracle] size={size} eval_dir={eval_dir.name}")
        payload = _stan_probs_from_eval_dir(_bundle_path(cfg, size), eval_dir)
        if payload is not None:
            payloads[size] = payload
            probs, labels = payload
            print(f"[Stan Oracle] size={size} n={labels.shape[0]} "
                  f"log_loss={float(np.mean(_per_example_nll(probs, labels))):.4f}")
        run_dir_name = eval_dir.name[: -len("_eval")]
        runtime = _runtime_from_launcher(
            cfg.stan_oracle_root / run_dir_name / "launcher_metadata.json", "fit_elapsed_seconds"
        )
        if runtime is not None:
            runtimes[size] = runtime / 60.0
    return payloads, runtimes


def _collect_misspec_payloads(
    cfg: DatasetConfig, suffix: str
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, float]]:
    payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    runtimes: dict[int, float] = {}
    if cfg.misspec_root is None:
        return payloads, runtimes
    axis = "Item" if cfg.slug == "item" else "Annot"
    for size in sorted(_available_misspec_sizes(cfg, suffix)):
        eval_dir = cfg.misspec_root / f"{SPLIT_PREFIX}_{axis}_T_{size}_{suffix}_eval"
        payload = _stan_probs_from_eval_dir(_bundle_path(cfg, size), eval_dir)
        if payload is not None:
            payloads[size] = payload
        run_dir_name = eval_dir.name[: -len("_eval")]
        runtime = _runtime_from_launcher(
            cfg.misspec_root / run_dir_name / "launcher_metadata.json", "fit_elapsed_seconds"
        )
        if runtime is not None:
            runtimes[size] = runtime / 60.0
    return payloads, runtimes


def _collect_baseline_payloads(
    cfg: DatasetConfig, cutoff: int
) -> tuple[dict[str, dict[int, tuple[np.ndarray, np.ndarray]]], dict[str, dict[int, float]]]:
    payloads: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        label: {} for label in shared.STRUCTURED_BASELINE_LABELS
    }
    runtimes: dict[str, dict[int, float]] = {label: {} for label in shared.STRUCTURED_BASELINE_LABELS}
    key_map = {
        "Unigram Pool ij": "unigram_ij",
        "Naive Bayes IJK": "ijk",
        "Structured NB": "snb",
        "Structured Log-Linear": "log_linear",
    }
    for size in [s for s in cfg.display_sizes if s <= cutoff]:
        bundle_path = _bundle_path(cfg, size)
        if not bundle_path.exists():
            print(f"[Baselines] size={size} missing bundle: {bundle_path}")
            continue
        print(f"[Baselines] size={size} fitting...")
        bundle, fitted = load_and_fit(
            bundle_path, fit_log_linear=True, log_linear_epochs=100, log_linear_l2_lambda=2.0
        )
        cal_payloads = calibration_probs_labels(fitted, bundle, split="test")
        if not cal_payloads:
            continue
        for label in shared.STRUCTURED_BASELINE_LABELS:
            t0 = time.perf_counter()
            payload = cal_payloads.get(key_map[label])
            if payload is None:
                continue
            probs, labels = payload
            runtimes[label][size] = (time.perf_counter() - t0) / 60.0
            payloads[label][size] = (probs.astype(np.float64), labels.astype(np.int64))
    return payloads, runtimes


# ── Cache ─────────────────────────────────────────────────────────────────────

def _save_marformer_cache(
    path: Path,
    payloads: dict[int, tuple[str, np.ndarray, np.ndarray]],
    nt_payloads: dict[int, tuple[str, np.ndarray, np.ndarray]],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for size, (ckpt, probs, labels) in payloads.items():
        arrays[f"ckpt__marformer__{size}"] = np.asarray([ckpt])
        arrays[f"probs__marformer__{size}"] = probs
        arrays[f"labels__marformer__{size}"] = labels
    for size, (ckpt, probs, labels) in nt_payloads.items():
        arrays[f"ckpt__marformer_nt__{size}"] = np.asarray([ckpt])
        arrays[f"probs__marformer_nt__{size}"] = probs
        arrays[f"labels__marformer_nt__{size}"] = labels
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"Saved Marformer cache → {path}")


def _load_marformer_cache(
    path: Path,
) -> tuple[dict[int, tuple[str, np.ndarray, np.ndarray]], dict[int, tuple[str, np.ndarray, np.ndarray]]]:
    data = np.load(path, allow_pickle=False)
    payloads: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    nt_payloads: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    for key in data.files:
        if not key.startswith("probs__"):
            continue
        _, model_key, size_s = key.split("__")
        size = int(size_s)
        probs = data[key]
        labels = data[f"labels__{model_key}__{size}"]
        ckpt_key = f"ckpt__{model_key}__{size}"
        ckpt = str(data[ckpt_key][0]) if ckpt_key in data.files else "unknown.ckpt"
        if model_key == "marformer":
            payloads[size] = (ckpt, probs, labels)
        elif model_key == "marformer_nt":
            nt_payloads[size] = (ckpt, probs, labels)
    return payloads, nt_payloads


def _save_baseline_cache(
    path: Path, baseline_payloads: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for model, payloads in baseline_payloads.items():
        mk = _safe_key(model)
        for size, (probs, labels) in payloads.items():
            arrays[f"probs__{mk}__{size}"] = probs
            arrays[f"labels__{mk}__{size}"] = labels
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"Saved baseline cache → {path}")


def _load_baseline_cache(path: Path) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    label_map = {
        "unigram_pool_ij": "Unigram Pool ij",
        "naive_bayes_ijk": "Naive Bayes IJK",
        "structured_nb": "Structured NB",
        "structured_log_linear": "Structured Log-Linear",
    }
    data = np.load(path, allow_pickle=False)
    payloads: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        label: {} for label in shared.STRUCTURED_BASELINE_LABELS
    }
    for key in data.files:
        if not key.startswith("probs__"):
            continue
        _, mk, size_s = key.split("__")
        size = int(size_s)
        label = label_map.get(mk)
        if label is None:
            continue
        labels_key = f"labels__{mk}__{size}"
        if labels_key in data.files:
            payloads[label][size] = (data[key], data[labels_key])
    return payloads


# ── Runtime metadata from Marformer JSON (used when loading from cache) ───────

def _marformer_runtime_from_json(
    cfg: DatasetConfig,
    payloads: dict[int, tuple[str, np.ndarray, np.ndarray]],
    nt_payloads: dict[int, tuple[str, np.ndarray, np.ndarray]],
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
    runtimes: dict[int, float] = {}
    nt_runtimes: dict[int, float] = {}
    logloss: dict[int, float] = {}
    nt_logloss: dict[int, float] = {}
    for size, (ckpt, _, _) in payloads.items():
        run_dir = _marformer_run_dir(cfg, size)
        total_s = _runtime_from_launcher(run_dir / "launcher_metadata.json", "elapsed_seconds")
        best_json = run_dir / "TEST_RESULTS" / "best.json"
        if best_json.exists():
            best = _read_json(best_json)
            ll = best.get("missing", {}).get("log_loss")
            if ll is not None:
                logloss[size] = float(ll)
            total_epochs = int(_read_json(run_dir / "train_config.json").get("training", {}).get("epochs", 300))
            if total_s is not None:
                runtimes[size] = float(total_s) * (_parse_best_epoch(ckpt, total_epochs) / total_epochs) / 60.0
    for size, (ckpt, _, _) in nt_payloads.items():
        run_dir = _marformer_nt_run_dir(cfg, size)
        if run_dir is None:
            continue
        total_s = _runtime_from_launcher(run_dir / "launcher_metadata.json", "elapsed_seconds")
        best_json = run_dir / "TEST_RESULTS" / "best.json"
        if best_json.exists():
            best = _read_json(best_json)
            ll = best.get("missing", {}).get("log_loss")
            if ll is not None:
                nt_logloss[size] = float(ll)
            total_epochs = int(_read_json(run_dir / "train_config.json").get("training", {}).get("epochs", 300))
            if total_s is not None:
                nt_runtimes[size] = float(total_s) * (_parse_best_epoch(ckpt, total_epochs) / total_epochs) / 60.0
    return runtimes, nt_runtimes, logloss, nt_logloss


# ── Row building ──────────────────────────────────────────────────────────────

def _metric_row(
    model: str, display_size: int, source_size: int, metric: str,
    values: np.ndarray, mean_override: float | None = None,
) -> dict:
    mean = float(np.mean(values)) if mean_override is None else float(mean_override)
    std = float(np.std(values, ddof=0))
    q10, q25, median, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "dataset": "DOMAIN3-SIGMOID",
        "model": model,
        "metric": metric,
        "size": int(display_size),
        "source_size": int(source_size),
        "mean": mean, "std": std,
        "q10": float(q10), "q25": float(q25), "median": float(median),
        "q75": float(q75), "q90": float(q90),
        "min": float(np.min(values)), "max": float(np.max(values)),
        "count": int(values.size),
    }


def _runtime_row(model: str, display_size: int, source_size: int, minutes: float) -> dict:
    return {
        "dataset": "DOMAIN3-SIGMOID",
        "model": model, "metric": "runtime_minutes",
        "size": int(display_size), "source_size": int(source_size), "mean": float(minutes),
    }


def _summary_lookup(rows: list[dict], metric: str, model: str, size: int) -> dict | None:
    for row in rows:
        if row.get("metric") == metric and row.get("model") == model and int(row.get("size", -1)) == size:
            return row
    return None


# ── Main data collection and row assembly ─────────────────────────────────────

def _build_rows_and_payloads(
    cfg: DatasetConfig,
    device_name: str,
    use_cache: bool,
    use_cache_baselines: bool,
) -> tuple[list[dict], dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str | None]]]:

    # Marformer
    if use_cache and cfg.cache_path.exists():
        print(f"Loading Marformer from cache: {cfg.cache_path}")
        marformer_payloads, nt_payloads = _load_marformer_cache(cfg.cache_path)
        marformer_runtimes, nt_runtimes, logloss_override, nt_logloss = _marformer_runtime_from_json(
            cfg, marformer_payloads, nt_payloads
        )
    else:
        marformer_payloads, marformer_runtimes, logloss_override = _collect_marformer_payloads(cfg, device_name)
        nt_payloads, nt_runtimes, nt_logloss = _collect_marformer_nt_payloads(cfg, device_name)
        _save_marformer_cache(cfg.cache_path, marformer_payloads, nt_payloads)

    # Stan Oracle + misspec (always re-read from CSVs — fast)
    oracle_payloads, oracle_runtimes = _collect_oracle_payloads(cfg)
    misspec_specs = [("Stan Both", "BOTH"), ("Stan Bin", "BIN"), ("Stan Proj", "PROJ")]
    misspec_payloads: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    misspec_runtimes: dict[str, dict[int, float]] = {}
    for label, suffix in misspec_specs:
        p, r = _collect_misspec_payloads(cfg, suffix)
        misspec_payloads[label] = p
        misspec_runtimes[label] = r

    # Baselines
    if use_cache_baselines and cfg.baseline_cache_path.exists():
        print(f"Loading baselines from cache: {cfg.baseline_cache_path}")
        baseline_payloads = _load_baseline_cache(cfg.baseline_cache_path)
        baseline_runtimes: dict[str, dict[int, float]] = {l: {} for l in shared.STRUCTURED_BASELINE_LABELS}
    else:
        cutoff = cfg.baseline_cutoff if cfg.baseline_cutoff is not None else max(cfg.display_sizes)
        baseline_payloads, baseline_runtimes = _collect_baseline_payloads(cfg, cutoff)
        _save_baseline_cache(cfg.baseline_cache_path, baseline_payloads)

    # Build metric rows
    rows: list[dict] = []
    avail_oracle = set(oracle_payloads.keys())
    avail_misspec = {l: set(p.keys()) for l, p in misspec_payloads.items()}
    avail_baseline = {l: set(p.keys()) for l, p in baseline_payloads.items()}

    for size in cfg.display_sizes:
        if size in marformer_payloads:
            ckpt, probs, labels = marformer_payloads[size]
            rows.append(_metric_row("Marformer", size, size, "log_loss",
                                    _per_example_nll(probs, labels), logloss_override.get(size)))
            rows.append(_metric_row("Marformer", size, size, "mbr_l2", _per_example_mse(probs, labels)))
            if size in marformer_runtimes:
                rows.append(_runtime_row("Marformer", size, size, marformer_runtimes[size]))

        if size in nt_payloads:
            ckpt, probs, labels = nt_payloads[size]
            rows.append(_metric_row("Marformer NT", size, size, "log_loss",
                                    _per_example_nll(probs, labels), nt_logloss.get(size)))
            rows.append(_metric_row("Marformer NT", size, size, "mbr_l2", _per_example_mse(probs, labels)))
            if size in nt_runtimes:
                rows.append(_runtime_row("Marformer NT", size, size, nt_runtimes[size]))

        oracle_src = _source_size_for_cfg(cfg, size, avail_oracle)
        if oracle_src is not None:
            probs, labels = oracle_payloads[oracle_src]
            rows.append(_metric_row("Stan Oracle", size, oracle_src, "log_loss", _per_example_nll(probs, labels)))
            rows.append(_metric_row("Stan Oracle", size, oracle_src, "mbr_l2", _per_example_mse(probs, labels)))
            if size in oracle_runtimes:
                rows.append(_runtime_row("Stan Oracle", size, size, oracle_runtimes[size]))

        for label, _ in misspec_specs:
            src = _source_size_for_cfg(cfg, size, avail_misspec[label])
            if src is None:
                continue
            probs, labels = misspec_payloads[label][src]
            rows.append(_metric_row(label, size, src, "log_loss", _per_example_nll(probs, labels)))
            rows.append(_metric_row(label, size, src, "mbr_l2", _per_example_mse(probs, labels)))
            if size in misspec_runtimes[label]:
                rows.append(_runtime_row(label, size, size, misspec_runtimes[label][size]))

        for label in shared.STRUCTURED_BASELINE_LABELS:
            src = _source_size_for_baseline(cfg, size, avail_baseline[label])
            if src is None:
                continue
            probs, labels = baseline_payloads[label][src]
            rows.append(_metric_row(label, size, src, "log_loss", _per_example_nll(probs, labels)))
            rows.append(_metric_row(label, size, src, "mbr_l2", _per_example_mse(probs, labels)))
            if size in baseline_runtimes[label]:
                rows.append(_runtime_row(label, size, size, baseline_runtimes[label][size]))

    _write_json(cfg.summary_path, {
        "dataset": "DOMAIN3-SIGMOID",
        "display_sizes": cfg.display_sizes,
        "rows": rows,
        "marformer_selection": {
            str(size): {"checkpoint": ckpt}
            for size, (ckpt, _, _) in marformer_payloads.items()
        },
        "marformer_nt_selection": {
            str(size): {"checkpoint": ckpt}
            for size, (ckpt, _, _) in nt_payloads.items()
        },
    })

    # Combined payload dict for calibration
    all_payloads: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str | None]] = {}
    for size, (ckpt, probs, labels) in marformer_payloads.items():
        all_payloads[("Marformer", size)] = (probs, labels, ckpt)
    for size, (ckpt, probs, labels) in nt_payloads.items():
        all_payloads[("Marformer NT", size)] = (probs, labels, ckpt)
    for size, (probs, labels) in oracle_payloads.items():
        all_payloads[("Stan Oracle", size)] = (probs, labels, None)
    for label, p in misspec_payloads.items():
        for size, (probs, labels) in p.items():
            all_payloads[(label, size)] = (probs, labels, None)
    for label, p in baseline_payloads.items():
        for size, (probs, labels) in p.items():
            all_payloads[(label, size)] = (probs, labels, None)

    return rows, all_payloads


# ── Plotting ──────────────────────────────────────────────────────────────────

def _series_from_rows(cfg: DatasetConfig, rows: list[dict], metric: str) -> dict[str, dict[str, list[float]]]:
    stats = {
        stat: {label: [np.nan] * len(cfg.display_sizes) for label in SERIES_ORDER}
        for stat in ("mean", "std")
    }
    index_by_size = {size: idx for idx, size in enumerate(cfg.display_sizes)}
    for row in rows:
        if row.get("metric") != metric:
            continue
        label = row.get("model")
        size = int(row.get("size"))
        if label not in stats["mean"] or size not in index_by_size:
            continue
        idx = index_by_size[size]
        stats["mean"][label][idx] = float(row["mean"])
        stats["std"][label][idx] = float(row["std"])
    return stats


def _plot_series_with_stan_extension(
    ax: plt.Axes,
    xs: np.ndarray,
    display_sizes: np.ndarray,
    ys: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    label: str,
) -> None:
    valid = ~np.isnan(ys)
    if not np.any(valid):
        return
    xs_v, ys_v, lo_v, hi_v = xs[valid], ys[valid], lo[valid], hi[valid]
    linestyle, color, marker = shared._series_style(label)
    line_kw = dict(color=color, marker=marker,
                   linewidth=shared._series_linewidth(label),
                   alpha=shared._series_alpha(label), label=label)
    is_stan = label in {"Stan Oracle", "Stan Both", "Stan Bin", "Stan Proj"}
    if not is_stan or np.all(display_sizes[valid] <= ITEM_STAN_CUTOFF):
        ax.plot(xs_v, ys_v, linestyle=linestyle, **line_kw)
        ax.fill_between(xs_v, lo_v, hi_v, color=color, alpha=shared._series_fill_alpha(label), linewidth=0)
        return
    # Split at cutoff: solid up to 500, dotted beyond
    sizes_v = display_sizes[valid]
    pre = sizes_v <= ITEM_STAN_CUTOFF
    post = sizes_v >= ITEM_STAN_CUTOFF
    if np.any(pre):
        ax.plot(xs_v[pre], ys_v[pre], linestyle=linestyle, **line_kw)
        ax.fill_between(xs_v[pre], lo_v[pre], hi_v[pre], color=color,
                        alpha=shared._series_fill_alpha(label), linewidth=0)
    if np.any(post):
        ax.plot(xs_v[post], ys_v[post], linestyle=":", color=color,
                linewidth=shared._series_linewidth(label),
                alpha=shared._series_alpha(label), label="_nolegend_")
        ax.fill_between(xs_v[post], lo_v[post], hi_v[post], color=color,
                        alpha=shared._series_fill_alpha(label), linewidth=0)


def _plot_metric(
    cfg: DatasetConfig, rows: list[dict], metric: str, ylabel: str, title: str, output_path: Path
) -> None:
    stats = _series_from_rows(cfg, rows, metric)
    present_labels = [l for l in SERIES_ORDER if any(not math.isnan(v) for v in stats["mean"][l])]
    has_dense = len(present_labels) >= 9 or any(len(l) >= 20 for l in present_labels)

    if cfg.slug == "annot":
        legend_ratio = 1.75 if has_dense else 1.5
        left_ratio = 6.9 if has_dense else 6.7
        fig_w, fig_h = (18.6 if has_dense else 17.8), 7.15
        tick_fs, axis_fs, title_fs = 9.5, 11.5, 14.0
        legend_fs = 9.5 if has_dense else 10.0
        legend_ls, legend_hl = (0.52 if has_dense else 0.62), (2.0 if has_dense else 2.2)
    else:
        legend_ratio = 2.15 if has_dense else 1.6
        left_ratio = 6.15 if has_dense else 6.3
        fig_w, fig_h = (18.8 if has_dense else 17.6), 7.6
        tick_fs, axis_fs, title_fs = 10.5, 12.5, 16.0
        legend_fs = 10.0 if has_dense else 11.0
        legend_ls, legend_hl = (0.58 if has_dense else 0.7), (2.2 if has_dense else 2.4)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 2, width_ratios=[left_ratio, legend_ratio],
                          height_ratios=[1.0, 4.8], hspace=0.05, wspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_leg = fig.add_subplot(gs[:, 1])
    display_sizes = np.asarray(cfg.display_sizes, dtype=int)
    xs_all = np.arange(len(display_sizes), dtype=float)

    for label in SERIES_ORDER:
        ys = np.asarray(stats["mean"][label], dtype=float)
        stds = np.asarray(stats["std"][label], dtype=float)
        lo = ys - 0.05 * stds
        hi = ys + 0.05 * stds
        if cfg.slug == "item":
            _plot_series_with_stan_extension(ax_top, xs_all, display_sizes, ys, lo, hi, label)
            _plot_series_with_stan_extension(ax_bottom, xs_all, display_sizes, ys, lo, hi, label)
        else:
            valid = ~np.isnan(ys)
            if not np.any(valid):
                continue
            linestyle, color, marker = shared._series_style(label)
            lkw = dict(color=color, marker=marker,
                       linewidth=shared._series_linewidth(label),
                       alpha=shared._series_alpha(label), label=label)
            ax_top.plot(xs_all[valid], ys[valid], linestyle=linestyle, **lkw)
            ax_bottom.plot(xs_all[valid], ys[valid], linestyle=linestyle, **lkw)
            ax_top.fill_between(xs_all[valid], lo[valid], hi[valid], color=color,
                                alpha=shared._series_fill_alpha(label), linewidth=0)
            ax_bottom.fill_between(xs_all[valid], lo[valid], hi[valid], color=color,
                                   alpha=shared._series_fill_alpha(label), linewidth=0)

    if metric == "log_loss":
        ax_bottom.set_ylim(0.58, 1.02)
        ax_bottom.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax_top.set_ylim(1.15, 1.65)
        ax_top.set_yticks([1.2, 1.4, 1.6])
    else:
        ax_bottom.set_ylim(0.0, 0.52)
        ax_bottom.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax_top.set_ylim(0.85, 2.55)
        ax_top.set_yticks([1.0, 1.5, 2.0, 2.5])

    # Double spine line at the scale break — visible spines on both sides of the gap
    ax_top.spines["bottom"].set_visible(True)
    ax_bottom.spines["top"].set_visible(True)
    ax_top.tick_params(labeltop=False, bottom=False, labelbottom=False)
    ax_bottom.tick_params(top=False)
    ax_bottom.set_xlim(-0.4, len(display_sizes) - 0.6)
    ax_bottom.set_xticks(xs_all)
    ax_bottom.set_xticklabels([str(s) for s in display_sizes])
    ax_bottom.set_xlabel(cfg.x_label)
    ax_bottom.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=title_fs, y=0.982 if cfg.slug == "annot" else 0.985)
    ax_top.tick_params(axis="y", labelsize=tick_fs)
    ax_bottom.tick_params(axis="both", labelsize=tick_fs)
    ax_bottom.xaxis.label.set_size(axis_fs)
    ax_bottom.yaxis.label.set_size(axis_fs)
    for ax in (ax_top, ax_bottom):
        ax.grid(True, alpha=0.35)

    shared._style_legend_right(ax_leg, ax_bottom, fontsize=legend_fs,
                                labelspacing=legend_ls, handlelength=legend_hl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.89, bottom=0.11, hspace=0.05, wspace=0.04)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved → {output_path}")


def _plot_runtime(cfg: DatasetConfig, rows: list[dict], output_path: Path) -> None:
    runtime_labels = [
        l for l in ("Marformer", "Stan Oracle")
        if any(_summary_lookup(rows, "runtime_minutes", l, s) is not None for s in cfg.display_sizes)
    ]
    fig_w = 17.6 if len(runtime_labels) >= 2 else 16.8
    if cfg.slug == "annot":
        fig = plt.figure(figsize=(17.8, 5.45))
        gs = fig.add_gridspec(1, 2, width_ratios=[6.8, 1.55], wspace=0.05)
        tick_fs, axis_fs, title_fs, legend_fs = 9.5, 11.5, 14.0, 9.75
    else:
        fig = plt.figure(figsize=(fig_w, 5.9))
        gs = fig.add_gridspec(1, 2, width_ratios=[6.3, 1.9], wspace=0.05)
        tick_fs, axis_fs, title_fs, legend_fs = 10.5, 12.5, 16.0, 10.5
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    xs_pos = {size: idx for idx, size in enumerate(cfg.display_sizes)}
    for label in ("Marformer", "Stan Oracle"):
        xs: list[float] = []
        ys: list[float] = []
        for size in cfg.display_sizes:
            row = _summary_lookup(rows, "runtime_minutes", label, size)
            if row is None:
                continue
            if label != "Marformer" and int(row.get("source_size", size)) != size:
                continue
            xs.append(float(xs_pos[size]))
            ys.append(float(row["mean"]))
        if not xs:
            continue
        linestyle, color, marker = shared._series_style(label)
        ax.plot(xs, ys, color=color, marker=marker, linestyle=linestyle,
                linewidth=shared._series_linewidth(label), alpha=shared._series_alpha(label), label=label)
    display_sizes = np.asarray(cfg.display_sizes, dtype=int)
    ax.set_xlim(-0.4, len(display_sizes) - 0.6)
    ax.set_xticks(np.arange(len(display_sizes), dtype=float))
    ax.set_xticklabels([str(s) for s in display_sizes])
    ax.set_xlabel(cfg.x_label)
    ax.set_ylabel("Runtime (minutes)")
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.xaxis.label.set_size(axis_fs)
    ax.yaxis.label.set_size(axis_fs)
    fig.suptitle(f"{cfg.title_prefix}: Runtime by {cfg.x_label}", fontsize=title_fs, y=0.98)
    ax.grid(True, alpha=0.35)
    shared._style_legend_right(ax_leg, ax, fontsize=legend_fs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.13, wspace=0.04)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved → {output_path}")


def _plot_calibration(
    cfg: DatasetConfig,
    rows: list[dict],
    all_payloads: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str | None]],
) -> None:
    cal_dir = cfg.out_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_models = [l for l in SERIES_ORDER if l != "Marformer NT"]
    for size in cfg.display_sizes:
        payloads_for_size: list[tuple[str, tuple[np.ndarray, np.ndarray], str]] = []
        for label in cal_models:
            row = _summary_lookup(rows, "log_loss", label, size)
            if row is None:
                continue
            source_size = int(row["source_size"])
            entry = all_payloads.get((label, source_size))
            if entry is None:
                continue
            probs, labels, _ = entry
            payloads_for_size.append((label, (probs, labels), label))
        if not payloads_for_size:
            continue
        ncols = 4 if len(payloads_for_size) > 6 else 3
        nrows = math.ceil(len(payloads_for_size) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 4.7 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for ax, (title, payload, color_key) in zip(axes, payloads_for_size):
            probs, labels = payload
            shared._plot_ece(ax, probs, labels, title, shared.COLORS.get(color_key, "#888888"))
        for ax in axes[len(payloads_for_size):]:
            ax.axis("off")
        fig.suptitle(f"{cfg.title_prefix}: Calibration at {size} {cfg.x_label}", fontsize=18, y=0.98)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92], pad=1.0, w_pad=1.8, h_pad=2.0)
        out_path = cal_dir / f"{cfg.slug}_calibration_size{size}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _plot_dataset(cfg: DatasetConfig, args: argparse.Namespace, device_name: str) -> None:
    rows, all_payloads = _build_rows_and_payloads(
        cfg, device_name, use_cache=args.cache, use_cache_baselines=args.cache_baselines
    )
    print(f"Saved summary → {cfg.summary_path}")
    _plot_metric(cfg, rows, "log_loss", "Test Log Loss",
                 f"{cfg.title_prefix}: Test Log Loss (mean ± 0.05 SD)",
                 cfg.out_dir / f"{cfg.slug}_generalization_log_loss.png")
    _plot_metric(cfg, rows, "mbr_l2", "MBR L2 (MSE)",
                 f"{cfg.title_prefix}: MBR L2 (mean ± 0.05 SD)",
                 cfg.out_dir / f"{cfg.slug}_generalization_mbr_l2.png")
    _plot_runtime(cfg, rows, cfg.out_dir / f"{cfg.slug}_generalization_runtime.png")
    if args.calibration:
        _plot_calibration(cfg, rows, all_payloads)


def main() -> None:
    args = parse_args()
    device_name = _choose_device(args.device)
    if args.part in {"all", "item"}:
        _plot_dataset(ITEM_CFG, args, device_name)
    if args.part in {"all", "annot"}:
        _plot_dataset(ANNOT_CFG, args, device_name)


if __name__ == "__main__":
    main()
