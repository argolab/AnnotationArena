#!/usr/bin/env python3
"""
Plot DOMAIN3-OLD generalization figures in the same style as DOMAIN3-FINAL.

Item generalization:
  - Marformer exact sizes: 100..1000, 2000, 3000, 4000
  - Stan / misspec / structured baselines: exact only through 500
  - For display sizes > 500, extend the last available <= 500 source result
  - Use a small x-axis jump marker between 1000 and 2000
  - Use a broken y-axis for log loss and MBR L2

Annotator generalization:
  - Marformer exact sizes: 10..50
  - Stan / misspec: plot exact existing sizes and skip missing ones
  - No x-axis cut

Outputs:
  - PLOTS/DOMAIN3-OLD/ITEM_GENERALIZATION/*
  - PLOTS/DOMAIN3-OLD/ANNOT_GENERALIZATION/*
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
from imputer.entity_mf.model import EntityMarformer
from imputer.entity_mf.test import _load_checkpoint, _reconstruct
from scripts.utils import plot_domain3_generalization as shared


ITEM_DISPLAY_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
ITEM_STAN_CUTOFF = 500
ITEM_MARFORMER_MAX_ITEM = 100
ITEM_X_LEFT = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
ITEM_X_RIGHT = [2000, 3000, 4000]
PROB_PREFIX = "prob_cat_"

SERIES_ORDER = [
    "Marformer",
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
    runtime_cache_path: Path
    stan_oracle_root: Path | None = None
    stan_oracle_eval_suffix: str | None = None
    misspec_root: Path | None = None
    x_break_enabled: bool = False
    x_left_sizes: list[int] | None = None
    x_right_sizes: list[int] | None = None
    stan_cutoff: int | None = None
    baseline_cutoff: int | None = None


ITEM_CFG = DatasetConfig(
    slug="item",
    title_prefix="Item Generalization",
    x_label="Training Items",
    display_sizes=ITEM_DISPLAY_SIZES,
    data_root=ROOT / "DATA/STAN/DOMAIN3-OLD/ItemSplits/Transductive",
    marformer_root=ROOT / "RESULTS/MARFORMER/DOMAIN3-OLD/ITEM",
    marformer_run_stem="DOMAIN3-OLD_Item_T_{size}_MARFORMER",
    out_dir=ROOT / "PLOTS/DOMAIN3-OLD/ITEM_GENERALIZATION",
    summary_path=ROOT / "PLOTS/DOMAIN3-OLD/ITEM_GENERALIZATION/item_generalization_summary.json",
    cache_path=ROOT / "PLOTS/DOMAIN3-OLD/ITEM_GENERALIZATION/item_generalization_cache.npz",
    runtime_cache_path=ROOT / "PLOTS/DOMAIN3-OLD/ITEM_GENERALIZATION/item_runtime_cache.json",
    stan_oracle_root=ROOT / "RESULTS/STAN/DOMAIN3-OLD/ORACLE/ITEM",
    stan_oracle_eval_suffix="TENSOR_eval",
    misspec_root=ROOT / "RESULTS/STAN/DOMAIN3-OLD/MISSPEC/ITEM",
    x_break_enabled=True,
    x_left_sizes=ITEM_X_LEFT,
    x_right_sizes=ITEM_X_RIGHT,
    stan_cutoff=ITEM_STAN_CUTOFF,
    baseline_cutoff=1000,
)

ANNOT_CFG = DatasetConfig(
    slug="annot",
    title_prefix="Annotator Generalization",
    x_label="Training Annotators",
    display_sizes=[10, 15, 20, 25, 30, 35, 40, 45, 50],
    data_root=ROOT / "DATA/STAN/DOMAIN3-OLD/AnnotSplits/Transductive",
    marformer_root=ROOT / "RESULTS/MARFORMER/DOMAIN3-OLD/ANNOT",
    marformer_run_stem="DOMAIN3-OLD_Annot_T_{size}_MARFORMER",
    out_dir=ROOT / "PLOTS/DOMAIN3-OLD/ANNOT_GENERALIZATION",
    summary_path=ROOT / "PLOTS/DOMAIN3-OLD/ANNOT_GENERALIZATION/annot_generalization_summary.json",
    cache_path=ROOT / "PLOTS/DOMAIN3-OLD/ANNOT_GENERALIZATION/annot_generalization_cache.npz",
    runtime_cache_path=ROOT / "PLOTS/DOMAIN3-OLD/ANNOT_GENERALIZATION/annot_runtime_cache.json",
    stan_oracle_root=ROOT / "RESULTS/STAN/DOMAIN3-OLD/ORACLE/ANNOT",
    stan_oracle_eval_suffix="TENSOR_eval",
    misspec_root=ROOT / "RESULTS/STAN/DOMAIN3-OLD/MISSPEC/ANNOT",
    baseline_cutoff=None,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DOMAIN3-OLD generalization figures.")
    parser.add_argument(
        "--use-summary",
        action="store_true",
        help="Reuse saved summary + cache instead of recomputing metrics and calibration payloads.",
    )
    parser.add_argument(
        "--use-marformer-summary-only",
        action="store_true",
        help="Reuse only cached Marformer summary/cache, while rebuilding Stan and structured baselines.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for Marformer evaluation; default is cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--part",
        choices=["all", "item", "annot"],
        default="all",
        help="Which portion of DOMAIN3-OLD to plot.",
    )
    return parser.parse_args()


def _choose_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
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
    payload = _read_json(path)
    value = payload.get(key)
    return None if value is None else float(value)


def _parse_best_epoch(checkpoint_name: str | None, total_epochs: int) -> int:
    if not checkpoint_name or checkpoint_name == "last.ckpt":
        return total_epochs
    match = re.search(r"epoch=(\d+)", checkpoint_name)
    if match is None:
        return total_epochs
    return int(match.group(1)) + 1


def _source_size_for_exact_or_extend(display_size: int, available_sizes: set[int], cutoff: int) -> int | None:
    if display_size <= cutoff:
        return display_size if display_size in available_sizes else None
    candidates = [size for size in available_sizes if size <= cutoff]
    return max(candidates) if candidates else None


def _source_size_for_cfg(cfg: DatasetConfig, display_size: int, available_sizes: set[int]) -> int | None:
    if cfg.slug == "annot" or cfg.stan_cutoff is None:
        return display_size if display_size in available_sizes else None
    return _source_size_for_exact_or_extend(display_size, available_sizes, cfg.stan_cutoff)


def _source_size_for_baseline_cfg(cfg: DatasetConfig, display_size: int, available_sizes: set[int]) -> int | None:
    if cfg.slug == "annot" or cfg.baseline_cutoff is None:
        return display_size if display_size in available_sizes else None
    return _source_size_for_exact_or_extend(display_size, available_sizes, cfg.baseline_cutoff)


def _marformer_run_dir(cfg: DatasetConfig, size: int) -> Path:
    return cfg.marformer_root / cfg.marformer_run_stem.format(size=size)


def _bundle_path(cfg: DatasetConfig, size: int) -> Path:
    return cfg.data_root / f"DOMAIN3-OLD_{'Item' if cfg.slug == 'item' else 'Annot'}_T_{size}" / "data_bundle.json"


def _test_missing_indices_and_labels(bundle: dict) -> tuple[list[int], np.ndarray]:
    idxs = list(bundle.get("missing_ratings_indexes_in_test_instance", []))
    labels = np.asarray([bundle["missing_ratings"][idx]["value"] - 1 for idx in idxs], dtype=np.int64)
    return idxs, labels


def _per_example_nll(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0)
    return -np.log(clipped)


def _per_example_mse(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    expected = probs @ classes
    truth = labels.astype(np.float64) + 1.0
    return (expected - truth) ** 2


def _available_marformer_sizes(cfg: DatasetConfig) -> set[int]:
    available: set[int] = set()
    for size in cfg.display_sizes:
        if _marformer_run_dir(cfg, size).exists():
            available.add(size)
    return available


def _available_oracle_sizes(cfg: DatasetConfig) -> set[int]:
    if cfg.stan_oracle_root is None or cfg.stan_oracle_eval_suffix is None:
        return set()
    available: set[int] = set()
    regex = re.compile(rf"DOMAIN3-OLD_{'Item' if cfg.slug == 'item' else 'Annot'}_T_(\d+)_{cfg.stan_oracle_eval_suffix}$")
    for path in cfg.stan_oracle_root.glob("*"):
        if not path.is_dir():
            continue
        match = regex.match(path.name)
        if match:
            available.add(int(match.group(1)))
    return available


def _available_misspec_sizes(cfg: DatasetConfig, suffix: str) -> set[int]:
    if cfg.misspec_root is None:
        return set()
    available: set[int] = set()
    regex = re.compile(rf"DOMAIN3-OLD_{'Item' if cfg.slug == 'item' else 'Annot'}_T_(\d+)_{suffix}_eval$")
    for path in cfg.misspec_root.glob("*"):
        if not path.is_dir():
            continue
        match = regex.match(path.name)
        if match:
            available.add(int(match.group(1)))
    return available


def _write_best_json_from_test_results(run_dir: Path) -> None:
    test_dir = run_dir / "TEST_RESULTS"
    candidates: list[tuple[float, float, str, dict]] = []
    for path in sorted(test_dir.glob("*.json")):
        if path.name in {"best.json", "launcher_metadata.json"}:
            continue
        data = _read_json(path)
        missing = data.get("missing", {})
        log_loss = missing.get("log_loss")
        rmse = missing.get("rmse")
        count = missing.get("n", 0)
        if log_loss is None or count == 0:
            continue
        candidates.append((
            float(log_loss),
            float(rmse) if rmse is not None else math.inf,
            path.name,
            data,
        ))

    if not candidates:
        raise RuntimeError(f"No valid checkpoint evaluation JSON found in {test_dir}")

    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    best_log_loss, best_rmse, best_name, best_data = candidates[0]
    best_data["selected_from"] = best_name
    best_data["selection_metric"] = "missing.log_loss"
    _write_json(test_dir / "best.json", best_data)
    print(f"  best checkpoint: {best_name}")
    print(f"  best missing log_loss={best_log_loss:.6f} rmse={best_rmse:.6f}")


def _ensure_marformer_best_json(run_dir: Path, device_name: str) -> Path:
    best_json = run_dir / "TEST_RESULTS" / "best.json"
    if best_json.exists():
        return best_json

    print(f"Running test.py checkpoint sweep for {run_dir.name}")
    (run_dir / "TEST_RESULTS").mkdir(parents=True, exist_ok=True)
    for checkpoint_mode in ("all", "last"):
        cmd = [
            sys.executable,
            "-m",
            "imputer.entity_mf.test",
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            checkpoint_mode,
            "--device",
            device_name,
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    _write_best_json_from_test_results(run_dir)
    return best_json


def _chunk_variables_by_item_limit(variables: list, max_item: int) -> list[list]:
    item_ids = sorted({iid for var in variables for iid in var.item_ids})
    if len(item_ids) <= max_item:
        return [variables]
    chunks: list[list] = []
    for start in range(0, len(item_ids), max_item):
        item_set = set(item_ids[start : start + max_item])
        chunk_vars = [var for var in variables if all(iid in item_set for iid in var.item_ids)]
        if chunk_vars:
            chunks.append(chunk_vars)
    return chunks


def _marformer_eval_max_item(cfg: DatasetConfig, size: int) -> int | None:
    if cfg.slug == "item":
        return None if size <= 700 else 300
    return None


@torch.no_grad()
def _marformer_probs_all_missing(run_dir: Path, checkpoint_name: str, max_item: int, device_name: str) -> tuple[np.ndarray, np.ndarray]:
    model, eval_vars, _train_cfg = _reconstruct(run_dir)
    device = torch.device(device_name)
    ckpt_path = run_dir / "checkpoints" / checkpoint_name
    _load_checkpoint(model, ckpt_path, device)
    model.to(device)
    model.eval()

    num_classes = model.types["rating"].num_classes
    prob_rows: list[np.ndarray] = []
    labels: list[int] = []

    chunked_vars = [eval_vars] if max_item is None else _chunk_variables_by_item_limit(eval_vars, max_item)
    for chunk_vars in chunked_vars:
        graph = variable_list_to_entity_graph(chunk_vars, model.types)
        params = model(graph, device=device)
        for idx, var in enumerate(chunk_vars):
            if not getattr(var, "is_missing", False):
                continue
            logits = params[0, idx, 1 : 1 + num_classes]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float64)
            prob_rows.append(probs)
            labels.append(int(var.rating_value))

    if not prob_rows:
        raise RuntimeError(f"No missing-token probabilities recovered for {run_dir.name} @ {checkpoint_name}")
    return np.vstack(prob_rows), np.asarray(labels, dtype=np.int64)


def _collect_marformer_payloads(cfg: DatasetConfig, device_name: str) -> tuple[dict[int, tuple[str, np.ndarray, np.ndarray]], dict[int, float], dict[int, float]]:
    payloads: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    runtime_minutes: dict[int, float] = {}
    mean_log_loss_override: dict[int, float] = {}

    for size in cfg.display_sizes:
        run_dir = _marformer_run_dir(cfg, size)
        if not run_dir.exists():
            continue
        best_json = _ensure_marformer_best_json(run_dir, device_name)
        best_payload = _read_json(best_json)
        checkpoint_name = best_payload.get("checkpoint")
        if not checkpoint_name:
            continue
        probs, labels = _marformer_probs_all_missing(
            run_dir,
            checkpoint_name,
            _marformer_eval_max_item(cfg, size),
            device_name,
        )
        payloads[size] = (checkpoint_name, probs, labels)
        missing_metrics = best_payload.get("missing", {})
        if missing_metrics.get("log_loss") is not None:
            mean_log_loss_override[size] = float(missing_metrics["log_loss"])

        total_seconds = _runtime_from_launcher(run_dir / "launcher_metadata.json", "elapsed_seconds")
        total_epochs = int(_read_json(run_dir / "train_config.json").get("training", {}).get("epochs", 300))
        if total_seconds is not None:
            epoch_count = _parse_best_epoch(checkpoint_name, total_epochs)
            runtime_minutes[size] = float(total_seconds) * (epoch_count / total_epochs) / 60.0

    return payloads, runtime_minutes, mean_log_loss_override


def _stan_probs_from_eval_dir(bundle_path: Path, eval_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    probs_path = eval_dir / "rating_probabilities.csv"
    if not probs_path.exists():
        return None
    bundle = _read_json(bundle_path)
    test_idxs, labels = _test_missing_indices_and_labels(bundle)
    if not test_idxs:
        return None
    df = pd.read_csv(probs_path)
    prob_cols = sorted(col for col in df.columns if col.startswith(PROB_PREFIX))
    grouped = (
        df[df["missing_rating_idx"].isin(test_idxs)]
        .groupby("missing_rating_idx")[prob_cols]
        .mean()
        .reindex(test_idxs)
    )
    if grouped.isnull().any().any():
        return None
    probs = grouped.to_numpy(dtype=np.float64)
    return probs, labels


def _collect_oracle_payloads(cfg: DatasetConfig) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, float]]:
    payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    runtime_minutes: dict[int, float] = {}
    if cfg.stan_oracle_root is None or cfg.stan_oracle_eval_suffix is None:
        return payloads, runtime_minutes

    for size in sorted(_available_oracle_sizes(cfg)):
        eval_dir = cfg.stan_oracle_root / f"DOMAIN3-OLD_{'Item' if cfg.slug == 'item' else 'Annot'}_T_{size}_{cfg.stan_oracle_eval_suffix}"
        payload = _stan_probs_from_eval_dir(_bundle_path(cfg, size), eval_dir)
        if payload is not None:
            payloads[size] = payload
        runtime = _runtime_from_launcher(cfg.stan_oracle_root / f"DOMAIN3-OLD_{'Item' if cfg.slug == 'item' else 'Annot'}_T_{size}_TENSOR" / "launcher_metadata.json", "fit_elapsed_seconds")
        if runtime is not None:
            runtime_minutes[size] = runtime / 60.0
    return payloads, runtime_minutes


def _collect_misspec_payloads(cfg: DatasetConfig, suffix: str) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, float]]:
    payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    runtime_minutes: dict[int, float] = {}
    if cfg.misspec_root is None:
        return payloads, runtime_minutes

    for size in sorted(_available_misspec_sizes(cfg, suffix)):
        eval_dir = cfg.misspec_root / f"DOMAIN3-OLD_{'Item' if cfg.slug == 'item' else 'Annot'}_T_{size}_{suffix}_eval"
        payload = _stan_probs_from_eval_dir(_bundle_path(cfg, size), eval_dir)
        if payload is not None:
            payloads[size] = payload
        runtime = _runtime_from_launcher(cfg.misspec_root / f"DOMAIN3-OLD_{'Item' if cfg.slug == 'item' else 'Annot'}_T_{size}_{suffix}" / "launcher_metadata.json", "fit_elapsed_seconds")
        if runtime is not None:
            runtime_minutes[size] = runtime / 60.0
    return payloads, runtime_minutes


def _collect_baseline_payloads(cfg: DatasetConfig, cutoff: int) -> tuple[dict[str, dict[int, tuple[np.ndarray, np.ndarray]]], dict[str, dict[int, float]]]:
    payloads: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {label: {} for label in shared.STRUCTURED_BASELINE_LABELS}
    runtime_minutes: dict[str, dict[int, float]] = {label: {} for label in shared.STRUCTURED_BASELINE_LABELS}
    key_map = {
        "Unigram Pool ij": "unigram_ij",
        "Naive Bayes IJK": "ijk",
        "Structured NB": "snb",
        "Structured Log-Linear": "log_linear",
    }
    for size in [s for s in cfg.display_sizes if s <= cutoff]:
        bundle_path = _bundle_path(cfg, size)
        if not bundle_path.exists():
            continue
        bundle, fitted = load_and_fit(
            bundle_path,
            fit_log_linear=True,
            log_linear_epochs=100,
            log_linear_l2_lambda=2.0,
        )
        calibration_payloads = calibration_probs_labels(fitted, bundle, split="test")
        if not calibration_payloads:
            continue
        for label in shared.STRUCTURED_BASELINE_LABELS:
            start = time.perf_counter()
            payload = calibration_payloads.get(key_map[label])
            if payload is None:
                continue
            probs, labels = payload
            runtime_minutes[label][size] = (time.perf_counter() - start) / 60.0
            payloads[label][size] = (probs.astype(np.float64), labels.astype(np.int64))
    return payloads, runtime_minutes


def _metric_row(model: str, display_size: int, source_size: int, metric: str, values: np.ndarray, mean_override: float | None = None) -> dict:
    mean = float(np.mean(values)) if mean_override is None else float(mean_override)
    std = float(np.std(values, ddof=0))
    q10, q25, median, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "dataset": "DOMAIN3-OLD",
        "model": model,
        "metric": metric,
        "size": int(display_size),
        "source_size": int(source_size),
        "mean": mean,
        "std": std,
        "q10": float(q10),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q90": float(q90),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "count": int(values.size),
    }


def _runtime_row(model: str, display_size: int, source_size: int, minutes: float) -> dict:
    return {
        "dataset": "DOMAIN3-OLD",
        "model": model,
        "metric": "runtime_minutes",
        "size": int(display_size),
        "source_size": int(source_size),
        "mean": float(minutes),
    }


def _save_cache(
    cache_path: Path,
    marformer_payloads: dict[int, tuple[str, np.ndarray, np.ndarray]],
    oracle_payloads: dict[int, tuple[np.ndarray, np.ndarray]],
    misspec_payloads: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    baseline_payloads: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for size, (checkpoint_name, probs, labels) in marformer_payloads.items():
        arrays[f"ckpt__marformer__{size}"] = np.asarray([checkpoint_name])
        arrays[f"probs__marformer__{size}"] = probs
        arrays[f"labels__marformer__{size}"] = labels
    for size, (probs, labels) in oracle_payloads.items():
        arrays[f"probs__stan_oracle__{size}"] = probs
        arrays[f"labels__stan_oracle__{size}"] = labels
    for model, payloads in misspec_payloads.items():
        model_key = _safe_key(model)
        for size, (probs, labels) in payloads.items():
            arrays[f"probs__{model_key}__{size}"] = probs
            arrays[f"labels__{model_key}__{size}"] = labels
    for model, payloads in baseline_payloads.items():
        model_key = _safe_key(model)
        for size, (probs, labels) in payloads.items():
            arrays[f"probs__{model_key}__{size}"] = probs
            arrays[f"labels__{model_key}__{size}"] = labels
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)


def _load_cache(cache_path: Path) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str | None]] | None:
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=False)
    out: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str | None]] = {}
    for key in data.files:
        if not key.startswith("probs__"):
            continue
        _, model_key, size_s = key.split("__")
        size = int(size_s)
        label = {
            "marformer": "Marformer",
            "stan_oracle": "Stan Oracle",
            "stan_both": "Stan Both",
            "stan_bin": "Stan Bin",
            "stan_proj": "Stan Proj",
            "unigram_pool_ij": "Unigram Pool ij",
            "naive_bayes_ijk": "Naive Bayes IJK",
            "structured_nb": "Structured NB",
            "structured_log_linear": "Structured Log-Linear",
        }.get(model_key)
        if label is None:
            continue
        labels_key = f"labels__{model_key}__{size}"
        if labels_key not in data:
            continue
        ckpt_name = None
        ckpt_key = f"ckpt__{model_key}__{size}"
        if ckpt_key in data:
            ckpt_name = str(data[ckpt_key][0])
        out[(label, size)] = (data[key], data[labels_key], ckpt_name)
    return out


def _load_marformer_from_summary_cache(
    cfg: DatasetConfig,
) -> tuple[dict[int, tuple[str, np.ndarray, np.ndarray]], dict[int, float], dict[int, float]]:
    if not cfg.summary_path.exists():
        raise FileNotFoundError(f"Missing summary for {cfg.slug}: {cfg.summary_path}")
    if not cfg.cache_path.exists():
        raise FileNotFoundError(f"Missing cache for {cfg.slug}: {cfg.cache_path}")

    payload = _read_json(cfg.summary_path)
    rows = payload.get("rows", [])
    cache = _load_cache(cfg.cache_path)
    if cache is None:
        raise RuntimeError(f"Failed to load cache: {cfg.cache_path}")

    marformer_payloads: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    marformer_runtimes: dict[int, float] = {}
    marformer_logloss_override: dict[int, float] = {}

    for size in cfg.display_sizes:
        cached = cache.get(("Marformer", size))
        if cached is not None:
            probs, labels, checkpoint_name = cached
            marformer_payloads[size] = (checkpoint_name or "unknown.ckpt", probs, labels)

        runtime_row = _summary_lookup(rows, "runtime_minutes", "Marformer", size)
        if runtime_row is not None:
            marformer_runtimes[size] = float(runtime_row["mean"])

        logloss_row = _summary_lookup(rows, "log_loss", "Marformer", size)
        if logloss_row is not None:
            marformer_logloss_override[size] = float(logloss_row["mean"])

    print(f"Used cached Marformer summary/cache -> {cfg.summary_path}")
    return marformer_payloads, marformer_runtimes, marformer_logloss_override


def _summary_rows_by_metric(rows: list[dict], metric: str) -> list[dict]:
    return [row for row in rows if row.get("metric") == metric]


def _summary_lookup(rows: list[dict], metric: str, model: str, display_size: int) -> dict | None:
    for row in rows:
        if row.get("metric") == metric and row.get("model") == model and int(row.get("size", -1)) == display_size:
            return row
    return None


def _plot_x_jump_marker(ax: plt.Axes, x: float, y: float = 0.0) -> None:
    trans = ax.get_xaxis_transform()
    dx = 0.12
    dy = 0.02
    kwargs = dict(transform=trans, color="0.25", linewidth=1.0, clip_on=False)
    ax.plot([x - dx, x - dx / 4], [y - dy, y + dy], **kwargs)
    ax.plot([x + dx / 4, x + dx], [y - dy, y + dy], **kwargs)


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
    xs_v = xs[valid]
    ys_v = ys[valid]
    lo_v = lo[valid]
    hi_v = hi[valid]
    linestyle, color, marker = shared._series_style(label)
    line_kwargs = dict(
        color=color,
        marker=marker,
        linewidth=shared._series_linewidth(label),
        alpha=shared._series_alpha(label),
        label=label,
    )

    is_stan_family = label in {"Stan Oracle", "Stan Both", "Stan Bin", "Stan Proj"}
    if not is_stan_family or np.all(display_sizes[valid] <= ITEM_STAN_CUTOFF):
        ax.plot(xs_v, ys_v, linestyle=linestyle, **line_kwargs)
        ax.fill_between(xs_v, lo_v, hi_v, color=color, alpha=shared._series_fill_alpha(label), linewidth=0)
        return

    sizes_v = display_sizes[valid]
    pre_mask = sizes_v <= ITEM_STAN_CUTOFF
    post_mask = sizes_v >= ITEM_STAN_CUTOFF
    if np.any(pre_mask):
        ax.plot(xs_v[pre_mask], ys_v[pre_mask], linestyle=linestyle, **line_kwargs)
        ax.fill_between(
            xs_v[pre_mask], lo_v[pre_mask], hi_v[pre_mask],
            color=color, alpha=shared._series_fill_alpha(label), linewidth=0,
        )
    if np.any(post_mask):
        ax.plot(
            xs_v[post_mask],
            ys_v[post_mask],
            linestyle=":",
            color=color,
            marker=marker,
            linewidth=shared._series_linewidth(label),
            alpha=shared._series_alpha(label),
            label="_nolegend_",
        )
        ax.fill_between(
            xs_v[post_mask], lo_v[post_mask], hi_v[post_mask],
            color=color, alpha=shared._series_fill_alpha(label), linewidth=0,
        )


def _series_from_rows(cfg: DatasetConfig, rows: list[dict], metric: str) -> dict[str, dict[str, list[float]]]:
    stats = {
        stat_name: {label: [np.nan] * len(cfg.display_sizes) for label in SERIES_ORDER}
        for stat_name in ("mean", "std")
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


def _plot_metric(cfg: DatasetConfig, rows: list[dict], metric: str, ylabel: str, title: str, output_path: Path) -> None:
    stats = _series_from_rows(cfg, rows, metric)
    present_labels = [
        label
        for label in SERIES_ORDER
        if any(not math.isnan(v) for v in stats["mean"][label])
    ]
    has_dense_legend = len(present_labels) >= 9 or any(len(label) >= 20 for label in present_labels)
    if cfg.slug == "annot":
        legend_ratio = 1.75 if has_dense_legend else 1.5
        left_ratio = 6.9 if has_dense_legend else 6.7
        fig_width = 18.6 if has_dense_legend else 17.8
        fig_height = 7.15
        tick_fontsize = 9.5
        axis_label_fontsize = 11.5
        title_fontsize = 14.0
        legend_fontsize = 9.5 if has_dense_legend else 10.0
        legend_labelspacing = 0.52 if has_dense_legend else 0.62
        legend_handlelength = 2.0 if has_dense_legend else 2.2
    else:
        legend_ratio = 2.15 if has_dense_legend else 1.6
        left_ratio = 6.15 if has_dense_legend else 6.3
        fig_width = 18.8 if has_dense_legend else 17.6
        fig_height = 7.6
        tick_fontsize = 10.5
        axis_label_fontsize = 12.5
        title_fontsize = 16.0
        legend_fontsize = 10.0 if has_dense_legend else 11.0
        legend_labelspacing = 0.58 if has_dense_legend else 0.7
        legend_handlelength = 2.2 if has_dense_legend else 2.4
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[left_ratio, legend_ratio],
        height_ratios=[1.0, 4.8],
        hspace=0.05,
        wspace=0.05,
    )
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
            line_kwargs = dict(
                color=color,
                marker=marker,
                linewidth=shared._series_linewidth(label),
                alpha=shared._series_alpha(label),
                label=label,
            )
            ax_top.plot(xs_all[valid], ys[valid], linestyle=linestyle, **line_kwargs)
            ax_bottom.plot(xs_all[valid], ys[valid], linestyle=linestyle, **line_kwargs)
            ax_top.fill_between(
                xs_all[valid], lo[valid], hi[valid],
                color=color, alpha=shared._series_fill_alpha(label), linewidth=0,
            )
            ax_bottom.fill_between(
                xs_all[valid], lo[valid], hi[valid],
                color=color, alpha=shared._series_fill_alpha(label), linewidth=0,
            )

    if metric == "log_loss":
        lower_ticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        upper_ticks = [1.0, 1.5, 2.0, 2.5, 3.0]
        ax_bottom.set_ylim(0.2, 0.82)
        ax_bottom.set_yticks(lower_ticks)
        ax_top.set_ylim(0.95, 3.05)
        ax_top.set_yticks(upper_ticks)
    else:
        lower_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        upper_ticks = [1.0, 1.5, 2.0, 2.5]
        ax_bottom.set_ylim(0.0, 0.52)
        ax_bottom.set_yticks(lower_ticks)
        ax_top.set_ylim(0.85, 2.55)
        ax_top.set_yticks(upper_ticks)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False, labelbottom=False)
    ax_bottom.tick_params(top=False)

    ax_bottom.set_xlim(-0.4, len(display_sizes) - 0.6)
    ax_bottom.set_xticks(xs_all)
    ax_bottom.set_xticklabels([str(size) for size in display_sizes])
    ax_bottom.set_xlabel(cfg.x_label)
    ax_bottom.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=title_fontsize, y=0.982 if cfg.slug == "annot" else 0.985)

    ax_top.tick_params(axis="y", labelsize=tick_fontsize)
    ax_bottom.tick_params(axis="both", labelsize=tick_fontsize)
    ax_bottom.xaxis.label.set_size(axis_label_fontsize)
    ax_bottom.yaxis.label.set_size(axis_label_fontsize)

    xs_break = np.linspace(0.0, 1.0, 61)
    top_break = np.where(np.arange(xs_break.size) % 2 == 0, -0.005, 0.005)
    bottom_break = np.where(np.arange(xs_break.size) % 2 == 0, 1.005, 0.995)
    ax_top.plot(xs_break, top_break, transform=ax_top.transAxes, color="0.25", linewidth=1.0, clip_on=False)
    ax_bottom.plot(xs_break, bottom_break, transform=ax_bottom.transAxes, color="0.25", linewidth=1.0, clip_on=False)
    if cfg.slug == "item" and 1000 in cfg.display_sizes and 2000 in cfg.display_sizes:
        jump_idx = cfg.display_sizes.index(1000) + 0.5
        _plot_x_jump_marker(ax_bottom, x=float(jump_idx), y=0.0)

    for ax in (ax_top, ax_bottom):
        ax.grid(True, alpha=0.35)

    shared._style_legend_right(
        ax_leg,
        ax_bottom,
        fontsize=legend_fontsize,
        labelspacing=legend_labelspacing,
        handlelength=legend_handlelength,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        top=0.89,
        bottom=0.11,
        hspace=0.05,
        wspace=0.04,
    )
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _plot_runtime(cfg: DatasetConfig, rows: list[dict], output_path: Path) -> None:
    runtime_labels = [
        label
        for label in ("Marformer", "Stan Oracle")
        if _summary_lookup(rows, "runtime_minutes", label, cfg.display_sizes[0]) is not None
        or any(_summary_lookup(rows, "runtime_minutes", label, size) is not None for size in cfg.display_sizes)
    ]
    has_dense_legend = len(runtime_labels) >= 2 and cfg.slug == "annot"
    if cfg.slug == "annot":
        fig = plt.figure(figsize=(17.8, 5.45))
        gs = fig.add_gridspec(1, 2, width_ratios=[6.8, 1.55], wspace=0.05)
        tick_fontsize = 9.5
        axis_label_fontsize = 11.5
        title_fontsize = 14.0
        legend_fontsize = 9.75
    else:
        fig = plt.figure(figsize=(17.6 if has_dense_legend else 16.8, 5.9))
        gs = fig.add_gridspec(1, 2, width_ratios=[6.3, 1.9 if has_dense_legend else 1.6], wspace=0.05)
        tick_fontsize = 10.5
        axis_label_fontsize = 12.5
        title_fontsize = 16.0
        legend_fontsize = 10.5 if has_dense_legend else 11.0
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    display_sizes = np.asarray(cfg.display_sizes, dtype=int)
    xs_pos = {size: idx for idx, size in enumerate(cfg.display_sizes)}

    for label in ["Marformer", "Stan Oracle"]:
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
        ax.plot(
            xs, ys, color=color, marker=marker, linestyle=linestyle,
            linewidth=shared._series_linewidth(label), alpha=shared._series_alpha(label), label=label,
        )

    ax.set_xlim(-0.4, len(display_sizes) - 0.6)
    ax.set_xticks(np.arange(len(display_sizes), dtype=float))
    ax.set_xticklabels([str(size) for size in display_sizes])
    ax.set_xlabel(cfg.x_label)
    ax.set_ylabel("Runtime (minutes)")
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.xaxis.label.set_size(axis_label_fontsize)
    ax.yaxis.label.set_size(axis_label_fontsize)
    fig.suptitle(f"{cfg.title_prefix}: Runtime by {cfg.x_label}", fontsize=title_fontsize, y=0.98)
    if cfg.slug == "item" and 1000 in cfg.display_sizes and 2000 in cfg.display_sizes:
        jump_idx = cfg.display_sizes.index(1000) + 0.5
        _plot_x_jump_marker(ax, x=float(jump_idx), y=0.0)
    ax.grid(True, alpha=0.35)

    shared._style_legend_right(ax_leg, ax, fontsize=legend_fontsize)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.13, wspace=0.04)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved -> {output_path}")


def _payload_from_cache(cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str | None]], model: str, source_size: int) -> tuple[np.ndarray, np.ndarray] | None:
    payload = cache.get((model, source_size))
    if payload is None:
        return None
    return payload[0], payload[1]


def _plot_calibration(cfg: DatasetConfig, rows: list[dict], cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str | None]] | None) -> None:
    if cache is None:
        print(f"Skipping calibration for {cfg.slug}: cache is unavailable.")
        return

    cal_dir = cfg.out_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    calibration_models = ["Marformer", "Stan Oracle", "Stan Both", "Stan Bin", "Stan Proj", "Unigram Pool ij", "Naive Bayes IJK", "Structured NB", "Structured Log-Linear"]

    for size in cfg.display_sizes:
        payloads: list[tuple[str, tuple[np.ndarray, np.ndarray], str]] = []
        for label in calibration_models:
            row = _summary_lookup(rows, "log_loss", label, size)
            if row is None:
                continue
            source_size = int(row["source_size"])
            payload = _payload_from_cache(cache, label, source_size)
            if payload is None:
                continue
            payloads.append((label, payload, label))
        if not payloads:
            continue

        ncols = 4 if len(payloads) > 6 else 3
        nrows = math.ceil(len(payloads) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 4.7 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for ax, (title, payload, color_key) in zip(axes, payloads):
            probs, labels = payload
            shared._plot_ece(ax, probs, labels, title, shared.COLORS[color_key])
        for ax in axes[len(payloads):]:
            ax.axis("off")

        fig.suptitle(f"{cfg.title_prefix}: Calibration at {size} {cfg.x_label}", fontsize=18, y=0.98)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92], pad=1.0, w_pad=1.8, h_pad=2.0)
        output_path = cal_dir / f"{cfg.slug}_generalization_calibration_size{size}.png"
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved -> {output_path}")


def _build_rows_and_cache(
    cfg: DatasetConfig,
    device_name: str,
    use_marformer_summary_only: bool = False,
) -> tuple[list[dict], dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str | None]]]:
    rows: list[dict] = []

    if use_marformer_summary_only:
        marformer_payloads, marformer_runtimes, marformer_logloss_override = _load_marformer_from_summary_cache(cfg)
    else:
        marformer_payloads, marformer_runtimes, marformer_logloss_override = _collect_marformer_payloads(cfg, device_name)
    oracle_payloads, oracle_runtimes = _collect_oracle_payloads(cfg)
    misspec_specs = [
        ("Stan Both", "BOTH"),
        ("Stan Bin", "BIN"),
        ("Stan Proj", "PROJ"),
    ]
    misspec_payloads: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    misspec_runtimes: dict[str, dict[int, float]] = {}
    for label, suffix in misspec_specs:
        payloads, runtimes = _collect_misspec_payloads(cfg, suffix)
        misspec_payloads[label] = payloads
        misspec_runtimes[label] = runtimes

    baseline_payloads, baseline_runtimes = _collect_baseline_payloads(
        cfg,
        cfg.baseline_cutoff if cfg.baseline_cutoff is not None else max(cfg.display_sizes),
    )

    available_oracle_sizes = set(oracle_payloads.keys())
    available_misspec_sizes = {label: set(payloads.keys()) for label, payloads in misspec_payloads.items()}
    available_baseline_sizes = {label: set(payloads.keys()) for label, payloads in baseline_payloads.items()}

    for size in cfg.display_sizes:
        if size in marformer_payloads:
            checkpoint_name, probs, labels = marformer_payloads[size]
            rows.append(_metric_row(
                "Marformer", size, size, "log_loss", _per_example_nll(probs, labels),
                mean_override=marformer_logloss_override.get(size),
            ))
            rows.append(_metric_row("Marformer", size, size, "mbr_l2", _per_example_mse(probs, labels)))
            runtime = marformer_runtimes.get(size)
            if runtime is not None:
                rows.append(_runtime_row("Marformer", size, size, runtime))

        oracle_source = _source_size_for_cfg(cfg, size, available_oracle_sizes)
        if oracle_source is not None:
            probs, labels = oracle_payloads[oracle_source]
            rows.append(_metric_row("Stan Oracle", size, oracle_source, "log_loss", _per_example_nll(probs, labels)))
            rows.append(_metric_row("Stan Oracle", size, oracle_source, "mbr_l2", _per_example_mse(probs, labels)))
            if size in oracle_runtimes:
                rows.append(_runtime_row("Stan Oracle", size, size, oracle_runtimes[size]))

        for label, _suffix in misspec_specs:
            source_size = _source_size_for_cfg(cfg, size, available_misspec_sizes[label])
            if source_size is None:
                continue
            probs, labels = misspec_payloads[label][source_size]
            rows.append(_metric_row(label, size, source_size, "log_loss", _per_example_nll(probs, labels)))
            rows.append(_metric_row(label, size, source_size, "mbr_l2", _per_example_mse(probs, labels)))
            if size in misspec_runtimes[label]:
                rows.append(_runtime_row(label, size, size, misspec_runtimes[label][size]))

        for label in shared.STRUCTURED_BASELINE_LABELS:
            source_size = _source_size_for_baseline_cfg(cfg, size, available_baseline_sizes[label])
            if source_size is None:
                continue
            probs, labels = baseline_payloads[label][source_size]
            rows.append(_metric_row(label, size, source_size, "log_loss", _per_example_nll(probs, labels)))
            rows.append(_metric_row(label, size, source_size, "mbr_l2", _per_example_mse(probs, labels)))
            if size in baseline_runtimes[label]:
                rows.append(_runtime_row(label, size, size, baseline_runtimes[label][size]))

    _write_json(cfg.summary_path, {
        "dataset": cfg.slug,
        "display_sizes": cfg.display_sizes,
        "stan_cutoff": cfg.stan_cutoff,
        "baseline_cutoff": cfg.baseline_cutoff,
        "marformer_eval_max_item_rule": "item: None through 700, then 300; annot: None",
        "rows": rows,
        "marformer_selection": {
            str(size): {"checkpoint": checkpoint_name}
            for size, (checkpoint_name, _probs, _labels) in marformer_payloads.items()
        },
    })
    _save_cache(cfg.cache_path, marformer_payloads, oracle_payloads, misspec_payloads, baseline_payloads)
    cache = _load_cache(cfg.cache_path)
    return rows, {} if cache is None else cache


def _plot_dataset(
    cfg: DatasetConfig,
    use_summary: bool,
    device_name: str,
    use_marformer_summary_only: bool = False,
) -> None:
    if use_summary:
        if not cfg.summary_path.exists():
            raise FileNotFoundError(f"Missing summary for {cfg.slug}: {cfg.summary_path}")
        if not cfg.cache_path.exists():
            raise FileNotFoundError(f"Missing cache for {cfg.slug}: {cfg.cache_path}")
        payload = _read_json(cfg.summary_path)
        rows = payload.get("rows", [])
        cache = _load_cache(cfg.cache_path)
        if cache is None:
            raise RuntimeError(f"Failed to load cache: {cfg.cache_path}")
        print(f"Used cached summary -> {cfg.summary_path}")
    else:
        rows, cache = _build_rows_and_cache(
            cfg,
            device_name,
            use_marformer_summary_only=use_marformer_summary_only,
        )
        print(f"Saved summary -> {cfg.summary_path}")

    _plot_metric(
        cfg,
        rows,
        metric="log_loss",
        ylabel="Test Log Loss",
        title=f"{cfg.title_prefix}: Test Log Loss with Additional {cfg.x_label} (mean ± 0.05 SD)",
        output_path=cfg.out_dir / f"{cfg.slug}_generalization_log_loss.png",
    )
    _plot_metric(
        cfg,
        rows,
        metric="mbr_l2",
        ylabel="MBR L2 (MSE)",
        title=f"{cfg.title_prefix}: MBR L2 with Additional {cfg.x_label} (mean ± 0.05 SD)",
        output_path=cfg.out_dir / f"{cfg.slug}_generalization_mbr_l2.png",
    )
    _plot_runtime(cfg, rows, cfg.out_dir / f"{cfg.slug}_generalization_runtime.png")
    _plot_calibration(cfg, rows, cache)


def _plot_item(
    cfg: DatasetConfig,
    use_summary: bool,
    device_name: str,
    use_marformer_summary_only: bool = False,
) -> None:
    _plot_dataset(cfg, use_summary, device_name, use_marformer_summary_only=use_marformer_summary_only)


def _plot_annot(
    cfg: DatasetConfig,
    use_summary: bool,
    device_name: str,
    use_marformer_summary_only: bool = False,
) -> None:
    _plot_dataset(cfg, use_summary, device_name, use_marformer_summary_only=use_marformer_summary_only)


def main() -> None:
    args = parse_args()
    if args.use_summary and args.use_marformer_summary_only:
        raise ValueError("Use either --use-summary or --use-marformer-summary-only, not both.")
    device_name = _choose_device(args.device)
    shared.MISSPEC_ROOT = ROOT / "RESULTS/STAN/DOMAIN3-OLD/MISSPEC"
    shared.MISSPEC_MODELS = [
        ("Stan Both", "BOTH"),
        ("Stan Bin", "BIN"),
        ("Stan Proj", "PROJ"),
    ]
    if args.part in {"all", "item"}:
        _plot_item(
            ITEM_CFG,
            use_summary=args.use_summary,
            device_name=device_name,
            use_marformer_summary_only=args.use_marformer_summary_only,
        )
    if args.part in {"all", "annot"}:
        _plot_annot(
            ANNOT_CFG,
            use_summary=args.use_summary,
            device_name=device_name,
            use_marformer_summary_only=args.use_marformer_summary_only,
        )


if __name__ == "__main__":
    main()
