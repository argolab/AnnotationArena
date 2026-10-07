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
from scripts.utils.font_scale import fs, pfs, ts
from scripts.utils import plot_domain3_generalization as shared


ITEM_DISPLAY_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
# None = no cutoff. This was 500, a placeholder for item sizes Stan had not been
# fitted at: beyond it the Stan series were drawn as a FLAT DOTTED extension of
# their K=500 value, i.e. a line that is not data. The final rerun fits all four
# Stan variants (Oracle, Bin, Proj, Both) to K=4000 -- Oracle reaches 0.600 at
# 4000 against 0.679 at 500 -- so the placeholder understated Stan and hid that
# it keeps improving. Setting this to an int restores the old behaviour.
ITEM_STAN_CUTOFF: int | None = None
# No cap. This was 500, from a time when the item NT runs above 500 did not
# exist; it silently truncated the NT curve at K=500 even after the final run
# produced all 13 NT cells, so the figure showed five NT points against a
# thirteen-point table. _available_marformer_nt_sizes already requires both the
# run directory and the NT bundle to be on disk, which is the real gate.
ITEM_MARFORMER_NT_MAX: int | None = None
ITEM_X_BREAK_AFTER = 1000
ITEM_EXCLUDED_SERIES: frozenset[str] = frozenset({"Naive Bayes IJK"})
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

STAN_SERIES = ["Stan Oracle", "Stan Both", "Stan Bin", "Stan Proj"]
CLEAN_SERIES = ["Marformer", *STAN_SERIES]
CLEAN_NT_SERIES = ["Marformer", "Marformer NT", *STAN_SERIES]

# Local color overrides: Marformer→crimson, Stan→shades of blue, Baselines→greys
COLORS: dict[str, str] = {
    "Marformer":             "#7a1f3d",
    "Marformer NT":          "#c47088",  # lighter crimson, same hue
    "Stan Oracle":           "#0d47a1",  # darkest blue
    "Stan Both":             "#1565c0",
    "Stan Bin":              "#1976d2",
    "Stan Proj":             "#42a5f5",  # lightest blue
    "Unigram Pool ij":       "#616161",
    "Naive Bayes IJK":       "#757575",
    "Structured NB":         "#9e9e9e",
    "Structured Log-Linear": "#424242",
}


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
    marformer_nt_root=ROOT / "RESULTS/MARFORMER-NT/DOMAIN3-SIGMOID/ANNOT",
    marformer_nt_run_stem="DOMAIN3-SIGMOID_Annot_NT_{size}_MARFORMER",
    nt_data_root=ROOT / "DATA/STAN/DOMAIN3-SIGMOID/AnnotSplits/NonTransductive",
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
    parser.add_argument("--metric-population", choices=["corner", "pooled"],
                        default="corner",
                        help="corner (default) scores Marformer on the held-out "
                             "test block only, the same population Stan is "
                             "scored on. pooled mixes in the training "
                             "partition's own missing cells (D22/D23) and only "
                             "reproduces the older published figures.")
    parser.add_argument("--checkpoint-policy", choices=["last", "test-argmin"],
                        default="last",
                        help="last (default) reports the refit's final epoch, "
                             "which is the F13 protocol and matches the result "
                             "tables. test-argmin scores every periodic "
                             "checkpoint and keeps the best ON TEST -- that is "
                             "selection on test, and only reproduces the older "
                             "published figures.")
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
    nt_max = ITEM_MARFORMER_NT_MAX if cfg.slug == "item" else None
    available: set[int] = set()
    for size in cfg.display_sizes:
        if nt_max is not None and size > nt_max:
            continue
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

# Which checkpoint the figures report. "last" is the F13 protocol: the refit's
# final epoch, with the stopping epoch chosen beforehand on a validation block
# carved out of TRAINING data. "test-argmin" is what this script used to do
# unconditionally -- score every periodic checkpoint on test and keep the
# minimum -- which is selection on test (D14) and the exact thing F13 exists to
# remove. It is kept only to reproduce the previously published figures.
CHECKPOINT_POLICY = "last"

# Which cells the Marformer series is scored on. "corner" is the held-out test
# block only -- `var.instance == "test"` -- which is what Stan is already scored
# on (`missing_ratings_indexes_in_test_instance`). "pooled" is every missing
# token in the evaluation graph, i.e. the test block mixed with the training
# partition's own missing cells, which is what this script used to do: D22/D23.
#
# Those training cells are in-domain completion on items a transductive model has
# mostly seen, so they are easier, and their share GROWS with K -- 414 test cells
# against 13,836 training cells at Item_T_4000, so the pooled number is 97%
# training data there. Marformer was therefore scored on a progressively easier
# mixture while Stan was scored on the hard task, in the same figure.
METRIC_POPULATION = "corner"


def _write_best_json_from_test_results(run_dir: Path) -> None:
    """Write TEST_RESULTS/best.json: the checkpoint the figures will report.

    Under the default "last" policy this is a copy of last.json, so the figures
    agree with the tables rendered by RERUN/analysis/result_tables.py. Under
    "test-argmin" it is the minimum over every scored checkpoint, which is
    optimistic: measured on final-dual, by a median of 0.006 and up to 0.047
    nats across 23 of 34 cells.
    """
    test_dir = run_dir / "TEST_RESULTS"
    if CHECKPOINT_POLICY == "last":
        last = test_dir / "last.json"
        if not last.is_file():
            raise RuntimeError(f"{last} missing; cannot apply the 'last' policy")
        data = _read_json(last)
        data["selected_from"] = "last.json"
        data["selection_metric"] = "none (F13: last epoch of the refit)"
        _write_json(test_dir / "best.json", data)
        print(f"  checkpoint: last.ckpt (F13)  "
              f"log_loss={data.get('missing', {}).get('log_loss')}")
        return
    _write_best_json_test_argmin(run_dir)


def _write_best_json_test_argmin(run_dir: Path) -> None:
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


def _chunk_variables_by_item_limit(variables: list, max_item: int,
                                   keep_whole: set | None = None) -> list[list]:
    """Group variables into chunks of at most `max_item` distinct items.

    `keep_whole` is a set of item ids that must never be split across chunks and
    must always share a chunk with training context. The test corner is always
    the highest item ids, so a contiguous cut drops it at the tail, and whenever
    the item count lands just past a chunk boundary part of the corner ends up in
    an orphan chunk with no training context at all -- at Item_NT_4000 that put
    239 of 414 corner cells in a 36-item chunk and they scored 1.29 against 0.86
    for the cells that kept context. Mirrors
    RERUN/analysis/split_missing_metrics.py::chunk_by_item.
    """
    if keep_whole:
        ids = sorted({i for v in variables for i in v.item_ids})
        protected = sorted(set(keep_whole) & set(ids))
        rest = [i for i in ids if i not in set(protected)]
        if protected:
            budget = max(0, max_item - len(protected))
            if budget <= 0:
                groups = [set(protected)]
                groups += [set(rest[i:i + max_item]) for i in range(0, len(rest), max_item)]
            else:
                groups = [set(protected) | set(rest[-budget:])]
                head = rest[:-budget]
                groups += [set(head[i:i + max_item]) for i in range(0, len(head), max_item)]
            out = []
            for keep in groups:
                sel = [v for v in variables if all(j in keep for j in v.item_ids)]
                if sel:
                    out.append(sel)
            return out or [variables]
    return _chunk_variables_contiguous(variables, max_item)


def _chunk_variables_contiguous(variables: list, max_item: int) -> list[list]:
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
        return 500
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
    corner_only = METRIC_POPULATION == "corner"
    keep = ({i for v in eval_vars if getattr(v, "instance", None) == "test"
             for i in v.item_ids} if corner_only else None)
    chunks = ([eval_vars] if max_item is None
              else _chunk_variables_by_item_limit(eval_vars, max_item, keep_whole=keep))
    for chunk_vars in chunks:
        graph = variable_list_to_entity_graph(chunk_vars, model.types)
        params = model(graph, device=device)
        for idx, var in enumerate(chunk_vars):
            if not getattr(var, "is_missing", False):
                continue
            # The corner is the held-out test block, the population Stan is
            # scored on. Keeping every missing token instead pools in the
            # training partition's own cells (D22/D23).
            if corner_only and getattr(var, "instance", None) != "test":
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
        # best.json's "missing" field is the POOLED loss. Under the corner
        # policy the log-loss must come from the same corner payloads as every
        # other metric in the figure, so there is nothing to override -- using
        # it would reintroduce D22 for log-loss alone and leave log-loss and
        # mbr_l2 measured on different populations.
        ov = _pooled_logloss_override(best)
        if ov is not None:
            logloss_override[size] = ov
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
        payload, runtime, logloss = _collect_one_marformer_nt_payload(cfg, size, run_dir, device_name)
        if payload is None:
            continue
        payloads[size] = payload
        if logloss is not None:
            logloss_override[size] = logloss
        if runtime is not None:
            runtimes[size] = runtime
    return payloads, runtimes, logloss_override


def _collect_one_marformer_nt_payload(
    cfg: DatasetConfig, size: int, run_dir: Path, device_name: str
) -> tuple[tuple[str, np.ndarray, np.ndarray] | None, float | None, float | None]:
    best_json = _ensure_marformer_best_json(run_dir, device_name)
    best = _read_json(best_json)
    ckpt = best.get("checkpoint")
    if not ckpt:
        return None, None, None
    probs, labels = _marformer_probs_all_missing(
        run_dir, ckpt, _marformer_eval_max_item(cfg, size), device_name
    )
    # Same gate as the transductive path: best.json's "missing" block is the
    # POOLED metric, which mixes in the training partition's own held-out cells.
    # Under the default corner population the curve must come from the probs
    # computed above, or the NT series is measured on a different population
    # from the T series beside it -- which is exactly D22/D23, surviving on the
    # NT path alone after the T path was fixed.
    logloss = _pooled_logloss_override(best)
    runtime = None
    total_s = _runtime_from_launcher(run_dir / "launcher_metadata.json", "elapsed_seconds")
    total_epochs = int(_read_json(run_dir / "train_config.json").get("training", {}).get("epochs", 300))
    if total_s is not None:
        runtime_epochs = _parse_best_epoch(ckpt, total_epochs)
        runtime = float(total_s) * (runtime_epochs / total_epochs) / 60.0
    return (ckpt, probs, labels), runtime, logloss


def _collect_missing_marformer_nt_payloads(
    cfg: DatasetConfig,
    cached_nt_sizes: set[int],
    device_name: str,
) -> tuple[dict[int, tuple[str, np.ndarray, np.ndarray]], dict[int, float], dict[int, float]]:
    payloads: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    runtimes: dict[int, float] = {}
    logloss_override: dict[int, float] = {}
    missing_sizes = sorted(_available_marformer_nt_sizes(cfg) - cached_nt_sizes)
    for size in missing_sizes:
        run_dir = _marformer_nt_run_dir(cfg, size)
        if run_dir is None:
            continue
        print(f"[Marformer NT cache refresh] size={size} missing from cache; collecting...")
        payload, runtime, logloss = _collect_one_marformer_nt_payload(cfg, size, run_dir, device_name)
        if payload is None:
            continue
        payloads[size] = payload
        if logloss is not None:
            logloss_override[size] = logloss
        if runtime is not None:
            runtimes[size] = runtime
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
            bundle_path, fit_log_linear=True, log_linear_epochs=100
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

def _pooled_logloss_override(best: dict) -> float | None:
    """best.json's pooled "missing" log-loss, but only if pooled was asked for.

    This is the single place that decides. best.json reports the POOLED metric,
    which mixes the training partition's own held-out cells in with the test
    corner and whose share grows with K; the curve must otherwise come from the
    corner probabilities, or the figures measure a different quantity from the
    tables. There were FOUR copies of this decision -- the transductive
    collector, the NT collector, and both halves of the cached path -- and
    gating only the first left the NT arm wrong in the figures and left a
    --cache rebuild able to revert both arms silently.
    """
    if METRIC_POPULATION != "pooled":
        return None
    v = best.get("missing", {}).get("log_loss")
    return None if v is None else float(v)


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
            ll = _pooled_logloss_override(best)
            if ll is not None:
                logloss[size] = ll
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
            ll = _pooled_logloss_override(best)
            if ll is not None:
                nt_logloss[size] = ll
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
        new_nt_payloads, new_nt_runtimes, new_nt_logloss = _collect_missing_marformer_nt_payloads(
            cfg, set(nt_payloads), device_name
        )
        if new_nt_payloads:
            nt_payloads.update(new_nt_payloads)
            nt_runtimes.update(new_nt_runtimes)
            nt_logloss.update(new_nt_logloss)
            _save_marformer_cache(cfg.cache_path, marformer_payloads, nt_payloads)
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

        nt_capped = (cfg.slug == "item" and ITEM_MARFORMER_NT_MAX is not None
                     and size > ITEM_MARFORMER_NT_MAX)
        if size in nt_payloads and not nt_capped:
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
            if cfg.slug == "item" and label in ITEM_EXCLUDED_SERIES:
                continue
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
        if cfg.slug == "item" and label in ITEM_EXCLUDED_SERIES:
            continue
        for size, (probs, labels) in p.items():
            all_payloads[(label, size)] = (probs, labels, None)

    return rows, all_payloads


# ── Plotting ──────────────────────────────────────────────────────────────────

def _local_series_style(label: str) -> tuple[str, str, str]:
    """Return (linestyle, color, marker) using local COLORS override."""
    linestyle, _orig_color, marker = shared._series_style(label)
    color = COLORS.get(label, "#888888")
    if label == "Marformer NT":
        linestyle = ":"
    return linestyle, color, marker


def _draw_xaxis_break_marks(ax: plt.Axes, display_sizes: np.ndarray, break_after_size: int) -> None:
    """Draw '//' x-axis jump marker between break_after_size and the next size."""
    sizes = list(display_sizes)
    if break_after_size not in sizes:
        return
    idx = sizes.index(break_after_size)
    if idx + 1 >= len(sizes):
        return
    x = float(idx) + 0.5
    trans = ax.get_xaxis_transform()
    dx, dy = 0.12, 0.02
    kw = dict(transform=trans, color="0.25", linewidth=1.0, clip_on=False)
    ax.plot([x - dx, x - dx / 4], [-dy, dy], **kw)
    ax.plot([x + dx / 4, x + dx], [-dy, dy], **kw)


def _metric_y_ticks(cfg: DatasetConfig, metric: str) -> tuple[list[float], tuple[float, float]]:
    """Return y tick anchors for a single-axis compressed scale."""
    if cfg.slug == "item":
        if metric == "log_loss":
            # Starts at 0.60, not 0.70: Stan Oracle reaches 0.600 at K=4000 once
            # the cutoff placeholder is gone, and the old floor cut those points
            # off the bottom of the axis.
            return (
                [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.30, 1.60],
                (0.57, 1.68),
            )
        return (
            [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.00, 1.50],
            (0.22, 1.65),
        )

    if metric == "log_loss":
        return (
            [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.20, 1.40],
            (0.67, 1.50),
        )
    return (
        [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 1.00, 1.40],
        (0.27, 1.50),
    )


def _linear_metric_y_ticks(
    cfg: DatasetConfig,
    stats: dict[str, dict[str, list[float]]],
    series_labels: list[str],
    metric: str,
) -> tuple[list[float], tuple[float, float]]:
    values: list[float] = []
    for label in series_labels:
        values.extend(float(v) for v in stats["mean"][label] if not math.isnan(v))
    if not values:
        return [0.0, 0.5, 1.0], (0.0, 1.0)

    data_min, data_max = min(values), max(values)
    if metric == "log_loss":
        # The floor has to follow the data. It used to be pinned at 0.65, which
        # was safe only while the lowest series was Stan's 0.679; with Stan's
        # real curve reaching 0.600 a fixed floor silently clips the series off
        # the bottom instead of failing.
        floor_tick = math.floor(data_min / 0.05) * 0.05
        lower = min(0.65 if data_min < 0.75 else 0.70, floor_tick)
        upper = 1.05 if data_max <= 1.05 else math.ceil((data_max + 0.03) / 0.05) * 0.05
        ticks = list(np.round(np.arange(lower, upper + 0.001, 0.05), 2))
        return ticks, (lower - 0.02, upper)

    lower = 0.20 if cfg.slug == "item" else 0.25
    tick_step = 0.05
    upper = math.ceil((data_max + 0.03) / tick_step) * tick_step
    upper = min(max(upper, 0.55 if cfg.slug == "item" else 0.65), 1.05)
    ticks = list(np.round(np.arange(lower + 0.05, upper + 0.001, tick_step), 2))
    return ticks, (lower, upper)


def _compressed_y_transform(ticks: list[float]):
    """Map numeric y values onto equal-spaced tick positions."""
    tick_values = np.asarray(ticks, dtype=float)
    tick_positions = np.arange(len(tick_values), dtype=float)

    def transform(values):
        arr = np.asarray(values, dtype=float)
        out = np.interp(arr, tick_values, tick_positions)
        finite = np.isfinite(arr)
        if len(tick_values) >= 2:
            lo = finite & (arr < tick_values[0])
            hi = finite & (arr > tick_values[-1])
            lo_scale = (tick_positions[1] - tick_positions[0]) / (tick_values[1] - tick_values[0])
            hi_scale = (tick_positions[-1] - tick_positions[-2]) / (tick_values[-1] - tick_values[-2])
            out[lo] = tick_positions[0] + (arr[lo] - tick_values[0]) * lo_scale
            out[hi] = tick_positions[-1] + (arr[hi] - tick_values[-1]) * hi_scale
        out[~finite] = np.nan
        return out

    return transform


def _compressed_y_break_position(ticks: list[float]) -> float | None:
    diffs = np.diff(np.asarray(ticks, dtype=float))
    if len(diffs) == 0:
        return None
    small_step = float(np.min(diffs))
    large_gaps = np.flatnonzero(diffs > small_step * 2.5)
    if len(large_gaps) == 0:
        return None
    return float(large_gaps[0]) + 0.5


def _draw_yaxis_break_marks(ax: plt.Axes, y: float) -> None:
    """Draw a small '//' marker on the y-axis where the scale compresses."""
    trans = ax.get_yaxis_transform()
    x = 0.0
    dx, dy = 0.012, 0.12
    gap = 0.05
    kw = dict(transform=trans, color="0.25", linewidth=1.0, clip_on=False)
    ax.plot([x - dx, x + dx], [y - dy - gap, y + dy - gap], **kw)
    ax.plot([x - dx, x + dx], [y - dy + gap, y + dy + gap], **kw)


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


def _plot_series_on_item_axis(
    ax: plt.Axes,
    xs: np.ndarray,
    display_sizes: np.ndarray,
    ys: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    label: str,
) -> None:
    """Plot one series on the item generalization axis.

    Marformer/NT: solid line only where data exists (NaN gaps handled naturally).
    Stan Oracle/Misspecs: solid to ITEM_STAN_CUTOFF, flat dotted extension beyond;
      with ITEM_STAN_CUTOFF = None there is no extension and the real curve is
      drawn solid over its whole range.
    Baselines: solid where present (capped at baseline_cutoff upstream).
    """
    valid = ~np.isnan(ys)
    if not np.any(valid):
        return
    xs_v, ys_v, lo_v, hi_v = xs[valid], ys[valid], lo[valid], hi[valid]
    linestyle, color, marker = _local_series_style(label)
    line_kw = dict(color=color, marker=marker,
                   linewidth=shared._series_linewidth(label),
                   alpha=shared._series_alpha(label), label=label)
    is_stan = label in {"Stan Oracle", "Stan Both", "Stan Bin", "Stan Proj"}
    # `None` means every Stan point is real data, so take the plain path. Comparing
    # an array against None raises, so this guard has to come first.
    if (not is_stan or ITEM_STAN_CUTOFF is None
            or np.all(display_sizes[valid] <= ITEM_STAN_CUTOFF)):
        ax.plot(xs_v, ys_v, linestyle=linestyle, **line_kw)
        ax.fill_between(xs_v, lo_v, hi_v, color=color, alpha=shared._series_fill_alpha(label), linewidth=0)
        return
    # Stan: solid to ITEM_STAN_CUTOFF, flat dotted extension beyond
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


def _plot_metric(
    cfg: DatasetConfig,
    rows: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    *,
    series_labels: list[str] | None = None,
    compressed_y: bool = True,
) -> None:
    stats = _series_from_rows(cfg, rows, metric)
    excluded = ITEM_EXCLUDED_SERIES if cfg.slug == "item" else frozenset()
    label_order = SERIES_ORDER if series_labels is None else series_labels
    active_series = [l for l in label_order if l in SERIES_ORDER and l not in excluded]
    present_labels = [l for l in active_series if any(not math.isnan(v) for v in stats["mean"][l])]
    has_dense = len(present_labels) >= 9 or any(len(l) >= 20 for l in present_labels)

    if cfg.slug == "annot":
        legend_ratio = 1.5 if has_dense else 1.3
        left_ratio = 6.9 if has_dense else 6.7
        fig_w, fig_h = (18.2 if has_dense else 17.4), 7.15
        tick_fs, axis_fs, title_fs = fs(9.5), fs(11.5), ts(14.0)
        legend_fs = fs(9.5) if has_dense else fs(10.0)
        legend_ls, legend_hl = (0.52 if has_dense else 0.62), (2.0 if has_dense else 2.2)
    else:
        legend_ratio = 1.7 if has_dense else 1.4
        left_ratio = 6.3 if has_dense else 6.5
        fig_w, fig_h = (17.8 if has_dense else 17.0), 7.6
        tick_fs, axis_fs, title_fs = fs(10.5), fs(12.5), ts(16.0)
        legend_fs = fs(10.0) if has_dense else fs(11.0)
        legend_ls, legend_hl = (0.58 if has_dense else 0.7), (2.2 if has_dense else 2.4)

    if compressed_y:
        y_ticks, y_limits = _metric_y_ticks(cfg, metric)
        y_transform = _compressed_y_transform(y_ticks)
    else:
        y_ticks, y_limits = _linear_metric_y_ticks(cfg, stats, active_series, metric)

        def y_transform(values):
            return np.asarray(values, dtype=float)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[left_ratio, legend_ratio], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    display_sizes = np.asarray(cfg.display_sizes, dtype=int)
    xs_all = np.arange(len(display_sizes), dtype=float)

    for label in active_series:
        ys = np.asarray(stats["mean"][label], dtype=float)
        stds = np.asarray(stats["std"][label], dtype=float)
        lo = ys - 0.05 * stds
        hi = ys + 0.05 * stds
        ys_plot = y_transform(ys)
        lo_plot = y_transform(lo)
        hi_plot = y_transform(hi)
        if cfg.slug == "item":
            _plot_series_on_item_axis(ax, xs_all, display_sizes, ys_plot, lo_plot, hi_plot, label)
        else:
            valid = ~np.isnan(ys)
            if not np.any(valid):
                continue
            linestyle, color, marker = _local_series_style(label)
            lkw = dict(color=color, marker=marker,
                       linewidth=shared._series_linewidth(label),
                       alpha=shared._series_alpha(label), label=label)
            ax.plot(xs_all[valid], ys_plot[valid], linestyle=linestyle, **lkw)
            ax.fill_between(xs_all[valid], lo_plot[valid], hi_plot[valid], color=color,
                            alpha=shared._series_fill_alpha(label), linewidth=0)

    # A clipped series is invisible, not an error: the Stan curves ran off the
    # bottom of this axis for a whole figure build because the floor was a
    # constant. Say so loudly instead of shipping a figure with data outside it.
    clipped = []
    for label in active_series:
        for size, v in zip(display_sizes, stats["mean"][label]):
            if not math.isnan(v) and not (y_limits[0] <= v <= y_limits[1]):
                clipped.append(f"{label}@{size}={v:.4f}")
    if clipped:
        print(f"  *** WARNING: {len(clipped)} point(s) fall outside the y-limits "
              f"{y_limits} and will be CLIPPED from {cfg.slug}/{metric}: "
              f"{', '.join(clipped[:6])}{' ...' if len(clipped) > 6 else ''}", flush=True)

    ax.set_ylim(*y_transform(y_limits))
    ax.set_yticks(y_transform(y_ticks))
    ax.set_yticklabels([f"{tick:.2f}" for tick in y_ticks])
    ax.set_xlim(-0.4, len(display_sizes) - 0.6)
    ax.set_xticks(xs_all)
    ax.set_xticklabels([str(s) for s in display_sizes])
    ax.set_xlabel(cfg.x_label)
    ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=title_fs, y=0.982 if cfg.slug == "annot" else 0.985)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.xaxis.label.set_size(axis_fs)
    ax.yaxis.label.set_size(axis_fs)
    ax.grid(True, alpha=0.35)

    y_break = _compressed_y_break_position(y_ticks) if compressed_y else None
    if y_break is not None:
        _draw_yaxis_break_marks(ax, y_break)

    if cfg.slug == "item":
        _draw_xaxis_break_marks(ax, display_sizes, ITEM_X_BREAK_AFTER)

    shared._style_legend_right(ax_leg, ax, fontsize=legend_fs,
                                labelspacing=legend_ls, handlelength=legend_hl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.89, bottom=0.11, wspace=0.04)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved → {output_path}")


def _plot_runtime(cfg: DatasetConfig, rows: list[dict], output_path: Path) -> None:
    # For item axis cap display at 1000 (runtime beyond 1000 not collected)
    runtime_max = 1000 if cfg.slug == "item" else None
    plot_sizes = [s for s in cfg.display_sizes if runtime_max is None or s <= runtime_max]
    xs_pos = {size: idx for idx, size in enumerate(plot_sizes)}

    if cfg.slug == "annot":
        fig, ax = plt.subplots(figsize=(14.0, 5.45))
        tick_fs, axis_fs, title_fs, legend_fs = fs(9.5), fs(11.5), ts(14.0), fs(9.75)
    else:
        fig, ax = plt.subplots(figsize=(13.0, 5.9))
        tick_fs, axis_fs, title_fs, legend_fs = fs(10.5), fs(12.5), ts(16.0), fs(10.5)

    for label in ("Marformer", "Stan Oracle"):
        xs: list[float] = []
        ys: list[float] = []
        for size in plot_sizes:
            row = _summary_lookup(rows, "runtime_minutes", label, size)
            if row is None:
                continue
            if label != "Marformer" and int(row.get("source_size", size)) != size:
                continue
            xs.append(float(xs_pos[size]))
            ys.append(float(row["mean"]))
        if not xs:
            continue
        linestyle, color, marker = _local_series_style(label)
        ax.plot(xs, ys, color=color, marker=marker, linestyle=linestyle,
                linewidth=shared._series_linewidth(label), alpha=shared._series_alpha(label), label=label)

    display_sizes = np.asarray(plot_sizes, dtype=int)
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
    ax.legend(loc="best", fontsize=legend_fs, frameon=True, framealpha=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved → {output_path}")


def _plot_calibration(
    cfg: DatasetConfig,
    rows: list[dict],
    all_payloads: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, str | None]],
    subdir: str,
    model_labels: list[str],
) -> None:
    cal_dir = cfg.out_dir / subdir
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_excluded = ITEM_EXCLUDED_SERIES if cfg.slug == "item" else frozenset()
    cal_models = [l for l in model_labels if l not in cal_excluded]
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
        # Grown from 3.65x3.35 per panel for the same reason as the LLM-Rubric
        # grid: at one column width a panel is ~1.6in and the larger type needs room.
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 6.1 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for ax, (title, payload, color_key) in zip(axes, payloads_for_size):
            probs, labels = payload
            shared._plot_ece(ax, probs, labels, title, COLORS.get(color_key, "#888888"))
            ax.set_title(ax.get_title(), fontsize=ts(12), fontweight="normal", pad=8)
        for ax in axes[len(payloads_for_size):]:
            ax.axis("off")
        shared.share_panel_axis_labels(axes, nrows, ncols)
        # Padding widened with the type: at 3x the left panel's '1.0' tick and the
        # right panel's '0.0' met at the panel boundary.
        fig.tight_layout(pad=1.2, w_pad=2.6, h_pad=2.0)
        out_path = cal_dir / f"{cfg.slug}_calibration_size{size}.png"
        fig.savefig(out_path, dpi=300)
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
                 cfg.out_dir / f"{cfg.slug}_generalization_log_loss_clean.png",
                 series_labels=CLEAN_SERIES, compressed_y=False)
    _plot_metric(cfg, rows, "log_loss", "Test Log Loss",
                 f"{cfg.title_prefix}: Test Log Loss (mean ± 0.05 SD)",
                 cfg.out_dir / f"{cfg.slug}_generalization_log_loss_clean_nt.png",
                 series_labels=CLEAN_NT_SERIES, compressed_y=False)
    _plot_metric(cfg, rows, "log_loss", "Test Log Loss",
                 f"{cfg.title_prefix}: Test Log Loss (mean ± 0.05 SD)",
                 cfg.out_dir / f"{cfg.slug}_generalization_log_loss.png")

    _plot_metric(cfg, rows, "mbr_l2", "MBR L2 (MSE)",
                 f"{cfg.title_prefix}: MBR L2 (mean ± 0.05 SD)",
                 cfg.out_dir / f"{cfg.slug}_generalization_mbr_l2_clean.png",
                 series_labels=CLEAN_SERIES, compressed_y=False)
    _plot_metric(cfg, rows, "mbr_l2", "MBR L2 (MSE)",
                 f"{cfg.title_prefix}: MBR L2 (mean ± 0.05 SD)",
                 cfg.out_dir / f"{cfg.slug}_generalization_mbr_l2_clean_nt.png",
                 series_labels=CLEAN_NT_SERIES, compressed_y=False)
    _plot_metric(cfg, rows, "mbr_l2", "MBR L2 (MSE)",
                 f"{cfg.title_prefix}: MBR L2 (mean ± 0.05 SD)",
                 cfg.out_dir / f"{cfg.slug}_generalization_mbr_l2.png")
    _plot_runtime(cfg, rows, cfg.out_dir / f"{cfg.slug}_generalization_runtime.png")
    if args.calibration:
        cal_all_models = [l for l in SERIES_ORDER if l not in {"Marformer NT", "Unigram Pool ij"}]
        _plot_calibration(cfg, rows, all_payloads, "calibration_all", cal_all_models)
        _plot_calibration(cfg, rows, all_payloads, "calibration_clean", CLEAN_NT_SERIES)


def main() -> None:
    global CHECKPOINT_POLICY, METRIC_POPULATION
    args = parse_args()
    CHECKPOINT_POLICY = args.checkpoint_policy
    METRIC_POPULATION = args.metric_population
    print(f"metric population: {METRIC_POPULATION}"
          + ("  (test corner only -- comparable with Stan)"
             if METRIC_POPULATION == "corner"
             else "  *** POOLED with training cells (D22/D23) ***"), flush=True)
    print(f"checkpoint policy: {CHECKPOINT_POLICY}"
          + ("  (F13: the refit's last epoch, matching the result tables)"
             if CHECKPOINT_POLICY == "last"
             else "  *** SELECTION ON TEST -- reproduces the older figures only ***"),
          flush=True)
    device_name = _choose_device(args.device)
    if args.part in {"all", "item"}:
        _plot_dataset(ITEM_CFG, args, device_name)
    if args.part in {"all", "annot"}:
        _plot_dataset(ANNOT_CFG, args, device_name)


if __name__ == "__main__":
    main()
