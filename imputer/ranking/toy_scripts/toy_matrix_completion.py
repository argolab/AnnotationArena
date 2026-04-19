#!/usr/bin/env python3
from __future__ import annotations

"""
Toy matrix-completion experiment for EntityMarformer (multi-matrix regime).

Task:
- Two latent entity sets per graph: rows (N) and cols (M), each with latent dim D.
- Ground-truth scalar for matrix entry (i, j): y_ij = <U_i, V_j>.
  Latent rows/cols are drawn i.i.d. Uniform[-1, 1] per entry.
- A graph contains all N*M entry variables (no permanently missing entries).
- A fraction (mask_rate) is self-masked via token status for diagnostics.
- Training objective is MSE over all entries (observed + masked).
- Dataset is multi-matrix: each sample has freshly drawn latent (U, V).

Run from repo root:
  cd imputer/ranking
  PYTHONPATH=. python toy_scripts/toy_matrix_completion.py

EntityMarformer: shared relational bias, scale_shared_rel, pointer on, kaiming type init, dropout
0.1, Adam weight decay — same spirit as scripts/STAN/MARFORMER/.../run_train.sh, but this toy uses
deeper/wider defaults (embedding_dim, layers, heads, d_ff) and use_rel_value True (STAN often false;
pass --no-rel-value to match). Override with CLI flags. Real-data-only train flags (e.g.
--llm-input-dist) are not used here.

Input design: mc_entry uses one scalar in the param stream — initialized from input_value
(observed entries = target; masked = 0 or a tiny entry_input_tag_scale * cell_id tag) and
supervised on the same index after the forward pass. That propagates real matrix values through
row/col tokens for masked completion. Without distinct values per cell, the model tends toward a
global constant — especially under --resample-train-each-step.

Row/col entity tokens use per-entity deviations (mc_entry still has no deviations).
Default model depth/heads are raised
(see EMBEDDING_DIM / NUM_LAYERS / ATTN_HEADS / D_FF); override with CLI flags.

Optional experiments: --show-correct-vector (oracle U_i,V_j into row/col param stream via
input_value; SyntheticRegressionType with zero loss — not identifiable from Y alone under rotation),
--multiplication-head (H^2 pairwise head dot-products before each FFN).

Replot from saved curves without retraining:
  PYTHONPATH=. python toy_scripts/toy_matrix_completion.py --replot OUTPUT/toy_matrix_completion_curves/curves.json
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm

from imputer.entity_mf.config import EntityMarformerConfig
from imputer.entity_mf.data import EntityGraph, Relationship, Token
from imputer.entity_mf.model import EntityMarformer
from imputer.entity_mf.synthetic.types import RegressionSlices, SyntheticRegressionType
from imputer.entity_mf.types import EntityType, NullEntityType, VariationConfig


################################################################################
# Matrix-completion toy — EntityMarformer on synthetic low-rank grids.
#
# Flow: defaults → graph/schema builders → sample dataclass → MSE & debug helpers
# → plots → train/eval loop → argparse entrypoint.
################################################################################


################################################################################
# Defaults (fast-ish sanity settings) and output paths
################################################################################
SEED = 42
NUM_STEPS = 600
NUM_TRAIN_GRAPHS = 80
NUM_TEST_GRAPHS = 20

N_ROWS = 4
N_COLS = 4
LATENT_DIM = 1
# Uniform range for sampling U, V entries.
LATENT_SAMPLE_MIN = -1.0
LATENT_SAMPLE_MAX = 1.0
MASK_RATE = 0.15  # self-masked fraction (no permanently missing entries)

# Training/model knobs (deeper / multi-head defaults for the toy; override via CLI)
EMBEDDING_DIM = 512
NUM_LAYERS = 6
ATTN_HEADS = 4
DROPOUT = 0.1
D_FF = 2048
NUM_FFN_LAYERS = 1
LR = 2e-4
WEIGHT_DECAY = 0.01
TYPE_EMBEDDING_INIT = "kaiming"
USE_PER_HEAD_REL = False
USE_POINTER = False
USE_REL_VALUE = True
USE_ADDONE_ATTN = False
USE_DEVIATION_NORM = False
SCALE_SHARED_REL = True
USE_GRAPH_MASK = False
USE_LEARNED_EMBEDDING = False
# Unique per (i,j) bias in mc_entry param stream so tokens are not identical at t=0 (see docstring).
ENTRY_INPUT_TAG_SCALE = 0
# Extra relational channels: mc_entry↔mc_entry when same row / same column (pointer-like shortcuts).
MC_ENTRY_GRID_POINTER_RELS = False
# Oracle / architecture experiments (see EntityMarformerConfig + model.forward).
SHOW_CORRECT_VECTOR = False
USE_MULTIPLICATION_HEAD = False
READOUT_MLP_LAYER = 0
READOUT_MLP_DIM = 0
BATCH_GRAPHS = True
USE_MC_FIXED_POSITIONAL_FEATURE = False

OUT_DIR = Path("OUTPUT/toy_matrix_completion_curves")
_LOG_Y_FLOOR = 1e-12

# Write partial curves to out_dir/curves_live.json every N steps (0 = only write at end).
LIVE_CURVES_EVERY = 0
EVAL_EVERY = 1


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


################################################################################
# MatrixSample — one random matrix instance as an EntityGraph + bookkeeping
################################################################################


@dataclass
class MatrixSample:
    graph: EntityGraph
    # Entry metadata aligned with mc_entry token order.
    pairs: List[Tuple[int, int]]
    targets: List[float]
    masked_token_indices: List[int]
    masked_pairs: List[Tuple[int, int]]


################################################################################
# Schema, masking, and EntityGraph construction (row / col / mc_entry tokens)
################################################################################


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_types(
    n_rows: int,
    n_cols: int,
    *,
    latent_dim: int,
    use_mc_mask_bit: bool = False,
) -> Dict[str, EntityType]:
    # row/col: D-wide param stream (oracle via input_value when --show-correct-vector); has_target=False → no type loss.
    if n_rows < 1 or n_cols < 1:
        raise ValueError(f"n_rows and n_cols must be >= 1, got {n_rows}, {n_cols}")
    if latent_dim >= 1:
        latent_slices = RegressionSlices(input_dim=latent_dim, output_dim=latent_dim)
        row_type = SyntheticRegressionType(
            name="row",
            slices=latent_slices,
            has_target=False,
            variation=VariationConfig(enabled=True, num_entities=n_rows, reg_weight=0.0),
        )
        col_type = SyntheticRegressionType(
            name="col",
            slices=latent_slices,
            has_target=False,
            variation=VariationConfig(enabled=True, num_entities=n_cols, reg_weight=0.0),
        )
    else:
        row_type = NullEntityType(
            name="row",
            variation=VariationConfig(enabled=False, num_entities=0, reg_weight=0.0),
        )
        col_type = NullEntityType(
            name="col",
            variation=VariationConfig(enabled=False, num_entities=0, reg_weight=0.0),
        )
    # matrix-entry variable: scalar target y_ij.
    # Optionally add an explicit "is_masked" input feature to mc_entry tokens.
    entry_type = SyntheticRegressionType(
        name="mc_entry",
        slices=RegressionSlices(input_dim=2 if use_mc_mask_bit else 1, output_dim=1),
        has_target=True,
        variation=VariationConfig(enabled=False, num_entities=0, reg_weight=0.0),
    )
    return {"row": row_type, "col": col_type, "mc_entry": entry_type}


def _build_relationships(*, mc_entry_grid_pointer_rels: bool = False) -> List[Relationship]:
    rels = [
        Relationship(name="entry_to_row", source_type="mc_entry", target_type="row", inverse="row_to_entry"),
        Relationship(name="row_to_entry", source_type="row", target_type="mc_entry", inverse="entry_to_row"),
        Relationship(name="entry_to_col", source_type="mc_entry", target_type="col", inverse="col_to_entry"),
        Relationship(name="col_to_entry", source_type="col", target_type="mc_entry", inverse="entry_to_col"),
    ]
    if mc_entry_grid_pointer_rels:
        rels.extend(
            [
                Relationship(
                    name="entry_same_row",
                    source_type="mc_entry",
                    target_type="mc_entry",
                    inverse=None,
                ),
                Relationship(
                    name="entry_same_col",
                    source_type="mc_entry",
                    target_type="mc_entry",
                    inverse=None,
                ),
            ]
        )
    return rels


def _sample_masked_pairs(
    rng: random.Random,
    n_rows: int,
    n_cols: int,
    mask_rate: float,
) -> List[Tuple[int, int]]:
    if not (0.0 < mask_rate < 1.0):
        raise ValueError(f"mask_rate must be in (0, 1), got {mask_rate}")
    all_pairs = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    rng.shuffle(all_pairs)
    total = len(all_pairs)
    n_masked = int(round(mask_rate * total))
    n_masked = max(1, min(total - 1, n_masked))
    return all_pairs[:n_masked]


def _build_graph_for_pairs(
    *,
    types: Dict[str, EntityType],
    relationships: List[Relationship],
    y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    masked_pairs_set: set[Tuple[int, int]],
    entry_input_tag_scale: float = 0.0,
    mc_entry_grid_pointer_rels: bool = False,
    attach_ground_truth_latent: bool = False,
    use_mc_mask_bit: bool = False,
    use_mc_fixed_positional_feature: bool = False,
) -> Tuple[EntityGraph, List[Tuple[int, int]], List[float], List[int], List[Tuple[int, int]]]:
    n_rows, n_cols = y.shape
    if U.shape[0] != n_rows or V.shape[0] != n_cols or U.shape[1] != V.shape[1]:
        raise ValueError(f"U,V incompatible with y.shape {y.shape}: U{U.shape}, V{V.shape}")
    d = U.shape[1]
    tokens: List[Token] = []
    edges: List[Tuple[int, int, str]] = []

    def _oracle_param_raw(vec: np.ndarray) -> Dict[str, Any]:
        # Param stream only (see SyntheticRegressionType.build_param); keeps type feature stream separate.
        return {"input_value": [float(x) for x in vec.reshape(-1)]}

    # Entity tokens first (status 2 = observed; no masking on row/col).
    row_start = 0
    for r in range(n_rows):
        rd = _oracle_param_raw(U[r]) if attach_ground_truth_latent and d > 0 else None
        tokens.append(Token(type_name="row", entity_id=r, status=2, raw_data=rd))
    col_start = len(tokens)
    for c in range(n_cols):
        rd = _oracle_param_raw(V[c]) if attach_ground_truth_latent and d > 0 else None
        tokens.append(Token(type_name="col", entity_id=c, status=2, raw_data=rd))
    entry_start = len(tokens)

    # Entry variable tokens + edges to exactly one row and one col.
    pair_list: List[Tuple[int, int]] = []
    target_list: List[float] = []
    masked_token_indices: List[int] = []
    masked_pairs: List[Tuple[int, int]] = []
    for (i, j) in pairs:
        tgt = float(y[i, j])
        is_masked = (i, j) in masked_pairs_set
        # Observed entries carry their actual value in the input param slot so the attention
        # mechanism can propagate it to row/col tokens and from there to masked entries.
        # Masked entries receive 0 — the model must predict their value from context.
        # Without this, every entry looks identical to its neighbors, row/col tokens cannot
        # build a factored representation, and all predictions collapse to a global constant.
        if is_masked:
            input_val = entry_input_tag_scale * float(i * n_cols + j) if entry_input_tag_scale > 0.0 else 0.0
        else:
            input_val = tgt
        if use_mc_mask_bit:
            raw = {
                "input_value": [input_val, 1.0 if is_masked else 0.0],
                "target_value": [tgt],
            }
        else:
            raw = {
                "input_value": [input_val],
                "target_value": [tgt],
            }
        if use_mc_fixed_positional_feature:
            row_onehot = [1.0 if rr == i else 0.0 for rr in range(n_rows)]
            col_onehot = [1.0 if cc == j else 0.0 for cc in range(n_cols)]
            raw["fixed_feature"] = row_onehot + col_onehot
        entry_idx = len(tokens)
        status = 1 if is_masked else 2
        tokens.append(Token(type_name="mc_entry", entity_id=-1, status=status, raw_data=raw))
        pair_list.append((i, j))
        target_list.append(tgt)
        if is_masked:
            masked_token_indices.append(entry_idx)
            masked_pairs.append((i, j))

        row_token_idx = row_start + i
        col_token_idx = col_start + j
        edges.append((entry_idx, row_token_idx, "entry_to_row"))
        edges.append((row_token_idx, entry_idx, "row_to_entry"))
        edges.append((entry_idx, col_token_idx, "entry_to_col"))
        edges.append((col_token_idx, entry_idx, "col_to_entry"))

    if mc_entry_grid_pointer_rels:
        rel_names = {r.name for r in relationships}
        if "entry_same_row" not in rel_names or "entry_same_col" not in rel_names:
            raise ValueError(
                "mc_entry_grid_pointer_rels=True requires relationships from "
                "_build_relationships(mc_entry_grid_pointer_rels=True)."
            )
        # Bidirectional edges: all distinct mc_entry pairs sharing a row (resp. column).
        entry_slots: List[Tuple[int, int, int]] = []
        pair_ord = 0
        for (i, j) in pairs:
            entry_slots.append((entry_start + pair_ord, i, j))
            pair_ord += 1
        for a in range(len(entry_slots)):
            idx_a, ia, ja = entry_slots[a]
            for b in range(a + 1, len(entry_slots)):
                idx_b, ib, jb = entry_slots[b]
                if ia == ib and ja != jb:
                    edges.append((idx_a, idx_b, "entry_same_row"))
                    edges.append((idx_b, idx_a, "entry_same_row"))
                if ja == jb and ia != ib:
                    edges.append((idx_a, idx_b, "entry_same_col"))
                    edges.append((idx_b, idx_a, "entry_same_col"))

    graph = EntityGraph(types=types, relationships=relationships, tokens=tokens, edges=edges)

    # Sanity: each entry token has exactly one row edge and one col edge.
    row_deg = {}
    col_deg = {}
    for src, _tgt, rel in edges:
        if src >= entry_start:
            if rel == "entry_to_row":
                row_deg[src] = row_deg.get(src, 0) + 1
            if rel == "entry_to_col":
                col_deg[src] = col_deg.get(src, 0) + 1
    num_entries = len(tokens) - entry_start
    assert num_entries == len(pairs)
    for k in range(entry_start, entry_start + num_entries):
        assert row_deg.get(k, 0) == 1, "Each mc_entry must connect to exactly one row."
        assert col_deg.get(k, 0) == 1, "Each mc_entry must connect to exactly one col."

    return graph, pair_list, target_list, masked_token_indices, masked_pairs


def build_matrix_sample(
    *,
    device: torch.device,
    rng: random.Random,
    n_rows: int,
    n_cols: int,
    latent_dim: int,
    mask_rate: float,
    types: Dict[str, EntityType],
    relationships: List[Relationship],
    entry_input_tag_scale: float = ENTRY_INPUT_TAG_SCALE,
    mc_entry_grid_pointer_rels: bool = MC_ENTRY_GRID_POINTER_RELS,
    latent_sample_min: float = LATENT_SAMPLE_MIN,
    latent_sample_max: float = LATENT_SAMPLE_MAX,
    show_correct_vector: bool = SHOW_CORRECT_VECTOR,
    use_mc_mask_bit: bool = False,
    use_mc_fixed_positional_feature: bool = USE_MC_FIXED_POSITIONAL_FEATURE,
) -> MatrixSample:
    _ = device  # keep signature aligned with other toy builders
    if latent_sample_max <= latent_sample_min:
        raise ValueError(
            f"latent_sample_max must be > latent_sample_min, got min={latent_sample_min}, max={latent_sample_max}"
        )
    U = np.random.uniform(low=latent_sample_min, high=latent_sample_max, size=(n_rows, latent_dim))
    V = np.random.uniform(low=latent_sample_min, high=latent_sample_max, size=(n_cols, latent_dim))
    y = U @ V.T  # [N, M]

    all_pairs = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    masked_pairs = _sample_masked_pairs(
        rng=rng,
        n_rows=n_rows,
        n_cols=n_cols,
        mask_rate=mask_rate,
    )
    masked_pairs_set = set(masked_pairs)
    graph, pairs, targets, masked_token_indices, masked_pairs_out = _build_graph_for_pairs(
        types=types,
        relationships=relationships,
        y=y,
        U=U,
        V=V,
        pairs=all_pairs,
        masked_pairs_set=masked_pairs_set,
        entry_input_tag_scale=entry_input_tag_scale,
        mc_entry_grid_pointer_rels=mc_entry_grid_pointer_rels,
        attach_ground_truth_latent=show_correct_vector,
        use_mc_mask_bit=use_mc_mask_bit,
        use_mc_fixed_positional_feature=use_mc_fixed_positional_feature,
    )

    return MatrixSample(
        graph=graph,
        pairs=pairs,
        targets=targets,
        masked_token_indices=masked_token_indices,
        masked_pairs=masked_pairs_out,
    )


################################################################################
# Per-graph MSE on mc_entry, debug grids, optional deviation L2 regularizer
################################################################################


def _compute_entry_mse(
    model: EntityMarformer,
    graph: EntityGraph,
    device: torch.device,
    selected_token_indices: set[int] | None = None,
    params: torch.Tensor | None = None,
    readout_mlp: nn.Module | None = None,
) -> torch.Tensor:
    if params is None:
        if readout_mlp is None:
            params = model(graph, device=device)  # [1, L, P]
        else:
            _, combined = model(graph, device=device, return_combined=True)  # [1, L, model_dim]
            params = readout_mlp(combined)  # [1, L, P]
    entry_type = graph.types["mc_entry"]
    assert isinstance(entry_type, SyntheticRegressionType)
    out_slice = entry_type.slices.output_slice()

    preds: List[torch.Tensor] = []
    tgts: List[torch.Tensor] = []
    for idx, tok in enumerate(graph.tokens):
        if tok.type_name != "mc_entry":
            continue
        if selected_token_indices is not None and idx not in selected_token_indices:
            continue
        raw = tok.raw_data or {}
        t = raw.get("target_value", None)
        if t is None:
            continue
        p = params[0, idx, out_slice]
        preds.append(p)
        tgts.append(torch.tensor(t, dtype=p.dtype, device=device))

    if not preds:
        return torch.zeros((), device=device)
    pred = torch.stack(preds, dim=0)
    tgt = torch.stack(tgts, dim=0)
    return F.mse_loss(pred, tgt, reduction="mean")


def _compute_entry_mse_batch(
    *,
    params: torch.Tensor,  # [B, L, P]
    samples: Sequence[MatrixSample],
    graph: EntityGraph,
    device: torch.device,
    selected_token_indices_per_sample: Sequence[set[int] | None] | None = None,
) -> torch.Tensor:
    if not samples:
        return torch.zeros((), device=device)
    bsz = params.shape[0]
    if bsz != len(samples):
        raise ValueError(f"params batch size {bsz} does not match samples {len(samples)}")
    entry_type = graph.types["mc_entry"]
    assert isinstance(entry_type, SyntheticRegressionType)
    out_slice = entry_type.slices.output_slice()
    entry_indices = [idx for idx, tok in enumerate(graph.tokens) if tok.type_name == "mc_entry"]
    if not entry_indices:
        return torch.zeros((), device=device)
    pred = params[:, entry_indices, out_slice].squeeze(-1)  # [B, E]
    tgt = torch.tensor([s.targets for s in samples], dtype=pred.dtype, device=device)  # [B, E]

    if selected_token_indices_per_sample is None:
        return F.mse_loss(pred, tgt, reduction="mean")

    keep_mask = torch.zeros_like(pred, dtype=torch.bool, device=device)
    for b, keep_set in enumerate(selected_token_indices_per_sample):
        if keep_set is None:
            keep_mask[b, :] = True
            continue
        for e_pos, tok_idx in enumerate(entry_indices):
            if tok_idx in keep_set:
                keep_mask[b, e_pos] = True

    if keep_mask.any():
        return F.mse_loss(pred[keep_mask], tgt[keep_mask], reduction="mean")
    return torch.zeros((), device=device)


def _build_readout_mlp(
    *,
    input_dim: int,
    output_dim: int,
    num_layers: int,
    hidden_dim: int,
) -> nn.Module:
    if num_layers < 1:
        raise ValueError(f"readout_mlp_layer must be >= 1, got {num_layers}")
    if hidden_dim < 1:
        raise ValueError(f"readout_mlp_dim must be >= 1, got {hidden_dim}")

    layers: List[nn.Module] = []
    if num_layers == 1:
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def _debug_Y_Yhat_arrays(
    *,
    dbg: MatrixSample,
    params: torch.Tensor,
    out_slice: slice,
    n_rows: int,
    n_cols: int,
    masked_lookup: set[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tgt = np.zeros((n_rows, n_cols), dtype=np.float64)
    prd = np.zeros((n_rows, n_cols), dtype=np.float64)
    mask_ij = np.zeros((n_rows, n_cols), dtype=bool)
    pair_ord = 0
    for idx, tok in enumerate(dbg.graph.tokens):
        if tok.type_name != "mc_entry":
            continue
        i, j = dbg.pairs[pair_ord]
        pair_ord += 1
        prd[i, j] = float(params[0, idx, out_slice].item())
        tgt[i, j] = float((tok.raw_data or {}).get("target_value", [0.0])[0])
        if idx in masked_lookup:
            mask_ij[i, j] = True
    return tgt, prd, mask_ij


def _debug_print_Y_vs_Yhat_block(
    *,
    label: str,
    tgt: np.ndarray,
    prd: np.ndarray,
    mask_ij: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> None:
    """Two decimals, tab-separated columns; target cells are 7.2f + '*' or space (8 chars)."""
    print(f"  debug: {label} — Y (left) vs Ŷ (right), 2dp; * = masked status")
    hdr_left = "\t".join(f"j={j}".rjust(8) for j in range(n_cols))
    hdr_right = "\t".join(f"j={j}".rjust(8) for j in range(n_cols))
    print(f"  \t{hdr_left}\t||\t{hdr_right}")
    for i in range(n_rows):
        left_cells = []
        for j in range(n_cols):
            s = f"{tgt[i, j]:>7.2f}"
            s = s + ("*" if mask_ij[i, j] else " ")
            left_cells.append(s)
        right_cells = [f"{prd[i, j]:>8.2f}" for j in range(n_cols)]
        print(f"  i={i}\t" + "\t".join(left_cells) + "\t||\t" + "\t".join(right_cells))


def _compute_deviation_reg_loss(
    model: EntityMarformer,
    types: Dict[str, EntityType],
    device: torch.device,
) -> torch.Tensor:
    reg_loss = torch.zeros((), device=device)
    for type_name, t in types.items():
        if not t.variation.enabled or t.variation.reg_weight <= 0.0:
            continue
        table = model.deviation_tables.get(type_name, None)
        if table is None:
            continue
        reg_loss = reg_loss + t.variation.reg_weight * table.pow(2).sum()
    return reg_loss


################################################################################
# Save / reload learning curves as PNG (also used by --replot)
################################################################################


def render_plots_from_results(results: Dict[str, Any], out_dir: Path, *, log_y: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = np.asarray(results["train_mse"], dtype=np.float64)
    te = np.asarray(results["test_mse"], dtype=np.float64)
    n_plot = int(tr.shape[0])
    if te.shape[0] != n_plot:
        raise ValueError("train_mse and test_mse must have the same length")
    x = np.arange(1, n_plot + 1)
    if log_y:
        tr = np.maximum(tr, _LOG_Y_FLOOR)
        te = np.maximum(te, _LOG_Y_FLOOR)

    fig_tr, ax_tr = plt.subplots(figsize=(8.6, 5.2))
    fig_te, ax_te = plt.subplots(figsize=(8.6, 5.2))

    ax_tr.plot(x, tr, color="#1f77b4", linewidth=2.0)
    ax_te.plot(x, te, color="#d62728", linewidth=2.0)

    for ax, title, ylabel, fname in (
        (ax_tr, "Train MSE (matrix completion)", "train MSE", "matrix_completion_train_mse.png"),
        (ax_te, "Test MSE (matrix completion)", "test MSE", "matrix_completion_test_mse.png"),
    ):
        if log_y:
            ax.set_yscale("log")
            ylabel = f"{ylabel} (log scale)"
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title + (" - log y" if log_y else ""))
        ax.grid(True, which="both", alpha=0.3)
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)


################################################################################
# Full experiment: sample graphs, train EntityMarformer, log, write curves.json
################################################################################


def run_matrix_completion_experiment(
    *,
    out_dir: Path = OUT_DIR,
    seed: int = SEED,
    num_steps: int = NUM_STEPS,
    num_train_graphs: int = NUM_TRAIN_GRAPHS,
    num_test_graphs: int = NUM_TEST_GRAPHS,
    n_rows: int = N_ROWS,
    n_cols: int = N_COLS,
    latent_dim: int = LATENT_DIM,
    latent_sample_min: float = LATENT_SAMPLE_MIN,
    latent_sample_max: float = LATENT_SAMPLE_MAX,
    mask_rate: float = MASK_RATE,
    embedding_dim: int = EMBEDDING_DIM,
    num_layers: int = NUM_LAYERS,
    attn_heads: int = ATTN_HEADS,
    d_ff: int = D_FF,
    num_ffn_layers: int = NUM_FFN_LAYERS,
    dropout: float = DROPOUT,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    type_embedding_init: str = TYPE_EMBEDDING_INIT,
    use_per_head_rel: bool = USE_PER_HEAD_REL,
    use_pointer: bool = USE_POINTER,
    use_rel_value: bool = USE_REL_VALUE,
    use_addone_attn: bool = USE_ADDONE_ATTN,
    use_deviation_norm: bool = USE_DEVIATION_NORM,
    scale_shared_rel: bool = SCALE_SHARED_REL,
    use_graph_mask: bool = USE_GRAPH_MASK,
    use_learned_embedding: bool = USE_LEARNED_EMBEDDING,
    entry_input_tag_scale: float = ENTRY_INPUT_TAG_SCALE,
    mc_entry_grid_pointer_rels: bool = MC_ENTRY_GRID_POINTER_RELS,
    resample_train_each_step: bool = False,
    show_correct_vector: bool = SHOW_CORRECT_VECTOR,
    use_multiplication_head: bool = USE_MULTIPLICATION_HEAD,
    readout_mlp_layer: int = READOUT_MLP_LAYER,
    readout_mlp_dim: int = READOUT_MLP_DIM,
    live_curves_every: int = LIVE_CURVES_EVERY,
    eval_every: int = EVAL_EVERY,
    batch_graphs: bool = BATCH_GRAPHS,
    use_mc_mask_bit: bool = False,
    use_mc_fixed_positional_feature: bool = USE_MC_FIXED_POSITIONAL_FEATURE,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(seed)
    rng = random.Random(seed)

    if show_correct_vector and latent_dim < 1:
        raise ValueError("show_correct_vector requires --D >= 1")
    if use_learned_embedding and use_multiplication_head:
        raise ValueError("use_multiplication_head is not supported with use_learned_embedding")
    use_readout_mlp = readout_mlp_layer > 0 and readout_mlp_dim > 0
    if (readout_mlp_layer == 0) != (readout_mlp_dim == 0):
        raise ValueError(
            "readout_mlp_layer and readout_mlp_dim must both be zero (disabled) "
            "or both be positive (enabled)."
        )

    types = _build_types(
        n_rows=n_rows,
        n_cols=n_cols,
        latent_dim=latent_dim,
        use_mc_mask_bit=use_mc_mask_bit,
    )
    if use_mc_fixed_positional_feature:
        global_param_dim = max(t.param_dim for t in types.values())
        feature_dim = embedding_dim if use_learned_embedding else (embedding_dim - global_param_dim)
        required_pos_dim = n_rows + n_cols
        if feature_dim < required_pos_dim:
            raise ValueError(
                f"use_mc_fixed_positional_feature requires feature_dim >= N+M ({required_pos_dim}), "
                f"got feature_dim={feature_dim}. Increase embedding_dim or disable the flag."
            )
    relationships = _build_relationships(mc_entry_grid_pointer_rels=mc_entry_grid_pointer_rels)
    num_rels = len(relationships)
    head_dim = embedding_dim // attn_heads
    if use_per_head_rel and head_dim <= num_rels:
        raise ValueError(
            f"With use_per_head_rel, head_dim ({head_dim}) must be > num_relationships ({num_rels}). "
            "Increase embedding_dim, reduce attn_heads, disable --per-head-rel, or turn off "
            "mc_entry grid pointer relationships."
        )

    def _make_train_samples() -> List[MatrixSample]:
        samples: List[MatrixSample] = []
        for _ in range(num_train_graphs):
            samples.append(
                build_matrix_sample(
                    device=device,
                    rng=rng,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    latent_dim=latent_dim,
                    mask_rate=mask_rate,
                    types=types,
                    relationships=relationships,
                    entry_input_tag_scale=entry_input_tag_scale,
                    mc_entry_grid_pointer_rels=mc_entry_grid_pointer_rels,
                    latent_sample_min=latent_sample_min,
                    latent_sample_max=latent_sample_max,
                    show_correct_vector=show_correct_vector,
                    use_mc_mask_bit=use_mc_mask_bit,
                    use_mc_fixed_positional_feature=use_mc_fixed_positional_feature,
                )
            )
        return samples

    train_samples: List[MatrixSample] = _make_train_samples()
    test_samples: List[MatrixSample] = []
    for _ in range(num_test_graphs):
        test_samples.append(
            build_matrix_sample(
                device=device,
                rng=rng,
                n_rows=n_rows,
                n_cols=n_cols,
                latent_dim=latent_dim,
                mask_rate=mask_rate,
                types=types,
                relationships=relationships,
                entry_input_tag_scale=entry_input_tag_scale,
                mc_entry_grid_pointer_rels=mc_entry_grid_pointer_rels,
                latent_sample_min=latent_sample_min,
                latent_sample_max=latent_sample_max,
                show_correct_vector=show_correct_vector,
                use_mc_mask_bit=use_mc_mask_bit,
                use_mc_fixed_positional_feature=use_mc_fixed_positional_feature,
            )
        )

    # Build model from shared type/relationship schema.
    ref_graph = train_samples[0].graph
    cfg = EntityMarformerConfig(
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        attention_heads=attn_heads,
        dropout=dropout,
        d_ff=d_ff,
        num_ffn_layers=num_ffn_layers,
        use_per_head_rel=use_per_head_rel,
        use_pointer=use_pointer,
        use_rel_value=use_rel_value,
        use_addone_attn=use_addone_attn,
        type_embedding_init=type_embedding_init,
        use_deviation_norm=use_deviation_norm,
        scale_shared_rel=scale_shared_rel,
        use_learned_embedding=use_learned_embedding,
        use_graph_mask=use_graph_mask,
        use_multiplication_head=use_multiplication_head,
    )
    model = EntityMarformer(
        config=cfg,
        types=types,
        num_relationships=ref_graph.num_relationships,
    ).to(device)
    readout_mlp: nn.Module | None = None
    if use_readout_mlp:
        readout_mlp = _build_readout_mlp(
            input_dim=embedding_dim,
            output_dim=model.global_param_dim,
            num_layers=readout_mlp_layer,
            hidden_dim=readout_mlp_dim,
        ).to(device)
    trainable_params = list(model.parameters())
    if readout_mlp is not None:
        trainable_params += list(readout_mlp.parameters())
    opt = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    print(
        "Starting EntityMarformer matrix-completion toy\n"
        f"  N={n_rows}, M={n_cols}, D={latent_dim}, latent_sample_range=[{latent_sample_min}, {latent_sample_max}], "
        f"mask_rate={mask_rate}, no_missing=True\n"
        f"  num_train_graphs={num_train_graphs}, num_test_graphs={num_test_graphs}, steps={num_steps}\n"
        f"  model: emb={embedding_dim}, layers={num_layers}, heads={attn_heads}, d_ff={d_ff}, "
        f"ffn_layers={num_ffn_layers}, dropout={dropout}, lr={lr}, weight_decay={weight_decay}\n"
        f"  marformer: per_head_rel={use_per_head_rel}, pointer={use_pointer}, rel_value={use_rel_value}, "
        f"addone={use_addone_attn}, deviation_norm={use_deviation_norm}, scale_shared_rel={scale_shared_rel}, "
        f"graph_mask={use_graph_mask}, learned_emb={use_learned_embedding}, type_init={type_embedding_init!r}\n"
        f"  row/col deviation: on (per-entity tables enabled)\n"
        f"  mc_entry explicit mask-bit: {use_mc_mask_bit}\n"
        f"  mc_entry fixed positional feature: {use_mc_fixed_positional_feature}\n"
        f"  entry_input_tag_scale (per-cell param-stream tag)={entry_input_tag_scale}\n"
        f"  mc_entry_grid_pointer_rels={mc_entry_grid_pointer_rels} (R={num_rels})\n"
        f"  resample_train_each_step={resample_train_each_step}\n"
        f"  show_correct_vector={show_correct_vector}\n"
        f"  use_multiplication_head={use_multiplication_head}\n"
        f"  readout_mlp_layer={readout_mlp_layer}, readout_mlp_dim={readout_mlp_dim}, "
        f"enabled={use_readout_mlp}\n"
        f"  eval_every={eval_every}\n"
        f"  batch_graphs={batch_graphs}\n"
        f"  device={device}"
    )

    train_curve: List[float] = []
    test_curve: List[float] = []
    test_all_curve: List[float] = []

    def _results_payload(
        *,
        train_mse: List[float],
        test_mse: List[float],
        test_all_mse: List[float],
        completed_steps: int,
        live: bool,
    ) -> Dict[str, Any]:
        return {
            "seed": seed,
            "num_steps": num_steps,
            "completed_steps": completed_steps,
            "live": live,
            "num_train_graphs": num_train_graphs,
            "num_test_graphs": num_test_graphs,
            "N": n_rows,
            "M": n_cols,
            "D": latent_dim,
            "latent_sample_min": latent_sample_min,
            "latent_sample_max": latent_sample_max,
            "mask_rate": mask_rate,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "attn_heads": attn_heads,
            "d_ff": d_ff,
            "num_ffn_layers": num_ffn_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "row_col_deviation": True,
            "type_embedding_init": type_embedding_init,
            "use_per_head_rel": use_per_head_rel,
            "use_pointer": use_pointer,
            "use_rel_value": use_rel_value,
            "use_addone_attn": use_addone_attn,
            "use_deviation_norm": use_deviation_norm,
            "scale_shared_rel": scale_shared_rel,
            "use_graph_mask": use_graph_mask,
            "use_learned_embedding": use_learned_embedding,
            "entry_input_tag_scale": entry_input_tag_scale,
            "mc_entry_grid_pointer_rels": mc_entry_grid_pointer_rels,
            "num_relationships": num_rels,
            "resample_train_each_step": resample_train_each_step,
            "show_correct_vector": show_correct_vector,
            "use_multiplication_head": use_multiplication_head,
            "readout_mlp_layer": readout_mlp_layer,
            "readout_mlp_dim": readout_mlp_dim,
            "use_readout_mlp": use_readout_mlp,
            "eval_every": eval_every,
            "batch_graphs": batch_graphs,
            "use_mc_mask_bit": use_mc_mask_bit,
            "use_mc_fixed_positional_feature": use_mc_fixed_positional_feature,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "test_all_mse": test_all_mse,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    if live_curves_every > 0:
        # Same OUT_DIR as a prior job would leave curves_live.json until the next multiple of
        # live_curves_every; monitors then show stale completed_steps vs live tqdm.
        (out_dir / "curves_live.json").unlink(missing_ok=True)

    steps = tqdm(range(num_steps), total=num_steps, desc="train", leave=False)
    last_test_mse: float | None = None
    last_test_all_mse: float | None = None
    for step in steps:
        if resample_train_each_step:
            train_samples = _make_train_samples()
        model.train()
        opt.zero_grad(set_to_none=True)

        if batch_graphs:
            if readout_mlp is None:
                p_tr = model.forward_batch([s.graph for s in train_samples], device=device)
            else:
                _, c_tr = model.forward_batch(
                    [s.graph for s in train_samples], device=device, return_combined=True
                )
                p_tr = readout_mlp(c_tr)
            train_mse = _compute_entry_mse_batch(
                params=p_tr,
                samples=train_samples,
                graph=train_samples[0].graph,
                device=device,
            )
        else:
            train_sum = torch.zeros((), device=device)
            for sample in train_samples:
                train_sum = train_sum + _compute_entry_mse(
                    model,
                    sample.graph,
                    device=device,
                    readout_mlp=readout_mlp,
                )
            train_mse = train_sum / float(len(train_samples))

        reg_loss = _compute_deviation_reg_loss(model, types, device=device)
        loss = train_mse + reg_loss
        loss.backward()
        opt.step()

        do_eval = (
            step == 0
            or (step + 1) % max(1, eval_every) == 0
            or step == num_steps - 1
        )
        if do_eval:
            model.eval()
            with torch.no_grad():
                if batch_graphs:
                    if readout_mlp is None:
                        p_te = model.forward_batch([s.graph for s in test_samples], device=device)
                    else:
                        _, c_te = model.forward_batch(
                            [s.graph for s in test_samples], device=device, return_combined=True
                        )
                        p_te = readout_mlp(c_te)
                    test_mse = _compute_entry_mse_batch(
                        params=p_te,
                        samples=test_samples,
                        graph=test_samples[0].graph,
                        device=device,
                        selected_token_indices_per_sample=[set(s.masked_token_indices) for s in test_samples],
                    )
                    test_all_mse = _compute_entry_mse_batch(
                        params=p_te,
                        samples=test_samples,
                        graph=test_samples[0].graph,
                        device=device,
                    )
                else:
                    test_sum = torch.zeros((), device=device)
                    test_all_sum = torch.zeros((), device=device)
                    for sample in test_samples:
                        masked_idx = set(sample.masked_token_indices)
                        if readout_mlp is None:
                            p_te = model(sample.graph, device=device)
                        else:
                            _, c_te = model(sample.graph, device=device, return_combined=True)
                            p_te = readout_mlp(c_te)
                        test_sum = test_sum + _compute_entry_mse(
                            model,
                            sample.graph,
                            device=device,
                            selected_token_indices=masked_idx,
                            params=p_te,
                        )
                        test_all_sum = test_all_sum + _compute_entry_mse(
                            model, sample.graph, device=device, params=p_te
                        )
                    test_mse = test_sum / float(len(test_samples))
                    test_all_mse = test_all_sum / float(len(test_samples))
                last_test_mse = float(test_mse.detach().cpu().item())
                last_test_all_mse = float(test_all_mse.detach().cpu().item())
        else:
            assert last_test_mse is not None and last_test_all_mse is not None
            test_mse = torch.tensor(last_test_mse, device=device)
            test_all_mse = torch.tensor(last_test_all_mse, device=device)

        train_curve.append(float(train_mse.detach().cpu().item()))
        test_curve.append(float(test_mse.detach().cpu().item()))
        test_all_curve.append(float(test_all_mse.detach().cpu().item()))

        completed = step + 1
        if live_curves_every > 0 and (
            completed == 1
            or completed % live_curves_every == 0
            or completed == num_steps
        ):
            _atomic_write_json(
                out_dir / "curves_live.json",
                _results_payload(
                    train_mse=list(train_curve),
                    test_mse=list(test_curve),
                    test_all_mse=list(test_all_curve),
                    completed_steps=completed,
                    live=True,
                ),
            )

        if (step + 1) % 20 == 0 or step == 0:
            print(
                f"step {step+1:4d} | "
                f"train_mse={train_curve[-1]:.6f} "
                f"test_masked_mse={test_curve[-1]:.6f} "
                f"test_all_mse={float(test_all_mse.detach().cpu().item()):.6f} "
                f"reg={float(reg_loss.detach().cpu().item()):.6f} "
                f"total={float(loss.detach().cpu().item()):.6f}"
            )

        if (step + 1) % 100 == 0 or step == num_steps - 1:
            dbg_te = test_samples[0]
            entry_type = dbg_te.graph.types["mc_entry"]
            assert isinstance(entry_type, SyntheticRegressionType)
            out_slice = entry_type.slices.output_slice()
            # Small enough to read: full grids (e.g. 4×4, 5×5), train + test, 2dp + tabs.
            if n_rows * n_cols <= 25:
                with torch.no_grad():
                    if readout_mlp is None:
                        p_tr = model(train_samples[0].graph, device=device)
                        p_te = model(dbg_te.graph, device=device)
                    else:
                        _, c_tr = model(train_samples[0].graph, device=device, return_combined=True)
                        _, c_te = model(dbg_te.graph, device=device, return_combined=True)
                        p_tr = readout_mlp(c_tr)
                        p_te = readout_mlp(c_te)
                tr_tgt, tr_prd, tr_m = _debug_Y_Yhat_arrays(
                    dbg=train_samples[0],
                    params=p_tr,
                    out_slice=out_slice,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    masked_lookup=set(train_samples[0].masked_token_indices),
                )
                te_tgt, te_prd, te_m = _debug_Y_Yhat_arrays(
                    dbg=dbg_te,
                    params=p_te,
                    out_slice=out_slice,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    masked_lookup=set(dbg_te.masked_token_indices),
                )
                _debug_print_Y_vs_Yhat_block(
                    label="first train graph (this step)",
                    tgt=tr_tgt,
                    prd=tr_prd,
                    mask_ij=tr_m,
                    n_rows=n_rows,
                    n_cols=n_cols,
                )
                _debug_print_Y_vs_Yhat_block(
                    label="first test graph",
                    tgt=te_tgt,
                    prd=te_prd,
                    mask_ij=te_m,
                    n_rows=n_rows,
                    n_cols=n_cols,
                )
                print("  (* = masked token status in graph)")
            else:
                dbg = dbg_te
                with torch.no_grad():
                    if readout_mlp is None:
                        params = model(dbg.graph, device=device)
                    else:
                        _, c_dbg = model(dbg.graph, device=device, return_combined=True)
                        params = readout_mlp(c_dbg)
                masked_lookup = set(dbg.masked_token_indices)
                masked_shown = 0
                obs_shown = 0
                pair_ord = 0
                print("  debug sample preds vs tgts (masked subset):")
                for idx, tok in enumerate(dbg.graph.tokens):
                    if tok.type_name != "mc_entry":
                        continue
                    pair = dbg.pairs[pair_ord]
                    pair_ord += 1
                    pred = float(params[0, idx, out_slice].item())
                    tgt = float((tok.raw_data or {}).get("target_value", [0.0])[0])
                    if idx in masked_lookup:
                        if masked_shown < 5:
                            print(f"    pair={pair} pred={pred:.4f} tgt={tgt:.4f}")
                            masked_shown += 1
                    else:
                        if obs_shown < 5:
                            if obs_shown == 0:
                                print("  debug sample preds vs tgts (observed subset):")
                            print(f"    pair={pair} pred={pred:.4f} tgt={tgt:.4f}")
                            obs_shown += 1
                    if masked_shown >= 5 and obs_shown >= 5:
                        break

    results = _results_payload(
        train_mse=train_curve,
        test_mse=test_curve,
        test_all_mse=test_all_curve,
        completed_steps=num_steps,
        live=False,
    )
    (out_dir / "curves.json").write_text(json.dumps(results, indent=2))
    if live_curves_every > 0:
        _atomic_write_json(out_dir / "curves_live.json", results)
    render_plots_from_results(results, out_dir, log_y=True)
    print(f"Wrote curves and log-scale plots to {out_dir}")
    return results


################################################################################
# CLI — train from scratch or --replot from curves.json only
################################################################################


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrix-completion toy curves for EntityMarformer.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    parser.add_argument("--num-train-graphs", type=int, default=NUM_TRAIN_GRAPHS)
    parser.add_argument("--num-test-graphs", type=int, default=NUM_TEST_GRAPHS)
    parser.add_argument("--N", type=int, default=N_ROWS)
    parser.add_argument("--M", type=int, default=N_COLS)
    parser.add_argument("--D", type=int, default=LATENT_DIM)
    parser.add_argument(
        "--latent-sample-min",
        type=float,
        default=LATENT_SAMPLE_MIN,
        help="Lower bound for Uniform sampling of each latent entry.",
    )
    parser.add_argument(
        "--latent-sample-max",
        type=float,
        default=LATENT_SAMPLE_MAX,
        help="Upper bound for Uniform sampling of each latent entry.",
    )
    parser.add_argument("--mask-rate", type=float, default=MASK_RATE)
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--attn-heads", type=int, default=ATTN_HEADS)
    parser.add_argument("--d-ff", type=int, default=D_FF)
    parser.add_argument("--num-ffn-layers", type=int, default=NUM_FFN_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--type-embedding-init",
        type=str,
        default=TYPE_EMBEDDING_INIT,
        choices=("normal", "scaled_normal", "kaiming"),
    )
    # MARFORMER flags (defaults match scripts/STAN/MARFORMER/.../run_train.sh)
    parser.add_argument(
        "--per-head-rel",
        action="store_true",
        help="Use per-head relational bias (default: shared-bias / --no-per-head-rel in run_train).",
    )
    parser.add_argument(
        "--no-pointer",
        action="store_true",
        help="Disable pointer mechanism (default: pointer on, like run_train USE_POINTER=true).",
    )
    parser.add_argument(
        "--no-rel-value",
        dest="use_rel_value",
        action="store_false",
        help="Disable relation-specific value augmentation (default: enabled for this toy).",
    )
    parser.set_defaults(use_rel_value=True)
    parser.add_argument("--use-addone-attn", action="store_true")
    parser.add_argument("--use-deviation-norm", action="store_true")
    parser.add_argument(
        "--no-scale-shared-rel",
        action="store_true",
        help="Disable sqrt(head_dim) scaling of shared relational scores.",
    )
    parser.add_argument("--use-graph-mask", action="store_true")
    parser.add_argument("--use-learned-embedding", action="store_true")
    parser.add_argument(
        "--entry-input-tag-scale",
        type=float,
        default=ENTRY_INPUT_TAG_SCALE,
        help="mc_entry input_value = scale * (i*M+j); 0 disables (can restore degenerate identical preds in eval).",
    )
    parser.add_argument(
        "--use-mc-mask-bit",
        action="store_true",
        help="Add an explicit maskedness bit to mc_entry input_value: input_value becomes [value, is_masked].",
    )
    parser.add_argument(
        "--use-mc-fixed-positional-feature",
        action="store_true",
        help=(
            "Append fixed absolute row/column one-hot positional feature to mc_entry feature stream "
            "(concat one-hot(row_i), one-hot(col_j) in the tail of the feature vector)."
        ),
    )
    parser.add_argument(
        "--resample-train-each-step",
        action="store_true",
        help="If set, resample num-train-graphs fresh matrix samples at every training step.",
    )
    parser.add_argument(
        "--mc-entry-grid-pointer-rels",
        action="store_true",
        help=(
            "Add bidirectional entry_same_row / entry_same_col edges between all mc_entry pairs "
            "sharing a row or column (R grows by 2). With --per-head-rel, need head_dim > R."
        ),
    )
    parser.add_argument(
        "--show-correct-vector",
        action="store_true",
        help="Put ground-truth U_i / V_j in row/col param stream (input_value); row/col types use zero loss.",
    )
    parser.add_argument(
        "--multiplication-head",
        action="store_true",
        help="Each block: concat H^2 pairwise head dot-products from attn output before FFN (incompatible with --use-learned-embedding).",
    )
    parser.add_argument(
        "--readout-mlp-layer",
        type=int,
        default=READOUT_MLP_LAYER,
        help=(
            "Number of layers in top readout MLP over final combined representation. "
            "Set 0 to keep current direct param readout."
        ),
    )
    parser.add_argument(
        "--readout-mlp-dim",
        type=int,
        default=READOUT_MLP_DIM,
        help=(
            "Hidden width of top readout MLP over final combined representation. "
            "Set 0 to keep current direct param readout."
        ),
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--live-curves-every",
        type=int,
        default=LIVE_CURVES_EVERY,
        help="Write curves_live.json every N steps (0 = only at end). Enables monitoring tail metrics while training.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=EVAL_EVERY,
        help="Run test-set evaluation every N steps (always evaluates at step 1 and final step).",
    )
    parser.add_argument(
        "--no-batch-graphs",
        action="store_true",
        help="Disable batched graph forward (fallback to one graph forward per sample).",
    )
    parser.add_argument("--replot", type=str, default=None)
    args = parser.parse_args()

    if args.replot:
        p = Path(args.replot)
        results = json.loads(p.read_text())
        out = Path(args.out_dir) if args.out_dir else p.parent
        render_plots_from_results(results, out, log_y=True)
        print(f"Wrote log-scale plots to {out}")
        return

    out = Path(args.out_dir) if args.out_dir else OUT_DIR
    run_matrix_completion_experiment(
        out_dir=out,
        seed=args.seed,
        num_steps=args.steps,
        num_train_graphs=args.num_train_graphs,
        num_test_graphs=args.num_test_graphs,
        n_rows=args.N,
        n_cols=args.M,
        latent_dim=args.D,
        latent_sample_min=args.latent_sample_min,
        latent_sample_max=args.latent_sample_max,
        mask_rate=args.mask_rate,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        attn_heads=args.attn_heads,
        d_ff=args.d_ff,
        num_ffn_layers=args.num_ffn_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        type_embedding_init=args.type_embedding_init,
        use_per_head_rel=args.per_head_rel,
        use_pointer=not args.no_pointer,
        use_rel_value=args.use_rel_value,
        use_addone_attn=args.use_addone_attn,
        use_deviation_norm=args.use_deviation_norm,
        scale_shared_rel=not args.no_scale_shared_rel,
        use_graph_mask=args.use_graph_mask,
        use_learned_embedding=args.use_learned_embedding,
        entry_input_tag_scale=args.entry_input_tag_scale,
        use_mc_mask_bit=args.use_mc_mask_bit,
        use_mc_fixed_positional_feature=args.use_mc_fixed_positional_feature,
        mc_entry_grid_pointer_rels=args.mc_entry_grid_pointer_rels,
        resample_train_each_step=args.resample_train_each_step,
        show_correct_vector=args.show_correct_vector,
        use_multiplication_head=args.multiplication_head,
        readout_mlp_layer=args.readout_mlp_layer,
        readout_mlp_dim=args.readout_mlp_dim,
        live_curves_every=args.live_curves_every,
        eval_every=args.eval_every,
        batch_graphs=not args.no_batch_graphs,
    )


if __name__ == "__main__":
    main()
