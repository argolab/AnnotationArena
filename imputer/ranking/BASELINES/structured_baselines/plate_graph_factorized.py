"""
Global transductive plate graph for structured NB.

All observed cells in the selected splits form one **global plate**. Slot counts
(P(y), P(i|y), …) use each cell once. Pair counts use every ordered (target, source)
cell pair with a 7-way structural relation (``feature_utils.relation_label``).

At prediction time, eval examples still use sources = observed cells on the same
item in the same split (see ``dataset_adapter``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dataset_adapter import Cell, LocalExample
from .feature_utils import NUM_RELATIONS, relation_label


@dataclass
class FactorizedPlateCounts:
    """Sufficient statistics for the global transductive plate graph."""

    num_classes: int
    num_attrs: int
    num_anns: int
    num_items: int
    num_relations: int
    n_y: np.ndarray  # (C,)
    n_i: np.ndarray  # (C, I)
    n_j: np.ndarray  # (C, J)
    n_k: np.ndarray  # (C, K)
    n_yy_rel: np.ndarray  # (R, C, C)  parent row = target y, col = source y'


def _infer_dims_from_cells(cells: Sequence[Cell], num_attrs: int, num_anns: int, num_items: int) -> tuple[int, int, int]:
    max_i = num_attrs - 1
    max_j = num_anns - 1
    max_k = num_items - 1
    for (ii, jj, kk, _v) in cells:
        max_i = max(max_i, ii)
        max_j = max(max_j, jj)
        max_k = max(max_k, kk)
    return max_i + 1, max_j + 1, max_k + 1


def accumulate_transductive_counts(
    cells: Sequence[Cell],
    num_attrs: int,
    num_classes: int,
    num_anns: int | None = None,
    num_items: int | None = None,
) -> FactorizedPlateCounts:
    """
    Fit counts from one global plate: each observed cell once for slots; all ordered
    distinct pairs for P(y' | y, relation).
    """
    c, r = num_classes, NUM_RELATIONS
    i_inf, j_inf, k_inf = _infer_dims_from_cells(
        cells,
        int(num_attrs),
        int(num_anns) if num_anns is not None else 1,
        int(num_items) if num_items is not None else 1,
    )
    i_dim = max(int(num_attrs), i_inf)
    j_dim = int(num_anns) if num_anns is not None else j_inf
    k_dim = int(num_items) if num_items is not None else k_inf

    n_y = np.zeros(c, dtype=np.float64)
    n_i = np.zeros((c, i_dim), dtype=np.float64)
    n_j = np.zeros((c, j_dim), dtype=np.float64)
    n_k = np.zeros((c, k_dim), dtype=np.float64)
    n_yy_rel = np.zeros((r, c, c), dtype=np.float64)

    for (ii, jj, kk, y) in cells:
        n_y[y] += 1.0
        n_i[y, ii] += 1.0
        n_j[y, jj] += 1.0
        n_k[y, kk] += 1.0

    n_cells = len(cells)
    for a in range(n_cells):
        i_t, j_t, k_t, y_t = cells[a]
        for b in range(n_cells):
            if a == b:
                continue
            i_s, j_s, k_s, y_s = cells[b]
            rel = relation_label(i_s, j_s, k_s, i_t, j_t, k_t)
            n_yy_rel[rel, y_t, y_s] += 1.0

    return FactorizedPlateCounts(
        num_classes=c,
        num_attrs=i_dim,
        num_anns=j_dim,
        num_items=k_dim,
        num_relations=r,
        n_y=n_y,
        n_i=n_i,
        n_j=n_j,
        n_k=n_k,
        n_yy_rel=n_yy_rel,
    )


def _log_parent_predicts_child(mat: np.ndarray, child_idx: int, alpha: float, vocab: int) -> np.ndarray:
    num = mat[:, child_idx] + alpha
    den = mat.sum(axis=1) + alpha * float(vocab)
    return np.log(num) - np.log(den)


def log_posterior_unnorm(counts: FactorizedPlateCounts, ex: LocalExample, alpha: float) -> np.ndarray:
    """Log P(y | i,j,k, sources) + const over y = 0..C-1."""
    c = counts.num_classes
    it, jt, kt = ex.target_i, ex.target_j, ex.target_k
    a = float(alpha)

    log_py = np.log(counts.n_y + a) - np.log(counts.n_y.sum() + a * c)
    log_pi = _log_parent_predicts_child(counts.n_i, it, a, counts.num_attrs)
    log_pj = _log_parent_predicts_child(counts.n_j, jt, a, counts.num_anns)
    log_pk = _log_parent_predicts_child(counts.n_k, kt, a, counts.num_items)
    scores = log_py + log_pi + log_pj + log_pk

    for (i_s, j_s, k_s, y_s) in ex.sources:
        rel = relation_label(i_s, j_s, k_s, it, jt, kt)
        scores += _log_parent_predicts_child(counts.n_yy_rel[rel], y_s, a, c)

    return scores


def log_proba_normalized(counts: FactorizedPlateCounts, ex: LocalExample, alpha: float) -> np.ndarray:
    scores = log_posterior_unnorm(counts, ex, alpha)
    m = float(scores.max())
    log_norm = m + np.log(np.sum(np.exp(scores - m)))
    return scores - log_norm
