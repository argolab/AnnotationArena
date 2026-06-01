"""
Global transductive plate graph for structured NB — new factorization.

All observed cells in train, val, and test form one global plate. Slot counts
(P(y), P(i|y), P(j|y), P(k|y)) use each cell once. Pair counts use every
ordered distinct (target, source) pair in the pool, routed by factor_routing:

  n_attr[i', i, y_target, y_source]  — per attribute-pair factor
  n_change_j[y_target, y_source]     — shared CHANGEJ factor

At prediction time, sources are ALL transductive observed cells except the
target cell (see dataset_adapter). Split only gates which missing rows are
evaluated, not which cells are sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dataset_adapter import Cell, LocalExample
from .factor_routing import route_sources


@dataclass(frozen=True)
class StructuredFactorMask:
    """Which pairwise factor families to use at fit and predict time."""

    attr_pair: bool = True
    change_j: bool = True

    @classmethod
    def all_on(cls) -> "StructuredFactorMask":
        return cls(True, True)

    @classmethod
    def all_off(cls) -> "StructuredFactorMask":
        return cls(False, False)


@dataclass
class FactorizedPlateCounts:
    """Sufficient statistics for the new structured plate graph."""

    num_classes: int
    num_attrs: int
    num_anns: int
    num_items: int

    n_y: np.ndarray       # (C,)
    n_i: np.ndarray       # (C, I)
    n_j: np.ndarray       # (C, J)
    n_k: np.ndarray       # (C, K)
    n_attr: np.ndarray    # (I, I, C, C)  [i', i, y_target, y_source]
    n_change_j: np.ndarray  # (C, C)  [y_target, y_source]


def _infer_dims_from_cells(
    cells: Sequence[Cell],
    num_attrs: int,
    num_anns: int,
    num_items: int,
) -> tuple[int, int, int]:
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
    *,
    factor_mask: StructuredFactorMask | None = None,
) -> FactorizedPlateCounts:
    """
    Fit counts from one global plate: each observed cell once for slots;
    all ordered distinct pairs for pairwise factors.
    """
    mask = factor_mask if factor_mask is not None else StructuredFactorMask.all_on()
    c = num_classes
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
    n_attr = np.zeros((i_dim, i_dim, c, c), dtype=np.float64)
    n_change_j = np.zeros((c, c), dtype=np.float64)

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
            # ATTR_PAIR: same (j, k), different i
            if mask.attr_pair and j_s == j_t and k_s == k_t and i_s != i_t:
                n_attr[i_s, i_t, y_t, y_s] += 1.0
            # CHANGE_J: same (i, k), different j
            elif mask.change_j and i_s == i_t and k_s == k_t and j_s != j_t:
                n_change_j[y_t, y_s] += 1.0
            # else: IGNORED (includes same (i, j), different k)

    return FactorizedPlateCounts(
        num_classes=c,
        num_attrs=i_dim,
        num_anns=j_dim,
        num_items=k_dim,
        n_y=n_y,
        n_i=n_i,
        n_j=n_j,
        n_k=n_k,
        n_attr=n_attr,
        n_change_j=n_change_j,
    )


def _log_conditional(table_row: np.ndarray, child_idx: int, alpha: float, vocab: int) -> np.ndarray:
    """
    Log P(child_idx | parent = y) for each y, from a (C, vocab) count table.

    table_row[y, child_idx] = count of (parent=y, child=child_idx).
    """
    num = table_row[:, child_idx] + alpha
    den = table_row.sum(axis=1) + alpha * float(vocab)
    return np.log(num) - np.log(den)


def log_posterior_unnorm(
    counts: FactorizedPlateCounts,
    ex: LocalExample,
    alpha: float,
    *,
    factor_mask: StructuredFactorMask | None = None,
) -> np.ndarray:
    """Log P(y | i,j,k, sources) + const over y = 0..C-1."""
    c = counts.num_classes
    it, jt, kt = ex.target_i, ex.target_j, ex.target_k
    a = float(alpha)
    mask = factor_mask if factor_mask is not None else StructuredFactorMask.all_on()

    # Slot factors
    log_py = np.log(counts.n_y + a) - np.log(counts.n_y.sum() + a * c)
    log_pi = _log_conditional(counts.n_i, it, a, counts.num_attrs)
    log_pj = _log_conditional(counts.n_j, jt, a, counts.num_anns)
    log_pk = _log_conditional(counts.n_k, kt, a, counts.num_items)
    scores = log_py + log_pi + log_pj + log_pk

    # Pairwise factors via routing
    routed = route_sources(ex.sources, it, jt, kt)

    # ATTR_PAIR: per (i', i) table — each source contributes independently
    if mask.attr_pair:
        for (i_src, y_src) in routed.attr_pairs:
            scores += _log_conditional(counts.n_attr[i_src, it], y_src, a, c)

    # CHANGE_J: shared table, weighted by multiplicity
    if mask.change_j:
        for y_src, cnt in routed.change_j.items():
            scores += cnt * _log_conditional(counts.n_change_j, y_src, a, c)

    return scores


def log_proba_normalized(
    counts: FactorizedPlateCounts,
    ex: LocalExample,
    alpha: float,
    *,
    factor_mask: StructuredFactorMask | None = None,
) -> np.ndarray:
    scores = log_posterior_unnorm(counts, ex, alpha, factor_mask=factor_mask)
    m = float(scores.max())
    log_norm = m + np.log(np.sum(np.exp(scores - m)))
    return scores - log_norm
