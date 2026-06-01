"""
Route source cells to their factor in the structured model.

For target cell (i, j, k) and source cell (i_s, j_s, k_s, y_s):

  ATTR_PAIR  : j_s == j and k_s == k and i_s != i
               → per-(i_s, i) table, key = (i_s, y_s)
  CHANGE_J   : i_s == i and k_s == k and j_s != j
               → shared CHANGEJ table, key = y_s
  IGNORED    : everything else (cross-item, cross-annotator, cross-attr, and
               same (i, j) with different k)

Only one of the two useful classes fires per source; patterns are mutually exclusive.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Sequence, Tuple

from .dataset_adapter import Cell


class FactorKind(IntEnum):
    ATTR_PAIR = 0
    CHANGE_J = 1
    IGNORED = 2


def route_source(
    i_s: int, j_s: int, k_s: int,
    i_t: int, j_t: int, k_t: int,
) -> FactorKind:
    """Classify one source cell relative to the target cell."""
    if j_s == j_t and k_s == k_t and i_s != i_t:
        return FactorKind.ATTR_PAIR
    if i_s == i_t and k_s == k_t and j_s != j_t:
        return FactorKind.CHANGE_J
    return FactorKind.IGNORED


@dataclass
class RoutedSources:
    """Pre-classified sources for one LocalExample."""

    # ATTR_PAIR: list of (i_src, y_src) — one entry per source cell
    attr_pairs: List[Tuple[int, int]]

    # CHANGE_J: Counter of y_src values (multiplicity handled)
    change_j: Counter


def route_sources(
    sources: Sequence[Cell],
    i_t: int,
    j_t: int,
    k_t: int,
) -> RoutedSources:
    """
    Partition all source cells for a given target (i_t, j_t, k_t).

    CHANGEJ multiplicities are folded into a Counter so that
    the log-likelihood contribution is count * log P(y_src | y_target).
    """
    attr_pairs: List[Tuple[int, int]] = []
    change_j: Counter = Counter()

    for (i_s, j_s, k_s, y_s) in sources:
        kind = route_source(i_s, j_s, k_s, i_t, j_t, k_t)
        if kind == FactorKind.ATTR_PAIR:
            attr_pairs.append((i_s, y_s))
        elif kind == FactorKind.CHANGE_J:
            change_j[y_s] += 1

    return RoutedSources(attr_pairs=attr_pairs, change_j=change_j)
