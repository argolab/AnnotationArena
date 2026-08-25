"""
Structural relation labels between a source rating cell and a target cell.

Each cell is X_{i,j,k} with 0-based indices (attribute i, annotator j, item k).
The mapping is deterministic and used by structured naive Bayes pair factors.

Precedence: two-index coincidences (same item+annot, same item+attr, same annot+attr)
are distinguished before single-index cases, before UNRELATED.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Tuple


class RelationKind(IntEnum):
    """Discrete edge label for (source cell, target cell)."""

    SAME_ITEM_SAME_ANNOT_DIFF_ATTR = 0  # j,k match; i differs
    SAME_ITEM_SAME_ATTR_DIFF_ANNOT = 1  # i,k match; j differs
    SAME_ANNOT_SAME_ATTR_DIFF_ITEM = 2  # i,j match; k differs
    SAME_ITEM_ONLY = 3  # k matches; i and j both differ from target
    SAME_ANNOT_ONLY = 4  # j matches only (i,k differ)
    SAME_ATTR_ONLY = 5  # i matches only (j,k differ)
    UNRELATED = 6  # i, j, k all differ


NUM_RELATIONS = len(RelationKind)

# Stable string names for logging / debugging
RELATION_NAMES: Tuple[str, ...] = tuple(r.name for r in RelationKind)


def relation_label(
    i_src: int,
    j_src: int,
    k_src: int,
    i_tgt: int,
    j_tgt: int,
    k_tgt: int,
) -> int:
    """
    Map (source indices) -> RelationKind value (int 0..NUM_RELATIONS-1).

    Args:
        i_src, j_src, k_src: source attribute, annotator, item (0-based)
        i_tgt, j_tgt, k_tgt: target attribute, annotator, item (0-based)

    Returns:
        Integer relation code. Caller must ensure (i_src,j_src,k_src) != (i_tgt,j_tgt,k_tgt).
    """
    si = i_src == i_tgt
    sj = j_src == j_tgt
    sk = k_src == k_tgt

    if sk and sj and not si:
        return int(RelationKind.SAME_ITEM_SAME_ANNOT_DIFF_ATTR)
    if sk and si and not sj:
        return int(RelationKind.SAME_ITEM_SAME_ATTR_DIFF_ANNOT)
    if si and sj and not sk:
        return int(RelationKind.SAME_ANNOT_SAME_ATTR_DIFF_ITEM)
    if sk and not si and not sj:
        return int(RelationKind.SAME_ITEM_ONLY)
    if sj and not si and not sk:
        return int(RelationKind.SAME_ANNOT_ONLY)
    if si and not sj and not sk:
        return int(RelationKind.SAME_ATTR_ONLY)
    return int(RelationKind.UNRELATED)


def assert_distinct_cells(
    i_src: int, j_src: int, k_src: int, i_tgt: int, j_tgt: int, k_tgt: int
) -> None:
    if (i_src, j_src, k_src) == (i_tgt, j_tgt, k_tgt):
        raise ValueError("source and target cells must differ")
