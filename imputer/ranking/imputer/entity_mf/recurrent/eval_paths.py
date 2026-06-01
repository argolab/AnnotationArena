"""Naming helpers for eval outputs keyed by max_item."""
from __future__ import annotations


def test_results_dir_name(max_item: int | None, *, full_graph: bool = False) -> str:
    if full_graph:
        return "TEST_RESULTS_FULLGRAPH"
    if max_item is None:
        return "TEST_RESULTS"
    return f"TEST_RESULTS_MAXITEM{max_item}"


def recurrence_scaling_dir_name(max_item: int | None, *, full_graph: bool = False) -> str:
    if full_graph:
        return "RECURRENCE_SCALING_FULLGRAPH"
    if max_item is None:
        return "RECURRENCE_SCALING"
    return f"RECURRENCE_SCALING_MAXITEM{max_item}"


def max_item_plot_tag(max_item: int | None) -> str:
    if max_item is None:
        return "fullgraph"
    return f"maxitem{max_item}"


def max_item_plot_label(max_item: int | None) -> str:
    if max_item is None:
        return "full graph (max_item=None)"
    return f"max_item={max_item}"
