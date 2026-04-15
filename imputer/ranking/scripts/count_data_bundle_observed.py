#!/usr/bin/env python3
"""Count observed and missing ratings in a data_bundle.json by split (train / eval=val / test)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_ALLOWED = frozenset({"train", "val", "test"})


def _counts_by_split(records: list) -> tuple[int, int, int, int]:
    by_instance = Counter(str(r["instance"]) for r in records)
    other = set(by_instance) - _ALLOWED
    if other:
        raise ValueError(f"unexpected instance values: {sorted(other)}")
    train = by_instance.get("train", 0)
    eval_ = by_instance.get("val", 0)
    test = by_instance.get("test", 0)
    total = len(records)
    if train + eval_ + test != total:
        raise ValueError("internal error: split sum != row count")
    return train, eval_, test, total


def _print_block(label: str, train: int, eval_: int, test: int, total: int) -> None:
    print(f"{label}:")
    print(f"  train: {train}")
    print(f"  eval:  {eval_}  (instance tag: val)")
    print(f"  test:  {test}")
    print(f"  total: {total}")


def main() -> None:
    default_path = (
        Path(__file__).resolve().parent.parent
        / "DATA"
        / "LLM_RUBRIC"
        / "LLMRubric_225_25_9_175"
        / "data_bundle.json"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "bundle",
        nargs="?",
        type=Path,
        default=default_path,
        help=f"path to data_bundle.json (default: {default_path})",
    )
    args = p.parse_args()
    path: Path = args.bundle

    with path.open() as f:
        data = json.load(f)

    observed = data.get("observed_ratings")
    if observed is None:
        raise SystemExit(f"{path}: missing 'observed_ratings' key")
    missing = data.get("missing_ratings")
    if missing is None:
        raise SystemExit(f"{path}: missing 'missing_ratings' key")

    try:
        o_train, o_eval, o_test, o_total = _counts_by_split(observed)
        m_train, m_eval, m_test, m_total = _counts_by_split(missing)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    print(f"file: {path}")
    _print_block("observed", o_train, o_eval, o_test, o_total)
    print()
    _print_block("missing", m_train, m_eval, m_test, m_total)
    print()
    _print_block(
        "observed + missing",
        o_train + m_train,
        o_eval + m_eval,
        o_test + m_test,
        o_total + m_total,
    )


if __name__ == "__main__":
    main()
