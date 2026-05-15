#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structured_baselines.dataset_adapter import LocalExample
from structured_baselines.naive_bayes_structured import StructuredNaiveBayes
from structured_baselines.plate_graph_factorized import accumulate_transductive_counts


def main() -> None:
    cells = [(0, 0, 0, 1), (1, 1, 0, 2)]
    exs = [
        LocalExample(0, 0, 0, 1, ((1, 1, 0, 2),)),
        LocalExample(1, 1, 0, 2, ((0, 0, 0, 1),)),
    ]
    counts = accumulate_transductive_counts(cells, num_attrs=2, num_classes=3)
    snb = StructuredNaiveBayes(counts=counts, alpha=1.0)
    m = snb.evaluate(exs)
    assert m["n"] == 2.0
    print("smoke OK", m)


if __name__ == "__main__":
    main()
