"""Defaults for structured baselines (always transductive: train+val+test observed)."""

from __future__ import annotations

DEFAULT_UNIGRAM_ALPHA = 1.0
DEFAULT_IJK_ALPHA = 1.0
DEFAULT_SNB_ALPHA = 1.0

TRANSDUCTIVE_INSTANCES = frozenset({"train", "val", "test"})
