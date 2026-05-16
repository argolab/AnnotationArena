"""Defaults for structured baselines (always transductive: train+val+test observed)."""

from __future__ import annotations

DEFAULT_UNIGRAM_ALPHA = 1.0
DEFAULT_IJK_ALPHA = 1.0
DEFAULT_SNB_ALPHA = 1.0

# Structured log-linear (PyTorch); optional — see ``fit_baselines(..., fit_log_linear=True)``.
DEFAULT_LOG_LINEAR_EPOCHS = 64
DEFAULT_LOG_LINEAR_LR = 0.05
DEFAULT_LOG_LINEAR_BATCH = 256
DEFAULT_LOG_LINEAR_PATIENCE = 5

TRANSDUCTIVE_INSTANCES = frozenset({"train", "val", "test"})
