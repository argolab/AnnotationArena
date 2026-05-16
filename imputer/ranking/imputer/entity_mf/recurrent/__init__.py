"""Recurrent Entity Marformer: weight-shared core with prelude/coda stacks."""

from .config import RecurrentMarformerConfig
from .model import RecurrentEntityMarformer

__all__ = ["RecurrentMarformerConfig", "RecurrentEntityMarformer"]
