"""Legacy training modules.

These modules are kept for backward compatibility and reference but are no longer
actively maintained. New code should use imputer.lightning_trainer instead.
"""

from .trainer import ImputerTrainer
from .multi_instance_trainer import (
    MultiInstanceTrainerBase,
    SequentialMIT,
    MixedMIT,
    GeneralMIT,
)

__all__ = [
    'ImputerTrainer',
    'MultiInstanceTrainerBase',
    'SequentialMIT',
    'MixedMIT',
    'GeneralMIT',
]
