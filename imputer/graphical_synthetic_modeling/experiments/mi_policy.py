"""
Mechanistic Interpretability Policy for MARFORMER analysis.

Provides a policy that selects specific budget points for layer-wise analysis
rather than progressive increments.
"""

import numpy as np
import logging
from typing import List, Iterator, Tuple
from experiments.policies import BaseObservationPolicy, SampleTuple

logger = logging.getLogger(__name__)


class MechanisticInterpretabilityPolicy(BaseObservationPolicy):
    """
    Policy that selects training data at specific budget points for MI analysis.

    Instead of progressive increments, this policy evaluates at exact budgets
    (e.g., [50, 500, 2000]) to analyze how layer-wise representations change
    with training data size.
    """

    def __init__(self, budgets: List[int], seed: int = 42):
        """
        Initialize MI policy with specific budget points.

        Args:
            budgets: List of exact budget values (e.g., [50, 500, 2000])
            seed: Random seed for reproducible selection
        """
        self.budgets = sorted(budgets)
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Initialize base class with first and last budgets
        super().__init__(
            name="MI_Analysis",
            start_budget=self.budgets[0],
            increment=0,  # Not used, we have explicit budgets
            max_budget=self.budgets[-1]
        )

        logger.info(f"MI Policy initialized with budgets: {self.budgets}, seed={seed}")

    def get_budget_sequence(self) -> List[int]:
        """
        Return exact budget points for MI analysis.

        Overrides the base class progressive sequence generation.

        Returns:
            List of exact budget values
        """
        logger.debug(f"MI budget sequence: {self.budgets}")
        return self.budgets

    def select_observations(self, sample_pool: List[SampleTuple], budget: int) -> List[SampleTuple]:
        """
        Select random subset of samples for given budget.

        Args:
            sample_pool: Available training samples
            budget: Number of samples to select

        Returns:
            Randomly selected training samples
        """
        n_available = len(sample_pool)
        n_select = min(budget, n_available)

        logger.debug(f"Selecting {n_select} examples from {n_available} available (budget={budget})")

        # If budget exceeds available samples, return all
        if n_select >= n_available:
            logger.debug(f"Budget {budget} >= available samples {n_available}, returning all")
            return sample_pool

        # Random selection without replacement
        indices = self._rng.choice(n_available, n_select, replace=False)
        selected_samples = [sample_pool[i] for i in sorted(indices)]

        logger.debug(f"Selected {len(selected_samples)} examples")

        return selected_samples

    def reset_random_state(self) -> None:
        """Reset random state for reproducible reruns."""
        self._rng = np.random.RandomState(self.seed)
        logger.debug(f"Reset random state with seed {self.seed}")

    def get_selection_info(self, sample_pool: List[SampleTuple]) -> dict:
        """
        Get information about selection behavior.

        Args:
            sample_pool: Available sample pool

        Returns:
            Dictionary with selection statistics
        """
        n_available = len(sample_pool)

        selection_info = {
            'policy_name': self.name,
            'policy_type': 'mechanistic_interpretability',
            'seed': self.seed,
            'n_available_samples': n_available,
            'budget_sequence': self.budgets,
            'n_budget_points': len(self.budgets),
            'selection_rates': [min(budget / n_available, 1.0) for budget in self.budgets]
        }

        logger.debug(f"MI selection info: {len(self.budgets)} budget points")

        return selection_info

    def __str__(self) -> str:
        return f"MI_Analysis(budgets={self.budgets}, seed={self.seed})"
