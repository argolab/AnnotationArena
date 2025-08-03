"""
Random example selection policy for progressive imputation.

Selects complete examples randomly at each budget step.
Cost: 1 unit = 1 complete example
"""

import numpy as np
import logging
from .base_policy import BaseObservationPolicy

logger = logging.getLogger(__name__)

class RandomExamplePolicy(BaseObservationPolicy):
    """
    Policy that randomly selects complete examples.
    
    Budget is interpreted as number of complete examples to observe.
    """
    
    def __init__(self, start_examples=10, increment=150, max_examples=3000, seed=42):
        super().__init__(
            name="RandomExample",
            start_budget=start_examples,
            increment=increment, 
            max_budget=max_examples
        )
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        
    def select_observations(self, sample_pool, budget):
        """
        Select a random subset of samples with missing data.
        
        Args:
            sample_pool: List of samples with missing data
            budget: Number of examples to select
            
        Returns:
            List of selected training samples
        """
        n_available = len(sample_pool)
        n_select = min(budget, n_available)
        
        if n_select >= n_available:
            logger.debug(f"Budget {budget} >= available samples {n_available}, returning all")
            return sample_pool
        
        # Random selection without replacement
        indices = self._rng.choice(n_available, n_select, replace=False)
        selected_samples = [sample_pool[i] for i in sorted(indices)]
        
        logger.debug(f"Selected {len(selected_samples)}/{n_available} examples (budget={budget})")
        
        return selected_samples
    
    def reset_random_state(self):
        """Reset random state for reproducible experiments."""
        self._rng = np.random.RandomState(self.seed)
        logger.debug(f"Reset random state with seed {self.seed}")