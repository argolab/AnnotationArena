"""
Selection policies for progressive imputation experiments.

Provides abstract policy interface and concrete random example selection policy
for progressive training data acquisition in imputation experiments.
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Iterator, Tuple, Any

logger = logging.getLogger(__name__)

# Type alias for sample tuple
SampleTuple = Tuple[Any, Any, Any, Any, Any, Any]  # Will be properly typed when imported


class BaseObservationPolicy(ABC):
    """
    Abstract base class for progressive observation policies.
    
    Defines the interface for selecting training observations progressively
    with increasing budget allocations for active learning experiments.
    """
    
    def __init__(self, name: str, start_budget: int, increment: int, max_budget: int):
        """
        Initialize policy with budget configuration.
        
        Args:
            name: Human-readable policy name
            start_budget: Initial budget allocation
            increment: Budget increment per step
            max_budget: Maximum budget allocation
        """
        self.name = name
        self.start_budget = start_budget
        self.increment = increment
        self.max_budget = max_budget
        
        logger.debug(f"Initialized policy {name}: start={start_budget}, "
                    f"increment={increment}, max={max_budget}")
        
    @abstractmethod
    def select_observations(self, sample_pool: List[SampleTuple], budget: int) -> List[SampleTuple]:
        """
        Select observations from sample pool given a budget allocation.
        
        Args:
            sample_pool: Available samples to select from
            budget: Budget allocation for this selection round
            
        Returns:
            Selected samples for training
        """
        pass
    
    def get_budget_sequence(self) -> List[int]:
        """
        Generate the sequence of budget values for progressive observation.
        
        Creates a sequence starting from start_budget, incrementing by increment,
        up to max_budget (inclusive).
        
        Returns:
            List of budget values for progressive experiment
        """
        budgets = []
        budget = self.start_budget
        
        # Generate sequence with regular increments
        while budget <= self.max_budget:
            budgets.append(budget)
            budget += self.increment
            
        # Ensure max_budget is included (handle edge case where increments don't align)
        if budgets[-1] != self.max_budget:
            budgets.append(self.max_budget)
            
        logger.debug(f"Budget sequence: {budgets}")
        return budgets
    
    def observe_progressively(self, sample_pool: List[SampleTuple]) -> Iterator[Tuple[int, List[SampleTuple]]]:
        """
        Generator for progressive observation with increasing budgets.
        
        Yields (budget, observations) pairs for each step in the progressive
        experiment, allowing for clean experiment coordination.
        
        Args:
            sample_pool: Available samples to select from
            
        Yields:
            Tuple of (current_budget, selected_observations)
        """
        budgets = self.get_budget_sequence()
        
        logger.info(f"Progressive observation with {self.name}: "
                   f"{len(budgets)} steps from {self.start_budget} to {self.max_budget}")
        
        for budget in budgets:
            observations = self.select_observations(sample_pool, budget)
            
            logger.debug(f"Budget {budget}: Selected {len(observations)} observations")
            yield budget, observations
            
    def __str__(self) -> str:
        return f"{self.name}(start={self.start_budget}, inc={self.increment}, max={self.max_budget})"


class RandomExamplePolicy(BaseObservationPolicy):
    """
    Policy that randomly selects complete examples from the sample pool.
    
    Budget is interpreted as the number of complete examples to observe.
    Uses controlled randomization with NumPy RandomState for reproducibility.
    """
    
    def __init__(self, start_examples: int = 10, increment: int = 100, 
                 max_examples: int = 500, seed: int = 42):
        """
        Initialize random example selection policy.
        
        Args:
            start_examples: Number of examples to start with
            increment: Number of examples to add each step
            max_examples: Maximum number of examples 
            seed: Random seed for reproducible selection
        """
        super().__init__(
            name="RandomExample",
            start_budget=start_examples,
            increment=increment, 
            max_budget=max_examples
        )
        
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        
        logger.debug(f"RandomExamplePolicy initialized with seed {seed}")
        
    def select_observations(self, sample_pool: List[SampleTuple], budget: int) -> List[SampleTuple]:
        """
        Select a random subset of samples without replacement.
        
        Args:
            sample_pool: List of available training samples
            budget: Number of examples to select
            
        Returns:
            List of randomly selected training samples
        """
        n_available = len(sample_pool)
        n_select = min(budget, n_available)
        
        logger.debug(f"Selecting {n_select} examples from {n_available} available (budget={budget})")
        
        # If budget exceeds available samples, return all
        if n_select >= n_available:
            logger.debug(f"Budget {budget} >= available samples {n_available}, returning all")
            return sample_pool
        
        # Random selection without replacement using controlled randomization
        indices = self._rng.choice(n_available, n_select, replace=False)
        selected_samples = [sample_pool[i] for i in sorted(indices)]
        
        logger.debug(f"Selected {len(selected_samples)} examples using indices: {sorted(indices)[:10]}...")
        
        return selected_samples
    
    def reset_random_state(self) -> None:
        """
        Reset random state for reproducible experiment reruns.
        
        Useful for ensuring consistent behavior across experiment repetitions
        or when debugging specific selection patterns.
        """
        self._rng = np.random.RandomState(self.seed)
        logger.debug(f"Reset random state with seed {self.seed}")
    
    def get_selection_info(self, sample_pool: List[SampleTuple]) -> dict:
        """
        Get information about selection behavior without actually selecting.
        
        Args:
            sample_pool: Available sample pool
            
        Returns:
            Dictionary with selection statistics and budget information
        """
        budgets = self.get_budget_sequence()
        n_available = len(sample_pool)
        
        selection_info = {
            'policy_name': self.name,
            'seed': self.seed,
            'n_available_samples': n_available,
            'budget_sequence': budgets,
            'n_steps': len(budgets),
            'selection_rates': [min(budget / n_available, 1.0) for budget in budgets],
            'total_selections': sum(min(budget, n_available) for budget in budgets)
        }
        
        logger.debug(f"Selection info: {self.name} will make {selection_info['n_steps']} selections")
        
        return selection_info