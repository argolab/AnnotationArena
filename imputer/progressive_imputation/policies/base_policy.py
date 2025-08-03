"""
Base policy interface for progressive observation.
"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseObservationPolicy(ABC):
    """
    Abstract base class for observation policies.
    """
    
    def __init__(self, name, start_budget, increment, max_budget):
        self.name = name
        self.start_budget = start_budget
        self.increment = increment
        self.max_budget = max_budget
        self.current_budget = 0
        
    @abstractmethod
    def select_observations(self, sample_pool, budget):
        """
        Select observations from sample pool given a budget.
        
        Args:
            sample_pool: Sample pool to sample from
            budget: Budget for observations (in policy-specific units)
            
        Returns:
            Selected observations (format depends on policy)
        """
        pass
    
    def get_budget_sequence(self):
        """
        Get the sequence of budgets for progressive observation.
        
        Returns:
            List of budget values
        """
        budgets = []
        budget = self.start_budget
        
        while budget <= self.max_budget:
            budgets.append(budget)
            budget += self.increment
            
        # Ensure we include max_budget
        if budgets[-1] != self.max_budget:
            budgets.append(self.max_budget)
            
        return budgets
    
    def observe_progressively(self, sample_pool):
        """
        Generator that yields (budget, observations) pairs progressively.
        
        Args:
            sample_pool: Sample pool to sample from
            
        Yields:
            tuple: (budget, observations) at each step
        """
        budgets = self.get_budget_sequence()
        logger.info(f"Progressive observation with {self.name}: {len(budgets)} steps from {self.start_budget} to {self.max_budget}")
        
        for budget in budgets:
            observations = self.select_observations(sample_pool, budget)
            logger.debug(f"Budget {budget}: Selected {len(observations)} observations")
            yield budget, observations
            
    def __str__(self):
        return f"{self.name}(start={self.start_budget}, inc={self.increment}, max={self.max_budget})"