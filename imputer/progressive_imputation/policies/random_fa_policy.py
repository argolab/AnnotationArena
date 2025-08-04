"""
Random Feature Acquisition (Random FA) policy.

This policy selects individual node observations randomly, providing a 
baseline for comparison with VOI-based Active Feature Acquisition.

Like AFA, it operates at node-level granularity but uses random selection
instead of information-theoretic criteria.

Author: Prabhav Singh
"""

import numpy as np
import torch
import logging
from typing import List, Tuple, Set
from .base_policy import BaseObservationPolicy

logger = logging.getLogger(__name__)

class RandomFAPolicy(BaseObservationPolicy):
    """
    Random Feature Acquisition policy.
    
    Selects individual node observations randomly from all available 
    unobserved nodes across all samples. Provides a fair baseline for 
    comparing against VOI-based Active Feature Acquisition.
    """
    
    def __init__(self, start_budget=10, nodes_per_cycle=150, max_budget=3000, seed=42):
        super().__init__(
            name="Random_FA",
            start_budget=start_budget,
            increment=nodes_per_cycle,
            max_budget=max_budget
        )
        self.nodes_per_cycle = nodes_per_cycle
        self.seed = seed
        self.observed_nodes = set()  # Track (sample_idx, node_idx) pairs
        self._rng = np.random.RandomState(seed)
        
        logger.info(f"Initialized Random FA policy: {nodes_per_cycle} nodes per cycle, "
                   f"budget {start_budget} to {max_budget}")
    
    def select_observations(self, sample_pool, budget):
        """
        Select individual node observations randomly.
        
        Args:
            sample_pool: List of samples with missing data
            budget: Number of nodes to observe this cycle
            
        Returns:
            Updated sample_pool with selected nodes marked as observed
        """
        logger.info(f"Random FA selecting {budget} node observations from {len(sample_pool)} samples")
        
        # This will be called by the experiment runner - but Random FA works differently
        # We need access to the full sample pool and node information
        # These will be passed via the experiment runner
        return sample_pool  # Placeholder - actual logic in select_nodes_randomly
    
    def select_nodes_randomly(self, sample_pool, budget, n_nodes):
        """
        Select nodes using random selection.
        
        Args:
            sample_pool: List of samples with missing data
            budget: Number of nodes to observe
            n_nodes: Number of nodes in the graph
            
        Returns:
            List of (sample_idx, node_idx, random_score) tuples for selected observations
        """
        logger.debug(f"Randomly selecting nodes, budget={budget}")
        
        # 1. Find all unobserved (sample_idx, node_idx) candidates
        candidates = []
        for sample_idx, sample in enumerate(sample_pool):
            # Handle both old format (5 elements) and new AFA format (6 elements)
            if len(sample) == 6:
                inputs, structure_info, dims, mask, targets, ground_truth = sample
            else:
                inputs, structure_info, dims, mask, targets = sample
            
            for node_idx in range(n_nodes):
                if mask[node_idx] == 1 and (sample_idx, node_idx) not in self.observed_nodes:
                    candidates.append((sample_idx, node_idx))
        
        logger.debug(f"Found {len(candidates)} candidate nodes for random selection")
        
        if not candidates:
            logger.warning("No candidate nodes available for random selection")
            return []
        
        # 2. Randomly select up to 'budget' candidates
        n_select = min(budget, len(candidates))
        selected_indices = self._rng.choice(len(candidates), n_select, replace=False)
        
        # 3. Create selected list with random scores (for consistency with VOI interface)
        selected = []
        for idx in selected_indices:
            sample_idx, node_idx = candidates[idx]
            random_score = self._rng.random()  # Random score between 0 and 1
            selected.append((sample_idx, node_idx, random_score))
        
        logger.info(f"Randomly selected {len(selected)} nodes")
        
        return selected
    
    def apply_observations(self, sample_pool, selected_observations):
        """
        Apply selected observations to the sample pool.
        
        Args:
            sample_pool: List of samples to modify
            selected_observations: List of (sample_idx, node_idx, score) tuples
            
        Returns:
            Updated sample_pool
        """
        logger.debug(f"Applying {len(selected_observations)} random observations to sample pool")
        
        for sample_idx, node_idx, score in selected_observations:
            # Mark as observed in our tracking
            self.observed_nodes.add((sample_idx, node_idx))
            
            # Get the true value for this node from ground truth
            sample = sample_pool[sample_idx]
            
            # Check if this is the new AFA format (with ground truth)
            if len(sample) == 6:
                from data.sample_generator_afa import get_ground_truth_value, apply_afa_observation
                true_value = get_ground_truth_value(sample, node_idx)
                sample_pool[sample_idx] = apply_afa_observation(sample, node_idx, true_value)
            else:
                # Old format - extract from targets
                if len(sample) == 6:
                    inputs, structure_info, dims, mask, targets, ground_truth = sample
                else:
                    inputs, structure_info, dims, mask, targets = sample
                true_probs = targets[node_idx].numpy()
                true_value = np.argmax(true_probs)
                
                # Update the sample
                new_inputs = inputs.clone()
                new_mask = mask.clone()
                
                # Mark as observed
                new_mask[node_idx] = 0
                new_inputs[node_idx, 0] = 0  # Remove mask bit
                new_inputs[node_idx, 1:] = 0  # Clear old values
                new_inputs[node_idx, 1 + true_value] = 1  # Set true value
                
                # Update sample in pool
                sample_pool[sample_idx] = (new_inputs, structure_info, dims, new_mask, targets)
        
        logger.debug(f"Total observed nodes: {len(self.observed_nodes)}")
        return sample_pool
    
    def reset_random_state(self):
        """Reset random state for reproducible experiments."""
        self._rng = np.random.RandomState(self.seed)
        logger.debug(f"Reset Random FA random state with seed {self.seed}")
    
    def get_budget_sequence(self):
        """
        Get the sequence of budgets for progressive observation.
        
        For Random FA, this represents cumulative node observations.
        
        Returns:
            List of budget values (cumulative node observations)
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
        Generator for progressive observation - Random FA version.
        
        Note: This is a placeholder. The actual Random FA logic is handled
        by the VOI experiment runner which calls select_nodes_randomly directly.
        
        Args:
            sample_pool: Sample pool to observe from
            
        Yields:
            tuple: (budget, sample_pool) at each step
        """
        budgets = self.get_budget_sequence()
        logger.info(f"Random FA progressive observation: {len(budgets)} steps from "
                   f"{self.start_budget} to {self.max_budget} node observations")
        
        for budget in budgets:
            # For Random FA, we return the current sample pool
            # The actual selection is done by the VOI experiment runner
            logger.debug(f"Random FA budget step: {budget} total node observations")
            yield budget, sample_pool