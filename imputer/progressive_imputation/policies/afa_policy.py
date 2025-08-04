"""
Active Feature Acquisition (AFA) policy using Value of Information (VOI).

This policy selects individual node observations based on their expected 
information gain, measured by the reduction in uncertainty (L2 loss) about 
other unobserved nodes.

Author: Prabhav Singh
"""

import numpy as np
import torch
import torch.nn.functional as F
import logging
from typing import List, Tuple, Set
from .base_policy import BaseObservationPolicy

logger = logging.getLogger(__name__)

class VOICalculator:
    """
    Value of Information calculator for Bayesian Network imputation.
    
    Computes the expected reduction in L2 loss from observing a candidate node,
    measuring how much uncertainty about target nodes would be reduced.
    """
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_l2_loss(self, predictions):
        """
        Compute L2 loss as variance of predicted distributions.
        
        Args:
            predictions: Tensor of shape (batch, n_nodes, n_states) with probability distributions
            
        Returns:
            float: L2 loss (variance) across all predictions
        """
        # For binary nodes: compute variance of Bernoulli distributions
        # Var(Bernoulli(p)) = p(1-p)
        probs = F.softmax(predictions, dim=-1)
        
        # Extract p for state 1
        p = probs[..., 1]  # Probability of state 1
        
        # Compute variance: p(1-p)
        variance = p * (1 - p)
        
        # Return mean variance across all nodes
        return variance.mean().item()
    
    def compute_voi(self, sample, candidate_node, target_nodes, imputer, bn, adj_matrix, n_nodes):
        """
        Compute VOI for observing candidate_node in a BN imputation sample.
        
        Args:
            sample: Single sample (inputs, structure_info, dims, mask, targets)
            candidate_node: Node index we're considering observing
            target_nodes: List of other unobserved nodes (targets)
            imputer: Trained imputer model
            bn: BayesNet object (for CPT extraction)
            adj_matrix: Adjacency matrix
            n_nodes: Number of nodes
            
        Returns:
            float: VOI value (expected reduction in L2 loss)
        """
        if not target_nodes:
            return 0.0
            
        # Handle both old format (5 elements) and new AFA format (6 elements)
        if len(sample) == 6:
            inputs, structure_info, dims, mask, targets, ground_truth = sample
        else:
            inputs, structure_info, dims, mask, targets = sample
        
        # Get observed nodes for CPT extraction
        observed_nodes = [i for i in range(n_nodes) if mask[i] == 0]
        
        # 1. Get initial predictions and compute L2 loss on target nodes
        try:
            from models.imputer import extract_cpts_for_nodes, compute_max_cpt_size
            
            max_cpt_size = compute_max_cpt_size(bn)
            cpt_info = extract_cpts_for_nodes(bn, observed_nodes, n_nodes, max_cpt_size)
            cpt_tensor = torch.FloatTensor(cpt_info).to(self.device)
            
            # Add batch dimension for model input
            inputs_batch = inputs.unsqueeze(0).to(self.device)
            structure_batch = structure_info.unsqueeze(0).to(self.device)
            cpt_batch = cpt_tensor.unsqueeze(0)
            dims_batch = dims.unsqueeze(0).to(self.device)
            
            # Get initial predictions
            initial_preds = imputer(inputs_batch, structure_batch, cpt_batch, dims_batch)
            
            # Extract predictions for target nodes only
            target_preds = initial_preds[:, target_nodes, :]
            initial_loss = self.compute_l2_loss(target_preds)
            
            # Get candidate node's predicted probabilities
            candidate_preds = initial_preds[0, candidate_node, :]
            candidate_probs = F.softmax(candidate_preds, dim=-1)
            
        except Exception as e:
            logger.debug(f"Initial prediction failed for candidate {candidate_node}: {e}")
            return 0.0
        
        # 2. Compute expected posterior loss
        expected_posterior_loss = 0.0
        
        for value in [0, 1]:  # Binary nodes
            try:
                # Create hypothetical input with candidate_node = value
                hyp_inputs = inputs.clone()
                hyp_inputs[candidate_node, 0] = 0  # Mark as observed
                hyp_inputs[candidate_node, 1:] = 0  # Clear old values
                hyp_inputs[candidate_node, 1 + value] = 1  # Set to value
                
                # Update mask
                hyp_mask = mask.clone()
                hyp_mask[candidate_node] = 0  # Mark as observed
                
                # Update observed nodes for CPT extraction
                hyp_observed_nodes = observed_nodes + [candidate_node]
                hyp_cpt_info = extract_cpts_for_nodes(bn, hyp_observed_nodes, n_nodes, max_cpt_size)
                hyp_cpt_tensor = torch.FloatTensor(hyp_cpt_info).to(self.device)
                
                # Get predictions with this additional observation
                hyp_inputs_batch = hyp_inputs.unsqueeze(0).to(self.device)
                hyp_cpt_batch = hyp_cpt_tensor.unsqueeze(0)
                
                hyp_preds = imputer(hyp_inputs_batch, structure_batch, hyp_cpt_batch, dims_batch)
                
                # Extract predictions for target nodes (excluding candidate)
                remaining_targets = [t for t in target_nodes if t != candidate_node]
                if remaining_targets:
                    hyp_target_preds = hyp_preds[:, remaining_targets, :]
                    hyp_loss = self.compute_l2_loss(hyp_target_preds)
                else:
                    hyp_loss = 0.0  # No remaining targets
                
                # Weight by predicted probability of candidate_node = value
                prob = candidate_probs[value].item()
                expected_posterior_loss += prob * hyp_loss
                
            except Exception as e:
                logger.debug(f"Hypothetical prediction failed for candidate {candidate_node}, value {value}: {e}")
                # If prediction fails, assume no improvement
                expected_posterior_loss += candidate_probs[value].item() * initial_loss
        
        # VOI = expected reduction in L2 loss
        voi = initial_loss - expected_posterior_loss
        
        return max(0.0, voi)  # Ensure non-negative VOI


class AFAPolicy(BaseObservationPolicy):
    """
    Active Feature Acquisition policy using Value of Information.
    
    Selects individual node observations based on their expected information gain,
    measured by reduction in L2 loss about other unobserved nodes.
    """
    
    def __init__(self, start_budget=10, nodes_per_cycle=150, max_budget=3000, seed=42):
        super().__init__(
            name="AFA_VOI",
            start_budget=start_budget,
            increment=nodes_per_cycle,
            max_budget=max_budget
        )
        self.nodes_per_cycle = nodes_per_cycle
        self.seed = seed
        self.observed_nodes = set()  # Track (sample_idx, node_idx) pairs
        self.voi_calculator = VOICalculator()
        self._rng = np.random.RandomState(seed)
        
        logger.info(f"Initialized AFA policy: {nodes_per_cycle} nodes per cycle, "
                   f"budget {start_budget} to {max_budget}")
    
    def select_observations(self, sample_pool, budget):
        """
        Select individual node observations using VOI.
        
        Args:
            sample_pool: List of samples with missing data
            budget: Number of nodes to observe this cycle
            
        Returns:
            Updated sample_pool with selected nodes marked as observed
        """
        logger.info(f"AFA selecting {budget} node observations from {len(sample_pool)} samples")
        
        # This will be called by the experiment runner - but AFA works differently
        # We need access to the imputer model, BN, and adjacency matrix
        # These will be passed via the experiment runner
        return sample_pool  # Placeholder - actual logic in select_nodes_with_voi
    
    def select_nodes_with_voi(self, sample_pool, budget, imputer, bn, adj_matrix, n_nodes):
        """
        Select nodes using VOI computation.
        
        Args:
            sample_pool: List of samples with missing data
            budget: Number of nodes to observe
            imputer: Trained imputer model
            bn: BayesNet object
            adj_matrix: Adjacency matrix
            n_nodes: Number of nodes
            
        Returns:
            List of (sample_idx, node_idx, voi) tuples for selected observations
        """
        logger.debug(f"Computing VOI for node selection, budget={budget}")
        
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
        
        logger.debug(f"Found {len(candidates)} candidate nodes to evaluate")
        
        if not candidates:
            logger.warning("No candidate nodes available for VOI selection")
            return []
        
        # 2. Compute VOI for each candidate
        candidate_vois = []
        
        for sample_idx, candidate_node in candidates:
            sample = sample_pool[sample_idx]
            # Handle both old format (5 elements) and new AFA format (6 elements)
            if len(sample) == 6:
                inputs, structure_info, dims, mask, targets, ground_truth = sample
            else:
                inputs, structure_info, dims, mask, targets = sample
            
            # Find other unobserved nodes in this sample (targets for VOI)
            target_nodes = [i for i in range(n_nodes) 
                           if mask[i] == 1 and i != candidate_node]
            
            if target_nodes:  # Only compute VOI if there are targets
                try:
                    voi = self.voi_calculator.compute_voi(
                        sample, candidate_node, target_nodes, 
                        imputer, bn, adj_matrix, n_nodes
                    )
                    candidate_vois.append((sample_idx, candidate_node, voi))
                    
                except Exception as e:
                    logger.debug(f"VOI computation failed for sample {sample_idx}, "
                               f"node {candidate_node}: {e}")
                    # Assign small random VOI if computation fails
                    voi = self._rng.random() * 1e-6
                    candidate_vois.append((sample_idx, candidate_node, voi))
            else:
                # No targets - assign zero VOI
                candidate_vois.append((sample_idx, candidate_node, 0.0))
        
        # 3. Select top 'budget' candidates by VOI
        selected = sorted(candidate_vois, key=lambda x: x[2], reverse=True)[:budget]
        
        logger.info(f"Selected {len(selected)} nodes with VOI range: "
                   f"{selected[-1][2]:.6f} to {selected[0][2]:.6f}")
        
        return selected
    
    def apply_observations(self, sample_pool, selected_observations):
        """
        Apply selected observations to the sample pool.
        
        Args:
            sample_pool: List of samples to modify
            selected_observations: List of (sample_idx, node_idx, voi) tuples
            
        Returns:
            Updated sample_pool
        """
        logger.debug(f"Applying {len(selected_observations)} observations to sample pool")
        
        for sample_idx, node_idx, voi in selected_observations:
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
        logger.debug(f"Reset AFA random state with seed {self.seed}")
    
    def get_budget_sequence(self):
        """
        Get the sequence of budgets for progressive observation.
        
        For AFA, this represents cumulative node observations.
        
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
        Generator for progressive observation - AFA version.
        
        Note: This is a placeholder. The actual AFA logic is handled
        by the VOI experiment runner which calls select_nodes_with_voi directly.
        
        Args:
            sample_pool: Sample pool to observe from
            
        Yields:
            tuple: (budget, sample_pool) at each step
        """
        budgets = self.get_budget_sequence()
        logger.info(f"AFA progressive observation: {len(budgets)} steps from "
                   f"{self.start_budget} to {self.max_budget} node observations")
        
        for budget in budgets:
            # For AFA, we return the current sample pool
            # The actual selection is done by the VOI experiment runner
            logger.debug(f"AFA budget step: {budget} total node observations")
            yield budget, sample_pool