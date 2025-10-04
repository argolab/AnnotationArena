"""Logit lens analyzer for examining intermediate representations."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from imputer.data import RankingData, DataConverter
from imputer.ranking_imputer import MultiVariableImputer


@dataclass
class LayerAnalysis:
    """Results from analyzing a single layer."""
    layer_idx: int
    hidden_states: torch.Tensor  # [B, N, D]
    logits: Dict[str, torch.Tensor]  # {'rating': [B, N, C], 'ranking': [B, N, R]}
    metrics: Dict[str, float]  # Performance metrics for this layer


@dataclass
class VariableAnalysis:
    """Results from analyzing a single variable across all layers."""
    variable: RankingData
    layer_analyses: List[LayerAnalysis]  # One analysis per layer


@dataclass
class LogitLensResults:
    """Complete results from logit lens analysis."""
    all_variables: List[VariableAnalysis]  # All analyzed variables (status derived from variable properties)
    model_config: Dict[str, Any]
    data_config: Dict[str, Any]
    
    def filter_variables(self, **filters) -> List[VariableAnalysis]:
        """Filter variables based on status flags.
        
        Args:
            **filters: Keyword arguments for filtering (is_train, is_test, is_rating, etc.)
        
        Returns:
            List of VariableAnalysis objects matching the filters
        """
        filtered = self.all_variables
        
        for key, value in filters.items():
            if key == 'is_train':
                filtered = [var for var in filtered if (var.variable.instance == 'train') == value]
            elif key == 'is_test':
                filtered = [var for var in filtered if (var.variable.instance == 'test') == value]
            elif key == 'is_rating':
                filtered = [var for var in filtered if (not var.variable.is_listwise) == value]
            elif key == 'is_ranking':
                filtered = [var for var in filtered if var.variable.is_listwise == value]
            elif key == 'is_observed':
                filtered = [var for var in filtered if var.variable.is_observed == value]
            elif key == 'is_masked':
                filtered = [var for var in filtered if var.variable.is_masked == value]
            elif key == 'is_missing':
                filtered = [var for var in filtered if var.variable.is_missing == value]
            else:
                print(f"Warning: Unknown filter key '{key}' - skipping")
        
        return filtered
    
    def get_train_variables(self) -> List[VariableAnalysis]:
        """Get all training variables."""
        return self.filter_variables(is_train=True)
    
    def get_test_variables(self) -> List[VariableAnalysis]:
        """Get all test variables."""
        return self.filter_variables(is_test=True)
    
    def get_rating_variables(self) -> List[VariableAnalysis]:
        """Get all rating variables."""
        return self.filter_variables(is_rating=True)
    
    def get_ranking_variables(self) -> List[VariableAnalysis]:
        """Get all ranking variables."""
        return self.filter_variables(is_ranking=True)
    
    def get_observed_variables(self) -> List[VariableAnalysis]:
        """Get all observed variables."""
        return self.filter_variables(is_observed=True)
    
    def get_masked_variables(self) -> List[VariableAnalysis]:
        """Get all masked variables."""
        return self.filter_variables(is_masked=True)
    
    def get_missing_variables(self) -> List[VariableAnalysis]:
        """Get all missing variables."""
        return self.filter_variables(is_missing=True)


class LogitLensAnalyzer:
    """Analyzes intermediate representations using logit lens technique."""
    
    def __init__(self, model: MultiVariableImputer, converter: DataConverter, device: str = 'cuda'):
        self.model = model.to(device)
        self.converter = converter
        self.device = device
        self.model.eval()
        
    def analyze_all_variables_across_layers(self, 
                                          all_variables: List[RankingData]) -> List[VariableAnalysis]:
        """Analyze all variables across all layers in a single forward pass."""
        
        with torch.no_grad():
            # Run a single forward pass with intermediates captured
            logits_final, hidden_intermediates = self.model(all_variables, return_intermediate=True)
            # hidden_intermediates is a list of [features, params] per transformer block,
            # plus a final [features_normed, params] entry after normalization per model.forward

            layer_analyses: List[LayerAnalysis] = []
            for layer_idx, (features_snapshot, params_snapshot) in enumerate(hidden_intermediates):
                # Compute head logits from the params snapshot at this layer
                rating_logits = self.model.apply_head('rating', params_snapshot)
                ranking_logits = self.model.apply_head('ranking', params_snapshot)

                layer_analyses.append(
                    LayerAnalysis(
                        layer_idx=layer_idx,
                        hidden_states=features_snapshot,
                        logits={'rating': rating_logits, 'ranking': ranking_logits},
                        metrics={}
                    )
                )
            
            # Create VariableAnalysis for each variable
            variable_analyses = []
            for i, var in enumerate(all_variables):
                # Determine head type based on variable type
                head_type = 'ranking' if var.is_listwise else 'rating'
                
                # Extract metrics for this variable across all layers
                var_layer_analyses = []
                for layer_analysis in layer_analyses:
                    logits_full = layer_analysis.logits[head_type]
                    # Take per-variable slice to avoid storing logits for all variables
                    logits_slice = logits_full[0, i]
                    metrics = self._compute_single_variable_metrics(var, logits_slice, head_type)
                    
                    # Store only this variable's hidden state vector for the layer
                    if layer_analysis.hidden_states is not None:
                        # hidden_states shape expected [B, N, D]; take [0, i, :]
                        try:
                            hidden_slice = layer_analysis.hidden_states[0, i]
                        except Exception:
                            hidden_slice = layer_analysis.hidden_states
                    else:
                        hidden_slice = None

                    var_layer_analysis = LayerAnalysis(
                        layer_idx=layer_analysis.layer_idx,
                        hidden_states=hidden_slice,
                        logits={head_type: logits_slice},
                        metrics=metrics
                    )
                    var_layer_analyses.append(var_layer_analysis)
                
                variable_analysis = VariableAnalysis(
                    variable=var,
                    layer_analyses=var_layer_analyses
                )
                variable_analyses.append(variable_analysis)
            
            return variable_analyses
    
    def _compute_single_variable_metrics(self, 
                                        target_variable: RankingData,
                                        variable_prediction_logits: torch.Tensor, 
                                        head_type: str) -> Dict[str, float]:
        """Compute metrics for a single variable at a specific layer."""
        
        # Evaluate whenever we have a valid target, mirroring eval.py behavior:
        # - observed (train/test)
        # - masked (train held-out with ground truth)
        # - missing (test items that still carry reference in bundle)
        
        if head_type == 'rating' and not target_variable.is_listwise:
            # Rating metrics
            assert not target_variable.is_listwise, "rating variables should be provided"
            target = target_variable.rating_value
            if target is None:
                raise RuntimeError("no valid target")
            prediction = torch.argmax(variable_prediction_logits).item()
            accuracy = 1.0 if prediction == target else 0.0
            
            pred_rating = prediction + 1  # Convert to 1-5 scale
            true_rating = target + 1
            rmse = float(np.sqrt((pred_rating - true_rating) ** 2))
            
            # Compute soft prediction (expected value under softmax)
            probs = torch.softmax(variable_prediction_logits, dim=0).cpu().numpy()
            # Assume classes are 0,1,2,3,4 (for 1-5 scale)
            class_indices = np.arange(len(probs))
            expected_rating = float(np.sum(probs * (class_indices + 1)))
            l2_loss = float((expected_rating - true_rating) ** 2)
            
            return {'accuracy': accuracy, 'rmse': rmse, 'l2_loss': l2_loss, 'num_evaluations': 1}
            
        elif head_type == 'ranking' and target_variable.is_listwise:
            assert target_variable.is_listwise, "ranking variables should be provided here"
            # Ranking metrics
            scores = variable_prediction_logits.cpu().numpy()
            
            if len(target_variable.ranking_order or []) == 2:
                # Compute softmax probabilities for the two items
                probs = torch.softmax(torch.tensor(scores[:2]), dim=0).numpy()
                pred_first_wins = probs[0] > probs[1]
                pred_ranking = [1, 2] if pred_first_wins else [2, 1]
                
                pred_first = pred_ranking[0] < pred_ranking[1]
                true_first = target_variable.ranking_order[0] < target_variable.ranking_order[1]
                accuracy = 1.0 if pred_first == true_first else 0.0

                # Bradley-Terry log loss (bt_loss)
                # True label: 0 if first wins, 1 if second wins
                # We assume ranking_order gives the true order: lower is better
                true_label = 0 if true_first else 1
                # Clamp probabilities for numerical stability
                eps = 1e-8
                p = np.clip(probs[true_label], eps, 1 - eps)
                bt_loss = -np.log(p)
            else:
                # No valid target for ranking
                raise RuntimeError("no valid target")
            
            return {'accuracy': accuracy, 'bt_loss': float(bt_loss), 'num_evaluations': 1}
        
        # No valid target for this head/variable
        raise RuntimeError("no valid target")
    
    def analyze_all_layers(self, 
                          train_variables: List[RankingData],
                          test_variables: List[RankingData]) -> LogitLensResults:
        """Optimized logit lens analysis: separate forward passes for train and test instances."""
        
        print("Running logit lens analysis...")
        
        # Process train and test variables separately since they are different instances
        train_analyses = []
        test_analyses = []
        
        if train_variables:
            print(f"Processing {len(train_variables)} training variables...")
            train_analyses = self.analyze_all_variables_across_layers(train_variables)
        
        if test_variables:
            print(f"Processing {len(test_variables)} test variables...")
            test_analyses = self.analyze_all_variables_across_layers(test_variables)
        
        # Combine all analyses
        all_analyses = train_analyses + test_analyses
        
        # Extract configs
        model_config = {
            'num_attributes': self.model.num_attributes,
            'num_annotators': self.model.num_annotators,
            'num_items': self.model.num_items,
            'num_likert_classes': self.model.num_likert_classes,
            'max_rank_size': self.model.max_rank_size,
            'embedding_dim': self.model.embedding_dim,
            'num_layers': len(self.model.blocks)
        }
        
        # Count variables by status for data config
        train_count = len(train_variables)
        test_count = len(test_variables)
        observed_count = sum(1 for var in train_variables + test_variables if var.is_observed)
        masked_count = sum(1 for var in train_variables + test_variables if var.is_masked)
        missing_count = sum(1 for var in train_variables + test_variables if var.is_missing)
        rating_count = sum(1 for var in train_variables + test_variables if not var.is_listwise)
        ranking_count = sum(1 for var in train_variables + test_variables if var.is_listwise)
        
        data_config = {
            'num_train_variables': train_count,
            'num_test_variables': test_count,
            'num_observed_variables': observed_count,
            'num_masked_variables': masked_count,
            'num_missing_variables': missing_count,
            'num_rating_variables': rating_count,
            'num_ranking_variables': ranking_count
        }
        
        print(f"Analysis complete: {len(all_analyses)} variables analyzed (train: {len(train_analyses)}, test: {len(test_analyses)})")
        
        return LogitLensResults(
            all_variables=all_analyses,
            model_config=model_config,
            data_config=data_config
        )
