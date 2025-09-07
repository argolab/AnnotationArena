from typing import List, Dict
import torch
from scipy.stats import spearmanr, kendalltau
import numpy as np
import torch.optim as optim

from .losses import DefaultLossStrategy, adapt_batched_logits_to_predictions
from .data import RankingData



class ImputerTrainer:
    def __init__(self, model, learning_rate=1e-3, alpha=1.0, beta=1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_strategy = DefaultLossStrategy()

    def train_step(self, ranking_data_list: List[RankingData]):
        """Single training step using List[RankingData] directly."""
        self.optimizer.zero_grad()

        # Forward pass - model takes List[RankingData] and returns batched tensors
        out = self.model(ranking_data_list)
        
        # Convert batched model outputs to per-variable predictions using the adapter
        predictions = adapt_batched_logits_to_predictions(out)
        
        # Create masked flags list - True if variable is masked (no supervision)
        masked_flags = []
        for var in ranking_data_list:
            is_masked = var.masked
            masked_flags.append(is_masked)
        
        # Compute losses using the loss strategy
        losses = self.loss_strategy.compute(predictions, ranking_data_list, masked_flags)

        # Backprop
        total_loss_tensor = losses.get('_total_loss_tensor', None)
        if total_loss_tensor is None:
            # Create a differentiable tensor from the computed loss
            total_loss_tensor = torch.tensor(losses['total_loss'], device=self.device, requires_grad=True)
        
        total_loss_tensor.backward()
        self.optimizer.step()

        # Return only float metrics
        return {k: v for k, v in losses.items() if not k.startswith('_')}

    def evaluate_with_test_data(self, test_data_list: List[RankingData]):
        """Evaluate model on test data."""
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            out = self.model(test_data_list)
            rating_logits = out['rating']
            ranking_logits = out['ranking']
            
            # Convert batched outputs to per-variable predictions using the adapter
            predictions = adapt_batched_logits_to_predictions(out)
            
            # Create masked flags - for test data, all variables with supervision are "observed"
            masked_flags = []
            eval_predictions = []
            eval_references = []
            
            # Lists to store ranking predictions and ground truth for Spearman correlation
            ranking_predictions = []
            ranking_ground_truths = []
            
            # Lists to store rating predictions and ground truth for accuracy
            rating_predictions = []
            rating_ground_truths = []
            
            for i, var in enumerate(test_data_list):
                if not var.is_listwise:  # Rating
                    # Store for accuracy calculation
                    pred_rating = torch.argmax(rating_logits[0, i]).item()
                    rating_predictions.append(pred_rating)
                    rating_ground_truths.append(var.rating_value)
                    
                    # Only include variables with supervision in evaluation
                    eval_predictions.append(predictions[i])
                    eval_references.append(var)
                else:  # Ranking
                    # Store for Spearman/Kendall correlation calculation
                    pred_scores = ranking_logits[0, i].cpu().numpy()
                    valid_positions = len([x for x in var.ranking_order if x > 0])
                    if valid_positions > 1:  # Need at least 2 items for correlation
                        # Convert predicted scores to predicted ranks (1=best, 2=second, etc)
                        # Higher scores should get lower rank numbers
                        from scipy.stats import rankdata
                        pred_ranks = rankdata(-pred_scores[:valid_positions])  # Negative to make higher scores = lower ranks
                        ranking_predictions.append(pred_ranks)
                        ranking_ground_truths.append(var.ranking_order[:valid_positions])
                    
                    # Only include variables with supervision in evaluation
                    eval_predictions.append(predictions[i])
                    eval_references.append(var)
                if not var.masked:
                    masked_flags.append(False)  # All are "observed" for test evaluation
                else:
                    masked_flags.append(True)
            
            if not eval_predictions:
                # No test data with supervision
                test_rating_loss = 0.0
                test_ranking_loss = 0.0
            else:
                # Compute losses on supervised test data
                losses = self.loss_strategy.compute(eval_predictions, test_data_list, masked_flags)
                test_rating_loss = losses.get('rating_loss', 0.0)
                test_ranking_loss = losses.get('ranking_loss', 0.0)
            
            # Calculate rating accuracy
            rating_accuracy = None
            if len(rating_predictions) > 0:
                correct = sum(p == t for p, t in zip(rating_predictions, rating_ground_truths))
                rating_accuracy = correct / len(rating_predictions)

            # Calculate Spearman and Kendall tau correlation for rankings
            spearman_rho = None
            spearman_pvalue = None
            kendall_tau = None
            kendall_pvalue = None
            if len(ranking_predictions) > 0:
                # Flatten all predictions and ground truths for overall correlation
                import numpy as np
                from scipy.stats import spearmanr, kendalltau
                all_pred_flat = np.concatenate(ranking_predictions)
                all_truth_flat = np.concatenate(ranking_ground_truths)
                
                if len(all_pred_flat) > 1:
                    try:
                        spearman_rho, spearman_pvalue = spearmanr(all_pred_flat, all_truth_flat)
                        kendall_tau, kendall_pvalue = kendalltau(all_pred_flat, all_truth_flat)
                    except Exception as e:
                        print(f"Warning: Could not calculate rank correlations: {e}")
                        spearman_rho = None
                        spearman_pvalue = None
                        kendall_tau = None
                        kendall_pvalue = None
            
        self.model.train()
        
        return {
            'test_rating_loss': float(test_rating_loss),
            'test_ranking_loss': float(test_ranking_loss),
            'total_test_loss': float(test_rating_loss + test_ranking_loss),
            'rating_accuracy': float(rating_accuracy) if rating_accuracy is not None else None,
            'num_rating_evaluations': len(rating_predictions),
            'spearman_rho': float(spearman_rho) if spearman_rho is not None else None,
            'spearman_pvalue': float(spearman_pvalue) if spearman_pvalue is not None else None,
            'kendall_tau': float(kendall_tau) if kendall_tau is not None else None,
            'kendall_pvalue': float(kendall_pvalue) if kendall_pvalue is not None else None,
            'num_ranking_evaluations': len(ranking_predictions),
        }


    def _compute_evaluation_metrics(self, predictions: List["TopLayerPredictionResult"], 
                                references: List[RankingData]) -> Dict[str, float]:
        """Compute additional evaluation metrics like accuracy, correlation, etc."""
        if not predictions:
            return {}
        
        rating_accuracy = 0.0
        ranking_correlations = []
        rating_count = 0
        ranking_count = 0
        
        for pred, ref in zip(predictions, references):
            if not ref.is_listwise and ref.rating_value is not None:
                # Rating accuracy
                predicted_class = torch.argmax(pred.rating_logits).item()
                if predicted_class == ref.rating_value:
                    rating_accuracy += 1.0
                rating_count += 1
            elif ref.is_listwise and ref.ranking_order is not None:
                # Ranking correlation (Spearman)
                predicted_scores = pred.ranking_logits.cpu().numpy()
                true_ranks = ref.ranking_order
                
                # Convert scores to predicted ranks (higher score = better rank)
                predicted_ranks = len(predicted_scores) + 1 - np.argsort(np.argsort(predicted_scores))
                
                if len(true_ranks) > 1:  # Need at least 2 items for correlation
                    from scipy.stats import spearmanr
                    correlation, _ = spearmanr(predicted_ranks[:len(true_ranks)], true_ranks)
                    if not np.isnan(correlation):
                        ranking_correlations.append(correlation)
                ranking_count += 1
        
        metrics = {}
        if rating_count > 0:
            metrics['rating_accuracy'] = rating_accuracy / rating_count
        if ranking_correlations:
            metrics['ranking_spearman'] = np.mean(ranking_correlations)
        
        return metrics

    def print_predictions_by_attribute(
        self,
        rating_logits: torch.Tensor,
        ranking_logits: torch.Tensor,
        test_rating_targets: torch.Tensor,
        test_ranking_targets: torch.Tensor,
        test_rating_mask: torch.Tensor,
        test_ranking_mask: torch.Tensor,
        all_variables,
        converter,
    ):
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS BY ATTRIBUTE")
        print("=" * 80)

        for attr in range(converter.num_attributes):
            print(f"\n--- ATTRIBUTE {attr} ---")

            # Ratings
            rating_found = False
            for i, var in enumerate(all_variables):
                if var['type'] == 'rating' and var['attribute'] == attr and test_rating_mask[0, i]:
                    if not rating_found:
                        print("Ratings:")
                        rating_found = True
                    pred_probs = torch.softmax(rating_logits[0, i], dim=0)
                    pred_class = torch.argmax(pred_probs).item() + 1
                    true_class = torch.argmax(test_rating_targets[0, i]).item() + 1
                    print(
                        f"  Annotator {var['annotator']}, Item {var['item']}: "
                        f"Pred={pred_class}, True={true_class}, "
                        f"Confidence={pred_probs[pred_class-1]:.3f}"
                    )

            # Rankings
            ranking_found = False
            for i, var in enumerate(all_variables):
                if var['type'] == 'ranking' and var['attribute'] == attr and test_ranking_mask[0, i]:
                    if not ranking_found:
                        print("Rankings:")
                        ranking_found = True
                    pred_scores = ranking_logits[0, i, : converter.max_rank_size]
                    pred_ranking_indices = torch.argsort(pred_scores, descending=True)
                    pred_items = [var['items'][idx] for idx in pred_ranking_indices]

                    true_scores = test_ranking_targets[0, i, : converter.max_rank_size]
                    true_ranking_indices = torch.argsort(true_scores, descending=False)
                    valid_positions = true_scores > 0
                    if valid_positions.any():
                        true_items = [var['items'][idx] for idx in true_ranking_indices]
                    else:
                        true_items = var['items']
                    print(
                        f"  Annotator {var['annotator']}, Items {var['items']}: "
                        f"Pred={pred_items}, True={true_items}"
                    )