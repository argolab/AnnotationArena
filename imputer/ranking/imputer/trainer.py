from typing import List
import torch
from scipy.stats import spearmanr, kendalltau
import numpy as np
import torch.optim as optim

from .losses import DefaultLossStrategy, adapt_batched_logits_to_predictions
from .data import RankingData


class ImputerTrainer:
    def __init__(self, model, learning_rate=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_strategy = DefaultLossStrategy()

    def train_step(self, batch):
        """Single training step using legacy tensor batch + structured losses."""
        self.optimizer.zero_grad()

        # Move batch to device
        variable_data = batch['variable_data'].to(self.device)
        variable_types = batch['variable_types'].to(self.device)
        attribute_ids = batch['attribute_ids'].to(self.device)
        annotator_ids = batch['annotator_ids'].to(self.device)
        item_ids = batch['item_ids'].to(self.device)
        rating_targets = batch['rating_targets'].to(self.device)
        ranking_targets = batch['ranking_targets'].to(self.device)
        rating_mask = batch['rating_mask'].to(self.device)
        ranking_mask = batch['ranking_mask'].to(self.device)
        rating_masked = batch['rating_masked'].to(self.device)
        ranking_masked = batch['ranking_masked'].to(self.device)
        ranking_data_list = self.model._convert_legacy_tensors_to_ranking_data(
            variable_data, variable_types, attribute_ids, annotator_ids, item_ids
        )
        # Forward pass
        out = self.model(ranking_data_list)
        rating_logits = out['rating']
        ranking_logits = out['ranking']

        # Structured predictions and references for loss computation
        # Only create references for variables that have supervision (mask = True)
        predictions_full = adapt_batched_logits_to_predictions({'rating': rating_logits, 'ranking': ranking_logits})
        predictions: List["TopLayerPredictionResult"] = []
        references: List[RankingData] = []
        all_vars = batch['all_variables']

        # Reconstruct references from batch tensors (0-indexed) - only for supervised variables
        for i, var in enumerate(all_vars):
            if var['type'] == 'rating' and rating_mask[0, i]:  # Only if has supervision
                rating_val = int(torch.argmax(rating_targets[0, i]).item())
                predictions.append(predictions_full[i])
                references.append(RankingData(
                    annotator_id=var['annotator'] - 1,
                    attribute_id=var['attribute'] - 1,
                    is_listwise=False,
                    item_ids=[var['item'] - 1],
                    rating_value=rating_val,
                ))
            elif var['type'] == 'ranking' and ranking_mask[0, i]:  # Only if has supervision
                scores_vec = ranking_targets[0, i]
                ranking_order = []
                for j in range(scores_vec.shape[0]):
                    s = int(scores_vec[j].item())
                    if s > 0:
                        ranking_order.append(int(s))
                predictions.append(predictions_full[i])
                references.append(RankingData(
                    annotator_id=var['annotator'] - 1,
                    attribute_id=var['attribute'] - 1,
                    is_listwise=True,
                    item_ids=[it - 1 for it in var['items'][: self.model.max_rank_size]],
                    ranking_order=ranking_order,
                ))

        losses = self.loss_strategy.compute(predictions, references)

        # Backprop
        total_loss_tensor = losses.get('_total_loss_tensor', None)
        if total_loss_tensor is None:
            total_loss_tensor = (rating_logits.sum() * 0.0) + (ranking_logits.sum() * 0.0) + torch.tensor(losses['total_loss'], device=self.device)
        total_loss_tensor.backward()
        self.optimizer.step()

        # Return only float metrics
        return {k: v for k, v in losses.items() if not k.startswith('_')}

    def evaluate_with_test_data(self, batch, test_data, converter, verbose=True):
        """Pure imputation evaluation: predict ALL test variables from structure only."""
        self.model.eval()

        with torch.no_grad():
            # Process test data
            test_rating_data, test_ranking_data = converter.process_training_data(test_data)
            all_variables = batch['all_variables']

            # Create test input with NO supervision values (pure imputation)
            # All test variables get zero supervision, only structural embeddings
            test_variable_data = torch.zeros_like(batch['variable_data'])
            
            # Move to device
            test_variable_data = test_variable_data.to(self.device)
            variable_types = batch['variable_types'].to(self.device)
            attribute_ids = batch['attribute_ids'].to(self.device)
            annotator_ids = batch['annotator_ids'].to(self.device)
            item_ids = batch['item_ids'].to(self.device)

            ranking_data_list = self.model._convert_legacy_tensors_to_ranking_data(
                test_variable_data, variable_types, attribute_ids, annotator_ids, item_ids
            )

            out = self.model(ranking_data_list)
            rating_logits = out['rating']
            ranking_logits = out['ranking']

            # Build targets for ALL test variables that have ground truth
            test_rating_mask = torch.zeros(1, len(all_variables), dtype=torch.bool)
            test_ranking_mask = torch.zeros(1, len(all_variables), dtype=torch.bool)
            test_rating_targets = torch.zeros(1, len(all_variables), converter.num_likert_classes)
            test_ranking_targets = torch.zeros(1, len(all_variables), converter.max_rank_size)

            # Process all test variables (source == 'test')
            for i, var in enumerate(all_variables):
                if var.get('source') == 'test':
                    if var['type'] == 'rating':
                        key = (var['attribute'], var['annotator'], var['item'])
                        if key in test_rating_data:
                            test_rating_mask[0, i] = True
                            rating_value = test_rating_data[key] - 1
                            test_rating_targets[0, i, rating_value] = 1.0
                    else:  # ranking
                        items = var['items']
                        # Find matching ranking in the list
                        matching_ranking = None
                        for ranking_entry in test_ranking_data:
                            if (ranking_entry['attribute'] == var['attribute'] and
                                ranking_entry['annotator'] == var['annotator'] and
                                ranking_entry['items'] == items):
                                matching_ranking = ranking_entry
                                break
                        
                        if matching_ranking:
                            test_ranking_mask[0, i] = True
                            order = matching_ranking['order']
                            for j, pos in enumerate(order):
                                if j < converter.max_rank_size:
                                    test_ranking_targets[0, i, j] = pos

            # Structured loss on ALL test entries with ground truth
            predictions_full = adapt_batched_logits_to_predictions({'rating': rating_logits, 'ranking': ranking_logits})
            predictions: List["TopLayerPredictionResult"] = []
            references: List[RankingData] = []
            
            # Lists to store ranking predictions and ground truth for Spearman correlation
            ranking_predictions = []
            ranking_ground_truths = []
            
            # Lists to store rating predictions and ground truth for accuracy
            rating_predictions = []
            rating_ground_truths = []
            
            for i, var in enumerate(all_variables):
                if var.get('source') == 'test':
                    if var['type'] == 'rating' and test_rating_mask[0, i]:
                        rating_val = int(torch.argmax(test_rating_targets[0, i]).item())
                        
                        # Store for accuracy calculation
                        pred_rating = torch.argmax(rating_logits[0, i]).item()
                        rating_predictions.append(pred_rating)
                        rating_ground_truths.append(rating_val)
                        
                        predictions.append(predictions_full[i])
                        references.append(RankingData(
                            annotator_id=var['annotator'] - 1,
                            attribute_id=var['attribute'] - 1,
                            is_listwise=False,
                            item_ids=[var['item'] - 1],
                            rating_value=rating_val,
                        ))
                    elif var['type'] == 'ranking' and test_ranking_mask[0, i]:
                        scores_vec = test_ranking_targets[0, i]
                        ranking_order = []
                        for j in range(scores_vec.shape[0]):
                            s = int(scores_vec[j].item())
                            if s > 0:
                                ranking_order.append(int(s))
                        
                        # Store for Spearman/Kendall correlation calculation
                        pred_ranking = predictions_full[i]
                        # Get predicted ranking scores directly from ranking_logits
                        pred_scores = ranking_logits[0, i].cpu().numpy()
                        valid_positions = len([x for x in ranking_order if x > 0])
                        if valid_positions > 1:  # Need at least 2 items for correlation
                            # Convert predicted scores to predicted ranks (1=best, 2=second, etc)
                            # Higher scores should get lower rank numbers  
                            from scipy.stats import rankdata
                            pred_ranks = rankdata(-pred_scores[:valid_positions])  # Negative to make higher scores = lower ranks
                            ranking_predictions.append(pred_ranks)
                            ranking_ground_truths.append(ranking_order[:valid_positions])
                        
                        predictions.append(predictions_full[i])
                        references.append(RankingData(
                            annotator_id=var['annotator'] - 1,
                            attribute_id=var['attribute'] - 1,
                            is_listwise=True,
                            item_ids=[it - 1 for it in var['items'][: converter.max_rank_size]],
                            ranking_order=ranking_order,
                        ))

            if len(predictions) == 0:
                return {
                    'test_rating_loss': 0.0,
                    'test_ranking_loss': 0.0,
                    'total_test_loss': 0.0,
                    'rating_accuracy': None,
                    'num_rating_evaluations': 0,
                    'spearman_rho': None,
                    'spearman_pvalue': None,
                    'kendall_tau': None,
                    'kendall_pvalue': None,
                    'num_ranking_evaluations': 0,
                }

            losses_eval = self.loss_strategy.compute(predictions, references)
            test_rating_loss = losses_eval['rating_loss']
            test_ranking_loss = losses_eval['ranking_loss']

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

            if verbose:
                self.print_predictions_by_attribute(
                    rating_logits, ranking_logits,
                    test_rating_targets, test_ranking_targets,
                    test_rating_mask, test_ranking_mask,
                    all_variables, converter,
                )
                
                # Print rating accuracy
                print(f"\n=== Rating Evaluation Metrics ===")
                print(f"Number of rating evaluations: {len(rating_predictions)}")
                if rating_accuracy is not None:
                    print(f"Rating accuracy: {rating_accuracy:.4f} ({rating_accuracy*100:.1f}%)")
                else:
                    print("Rating accuracy could not be calculated")
                
                # Print correlation results
                print(f"\n=== Ranking Evaluation Metrics ===")
                print(f"Number of ranking evaluations: {len(ranking_predictions)}")
                if spearman_rho is not None:
                    print(f"Spearman's rho: {spearman_rho:.4f} (p={spearman_pvalue:.4f})")
                    print(f"Kendall's tau: {kendall_tau:.4f} (p={kendall_pvalue:.4f})")
                    avg_corr = (abs(spearman_rho) + abs(kendall_tau)) / 2
                    print(f"Average correlation strength: {'Strong' if avg_corr > 0.7 else 'Moderate' if avg_corr > 0.3 else 'Weak'}")
                else:
                    print("Rank correlations could not be calculated")

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
                    # For ground truth display: sort by ascending rank (1st place first)
                    valid_positions = true_scores > 0
                    if valid_positions.any():
                        valid_ranks = true_scores[valid_positions]
                        valid_items = [var['items'][idx] for idx in torch.where(valid_positions)[0]]
                        # Sort items by their rank values (ascending: 1st, 2nd, 3rd...)
                        sorted_indices = torch.argsort(valid_ranks, descending=False)
                        true_items = [valid_items[idx] for idx in sorted_indices]
                    else:
                        true_items = var['items']
                    print(
                        f"  Annotator {var['annotator']}, Items {var['items']}: "
                        f"Pred={pred_items}, True={true_items}"
                    )