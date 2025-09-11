from typing import List
import torch
# Removed scipy correlation imports - now using pairwise accuracy
import numpy as np
import torch.optim as optim

from .losses import DefaultLossStrategy, adapt_batched_logits_to_predictions
from .data import RankingData


class ImputerTrainer:
    def __init__(self, model, learning_rate=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu', embedding_anchor_reg: float = 0.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_strategy = DefaultLossStrategy()
        # Regularize embedding parameters towards their random initialization
        self.embedding_anchor_reg = float(embedding_anchor_reg)
        self._embedding_initial_params = {}
        if self.embedding_anchor_reg > 0.0:
            # Snapshot initial embedding parameters after model is moved to device
            for name, param in self.model.named_parameters():
                if param.requires_grad and self._is_embedding_param_name(name):
                    self._embedding_initial_params[name] = param.detach().clone()

    def _is_embedding_param_name(self, name: str) -> bool:
        # FIXME: this function seems not robust. Be careful.
        n = name.lower()
        return ("embedding" in n) or ("embed" in n)

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

        # Reconstruct references from batch tensors (0-indexed) - for ALL training variables with ground truth
        for i, var in enumerate(all_vars):
            if var.get('source') == 'train':  # All training variables (both masked and unmasked)
                if var['type'] == 'rating' and rating_mask[0, i]:  # Has ground truth
                    rating_val = int(torch.argmax(rating_targets[0, i]).item())
                    predictions.append(predictions_full[i])
                    references.append(RankingData(
                        annotator_id=var['annotator'] - 1,
                        attribute_id=var['attribute'] - 1,
                        is_listwise=False,
                        item_ids=[var['item'] - 1],
                        rating_value=rating_val,
                    ))
                elif var['type'] == 'ranking' and ranking_mask[0, i]:  # Has ground truth
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

        # Embedding anchor regularization: keep embeddings close to their random initialization
        reg_scaled = torch.tensor(0.0, device=self.device)
        if self.embedding_anchor_reg > 0.0 and self._embedding_initial_params:
            reg = torch.tensor(0.0, device=self.device)
            for name, p in self.model.named_parameters():
                if p.requires_grad and self._is_embedding_param_name(name):
                    init = self._embedding_initial_params.get(name)
                    if init is not None:
                        reg = reg + (p - init).pow(2).sum()
            reg_scaled = self.embedding_anchor_reg * reg
            # Log metric (float only in returned dict)
            losses['embedding_reg'] = float(reg_scaled.detach().item())
            # Ensure total_loss reflects the regularizer for logging
            if 'total_loss' in losses:
                losses['total_loss'] = float(losses['total_loss'] + losses['embedding_reg'])

        # Backprop
        total_loss_tensor = losses.get('_total_loss_tensor', None)
        if total_loss_tensor is None:
            total_loss_tensor = (rating_logits.sum() * 0.0) + (ranking_logits.sum() * 0.0) + torch.tensor(losses['total_loss'], device=self.device)
        # Add regularization term to tensor used for backprop
        total_loss_tensor = total_loss_tensor + reg_scaled
        total_loss_tensor.backward()
        self.optimizer.step()

        # Return only float metrics
        return {k: v for k, v in losses.items() if not k.startswith('_')}

    def evaluate_with_test_data(self, batch, test_data, converter, masking_rate=0.5, verbose=True):
        """Conditional imputation evaluation: same masking rate as training."""
        self.model.eval()

        with torch.no_grad():
            # Use the test batch which already has the correct masking applied
            all_variables = batch['all_variables']
            
            # Move batch to device (test batch already has correct masking)
            variable_data = batch['variable_data'].to(self.device)
            variable_types = batch['variable_types'].to(self.device)
            attribute_ids = batch['attribute_ids'].to(self.device)
            annotator_ids = batch['annotator_ids'].to(self.device)
            item_ids = batch['item_ids'].to(self.device)
            rating_targets = batch['rating_targets'].to(self.device)
            ranking_targets = batch['ranking_targets'].to(self.device)
            rating_mask = batch['rating_mask'].to(self.device)
            ranking_mask = batch['ranking_mask'].to(self.device)

            ranking_data_list = self.model._convert_legacy_tensors_to_ranking_data(
                variable_data, variable_types, attribute_ids, annotator_ids, item_ids
            )

            out = self.model(ranking_data_list)
            rating_logits = out['rating']
            ranking_logits = out['ranking']

            # Build targets for ALL test variables that have ground truth
            test_rating_mask = torch.zeros(1, len(all_variables), dtype=torch.bool)
            test_ranking_mask = torch.zeros(1, len(all_variables), dtype=torch.bool)
            test_rating_targets = torch.zeros(1, len(all_variables), converter.num_likert_classes)
            test_ranking_targets = torch.zeros(1, len(all_variables), converter.max_rank_size)

            # Extract test data for building targets
            test_rating_data, test_ranking_data = converter.process_training_data(test_data)

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
            
            # Lists to store pairwise ranking accuracy
            pairwise_correct = []
            pairwise_total = []
            
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
                        
                        # Store for pairwise ranking accuracy calculation
                        pred_ranking = predictions_full[i]
                        # Get predicted ranking scores directly from ranking_logits
                        pred_scores = ranking_logits[0, i].cpu().numpy()
                        valid_positions = len([x for x in ranking_order if x > 0])
                        if valid_positions == 2:  # Only evaluate pairwise rankings
                            # For pairwise rankings: ranking_logits[i, j] = score for item j being in position j+1
                            # Convert to predicted ranks using softmax probabilities
                            from scipy.special import softmax
                            position_probs = softmax(pred_scores[:valid_positions])
                            
                            # Predict: if prob[0] > prob[1], item 0 ranks first, otherwise item 1 ranks first
                            pred_first_wins = position_probs[0] > position_probs[1]
                            
                            # Ground truth: ranking_order[0] is rank of item 0, ranking_order[1] is rank of item 1
                            # If ranking_order = [1, 2], item 0 ranks first (rank 1 < rank 2)
                            # If ranking_order = [2, 1], item 1 ranks first (rank 2 > rank 1)
                            true_first_wins = ranking_order[0] < ranking_order[1]
                            
                            # Check if prediction matches ground truth
                            is_correct = pred_first_wins == true_first_wins
                            pairwise_correct.append(1 if is_correct else 0)
                            pairwise_total.append(1)
                        
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
                    'pairwise_accuracy': None,
                    'num_pairwise_evaluations': 0,
                }

            losses_eval = self.loss_strategy.compute(predictions, references)
            test_rating_loss = losses_eval['rating_loss']
            test_ranking_loss = losses_eval['ranking_loss']

            # Calculate rating accuracy
            rating_accuracy = None
            if len(rating_predictions) > 0:
                correct = sum(p == t for p, t in zip(rating_predictions, rating_ground_truths))
                rating_accuracy = correct / len(rating_predictions)

            # Calculate pairwise ranking accuracy
            pairwise_accuracy = None
            if len(pairwise_total) > 0:
                pairwise_accuracy = sum(pairwise_correct) / len(pairwise_total)

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
                
                # Print pairwise accuracy results
                print(f"\n=== Pairwise Ranking Evaluation Metrics ===")
                print(f"Number of pairwise evaluations: {len(pairwise_total)}")
                if pairwise_accuracy is not None:
                    print(f"Pairwise accuracy: {pairwise_accuracy:.4f} ({pairwise_accuracy*100:.1f}%)")
                else:
                    print("Pairwise accuracy could not be calculated")

            return {
                'test_rating_loss': float(test_rating_loss),
                'test_ranking_loss': float(test_ranking_loss),
                'total_test_loss': float(test_rating_loss + test_ranking_loss),
                'rating_accuracy': float(rating_accuracy) if rating_accuracy is not None else None,
                'num_rating_evaluations': len(rating_predictions),
                'pairwise_accuracy': float(pairwise_accuracy) if pairwise_accuracy is not None else None,
                'num_pairwise_evaluations': len(pairwise_total),
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
                    # For pairwise rankings: ranking_logits[0, i, j] represents the score that item j gets position (j+1)
                    # So ranking_logits[0, i, 0] = score for item 0 being in 1st place
                    #    ranking_logits[0, i, 1] = score for item 1 being in 2nd place
                    pred_scores = ranking_logits[0, i, : len(var['items'])]
                    # Convert scores to predicted ranking by assigning each item to most likely position
                    pred_ranks = torch.argmax(pred_scores.unsqueeze(0), dim=1)  # This is wrong approach
                    
                    # Correct approach: use softmax to get position probabilities, then create ranking
                    position_probs = torch.softmax(pred_scores, dim=0)
                    # For pairwise: if position_probs[0] > position_probs[1], then item 0 ranks higher
                    if len(var['items']) == 2:
                        if position_probs[0] > position_probs[1]:
                            pred_items = [var['items'][0], var['items'][1]]  # Item 0 first, Item 1 second
                        else:
                            pred_items = [var['items'][1], var['items'][0]]  # Item 1 first, Item 0 second
                    else:
                        # Fallback for non-pairwise (shouldn't happen in ICLR)
                        pred_ranking_indices = torch.argsort(position_probs, descending=True)
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