from typing import List
import torch
import torch.optim as optim

from .losses import DefaultLossStrategy, adapt_batched_logits_to_predictions
from .data import RankingData


class ImputerTrainer:
    def __init__(self, model, learning_rate=1e-3, alpha=1.0, beta=1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_strategy = DefaultLossStrategy(alpha=alpha, beta=beta)

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

        # Forward pass
        out = self.model(variable_data, variable_types, attribute_ids, annotator_ids, item_ids)
        rating_logits = out['rating']
        ranking_logits = out['ranking']

        # Structured predictions and references for loss computation
        # Only create references for variables that have supervision (mask = True)
        predictions_full = adapt_batched_logits_to_predictions({'rating': rating_logits, 'ranking': ranking_logits})
        predictions: List["TopLayerPredictionResult"] = []
        references: List[RankingData] = []
        masked_flags: List[bool] = []
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
                masked_flags.append(bool(rating_masked[0, i].item()))
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
                masked_flags.append(bool(ranking_masked[0, i].item()))

        losses = self.loss_strategy.compute(predictions, references, masked_flags)

        # Backprop
        total_loss_tensor = losses.get('_total_loss_tensor', None)
        if total_loss_tensor is None:
            total_loss_tensor = (rating_logits.sum() * 0.0) + (ranking_logits.sum() * 0.0) + torch.tensor(losses['total_loss'], device=self.device)
        total_loss_tensor.backward()
        self.optimizer.step()

        # Return only float metrics
        return {k: v for k, v in losses.items() if not k.startswith('_')}

    def evaluate_with_test_data(self, batch, test_data, converter, mask_rate=0.5, verbose=True):
        """Evaluate model on test data with proper imputation masking using structured losses."""
        self.model.eval()

        with torch.no_grad():
            # Process test data
            test_rating_data, test_ranking_data = converter.process_training_data(test_data)

            # Create test variables and apply masking for imputation
            all_variables = batch['all_variables']

            # Collect test variables that have data
            test_rating_vars = []
            test_ranking_vars = []
            for i, var in enumerate(all_variables):
                if var['type'] == 'rating':
                    key = (var['attribute'], var['annotator'], var['item'])
                    if key in test_rating_data:
                        test_rating_vars.append(i)
                elif var['type'] == 'ranking':
                    items = var['items']
                    # Check if ranking exists in the list
                    ranking_exists = any(
                        ranking_entry['attribute'] == var['attribute'] and
                        ranking_entry['annotator'] == var['annotator'] and
                        ranking_entry['items'] == items
                        for ranking_entry in test_ranking_data
                    )
                    if ranking_exists:
                        test_ranking_vars.append(i)

            import random
            random.seed(42)
            num_rating_masked = int(len(test_rating_vars) * mask_rate)
            num_ranking_masked = int(len(test_ranking_vars) * mask_rate)
            masked_test_rating_vars = set(random.sample(test_rating_vars, num_rating_masked)) if test_rating_vars else set()
            masked_test_ranking_vars = set(random.sample(test_ranking_vars, num_ranking_masked)) if test_ranking_vars else set()

            # Create input data for imputer (with masked positions set to zero)
            test_variable_data = batch['variable_data'].clone()
            for i in masked_test_rating_vars:
                test_variable_data[0, i, :] = 0.0
            for i in masked_test_ranking_vars:
                test_variable_data[0, i, :] = 0.0

            # Move to device
            test_variable_data = test_variable_data.to(self.device)
            variable_types = batch['variable_types'].to(self.device)
            attribute_ids = batch['attribute_ids'].to(self.device)
            annotator_ids = batch['annotator_ids'].to(self.device)
            item_ids = batch['item_ids'].to(self.device)

            out = self.model(test_variable_data, variable_types, attribute_ids, annotator_ids, item_ids)
            rating_logits = out['rating']
            ranking_logits = out['ranking']

            # Build targets/masks ONLY for the masked test variables
            test_rating_mask = torch.zeros(1, len(all_variables), dtype=torch.bool)
            test_ranking_mask = torch.zeros(1, len(all_variables), dtype=torch.bool)
            test_rating_targets = torch.zeros(1, len(all_variables), converter.num_likert_classes)
            test_ranking_targets = torch.zeros(1, len(all_variables), converter.max_rank_size)

            for i in masked_test_rating_vars:
                var = all_variables[i]
                key = (var['attribute'], var['annotator'], var['item'])
                if key in test_rating_data:
                    test_rating_mask[0, i] = True
                    rating_value = test_rating_data[key] - 1
                    test_rating_targets[0, i, rating_value] = 1.0

            for i in masked_test_ranking_vars:
                var = all_variables[i]
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

            # Structured loss on masked test entries
            # Only create references for variables that have test supervision (test_mask = True)
            predictions_full = adapt_batched_logits_to_predictions({'rating': rating_logits, 'ranking': ranking_logits})
            predictions: List["TopLayerPredictionResult"] = []
            references: List[RankingData] = []
            masked_flags: List[bool] = []
            for i, var in enumerate(all_variables):
                if var['type'] == 'rating' and test_rating_mask[0, i]:  # Only if has test supervision
                    rating_val = int(torch.argmax(test_rating_targets[0, i]).item())
                    predictions.append(predictions_full[i])
                    references.append(RankingData(
                        annotator_id=var['annotator'] - 1,
                        attribute_id=var['attribute'] - 1,
                        is_listwise=False,
                        item_ids=[var['item'] - 1],
                        rating_value=rating_val,
                    ))
                    masked_flags.append(True)  # All test evaluation entries are "masked" for loss computation
                elif var['type'] == 'ranking' and test_ranking_mask[0, i]:  # Only if has test supervision
                    scores_vec = test_ranking_targets[0, i]
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
                        item_ids=[it - 1 for it in var['items'][: converter.max_rank_size]],
                        ranking_order=ranking_order,
                    ))
                    masked_flags.append(True)  # All test evaluation entries are "masked" for loss computation

            losses_eval = self.loss_strategy.compute(predictions, references, masked_flags)
            test_rating_loss = losses_eval['rating_loss_masked']
            test_ranking_loss = losses_eval['ranking_loss_masked']

            if verbose:
                self.print_predictions_by_attribute(
                    rating_logits, ranking_logits,
                    test_rating_targets, test_ranking_targets,
                    test_rating_mask, test_ranking_mask,
                    all_variables, converter,
                )

            return {
                'test_rating_loss': float(test_rating_loss),
                'test_ranking_loss': float(test_ranking_loss),
                'total_test_loss': float(test_rating_loss + test_ranking_loss),
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
