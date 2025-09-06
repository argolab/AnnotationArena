from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import json
import torch


@dataclass
class RankingData:
    """Structured representation of a single variable for ranking/rating.

    All indices are 0-indexed for model consumption.
    - annotator_id: annotator index
    - attribute_id: attribute index
    - is_listwise: True for listwise ranking, False for rating
    - item_ids: for rating, a list with one item id; for ranking, the ranked item ids
    Optional supervision fields:
    - rating_value: class index [0..C-1] if rating observed
    - ranking_order: list of positions in [1..R] aligned with item_ids when ranking observed
    """
    annotator_id: int
    attribute_id: int
    is_listwise: bool
    item_ids: List[int]
    rating_value: Optional[int] = None
    ranking_order: Optional[List[int]] = None
    # TODO: do we need a observed flag?


class DataConverter:
    def __init__(self, num_attributes=10, num_annotators=5, num_items=10, num_likert_classes=5, max_rank_size=3):
        self.num_attributes = num_attributes
        self.num_annotators = num_annotators
        self.num_items = num_items
        self.num_likert_classes = num_likert_classes
        self.max_rank_size = max_rank_size

    def load_training_data(self, json_file: str) -> Dict[str, Any]:
        with open(json_file, 'r') as f:
            data = json.load(f)
        filtered_ratings = [r for r in data['ratings'] if r['item'] <= self.num_items]
        filtered_rankings = []
        for ranking in data['rankings']:
            items_to_check = ranking['items'][: self.max_rank_size]
            if all(item <= self.num_items for item in items_to_check):
                filtered_rankings.append(ranking)
        return {'ratings': filtered_ratings, 'rankings': filtered_rankings}

    def create_variables_from_actual_data(self, train_data: Dict[str, Any], test_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rating_variables: List[Dict[str, Any]] = []
        ranking_variables: List[Dict[str, Any]] = []

        # Create rating variables from training data
        for rating in train_data['ratings']:
            rating_variables.append({
                'type': 'rating', 
                'attribute': rating['attribute'], 
                'annotator': rating['annotator'], 
                'item': rating['item'],
                'source': 'train'
            })
        
        # Create rating variables from test data
        for rating in test_data['ratings']:
            rating_variables.append({
                'type': 'rating', 
                'attribute': rating['attribute'], 
                'annotator': rating['annotator'], 
                'item': rating['item'],
                'source': 'test'
            })

        # Create ranking variables from training data
        for ranking in train_data['rankings']:
            ranking_variables.append({
                'type': 'ranking',
                'attribute': ranking['attribute'],
                'annotator': ranking['annotator'], 
                'items': ranking['items'][: self.max_rank_size],
                'source': 'train'
            })
        
        # Create ranking variables from test data
        for ranking in test_data['rankings']:
            ranking_variables.append({
                'type': 'ranking',
                'attribute': ranking['attribute'],
                'annotator': ranking['annotator'], 
                'items': ranking['items'][: self.max_rank_size],
                'source': 'test'
            })

        return rating_variables, ranking_variables

    def process_training_data(self, data: Dict[str, Any]) -> Tuple[Dict[Tuple[int, int, int], int], List[Dict[str, Any]]]:
        rating_data: Dict[Tuple[int, int, int], int] = {}
        ranking_data: List[Dict[str, Any]] = []

        for rating in data['ratings']:
            key = (rating['attribute'], rating['annotator'], rating['item'])
            rating_data[key] = rating['value']

        for ranking in data['rankings']:
            items = ranking['items'][: self.max_rank_size]
            order = ranking['order'][: self.max_rank_size]
            ranking_data.append({
                'attribute': ranking['attribute'],
                'annotator': ranking['annotator'],
                'items': items,
                'order': order
            })
        return rating_data, ranking_data

    def build_structured_variables(self, rating_variables: List[Dict[str, Any]], ranking_variables: List[Dict[str, Any]]) -> List[RankingData]:
        variables: List[RankingData] = []
        for var in rating_variables:
            variables.append(RankingData(
                annotator_id=var['annotator'] - 1,
                attribute_id=var['attribute'] - 1,
                is_listwise=False,
                item_ids=[var['item'] - 1],
            ))
        for var in ranking_variables:
            variables.append(RankingData(
                annotator_id=var['annotator'] - 1,
                attribute_id=var['attribute'] - 1,
                is_listwise=True,
                item_ids=[i - 1 for i in var['items'][: self.max_rank_size]],
            ))
        return variables

    def build_structured_with_targets(
        self,
        rating_variables: List[Dict[str, Any]],
        ranking_variables: List[Dict[str, Any]],
        rating_data: Dict[Tuple[int, int, int], int],
        ranking_data: List[Dict[str, Any]],
    ) -> List[RankingData]:
        variables: List[RankingData] = []
        for var in rating_variables:
            key = (var['attribute'], var['annotator'], var['item'])
            rating_value = rating_data.get(key, None)
            variables.append(RankingData(
                annotator_id=var['annotator'] - 1,
                attribute_id=var['attribute'] - 1,
                is_listwise=False,
                item_ids=[var['item'] - 1],
                rating_value=(rating_value - 1) if rating_value is not None else None,
            ))
        for var in ranking_variables:
            items = var['items'][: self.max_rank_size]
            order = None
            # Find matching ranking in the list
            for ranking_entry in ranking_data:
                if (ranking_entry['attribute'] == var['attribute'] and 
                    ranking_entry['annotator'] == var['annotator'] and 
                    ranking_entry['items'] == items):
                    order = ranking_entry['order'][: self.max_rank_size]
                    break  # Take the first match
            variables.append(RankingData(
                annotator_id=var['annotator'] - 1,
                attribute_id=var['attribute'] - 1,
                is_listwise=True,
                item_ids=[i - 1 for i in items],
                ranking_order=order,
            ))
        return variables

    def create_training_batch(
        self,
        rating_variables: List[Dict[str, Any]],
        ranking_variables: List[Dict[str, Any]],
        rating_data: Dict[Tuple[int, int, int], int],
        ranking_data: List[Dict[str, Any]],
        test_data: Optional[Dict[str, Any]] = None,
        mask_rate: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Create a single training batch (legacy tensor format with masking).

        Returns a dict of tensors including inputs, targets, and masks along with 'all_variables'.
        """
        all_variables = rating_variables + ranking_variables
        num_variables = len(all_variables)

        variable_data = torch.zeros(1, num_variables, max(self.num_likert_classes, self.max_rank_size))
        variable_types = torch.zeros(1, num_variables, dtype=torch.long)
        attribute_ids = torch.zeros(1, num_variables, dtype=torch.long)
        annotator_ids = torch.zeros(1, num_variables, dtype=torch.long)
        item_ids = torch.full((1, num_variables, self.max_rank_size), -1, dtype=torch.long)

        rating_targets = torch.zeros(1, num_variables, self.num_likert_classes)
        ranking_targets = torch.zeros(1, num_variables, self.max_rank_size)
        rating_mask = torch.zeros(1, num_variables, dtype=torch.bool)
        ranking_mask = torch.zeros(1, num_variables, dtype=torch.bool)
        rating_masked = torch.zeros(1, num_variables, dtype=torch.bool)
        ranking_masked = torch.zeros(1, num_variables, dtype=torch.bool)

        # Collect available for masking - only training variables with data
        available_rating_vars = []
        available_ranking_vars = []
        for i, var in enumerate(all_variables):
            # Only consider variables from training data
            if var.get('source') == 'train':
                if var['type'] == 'rating':
                    key = (var['attribute'], var['annotator'], var['item'])
                    if key in rating_data:
                        available_rating_vars.append(i)
                else:
                    items = var['items']
                    # Check if ranking exists in the training data list
                    ranking_exists = any(
                        ranking_entry['attribute'] == var['attribute'] and
                        ranking_entry['annotator'] == var['annotator'] and
                        ranking_entry['items'] == items
                        for ranking_entry in ranking_data
                    )
                    if ranking_exists:
                        available_ranking_vars.append(i)

        import random
        num_rating_masked = int(len(available_rating_vars) * mask_rate)
        num_ranking_masked = int(len(available_ranking_vars) * mask_rate)
        masked_rating_indices = set(random.sample(available_rating_vars, num_rating_masked)) if available_rating_vars else set()
        masked_ranking_indices = set(random.sample(available_ranking_vars, num_ranking_masked)) if available_ranking_vars else set()

        for i, var in enumerate(all_variables):
            attribute_ids[0, i] = var['attribute'] - 1
            annotator_ids[0, i] = var['annotator'] - 1

            # Only process training variables for supervision
            if var.get('source') == 'train':
                if var['type'] == 'rating':
                    variable_types[0, i] = 0
                    item_ids[0, i, 0] = var['item'] - 1
                    key = (var['attribute'], var['annotator'], var['item'])
                    if key in rating_data:
                        rating_value = rating_data[key] - 1
                        rating_targets[0, i, rating_value] = 1.0
                        rating_mask[0, i] = True
                        if i in masked_rating_indices:
                            rating_masked[0, i] = True
                        else:
                            variable_data[0, i, rating_value] = 1.0
                else:
                    variable_types[0, i] = 1
                    items = var['items']
                    for j, item in enumerate(items):
                        if j < self.max_rank_size:
                            item_ids[0, i, j] = item - 1
                    # Find matching ranking in the list
                    matching_ranking = None
                    for ranking_entry in ranking_data:
                        if (ranking_entry['attribute'] == var['attribute'] and
                            ranking_entry['annotator'] == var['annotator'] and
                            ranking_entry['items'] == items):
                            matching_ranking = ranking_entry
                            break
                    
                    if matching_ranking:
                        order = matching_ranking['order']
                        for j, pos in enumerate(order):
                            if j < self.max_rank_size:
                                ranking_targets[0, i, j] = pos
                        ranking_mask[0, i] = True
                        if i in masked_ranking_indices:
                            ranking_masked[0, i] = True
                        else:
                            for j, pos in enumerate(order):
                                if j < self.max_rank_size:
                                    variable_data[0, i, j] = pos
            else:
                # Test variables - set basic info but no supervision
                if var['type'] == 'rating':
                    variable_types[0, i] = 0
                    item_ids[0, i, 0] = var['item'] - 1
                else:
                    variable_types[0, i] = 1
                    items = var['items']
                    for j, item in enumerate(items):
                        if j < self.max_rank_size:
                            item_ids[0, i, j] = item - 1

        return {
            'variable_data': variable_data,
            'variable_types': variable_types,
            'attribute_ids': attribute_ids,
            'annotator_ids': annotator_ids,
            'item_ids': item_ids,
            'rating_targets': rating_targets,
            'ranking_targets': ranking_targets,
            'rating_mask': rating_mask,
            'ranking_mask': ranking_mask,
            'rating_masked': rating_masked,
            'ranking_masked': ranking_masked,
            'all_variables': all_variables,
        }
