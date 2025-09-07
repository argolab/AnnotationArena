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
    masked: bool = False
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
    
    def _convert_tensors_to_ranking_data(self, variable_data, variable_types, attribute_ids, annotator_ids, item_ids) -> List[RankingData]:
        """Convert legacy tensor format to List[RankingData]."""
        batch_size, num_variables = variable_types.shape
        assert batch_size == 1, "Only batch size 1 supported for conversion"
        
        variables = []
        for i in range(num_variables):
            var_type = variable_types[0, i].item()
            attr_id = attribute_ids[0, i].item()
            annot_id = annotator_ids[0, i].item()
            
            if var_type == 0:  # Rating
                item_id = item_ids[0, i, 0].item()
                # Check if this rating has supervision (non-zero data)
                rating_value = None
                data_vec = variable_data[0, i]
                rating_value = torch.argmax(data_vec[1:]).item()
                if data_vec[0] == 0:
                    variables.append(RankingData(
                        annotator_id=annot_id,
                        attribute_id=attr_id,
                        is_listwise=False,
                        item_ids=[item_id],
                        rating_value=rating_value,
                        masked=False
                    ))
                else:
                    variables.append(RankingData(
                        annotator_id=annot_id,
                        attribute_id=attr_id,
                        is_listwise=False,
                        item_ids=[item_id],
                        rating_value=rating_value,
                        masked=True
                    ))
            else:  # Ranking
                # Extract valid item IDs (non-negative)
                item_list = []
                for j in range(self.max_rank_size):
                    item_id = item_ids[0, i, j].item()
                    if item_id >= 0:
                        item_list.append(item_id)
                
                # Check if this ranking has supervision (non-zero data)
                ranking_order = None
                data_vec = variable_data[0, i]
                ranking_order = []
                for j in range(len(item_list)):
                    if j < data_vec.shape[0]:
                        rank_pos = int(data_vec[j + 1].item())
                        if rank_pos > 0:
                            ranking_order.append(rank_pos)
                if data_vec[0] == 0:
                
                    variables.append(RankingData(
                        annotator_id=annot_id,
                        attribute_id=attr_id,
                        is_listwise=True,
                        item_ids=item_list,
                        ranking_order=ranking_order,
                        masked=False
                    ))
                else:
                    variables.append(RankingData(
                        annotator_id=annot_id,
                        attribute_id=attr_id,
                        is_listwise=True,
                        item_ids=item_list,
                        ranking_order=ranking_order,
                        masked=True
                    ))
                    
        
        return variables

    def create_ranking_data_list(
        self,
        rating_variables: List[Dict[str, Any]],
        ranking_variables: List[Dict[str, Any]],
        rating_data: Dict[Tuple[int, int, int], int],
        ranking_data: List[Dict[str, Any]],
        test_data: Optional[Dict[str, Any]] = None,
        mask_rate: float = 0.5,
        mode: str="train"
    ) -> Dict[str, torch.Tensor]:
        """Create a single training batch (legacy tensor format with masking).

        Returns a dict of tensors including inputs, targets, and masks along with 'all_variables'.
        """


        all_variables = [var for var in rating_variables + ranking_variables if var["source"] == mode]
        num_variables = len(all_variables)

        variable_data = torch.zeros(1, num_variables, max(self.num_likert_classes, self.max_rank_size) + 1)
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

            if var['type'] == 'rating':
                variable_types[0, i] = 0
                item_ids[0, i, 0] = var['item'] - 1
                key = (var['attribute'], var['annotator'], var['item'])
                if key in rating_data:
                    rating_value = rating_data[key] - 1
                    rating_targets[0, i, rating_value] = 1.0
                    rating_mask[0, i] = True
                    if i in masked_rating_indices:
                        variable_data[0, i, 0] = 1 #the mask bit
                    #for all variables set the value
                    variable_data[0, i, 1 + rating_value] = 1.0
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
                        variable_data[0, i, 0] = 1 #set the mask bit

                    #for all variables set the value
                    for j, pos in enumerate(order):
                        if j < self.max_rank_size:
                            variable_data[0, i, j + 1] = pos

        return self._convert_tensors_to_ranking_data(variable_data, variable_types, attribute_ids, annotator_ids, item_ids)
