from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import json
import torch
import random

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
        for ranking in data.get('pairwise_rankings', []):
            items_to_check = ranking['items'][: self.max_rank_size]
            if all(item <= self.num_items for item in items_to_check):
                filtered_rankings.append(ranking)
        return {'ratings': filtered_ratings, 'pairwise_rankings': filtered_rankings}

    def create_variables(self, data: Dict[str, Any]) -> List[RankingData]:
        """Convert raw data directly to List[RankingData]."""
        variables = []
        
        # Process ratings
        for rating in data['ratings']:
            variables.append(RankingData(
                annotator_id=rating['annotator'] - 1,
                attribute_id=rating['attribute'] - 1,
                is_listwise=False,
                item_ids=[rating['item'] - 1],
                rating_value=rating['value'] - 1
            ))
        
        # Process rankings
        for ranking in data['pairwise_rankings']:
            variables.append(RankingData(
                annotator_id=ranking['annotator'] - 1,
                attribute_id=ranking['attribute'] - 1,
                is_listwise=True,
                item_ids=[i - 1 for i in ranking['items'][:self.max_rank_size]],
                ranking_order=ranking['order'][:self.max_rank_size]
            ))
        
        return variables

    def create_masked_batch(self, variables: List[RankingData], masking_rate: float, batch_size: int) -> List[RankingData]:
        """Create a batch with random masking applied for self-supervised learning."""
        # Sample variables for the batch
        batch_variables = random.sample(variables, min(batch_size, len(variables)))
        
        # Apply masking: M% of variables are masked, (1-M)% are observed
        # The model will use observed variables to predict masked variables
        masked_variables = []
        masked_indices = random.sample(list(range(len(variables))), int(len(variables) * masking_rate))
        for i, var in enumerate(batch_variables):
            if i in masked_indices:
                # Create masked version (remove supervision)
                masked_var = RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=var.is_listwise,
                    item_ids=var.item_ids,
                    rating_value=None,  # Mask the rating value
                    ranking_order=None  # Mask the ranking order
                )
                masked_variables.append(masked_var)
            else:
                # Keep original (observed) for conditioning
                masked_variables.append(var)
        
        return masked_variables

    def create_evaluation_batch(self, variables: List[RankingData], masking_rate: float) -> Tuple[List[RankingData], List[RankingData]]:
        """Create evaluation batch with Test_M (masked) and Test_O (observed) split."""
        # Randomly select variables to mask
        num_to_mask = int(len(variables) * masking_rate)
        masked_indices = set(random.sample(range(len(variables)), num_to_mask))
        
        test_m_vars = []  # Masked variables
        test_o_vars = []  # Observed variables
        
        for i, var in enumerate(variables):
            if i in masked_indices:
                # Create masked version
                masked_var = RankingData(
                    annotator_id=var.annotator_id,
                    attribute_id=var.attribute_id,
                    is_listwise=var.is_listwise,
                    item_ids=var.item_ids,
                    rating_value=None if not var.is_listwise else var.rating_value,
                    ranking_order=None if var.is_listwise else var.ranking_order
                )
                test_m_vars.append(masked_var)
                test_o_vars.append(var)  # Keep original for reference
            else:
                test_o_vars.append(var)
        
        return test_m_vars, test_o_vars
