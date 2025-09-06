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

        all_ratings = train_data['ratings'] + test_data['ratings']
        rating_keys = set((r['attribute'], r['annotator'], r['item']) for r in all_ratings)
        for attr, annot, item in rating_keys:
            rating_variables.append({'type': 'rating', 'attribute': attr, 'annotator': annot, 'item': item})

        all_rankings = train_data['rankings'] + test_data['rankings']
        ranking_keys = set()
        for ranking in all_rankings:
            items = ranking['items'][: self.max_rank_size]
            ranking_keys.add((ranking['attribute'], ranking['annotator'], tuple(items)))
        for attr, annot, items_tuple in ranking_keys:
            ranking_variables.append({'type': 'ranking', 'attribute': attr, 'annotator': annot, 'items': list(items_tuple)})

        return rating_variables, ranking_variables

    def process_training_data(self, data: Dict[str, Any]) -> Tuple[Dict[Tuple[int, int, int], int], Dict[Tuple[int, int, Tuple[int, ...]], Dict[str, List[int]]]]:
        rating_data: Dict[Tuple[int, int, int], int] = {}
        ranking_data: Dict[Tuple[int, int, Tuple[int, ...]], Dict[str, List[int]]] = {}

        for rating in data['ratings']:
            key = (rating['attribute'], rating['annotator'], rating['item'])
            rating_data[key] = rating['value']

        for ranking in data['rankings']:
            items = ranking['items'][: self.max_rank_size]
            order = ranking['order'][: self.max_rank_size]
            key = (ranking['attribute'], ranking['annotator'], tuple(items))
            ranking_data[key] = {'items': items, 'order': order}
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
        ranking_data: Dict[Tuple[int, int, Tuple[int, ...]], Dict[str, List[int]]],
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
            key = (var['attribute'], var['annotator'], tuple(items))
            order = None
            if key in ranking_data:
                order = ranking_data[key]['order'][: self.max_rank_size]
            variables.append(RankingData(
                annotator_id=var['annotator'] - 1,
                attribute_id=var['attribute'] - 1,
                is_listwise=True,
                item_ids=[i - 1 for i in items],
                ranking_order=order,
            ))
        return variables

