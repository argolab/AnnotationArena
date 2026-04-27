from dataclasses import dataclass
from typing import List, Optional, Any
import json
import types


@dataclass
class RankingData:
    """Enhanced structured representation with clear status tracking.

    All indices are 0-indexed for model consumption.
    Status encoding:
    - 0: missing (not observed, target for prediction)
    - 1: masked (observed but hidden during training)
    - 2: observed (available for training)

    Fields:
    - annotator_id: annotator index
    - attribute_id: attribute index
    - is_listwise: True for listwise ranking, False for rating
    - item_ids: for rating, a list with one item id; for ranking, the ranked item ids
    - status: status code (0=missing, 1=masked, 2=observed)
    - instance: "train", "val", or "test"
    - rating_value: scalar score if rating observed
    - ranking_order: list of positions in [1..R] aligned with item_ids when ranking observed
    """
    annotator_id: int
    attribute_id: int
    is_listwise: bool
    item_ids: List[int]
    status: int
    instance: str
    rating_value: Optional[float] = None
    ranking_order: Optional[List[int]] = None
    # Optional soft target distribution over rating categories, used by some
    # non-scalar data sources.
    rating_dist: Optional[List[float]] = None

    @property
    def is_missing(self) -> bool:
        """True if this variable is missing (status=0)."""
        return self.status == 0

    @property
    def is_masked(self) -> bool:
        """True if this variable is masked (status=1)."""
        return self.status == 1

    @property
    def is_observed(self) -> bool:
        """True if this variable is observed (status=2)."""
        return self.status == 2


class DataConverter:
    def __init__(self, num_attributes=10, num_annotators=5, num_items=10, num_likert_classes=5, max_rank_size=3):
        self.num_attributes = num_attributes
        self.num_annotators = num_annotators
        self.num_items = num_items
        self.num_likert_classes = num_likert_classes
        self.max_rank_size = max_rank_size

    def load_bundle_data(self, bundle_file: str) -> Any:
        """Load complete data bundle from JSON file."""
        with open(bundle_file, 'r') as f:
            bundle_dict = json.load(f)
        return types.SimpleNamespace(**bundle_dict)

    def create_variables_from_bundle(self, bundle: Any,
                                    partition: str, status: str) -> List[RankingData]:
        """Convert bundle data to RankingData variables.

        Args:
            bundle: Complete data bundle
            partition: "train", "val", "test"
            status: "observed", "missing"

        Returns:
            List of RankingData objects
        """
        variables = []

        # Select rating data based on status
        if status == "observed":
            ratings = bundle.observed_ratings
            pairwise = bundle.observed_pairwise
            status_code = 2  # observed
        elif status == "missing":
            ratings = bundle.missing_ratings
            pairwise = bundle.missing_pairwise
            status_code = 0  # missing
        else:
            raise ValueError(f"Invalid status: {status}. Must be 'observed' or 'missing'")

        # Apply partition filtering
        if partition in ["train", "val", "test"]:
            ratings = [r for r in ratings if r['instance'] == partition]
            pairwise = [p for p in pairwise if p['instance'] == partition]
        else:
            raise ValueError(f"Invalid partition: {partition}. Must be 'train', 'val', 'test'")

        # Process ratings
        for rating in ratings:
            if rating['item'] <= self.num_items:
                variables.append(RankingData(
                    annotator_id=rating['annotator'] - 1,
                    attribute_id=rating['attribute'] - 1,
                    is_listwise=False,
                    item_ids=[rating['item'] - 1],
                    status=status_code,
                    instance=rating['instance'],
                    rating_value=float(rating['value']),
                    rating_dist=rating.get('rating_dist'),  # None for synthetic; list for real
                ))

        # Process pairwise rankings
        for ranking in pairwise:
            items_to_check = ranking['items'][:self.max_rank_size]
            if all(item <= self.num_items for item in items_to_check):
                variables.append(RankingData(
                    annotator_id=ranking['annotator'] - 1,
                    attribute_id=ranking['attribute'] - 1,
                    is_listwise=True,
                    item_ids=[i - 1 for i in items_to_check],
                    status=status_code,
                    instance=ranking['instance'],
                    ranking_order=ranking['order'][:self.max_rank_size]
                ))

        return variables
