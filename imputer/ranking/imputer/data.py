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
    - rating_value: class index [0..C-1] if rating observed
    - ranking_order: list of positions in [1..R] aligned with item_ids when ranking observed
    """
    annotator_id: int
    attribute_id: int
    is_listwise: bool
    item_ids: List[int]
    status: int
    instance: str
    rating_value: Optional[int] = None
    ranking_order: Optional[List[int]] = None
    # Soft target distribution over rating categories [length C, sums to 1].
    # None  → use one-hot of rating_value (synthetic / hard-label data).
    # List  → use as soft CE target (real data: LLM gives full distribution,
    #          human gives one-hot stored explicitly for uniform handling).
    rating_dist: Optional[List[float]] = None
    # Oracle diagnostics payloads (optional).
    oracle_eff_pref: Optional[List[float]] = None
    oracle_item_embedding: Optional[List[float]] = None

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

        embeddings = getattr(bundle, "embeddings", None)
        extra_ground_truth = getattr(bundle, "extra_ground_truth", None) or {}
        if not isinstance(extra_ground_truth, dict):
            extra_ground_truth = {}
        # generate_data.py flattens extra_ground_truth keys to top-level JSON; support both.
        eff_pref = getattr(bundle, "eff_pref", None) or extra_ground_truth.get("eff_pref")

        # Process ratings
        for rating in ratings:
            if rating['item'] <= self.num_items:
                item_id = rating['item'] - 1
                attr_id = rating['attribute'] - 1
                annot_id = rating['annotator'] - 1
                oracle_eff_pref = None
                oracle_item_embedding = None

                if eff_pref is not None:
                    ij_idx = attr_id * self.num_annotators + annot_id
                    oracle_eff_pref = eff_pref[ij_idx]
                if embeddings is not None:
                    oracle_item_embedding = embeddings[item_id]

                variables.append(RankingData(
                    annotator_id=annot_id,
                    attribute_id=attr_id,
                    is_listwise=False,
                    item_ids=[item_id],
                    status=status_code,
                    instance=rating['instance'],
                    rating_value=rating['value'] - 1,
                    rating_dist=rating.get('rating_dist'),  # None for synthetic; list for real
                    oracle_eff_pref=oracle_eff_pref,
                    oracle_item_embedding=oracle_item_embedding,
                ))

        # Process pairwise rankings
        for ranking in pairwise:
            items_to_check = ranking['items'][:self.max_rank_size]
            if all(item <= self.num_items for item in items_to_check):
                oracle_item_embedding = None
                if embeddings is not None and ranking['items']:
                    # Ranking tokens use first item as canonical item identity (same as pointer logic).
                    oracle_item_embedding = embeddings[ranking['items'][0] - 1]
                variables.append(RankingData(
                    annotator_id=ranking['annotator'] - 1,
                    attribute_id=ranking['attribute'] - 1,
                    is_listwise=True,
                    item_ids=[i - 1 for i in items_to_check],
                    status=status_code,
                    instance=ranking['instance'],
                    ranking_order=ranking['order'][:self.max_rank_size],
                    oracle_item_embedding=oracle_item_embedding,
                ))

        return variables
