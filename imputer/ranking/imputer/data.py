from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from stan.pipeline.bundle import GroundTruthBundle

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
    - instance: "train" or "test"
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

    def load_bundle_data(self, bundle_file: str) -> GroundTruthBundle:
        """Load complete data bundle from JSON file."""
        with open(bundle_file, 'r') as f:
            bundle_dict = json.load(f)
        
        return GroundTruthBundle.from_dict(bundle_dict)

    def create_variables_from_bundle(self, bundle: GroundTruthBundle, 
                                    partition: str, status: str) -> List[RankingData]:
        """Convert bundle data to RankingData variables.
        
        Args:
            bundle: Complete data bundle
            partition: "train", "test", or "all"
            status: "observed", "missing", or "all"
        
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
        if partition in ["train", "test"]:
            ratings = [r for r in ratings if r['instance'] == partition]
            pairwise = [p for p in pairwise if p['instance'] == partition]
        else:
            raise ValueError(f"Invalid partition: {partition}. Must be 'train', 'test'")
        
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
                    rating_value=rating['value'] - 1
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

    def validate_bundle(self, bundle: GroundTruthBundle) -> List[str]:
        """Validate bundle data integrity and return list of errors."""
        errors = []
        
        # Check dimensions
        K, D = bundle.embeddings.shape
        I, D_pref = bundle.mean_preferences.shape
        IJ, D_ann = bundle.annotator_preferences.shape
        
        if D != D_pref or D != D_ann:
            errors.append("Embedding dimensions inconsistent across embeddings, mean_preferences, annotator_preferences")
        
        if IJ != I * (IJ // I):
            errors.append("Annotator preferences dimension mismatch with I*J")
        
        # Check rating data integrity
        for rating in bundle.all_ratings:
            if rating['value'] < 1 or rating['value'] > self.num_likert_classes:
                errors.append(f"Invalid rating value {rating['value']} (must be 1-{self.num_likert_classes})")
            
            if rating['item'] < 1 or rating['item'] > K:
                errors.append(f"Invalid item index {rating['item']} (must be 1-{K})")
            
            if rating['attribute'] < 1 or rating['attribute'] > I:
                errors.append(f"Invalid attribute index {rating['attribute']} (must be 1-{I})")
            
            if rating['annotator'] < 1 or rating['annotator'] > self.num_annotators:
                errors.append(f"Invalid annotator index {rating['annotator']} (must be 1-{self.num_annotators})")
        
        # Check pairwise data integrity
        for ranking in bundle.all_pairwise:
            if len(ranking['items']) != 2:
                errors.append(f"Pairwise ranking must have exactly 2 items, got {len(ranking['items'])}")
            
            if len(ranking['order']) != 2:
                errors.append(f"Pairwise ranking must have exactly 2 order positions, got {len(ranking['order'])}")
            
            for item in ranking['items']:
                if item < 1 or item > K:
                    errors.append(f"Invalid item index {item} in pairwise ranking (must be 1-{K})")
            
            for pos in ranking['order']:
                if pos not in [1, 2]:
                    errors.append(f"Invalid order position {pos} in pairwise ranking (must be 1 or 2)")
            
            # Check annotator index for pairwise rankings
            if ranking['annotator'] < 1 or ranking['annotator'] > self.num_annotators:
                errors.append(f"Invalid annotator index {ranking['annotator']} in pairwise ranking (must be 1-{self.num_annotators})")
        
        # Check observed/missing partitions
        observed_item_ids = {(r['attribute'], r['annotator'], r['item']) for r in bundle.observed_ratings}
        missing_item_ids = {(r['attribute'], r['annotator'], r['item']) for r in bundle.missing_ratings}
        all_item_ids = {(r['attribute'], r['annotator'], r['item']) for r in bundle.all_ratings}
        
        if observed_item_ids | missing_item_ids != all_item_ids:
            errors.append("Observed and missing ratings do not partition all_ratings correctly")
        
        if observed_item_ids & missing_item_ids:
            errors.append("Found overlapping ratings between observed and missing sets")
        
        return errors

    def create_training_batch(self, variables: List[RankingData], batch_size: int) -> List[RankingData]:
        raise DeprecationWarning("This method is deprecated. Use create_variables_from_bundle instead.")

    def create_evaluation_batch(self, variables: List[RankingData]):
        """Create evaluation batch with Test_M (masked) and Test_O (observed) split."""
        raise DeprecationWarning("This method is deprecated. Use create_variables_from_bundle instead.")


# Usage Example:
# converter = DataConverter(num_items=8, num_likert_classes=5)
# bundle = converter.load_bundle_data("data_bundle.json")
# 
# # Validate data integrity
# errors = converter.validate_bundle(bundle)
# if errors:
#     print("Data validation errors:", errors)
# 
# # Create different variable sets - must specify exact partition and status
# train_obs = converter.create_variables_from_bundle(bundle, "train", "observed")
# test_obs = converter.create_variables_from_bundle(bundle, "test", "observed")
# train_missing = converter.create_variables_from_bundle(bundle, "train", "missing")
# test_missing = converter.create_variables_from_bundle(bundle, "test", "missing")
# 
# # Use helper properties for filtering
# observed_vars = [v for v in train_obs if v.is_observed]
# missing_vars = [v for v in train_missing if v.is_missing]
