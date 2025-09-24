# Data Converter Renovation Plan

## Overview

This document outlines the renovation plan for `imputer/ranking/imputer/data.py` to support the new STAN data bundle format while maintaining clean, readable code and proper separation of concerns.

## Key Design Decisions

1. **Status Encoding**: `0=missing`, `1=masked`, `2=observed` with helper properties for readability
2. **Use GroundTruthBundle**: Leverage existing `GroundTruthBundle` class instead of creating new `BundleData`
3. **Maintain Original API**: DataConverter still returns `List[RankingData]` 
4. **No Masking Logic**: DataConverter only handles loading, masking is done elsewhere
5. **Focused Methods**: Specific methods for different data partitions and statuses
6. **Validation**: Add bundle validation for code safety

## Phase 1: Enhanced RankingData Class

### 1.1 Updated RankingData Structure

```python
@dataclass
class RankingData:
    """Enhanced structured representation with clear status tracking.
    
    Status encoding:
    - 0: missing (not observed, target for prediction)
    - 1: masked (observed but hidden during training)
    - 2: observed (available for training)
    """
    annotator_id: int
    attribute_id: int
    is_listwise: bool
    item_ids: List[int]
    status: int  # 0=missing, 1=masked, 2=observed
    instance: str  # "train" or "test"
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
```

## Phase 2: DataConverter Renovation

### 2.1 Updated Constructor

```python
class DataConverter:
    def __init__(self, 
                 num_attributes: int = 10,
                 num_annotators: int = 5, 
                 num_items: int = 10,
                 num_likert_classes: int = 5,
                 max_rank_size: int = 3):
        self.num_attributes = num_attributes
        self.num_annotators = num_annotators
        self.num_items = num_items
        self.num_likert_classes = num_likert_classes
        self.max_rank_size = max_rank_size
```

### 2.2 Bundle Loading Method

```python
def load_bundle_data(self, bundle_file: str) -> GroundTruthBundle:
    """Load complete data bundle from JSON file."""
    with open(bundle_file, 'r') as f:
        bundle_dict = json.load(f)
    
    return GroundTruthBundle.from_dict(bundle_dict)
```

### 2.3 Core Variable Creation Method

```python
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
    elif status == "all":
        ratings = bundle.all_ratings
        pairwise = bundle.all_pairwise
        status_code = 2  # default to observed for "all"
    else:
        raise ValueError(f"Invalid status: {status}. Must be 'observed', 'missing', or 'all'")
    
    # Apply partition filtering
    if partition in ["train", "test"]:
        ratings = [r for r in ratings if r['instance'] == partition]
        pairwise = [p for p in pairwise if p['instance'] == partition]
    elif partition != "all":
        raise ValueError(f"Invalid partition: {partition}. Must be 'train', 'test', or 'all'")
    
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
```

### 2.4 Convenience Methods

```python
def create_train_observed_variables(self, bundle: GroundTruthBundle) -> List[RankingData]:
    """Create variables for observed training data."""
    return self.create_variables_from_bundle(bundle, "train", "observed")

def create_test_observed_variables(self, bundle: GroundTruthBundle) -> List[RankingData]:
    """Create variables for observed test data."""
    return self.create_variables_from_bundle(bundle, "test", "observed")

def create_train_missing_variables(self, bundle: GroundTruthBundle) -> List[RankingData]:
    """Create variables for missing training data (prediction targets)."""
    return self.create_variables_from_bundle(bundle, "train", "missing")

def create_test_missing_variables(self, bundle: GroundTruthBundle) -> List[RankingData]:
    """Create variables for missing test data (prediction targets)."""
    return self.create_variables_from_bundle(bundle, "test", "missing")

def create_all_observed_variables(self, bundle: GroundTruthBundle) -> List[RankingData]:
    """Create variables for all observed data (train + test)."""
    return self.create_variables_from_bundle(bundle, "all", "observed")

def create_all_missing_variables(self, bundle: GroundTruthBundle) -> List[RankingData]:
    """Create variables for all missing data (train + test)."""
    return self.create_variables_from_bundle(bundle, "all", "missing")
```

### 2.5 Legacy API Maintenance

```python
def load_training_data(self, json_file: str) -> Dict[str, Any]:
    """Legacy method - now loads bundle and returns observed data."""
    bundle = self.load_bundle_data(json_file)
    return {
        'ratings': bundle.observed_ratings,
        'pairwise_rankings': bundle.observed_pairwise
    }

def create_variables(self, data: Dict[str, Any]) -> List[RankingData]:
    """Legacy method - converts dict data to RankingData."""
    variables = []
    
    # Process ratings
    for rating in data['ratings']:
        variables.append(RankingData(
            annotator_id=rating['annotator'] - 1,
            attribute_id=rating['attribute'] - 1,
            is_listwise=False,
            item_ids=[rating['item'] - 1],
            status=2,  # Assume observed for legacy
            instance="train",  # Default for legacy
            rating_value=rating['value'] - 1
        ))
    
    # Process rankings
    for ranking in data.get('pairwise_rankings', []):
        variables.append(RankingData(
            annotator_id=ranking['annotator'] - 1,
            attribute_id=ranking['attribute'] - 1,
            is_listwise=True,
            item_ids=[i - 1 for i in ranking['items'][:self.max_rank_size]],
            status=2,  # Assume observed for legacy
            instance="train",  # Default for legacy
            ranking_order=ranking['order'][:self.max_rank_size]
        ))
    
    return variables
```

## Phase 3: Data Validation

### 3.1 Bundle Validation

```python
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
        
        if rating['annotator'] < 1 or rating['annotator'] > (IJ // I):
            errors.append(f"Invalid annotator index {rating['annotator']} (must be 1-{IJ // I})")
    
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
    
    # Check observed/missing partitions
    observed_item_ids = {(r['attribute'], r['annotator'], r['item']) for r in bundle.observed_ratings}
    missing_item_ids = {(r['attribute'], r['annotator'], r['item']) for r in bundle.missing_ratings}
    all_item_ids = {(r['attribute'], r['annotator'], r['item']) for r in bundle.all_ratings}
    
    if observed_item_ids | missing_item_ids != all_item_ids:
        errors.append("Observed and missing ratings do not partition all_ratings correctly")
    
    if observed_item_ids & missing_item_ids:
        errors.append("Found overlapping ratings between observed and missing sets")
    
    return errors
```

### 3.2 Usage Example

```python
# Load and validate bundle
converter = DataConverter(num_items=8, num_likert_classes=5)
bundle = converter.load_bundle_data("data_bundle.json")

# Validate data integrity
errors = converter.validate_bundle(bundle)
if errors:
    print("Data validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Bundle validation passed!")

# Create different variable sets
train_obs = converter.create_train_observed_variables(bundle)
test_obs = converter.create_test_observed_variables(bundle)
train_missing = converter.create_train_missing_variables(bundle)
test_missing = converter.create_test_missing_variables(bundle)

# Use helper properties for filtering
observed_vars = [v for v in train_obs if v.is_observed]
missing_vars = [v for v in train_missing if v.is_missing]
```

## Phase 4: Integration Benefits

### 4.1 Clean Separation of Concerns
- **DataConverter**: Only handles loading and basic conversion
- **Trainer Logic**: Handles masking and batch creation
- **STAN Pipeline**: Uses GroundTruthBundle directly

### 4.2 Improved Readability
- Clear status encoding with helper properties
- Explicit method names for different data partitions
- Validation ensures data integrity

### 4.3 Maintained Compatibility
- Original API preserved for existing code
- Legacy methods still work
- Gradual migration path available

## Implementation Status: ✅ COMPLETED

All phases have been successfully implemented:

1. **✅ Phase 1**: Updated RankingData class with new status encoding and helper properties
2. **✅ Phase 2**: Implemented core bundle loading and variable creation methods
3. **✅ Phase 3**: Added convenience methods for different partitions/statuses
4. **✅ Phase 4**: Implemented validation logic
5. **✅ Phase 5**: Added comprehensive tests

## Key Features Implemented

### Enhanced RankingData Class
- Status encoding: `0=missing`, `1=masked`, `2=observed`
- Helper properties: `is_missing`, `is_masked`, `is_observed`
- Instance tracking: `"train"` or `"test"`
- Clean, readable API

### DataConverter Methods
- `load_bundle_data()`: Load GroundTruthBundle from JSON
- `create_variables_from_bundle()`: Core method with partition/status filtering (only "train"/"test" and "observed"/"missing")
- `validate_bundle()`: Comprehensive data integrity validation

### Error Handling
- Clear error messages for invalid parameters
- No try/catch blocks - errors bubble up for debugging
- Validation catches data integrity issues

### Testing
- Unit tests for all methods
- Integration tests with realistic data
- Validation tests with corrupted data
- Error handling tests

## Usage Example

```python
# Load and validate bundle
converter = DataConverter(num_items=8, num_likert_classes=5)
bundle = converter.load_bundle_data("data_bundle.json")

# Validate data integrity
errors = converter.validate_bundle(bundle)
if errors:
    print("Data validation errors:", errors)

# Create different variable sets - must specify exact partition and status
train_obs = converter.create_variables_from_bundle(bundle, "train", "observed")
test_obs = converter.create_variables_from_bundle(bundle, "test", "observed")
train_missing = converter.create_variables_from_bundle(bundle, "train", "missing")
test_missing = converter.create_variables_from_bundle(bundle, "test", "missing")

# Use helper properties for filtering
observed_vars = [v for v in train_obs if v.is_observed]
missing_vars = [v for v in train_missing if v.is_missing]
```

## Files Modified

- `data.py`: Complete renovation with new RankingData and DataConverter
- `test_data.py`: Comprehensive unit tests
- `test_integration.py`: End-to-end integration tests
- `DATA_CONVERTER_RENOVATION_PLAN.md`: This documentation

The renovation is complete and ready for use with the STAN pipeline!
