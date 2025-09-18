# Ranking Imputation Experimental Framework - Implementation Plan

## Current Issues and Fixes Required

### Phase 1: Fix Masked Metrics and Sequential Training Clarity

**Critical Issues:**
1. **Finetuning Masked Metrics**: `GeneralMIT.finetune_on_instance()` final evaluation loses T_O/T_M split information, resulting in 0 masked metrics
2. **Heldout Evaluation Storage**: Pretraining results store raw `RankingData` objects instead of computed metrics in `heldout_evaluation`
3. **Sequential Training Logging**: Poor visibility into which instance is being processed and batch distribution
4. **Evaluation Callback Issues**: Heldout metrics may not be properly computed and stored

**Required Fixes:**

#### 1.1 Fix Finetuning Masked Metrics
**Problem**: In `GeneralMIT.finetune_on_instance()` (lines 309-315), final evaluation on full test instance doesn't preserve original masking information.

**Solution**:
- Preserve T_O/T_M variable lists throughout finetuning process
- Pass correct masking information to final evaluation
- Ensure `masked_metrics` reflects performance on T_M and `observed_metrics` on T_O

#### 1.2 Fix Heldout Evaluation Metrics
**Problem**: `heldout_evaluation` in results contains raw `RankingData` strings instead of evaluation metrics.

**Solution**:
- Store actual evaluation results from callbacks, not variable lists
- Investigate if heldout metrics are averaging across all training instances or computed individually
- **Preference**: Individual results per training instance for better analysis

#### 1.3 Improve Sequential Training Logging
**Current Behavior**: 100 total batches cycled across 8 training instances (~12-13 batches per instance)

**Required Improvements**:
- Add TQDM progress bars showing current instance and overall progress
- Clear logging of which instance is being processed
- Display batch distribution per instance
- Structured printing: "Training Instance 3/8: Batch 35/100 (Instance Batch 5/13)"

#### 1.4 Clarify Train/Heldout Split Usage
**Investigation Required**: Verify if `train_heldout_split` creates proper heldout sets and evaluation metrics are being computed correctly.

**Test Requirements**:
- Create test script that verifies masking logic with actual data samples
- Print and check samples from training/heldout sets for both pretraining and finetuning
- Verify masking rates are applied correctly (30% masked, 70% observed)

**Deliverables**:
- Fixed masked metrics for all strategies
- Proper heldout evaluation results storage
- Enhanced logging and progress tracking
- Test script: `tests/test_phase1_masked_metrics.py`

---

### Phase 2: Step-by-Step Evaluation Implementation

**Current State**:
- **Pretraining**: Only evaluates on heldout training data
- **Finetuning**: Only evaluates on T_O_heldout during finetuning
- **Domain Model**: Correctly evaluates at multiple sample counts ✅

**Required Implementation**:

#### 2.1 Add Test Set Evaluation During Training
**Pretraining**:
- Evaluate on test instances at regular intervals during pretraining
- Store intermediate test performance alongside heldout training performance
- Configurable evaluation frequency

**Finetuning**:
- Evaluate on full test instance (T_O + T_M) during finetuning steps
- Track progression of performance on masked variables

#### 2.2 Enhanced Progress Tracking
**Requirements**:
- TQDM progress bars for all training phases
- Structured printing after each evaluation step:
  ```
  Step 25/100 | Instance 3/8
  Train Loss: 0.456 | Heldout Loss: 0.523 | Test Loss: 0.612
  Train Acc: 0.78  | Heldout Acc: 0.74   | Test Acc: 0.69
  ```

#### 2.3 Comprehensive Results Storage
**Store in JSON**:
- Training metrics per step
- Heldout metrics per step
- Test metrics per step (when evaluated)
- All evaluation frequencies configurable

**Configuration Addition**:
```python
@dataclass
class EvaluationConfig:
    test_masking_rate: float = 0.5
    device: str = "cpu"
    eval_frequency_steps: int = 10      # NEW: Evaluate every N steps
    eval_on_test_during_training: bool = True  # NEW: Enable test evaluation during training
```

**Deliverables**:
- Test set evaluation during all training phases
- Enhanced progress tracking with TQDM and structured printing
- Comprehensive results storage with step-by-step metrics
- Test script: `tests/test_phase2_step_evaluation.py`

---

### Phase 3: Instance Parameter Diversification

**Current Problem**: All instances use identical parameters (K=30, I=10, J=5, etc.) differing only by random seed.

**Solution**: Allow per-instance parameter specification.

#### 3.1 Configuration Structure Update

**Modify `experiment_config.py`**:
```python
@dataclass
class DataConfig:
    num_instances: int = 5
    train_test_split: float = 0.7

    # NEW: Per-instance parameters (list format)
    K_list: List[int] = field(default_factory=lambda: [30, 25, 35, 20, 40])
    I_list: List[int] = field(default_factory=lambda: [10, 8, 12, 6, 15])
    J_list: List[int] = field(default_factory=lambda: [5, 4, 6, 3, 7])
    C: int = 5  # Keep constant

    # Validation: Ensure list lengths match num_instances
    def __post_init__(self):
        lists = [self.K_list, self.I_list, self.J_list]
        for lst in lists:
            if len(lst) != self.num_instances:
                raise ValueError(f"Parameter list length {len(lst)} doesn't match num_instances {self.num_instances}")
```

#### 3.2 Data Generation Update

**Modify `new_experiment_runner.py:generate_data()`**:
- Create different `ICLRDatasetConfig` per instance using list parameters
- Ensure proper indexing and parameter assignment
- Maintain backward compatibility with single-value parameters

#### 3.3 Test Configuration Update

**Update `test_config.json`**:
```json
{
  "data_config": {
    "num_instances": 3,
    "K_list": [10, 15, 12],
    "I_list": [3, 4, 3],
    "J_list": [3, 4, 3],
    "train_test_split": 0.6
  }
}
```

**Deliverables**:
- Per-instance parameter specification system
- Updated configuration validation
- Modified data generation to use instance-specific parameters
- Test script: `tests/test_phase3_instance_diversity.py`

---

## Testing Requirements

Each phase must include a comprehensive test script in `./tests/` directory:

### Test Script Requirements

**`tests/test_phase1_masked_metrics.py`**:
- Load actual generated data
- Print samples from training/heldout sets during pretraining
- Print samples from T_O_train/T_O_heldout/T_M during finetuning
- Verify masking rates (30% masked, 70% observed)
- Check that masked_metrics and observed_metrics are computed correctly
- Validate heldout evaluation results format

**`tests/test_phase2_step_evaluation.py`**:
- Run mini-experiment with step-by-step evaluation enabled
- Verify test set evaluation occurs during training
- Check metrics storage format and completeness
- Validate TQDM and structured printing output

**`tests/test_phase3_instance_diversity.py`**:
- Create test config with diverse instance parameters
- Verify different instances have different K, I, J values
- Check data generation creates appropriately sized instances
- Validate model compatibility with different instance sizes

## Implementation Priority

1. **✅ Phase 1 COMPLETED** (Critical): Fix masked metrics and improve training clarity
2. **Phase 2** (Important): Add comprehensive step-by-step evaluation
3. **Phase 3** (Enhancement): Enable instance parameter diversity

## Phase 1 - COMPLETED ✅

**✅ All Success Criteria Met**:
- ✅ Finetuning shows non-zero masked_metrics (fixed evaluation engine)
- ✅ Heldout evaluation stores actual metrics, not raw data (proper callback collection)
- ✅ Sequential training has clear progress indicators (TQDM + structured logging)
- ✅ All masking logic verified with test data (comprehensive test script)

**Implementation Summary**:
- **Fixed evaluation engine** (`imputer/eval.py`) to respect pre-existing `is_masked` flags
- **Enhanced multi-instance trainer** (`imputer/multi_instance_trainer.py`) with TQDM progress bars and callback collection
- **Fixed GeneralMIT evaluation** to preserve T_O/T_M masking information
- **Updated experiment config** with `eval_frequency`, `test_masking_rate`, and missing `D` parameter
- **Updated results storage** to capture callback metrics instead of raw variables
- **Created comprehensive test script** (`tests/test_phase1_masked_metrics.py`) that validates all fixes
- **Maintained backward compatibility** with existing `configs/test_config.json`

**Phase 2 Complete When**:
- Test set evaluation occurs during all training phases
- Comprehensive metrics stored at each evaluation step
- Enhanced logging with TQDM and structured output
- Configurable evaluation frequency working

**Phase 3 Complete When**:
- Instances can have different K, I, J parameters
- Configuration validates parameter list lengths
- Data generation works with diverse parameters
- All strategies handle variable instance sizes

---

*This plan addresses the critical issues identified in the current experimental framework while maintaining backward compatibility and ensuring thorough testing of each improvement.*