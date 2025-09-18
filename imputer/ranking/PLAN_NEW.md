# Ranking Imputation Experimental Framework - Implementation Plan

## Experimental Design Overview

**Instance-Level Organization**:
- N total instances split into train/test (e.g., 7 train, 3 test instances)
- Each test instance has M% variables masked (Test_M) and (1-M)% observed (Test_O)

**Four Evaluation Strategies**:
1. **Pretrained**: Use pretrained model directly on Test_M given Test_O
2. **Pretrained + Finetuning**: Pretrain → finetune on Test_O → evaluate on Test_M
3. **Fresh**: No pretraining → train only on Test_O → evaluate on Test_M
4. **Domain EM**: Fit EM to Test_O → evaluate on Test_M

**Training Strategies**:
- **SequentialMIT**: Train on instances sequentially (I1 → eval → I2 → eval...)
- **MixedMIT**: Create big combined instance, train on IID batches from all
- **GeneralMIT**: Single instance finetuning (for test instance finetuning)

---

## Current Implementation Status

### ✅ COMPLETED COMPONENTS

**Data Infrastructure**:
- `imputer/data.py` - Clean, unified implementation returning `List[RankingData]`
- `RankingData` class - Structured representation for variables
- No masking logic in data converter (correctly handled by trainer)
- Single data system (legacy tensor format removed)

**Random Embeddings**:
- `FullyRandomizedEmbeddingProvider` - Generates fresh random embeddings each forward pass
- Extends correct base class, dynamic sizing, non-trainable
- Config integration with "fully_random" option

**Basic Training Infrastructure**:
- `ImputerTrainer` - Handles masking, callback system, structured losses
- Masking logic in trainer (not data converter)
- Callback system for evaluation integration

**Basic Evaluation**:
- `EvaluationEngine` - M% masking, Test_M/Test_O splitting, comprehensive metrics
- `EvaluationResults` - Structured results with rating/ranking metrics
- Integration with loss strategies

**Multi-Instance Training (Partial)**:
- `MultiInstanceTrainerBase` - Generator-based architecture
- `SequentialMIT` and `MixedMIT` - Working implementations
- Integration with configuration system

**Configuration System**:
- Phase 5 parameters added (masking rates, evaluation types, etc.)
- Validation and testing complete
- Multi-instance experiment support

---

## REMAINING IMPLEMENTATION PHASES

## Phase 1: Data System Cleanup and Integration ✅ COMPLETED

**Objectives**: Clean up dual data systems, ensure consistent data flow

**Completed Tasks**:
1. **✅ Removed Legacy Data System**:
   - Deleted `imputer/data.py` completely
   - Updated all imports from `data_v2` to `data`
   - Updated `tests/test_data_converter.py` (5 import locations)
   - Updated `ranking_imputer.py:267` import

2. **✅ Renamed and Standardized**:
   - Renamed `imputer/data_v2.py` → `imputer/data.py`
   - All imports now use unified `imputer.data`
   - Left `CombineRandomTrainedEmbeddingProvider` (configurable, not used by default)

3. **✅ Cleaned Legacy Code**:
   - Removed commented tensor code from `trainer.py:186-200`
   - Removed `_convert_legacy_tensors_to_ranking_data()` method
   - Updated `forward_hidden()` to only accept `List[RankingData]`

4. **✅ Verified Integration**:
   - Data converter creates proper `List[RankingData]` format
   - Model forward pass works: Rating logits `[1, N, 5]`, Ranking logits `[1, N, 2]`
   - All components use unified data system

**Deliverables**:
- ✅ Single, clean data system
- ✅ All imports updated and working
- ✅ Integration tests passing
- ✅ No legacy tensor format support

---

## Phase 2: Complete Evaluation Engine ✅ COMPLETED

**Objectives**: Minor fixes to evaluation engine for complete metrics implementation

**Completed Tasks**:
1. **✅ Minor Implementation Fixes**:
   - Fixed type annotations for `List[RankingData]` consistency
   - Implemented proper pairwise ranking accuracy computation
   - Added device compatibility checks and model.to(device)
   - Enhanced error handling for edge cases (empty variables, invalid masking rates)
   - Added length validation for evaluation mask and variables

2. **✅ Integration Verification**:
   - Fixed EvaluationCallback parameter mismatch in trainer.py
   - Verified callback integration with comprehensive test
   - Tested evaluation format consistency across all metrics
   - Validated different masking rates (0.0, 0.5, 1.0)

3. **✅ Comprehensive Testing**:
   - Created `tests/test_eval_engine.py` with 9 comprehensive test functions
   - Tests masking logic, variable splitting, RMSE computation, ranking accuracy
   - Tests model integration, trainer callbacks, and edge cases
   - All tests pass successfully

**Key Achievements**:
- **NO TODO COMMENTS OR SIMPLIFIED VERSIONS** - All implementations complete
- Clean separation: eval only evaluates, doesn't handle data conversion or training
- Proper pairwise ranking accuracy using relative order comparison
- Robust error handling and device compatibility

**Deliverables**:
- ✅ Fully verified `EvaluationEngine` with comprehensive test coverage
- ✅ Complete trainer callback integration working correctly
- ✅ All metrics (loss, accuracy, RMSE) working correctly for ratings and rankings
- ✅ Proper Test_M/Test_O masking and evaluation

---

## Phase 3: Domain Model Integration ✅ COMPLETED

**Objectives**: Connect existing domain model implementation to experimental framework

**Completed Tasks**:
1. **✅ Simple Entrypoint Implementation**:
   - Added `evaluate_test_instance()` method to `DomainModelTrainer` class
   - Takes test instance data + MCMC parameters (chains, iter_sampling, iter_warmup)
   - Supports variable sample counts for training and inference experiments
   - Uses existing masking logic (`mask_test_data_for_evaluation`)

2. **✅ Perfect EvaluationResults Format Matching**:
   - Returns `EvaluationResults` object (same as neural models)
   - All required fields: total_loss, rating_loss, ranking_loss, accuracies, RMSE
   - Proper masked_metrics and observed_metrics breakdown
   - Consistent field types and structure

3. **✅ Integration with Existing Functionality**:
   - Reuses 100% of existing domain model methods
   - `mask_test_data_for_evaluation()` → splits test instance into observed/missing
   - `prepare_stan_data()` → converts to Stan format
   - `model.sample()` → MCMC training on observed data
   - `compute_training_log_loss()` → metrics on training data
   - `compute_log_loss_on_missing()` + `evaluate_imputation_accuracy()` → metrics on missing data

4. **✅ Comprehensive Testing**:
   - Created `tests/test_domain_model_entrypoint.py` with full test suite
   - Tests EvaluationResults format compatibility
   - Tests variable MCMC sample counts (50, 100, 200 samples)
   - Tests variable masking rates (30%, 50%, 70%)
   - All tests pass successfully

**Key Achievements**:
- **REUSED EXISTING IMPLEMENTATION** - Zero new metric computation logic
- **Perfect Format Matching** - Returns identical EvaluationResults structure as neural models
- **Variable Sample Count Support** - Easy to vary chains/iter_sampling for experiments
- **Clean Integration** - Self-contained ~120 lines in existing domain_model_trainer.py
- **Strategy 4 Ready** - Domain EM evaluation strategy fully functional

**Deliverables**:
- ✅ `evaluate_test_instance()` entrypoint method with perfect EvaluationResults format
- ✅ Support for variable MCMC sample counts (configurable training intensity)
- ✅ Complete test suite validating integration and format compatibility
- ✅ Strategy 4 (Domain EM) ready for experimental framework integration

---

## Phase 4: Complete Multi-Instance Training (MIT) 🚀

**Objectives**: Rewrite MIT to respect separation of concerns and complete all implementations

**Current Issues in multi_instance_trainer.py**:
- ❌ Line 41: TODO comment for saving results
- ❌ Line 125: NotImplementedError for GeneralMIT.finetune_on_instance
- ❌ Lines 62-78: Uses old tensor-based batch creation instead of List[RankingData]
- ❌ Creates own batches instead of using trainer's masking capabilities
- ❌ Calls eval_engine directly instead of using trainer callbacks
- ❌ Mixes data conversion, training, and evaluation logic

**Tasks**:
1. **Rewrite MIT for Clean Separation**:
   - Use data.py for all data conversion (List[RankingData] format)
   - Use trainer.py for masking and training with callbacks
   - Use eval_engine only through trainer callbacks
   - Remove direct batch creation logic

2. **Complete GeneralMIT**:
   - Implement `finetune_on_instance()` method properly
   - Support Test_O → Test_O_Masked/Test_O_Observed splitting via data.py
   - Train on Test_O_Observed using trainer callbacks for evaluation

3. **Fix SequentialMIT and MixedMIT**:
   - Remove direct batch creation, use trainer's masking
   - Use trainer callbacks for evaluation, not direct eval_engine calls
   - Remove TODO comments and implement result saving

4. **MIT Integration**:
   - Support all 4 evaluation strategies cleanly
   - Results collection through trainer callback system
   - Configuration-driven training parameters

**Key Requirements**:
- **RESPECT SEPARATION OF CONCERNS**:
  - Data gives data via List[RankingData]
  - Trainer handles masking and training with eval callbacks
  - Eval just evaluates when called by callbacks
- **NO TODO COMMENTS OR NOTIMPLEMENTEDERROR**
- Complete implementation of all methods

**Deliverables**:
- Rewritten `MultiInstanceTrainerBase` with clean separation
- Complete `GeneralMIT.finetune_on_instance()`
- Fixed `SequentialMIT` and `MixedMIT` using proper architecture
- Full integration with evaluation strategies through callbacks

---

## Phase 5: Experiment Runner and Configuration 🔧

**Objectives**: Orchestrate instance-level experiments with all evaluation strategies

**Tasks**:
1. **Main Experiment Runner**:
   - Instance-level train/test splitting
   - Execution of all 4 evaluation strategies
   - Results collection and comparison
   - Support for multiple experimental configurations

2. **Configuration Updates**:
   - Instance-level experiment configuration
   - Evaluation strategy selection
   - Training and evaluation parameters
   - Output and storage configuration

3. **Experiment Orchestration**:
   - Pretraining phase (SequentialMIT or MixedMIT)
   - Test instance evaluation (all 4 strategies)
   - Error bar computation across test instances
   - Experiment state management

**Deliverables**:
- Complete experiment runner
- Updated configuration system
- Multi-strategy experiment support

---

## Phase 6: Results Storage and Visualization 📈

**Objectives**: Store all experimental results and create visualization pipeline

**Tasks**:
1. **Results Storage System**:
   - JSON format for all results and experiment details
   - Structured storage of:
     - Model configurations and hyperparameters
     - Training histories and metrics
     - Evaluation results for all strategies
     - Instance-level performance breakdowns
     - Error bars and statistical summaries

2. **Results Schema**:
   - Experiment metadata (config, timestamps, environment)
   - Training results (loss curves, convergence)
   - Evaluation results (all 4 strategies, all metrics)
   - Instance-level breakdowns
   - Comparative analysis data

3. **Visualization Pipeline**:
   - Performance comparison across evaluation strategies
   - Instance-level error bars and distributions
   - Training convergence curves
   - Metric breakdowns (rating vs ranking performance)
   - Statistical significance testing

**Deliverables**:
- Complete results storage system
- Visualization generation pipeline
- Comparative analysis tools

---

## Implementation Priority

1. **✅ Phase 1** (Data Cleanup) - Critical foundation **COMPLETED**
2. **✅ Phase 2** (Evaluation Engine) - Minor fixes to complete metrics **COMPLETED**
3. **✅ Phase 3** (Domain Model Integration) - Connect existing domain trainer **COMPLETED**
4. **⏳ Phase 4** (Complete MIT) - Major rewrite for separation of concerns **NEXT**
5. **Phase 5** (Experiment Runner) - End-to-end experiments
6. **Phase 6** (Visualization) - Results analysis

**Rationale for Reordering**:
- **Phase 2**: Evaluation engine is mostly complete, just needs minor fixes
- **Phase 3**: Domain model trainer is fully implemented, just needs integration
- **Phase 4**: Multi-instance trainer has major separation violations and needs complete rewrite
- This ordering respects dependencies: clean eval → domain integration → MIT rewrite

## Key Implementation Standards

- **NO TODO COMMENTS OR SIMPLIFIED IMPLEMENTATIONS**
- Complete functionality in every phase
- Comprehensive testing for each component
- Clean separation of concerns
- Configuration-driven experiments
- Structured results and comprehensive metrics