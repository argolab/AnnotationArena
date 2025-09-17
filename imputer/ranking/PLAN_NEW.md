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

## Phase 2: Complete Evaluation Engine 🎯

**Objectives**: Implement comprehensive evaluation for all metrics on masked test sets

**Tasks**:
1. **Core Evaluation Metrics**:
   - Log loss computation on masked variables
   - Rating accuracy and RMSE
   - Ranking accuracy for pairwise comparisons
   - Metrics breakdown by masked/observed

2. **Evaluation Interface**:
   - Clean API for evaluating any model on any test set
   - M% masking handled within evaluation engine
   - Device-aware evaluation
   - Structured results output

3. **Integration Points**:
   - Called by trainer callbacks during training (on heldout sets)
   - Called by MIT for instance-level evaluation
   - Used by all 4 evaluation strategies

**Key Requirements**:
- **NO TODO COMMENTS OR SIMPLIFIED VERSIONS**
- Complete implementation of all metrics
- Masking logic consistent with training

**Deliverables**:
- Complete `EvaluationEngine` with all metrics
- Clean evaluation interface
- Integration with trainer callbacks

---

## Phase 3: Complete Multi-Instance Training (MIT) 🚀

**Objectives**: Implement all 3 MIT types supporting the 4 evaluation strategies

**Tasks**:
1. **Complete GeneralMIT**:
   - Implement `finetune_on_instance()` method
   - Support Test_O → Test_O_Masked/Test_O_Observed splitting
   - Train on Test_O_Observed, validate on Test_O_Masked
   - Integration with pretrained models

2. **Enhanced SequentialMIT**:
   - Ensure proper M% random masking during training
   - Evaluation on heldout sets after each instance
   - Support for both fresh and pretrained models

3. **Enhanced MixedMIT**:
   - IID batch sampling from combined instance pool
   - Consistent masking across instances
   - Evaluation on combined heldout set

4. **MIT Integration**:
   - Support all 4 evaluation strategies
   - Results collection and aggregation
   - Configuration-driven training parameters

**Key Requirements**:
- **NO TODO COMMENTS**
- Complete implementation of all methods
- Support for evaluation strategy switching

**Deliverables**:
- Complete `GeneralMIT.finetune_on_instance()`
- Enhanced `SequentialMIT` and `MixedMIT`
- Full integration with evaluation strategies

---

## Phase 4: Domain Model Integration 📊

**Objectives**: Implement Domain EM evaluation strategy

**Tasks**:
1. **Domain Model Runner**:
   - Interface to existing Stan domain model
   - Fit EM model to Test_O for each test instance
   - Generate predictions for Test_M variables

2. **Domain Model Evaluation**:
   - Compute same metrics as neural models (log loss, accuracy, RMSE)
   - Integration with evaluation engine
   - Results in same format as other strategies

3. **Stan Integration**:
   - Call existing Stan files
   - Handle domain model outputs
   - Error handling and validation

**Deliverables**:
- `DomainModelRunner` class
- Integration with evaluation engine
- Same metrics as neural approaches

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
2. **⏳ Phase 2** (Evaluation Engine) - Core functionality **NEXT**
3. **Phase 3** (Complete MIT) - Key experimental capability
4. **Phase 4** (Domain Model) - Baseline comparison
5. **Phase 5** (Experiment Runner) - End-to-end experiments
6. **Phase 6** (Visualization) - Results analysis

## Key Implementation Standards

- **NO TODO COMMENTS OR SIMPLIFIED IMPLEMENTATIONS**
- Complete functionality in every phase
- Comprehensive testing for each component
- Clean separation of concerns
- Configuration-driven experiments
- Structured results and comprehensive metrics