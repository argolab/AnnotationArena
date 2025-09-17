# Test Suite for Ranking Imputer

This directory contains comprehensive tests for the ranking imputer system using pytest.

## Test Files

- `test_corrected_understanding.py` - Tests for the corrected masking strategy and data flow
- `test_embedding_providers.py` - Tests for embedding providers including FullyRandomizedEmbeddingProvider
- `test_data_converter.py` - Tests for DataConverter functionality

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
# or
make test
```

### Run Specific Test File
```bash
pytest tests/test_corrected_understanding.py -v
# or
make test-understanding
```

### Run with Debugger (pdb)
```bash
pytest tests/ -v --pdb
# or
make test-debug
```

### Run Specific Test with Debugger
```bash
pytest tests/test_embedding_providers.py -v --pdb
# or
make test-embedding-debug
```

### Run Specific Test Function
```bash
pytest tests/ -v -k test_training_batch_creation
# or
make test-function FUNCTION=test_training_batch_creation
```

### Run with Coverage
```bash
pytest tests/ -v --cov=imputer --cov-report=html
# or
make test-coverage
```

## Test Structure

Each test file contains multiple test functions that can be run independently. The tests cover:

1. **Data Flow Understanding**: Correct train/test data separation
2. **Masking Strategy**: Self-supervised learning with M% masked, (1-M)% observed
3. **Multi-Instance Training**: SequentialMIT and MixedMIT implementations
4. **Embedding Providers**: FullyRandomizedEmbeddingProvider and assertions
5. **Batch Creation**: Training and evaluation batch creation

## Debugging

Use `--pdb` flag to enter the Python debugger at the start of each test. This is particularly useful for:

- Understanding the data flow
- Debugging batch creation
- Inspecting embedding provider behavior
- Verifying masking strategies

## Test Data

Tests use mock data that simulates the real data structure:
- Ratings: `{'annotator': int, 'attribute': int, 'item': int, 'value': int}`
- Rankings: `{'annotator': int, 'attribute': int, 'items': List[int], 'order': List[int]}`

## Expected Output

All tests should pass with output like:
```
✓ Training batch creation test passed!
✓ Evaluation batch creation test passed!
✓ SequentialMIT with corrected understanding test passed!
✓ MixedMIT with corrected understanding test passed!
✓ Data flow understanding test passed!
✓ Masking rates test passed!
✓ All corrected understanding tests passed!
```
