# Unit Tests for pre_select_data.py

This directory contains comprehensive unit tests for the `pre_select_data.py` module, covering both the `PreSelectDataPolars` and `PreSelectData` classes.

## Test Structure

The test suite is organized into several test classes:

1. **TestPreSelectDataPolars**: Tests for the Polars-based implementation
2. **TestPreSelectData**: Tests for the pandas-based implementation  
3. **TestComparisonBetweenImplementations**: Comparison tests between both implementations
4. **TestEdgeCasesAndErrorHandling**: Edge cases and error scenarios

## Test Coverage

### Core Functionality Tests

- **Initialization**: Tests parameter validation and default value handling
- **Parameter Parsing**: Tests for index column parsing, range parameters
- **File Reading**: Tests for CSV, TSV, Excel, and custom separator files
- **Data Selection**: Tests subset selection with various parameters
- **File Processing**: End-to-end processing workflow tests
- **Output Generation**: Tests for filename generation and range suffixes

### File Format Support

- ✅ CSV files (`.csv`)
- ✅ TSV files (`.tsv`) 
- ✅ Excel files (`.xlsx`, `.xls`) with multiple sheet support
- ✅ Custom separator files (pipe-delimited, etc.)
- ✅ Tab-separated files with explicit separator

### Parameter Testing

- ✅ Single and multiple index columns (`index_col`)
- ✅ Column range selection (`col_start`)
- ✅ Row range selection (`row_start`)
- ✅ Header row specification (`row_index`)
- ✅ Custom separators (`sep`)
- ✅ Excel sheet selection (`sheet`)
- ✅ Index separator for multiple columns (`index_separator`)

### Error Handling

- ✅ File not found errors
- ✅ Out-of-bounds parameter validation
- ✅ Invalid parameter format handling
- ✅ Empty file handling
- ✅ Unicode content support
- ✅ Large number handling
- ✅ Mixed data type support

### Comparison Tests

- ✅ Both implementations produce same data shapes
- ✅ Range parameter handling consistency
- ✅ Output file structure comparison

## Test Data

The test suite uses the following test data files located in `tests/data/`:

- `test_data.csv`: Main CSV test file with 10 rows of sample data
- `test_data.tsv`: TSV version of the test data
- `test_data.xlsx`: Excel file with two sheets (main data + products)
- `test_data_pipe.txt`: Pipe-delimited version of test data

## Running Tests

### Run All Tests
```bash
cd /root/uv_projects/btools
uv run pytest tests/scripts/pre/test_pre_select_data.py
```

### Run Specific Test Class
```bash
# Test only Polars implementation
uv run pytest tests/scripts/pre/test_pre_select_data.py::TestPreSelectDataPolars

# Test only pandas implementation  
uv run pytest tests/scripts/pre/test_pre_select_data.py::TestPreSelectData

# Test comparisons between implementations
uv run pytest tests/scripts/pre/test_pre_select_data.py::TestComparisonBetweenImplementations

# Test edge cases
uv run pytest tests/scripts/pre/test_pre_select_data.py::TestEdgeCasesAndErrorHandling
```

### Run Specific Test
```bash
# Test a specific functionality
uv run pytest tests/scripts/pre/test_pre_select_data.py::TestPreSelectDataPolars::test_process_basic

# Test with verbose output
uv run pytest tests/scripts/pre/test_pre_select_data.py -v

# Test with coverage report (if coverage installed)
uv run pytest tests/scripts/pre/test_pre_select_data.py --cov=btools.scripts.pre.pre_select_data
```

### Run Tests by Markers
```bash
# Skip slow tests
uv run pytest tests/scripts/pre/test_pre_select_data.py -m "not slow"

# Run only integration tests
uv run pytest tests/scripts/pre/test_pre_select_data.py -m "integration"
```

## Test Results Summary

✅ **54 total tests** - All passing  
✅ **30 Polars implementation tests**  
✅ **16 pandas implementation tests**  
✅ **2 comparison tests**  
✅ **6 edge case tests**  

## Key Test Categories

### 1. Initialization Tests (6 tests)
- Default parameter validation
- Custom parameter handling
- File path processing

### 2. Parameter Parsing Tests (10 tests)
- Index column parsing (single/multiple)
- Range parameter parsing (column/row)
- Invalid format handling
- Error conditions

### 3. File I/O Tests (12 tests)
- Multiple file format support
- Sheet selection for Excel files
- Custom separator handling
- File not found scenarios

### 4. Data Processing Tests (14 tests)
- Basic subset selection
- Range-based selection
- Multiple index columns
- Boundary validation

### 5. Integration Tests (8 tests)
- End-to-end processing workflows
- Output file generation
- Range suffix handling
- Cross-implementation comparison

### 6. Edge Case Tests (4 tests)
- Empty files
- Single row/column files
- Unicode content
- Large numbers and mixed types

## Dependencies

The test suite requires:
- pytest
- pandas
- polars 
- openpyxl (for Excel support)

All dependencies are included in the project's dev dependencies and can be installed with:
```bash
uv sync --group dev
```

## Notes

- Tests use temporary directories for output files to avoid cluttering the workspace
- Mock objects are used to suppress print statements during testing
- Tests are designed to be independent and can be run in any order
- The test suite validates both implementations produce consistent results
- Error handling tests ensure graceful failure modes
