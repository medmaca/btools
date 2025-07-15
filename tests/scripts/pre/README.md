# Unit Tests for pre_select_data.py

This directory contains comprehensive unit tests for the `pre_select_data.py` module, covering the `PreSelectDataPolars` class.

## Test Structure

The test suite is organized into several test classes:

1. **TestPreSelectDataPolars**: Tests for the Polars-based implementation
2. **TestEdgeCasesAndErrorHandling**: Edge cases and error scenarios

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

