# btools

A comprehensive package for bioinformatics tools and data processing utilities.

## Description

btools is a Python package that provides efficient tools for bioinformatics data processing, with a focus on data selection, manipulation, and analysis. The package leverages both pandas and Polars for optimal performance when working with large datasets.

## Features

- **Data Selection Tools**: Extract subsets of data from various file formats (CSV, TSV, Excel)
- **Dual Engine Support**: Choose between pandas and Polars for data processing based on your performance needs
- **Flexible File Handling**: Support for multiple delimiters and Excel sheets
- **Command Line Interface**: Easy-to-use CLI for common data processing tasks
- **Type Safe**: Full type annotations and strict type checking with pyright
- **Comprehensive Testing**: Extensive test suite with 100% coverage

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager and project manager. To install btools using uv:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install btools from the local project
uv add btools

# Or install in development mode if you're working on the project
uv pip install -e .

# From GitHub
uv pip install git+ssh://git@github.com/medmaca/btools.git
```

### Using pip

```bash
pip install btools
```

## Requirements

- Python >= 3.13
- pandas >= 2.0.0
- polars >= 1.31.0
- click >= 8.2.1
- fastexcel >= 0.14.0
- openpyxl >= 3.1.5

## Usage

### Command Line Interface

After installation, you can use the `btools` command:

```bash
# Show available commands
btools --help

# Show version
btools --version
```

### Python API

#### Data Selection with PreSelectData (pandas)

```python
from btools.scripts.pre.pre_select_data import PreSelectData

# Basic usage - select subset of data
selector = PreSelectData(
    input_file="data.csv",
    output_file="subset.csv",
    index_col=0,        # Use first column as index
    col_start=1,        # Start output from column 1
    row_start=1         # Start output from row 1
)

# Process the data
data = selector.process()
print(data.head())
```

#### Data Selection with PreSelectDataPolars (faster)

```python
from btools.scripts.pre.pre_select_data import PreSelectDataPolars

# High-performance data processing with Polars
selector = PreSelectDataPolars(
    input_file="large_data.csv",
    output_file="subset.csv",
    index_col="0,1",    # Use multiple columns as concatenated index
    col_start="1,100",  # Select columns 1-100
    row_start="1,1000"  # Select rows 1-1000
)

# Process large datasets efficiently
data = selector.process()
print(f"Processed data shape: {data.shape}")
```

#### Working with Different File Formats

```python
# Excel files
excel_selector = PreSelectData(
    input_file="data.xlsx",
    sheet="Sheet1",  # Specify sheet name
    sep=None         # Auto-detect separator
)

# Custom delimited files
custom_selector = PreSelectData(
    input_file="data.txt",
    sep="|"  # Pipe-delimited file
)

# TSV files
tsv_selector = PreSelectData(
    input_file="data.tsv",
    sep="\t"  # Tab-separated
)
```

#### Advanced Usage

```python
# Multiple index columns with custom separator
multi_index = PreSelectDataPolars(
    input_file="complex_data.csv",
    index_col="0,2,5",           # Columns 0, 2, and 5 as index
    index_separator="_",          # Use underscore to join indices
    col_start="10,50",           # Columns 10-50
    row_start="5,1000"           # Rows 5-1000
)

# Get information about the processed data
info = multi_index.get_info()
print(f"Data info: {info}")
```

## Development

### Setting up the Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd btools

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install development dependencies
uv sync --dev

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=btools --cov-report=html

# Run specific test file
uv run pytest tests/scripts/pre/test_pre_select_data.py

# Run tests with markers
uv run pytest -m "not slow"  # Skip slow tests
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check --fix

# Type checking
uv run pyright

# Run all quality checks
uv run poe all
```

### Available poe Tasks

```bash
uv run poe fmt      # Format code with ruff
uv run poe lint     # Lint and fix with ruff
uv run poe check    # Type check with pyright
uv run poe test     # Run tests with pytest
uv run poe all      # Run all of the above
```

## Project Structure

```text
btools/
├── src/btools/
│   ├── __init__.py
│   ├── cli/                    # Command line interface
│   │   ├── __init__.py
│   │   └── pre/
│   │       └── cli_pre.py
│   └── scripts/               # Core functionality
│       ├── __init__.py
│       └── pre/
│           ├── __init__.py
│           └── pre_select_data.py  # Data selection tools
├── tests/                     # Comprehensive test suite
│   ├── conftest.py
│   ├── data/                  # Test data files
│   └── scripts/pre/
│       └── test_pre_select_data.py
├── pyproject.toml            # Project configuration
├── uv.lock                   # Dependency lock file
└── README.md
```

## API Reference

### PreSelectData

A pandas-based data selector for general use cases.

**Methods:**

- `process()`: Process the input file and return selected data
- `get_info()`: Get information about the processed data
- `read_data()`: Read data from the input file
- `select_subset()`: Select a subset of the data

### PreSelectDataPolars

A Polars-based data selector optimized for large datasets.

**Methods:**

- `process()`: Process the input file and return selected data (faster than pandas)
- `get_info()`: Get information about the processed data
- `read_data()`: Read data from the input file using Polars
- `select_subset()`: Select a subset of the data with better memory efficiency

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass and code quality checks succeed:

   ```bash
   uv run poe all
   ```

5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**medmaca** - [matthew.care@gmail.com](mailto:matthew.care@gmail.com)

## Changelog

### v0.1.0

- Initial release
- PreSelectData and PreSelectDataPolars classes for data selection
- Command line interface
- Comprehensive test suite with 53+ tests
- Full type safety with pyright
- Support for CSV, TSV, Excel, and custom delimited files
