- [btools](#btools)
  - [Description](#description)
  - [Features](#features)
  - [Installation](#installation)
    - [Using uv (Recommended)](#using-uv-recommended)
  - [Usage](#usage)
    - [Command Line Interface](#command-line-interface)
      - [Data Viewing and Profiling](#data-viewing-and-profiling)
      - [Data Selection](#data-selection)
    - [Configuration System](#configuration-system)
      - [Available Configuration Variables](#available-configuration-variables)
      - [Example Configuration](#example-configuration)
    - [Python API](#python-api)
      - [Data Selection with PreSelectDataPolars](#data-selection-with-preselectdatapolars)
      - [Data Viewing and Profiling with PreViewData](#data-viewing-and-profiling-with-previewdata)
      - [Advanced Data Viewing](#advanced-data-viewing)
      - [Working with Different File Formats](#working-with-different-file-formats)
      - [Advanced Usage](#advanced-usage)
  - [Development](#development)
    - [Setting up the Development Environment](#setting-up-the-development-environment)
    - [Running Tests](#running-tests)
    - [Code Quality](#code-quality)
    - [Available poe Tasks](#available-poe-tasks)
  - [Project Structure](#project-structure)
  - [API Reference](#api-reference)
    - [PreViewData](#previewdata)
    - [PreSelectDataPolars](#preselectdatapolars)
  - [Contributing](#contributing)
  - [License](#license)
  - [Author](#author)

# btools

A comprehensive package for bioinformatics tools and data processing utilities.

## Description

btools is a Python package that provides efficient tools for bioinformatics data processing, with a focus on data selection, manipulation, and analysis. The package leverages Polars for optimal performance when working with large datasets.

## Features

- **Data Selection Tools**: Extract subsets of data from various file formats (CSV, TSV, Excel)
- **Data Viewing & Profiling**: Fast dataset exploration with rich terminal output and detailed analysis
- **Configurable Display Modes**: Multiple display options (auto, normal, rotated, wrapped) for optimal data visualization
- **TOML Export**: Generate detailed dataset reports with statistics and metadata
- **Fast Data Processing**: Uses Polars for efficient data manipulation and analysis
- **Flexible File Handling**: Support for multiple delimiters and Excel sheets
- **Environment Configuration**: Customizable display settings via .env files with priority-based loading
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

## Usage

### Command Line Interface

After installation, you can use the `btools` command:

```bash
# Show available commands
btools --help

# Show version
btools --version

# Show pre-processing commands
btools pre --help
```

#### Data Viewing and Profiling

Quickly explore and analyze datasets with beautiful terminal output:

```bash
# Basic usage - view first 50 rows and 25 columns
btools pre view data.csv

# View specific range of rows and columns
btools pre view data.csv --rows "10,100" --cols "5,15"

# Generate detailed TOML report with metadata and statistics
btools pre view data.csv --output-info --output-info-file analysis.toml

# Work with Excel files
btools pre view data.xlsx --sheet "Sheet2"

# Custom separator and display options
btools pre view data.txt --sep "|" --display-mode wrapped

# Minimal output - just data preview
btools pre view large_data.csv --no-types --no-stats --rows 10 --cols 5
```

**Display Modes:**

- `auto` (default): Automatically choose best display based on column count
- `normal`: Standard horizontal table (best for ≤5 columns)  
- `rotated`: Shortened column headers (good for 6-10 columns)
- `wrapped`: Multi-section tables (best for >10 columns)

**TOML Output Features:**

- Dataset overview (file info, dimensions, memory usage)
- Column details (types, missing values, unique counts)
- Statistical summaries for numeric columns
- Unique values list for columns with low cardinality (configurable threshold)
- Data quality metrics

#### Data Selection

Extract subsets of data from various file formats:

```bash
# Basic data selection
btools pre select_data input.csv --output subset.csv

# Advanced selection with ranges
btools pre select_data data.csv --col-start "1,50" --row-start "10,1000"

# Multiple index columns with custom separator
btools pre select_data data.csv --index-col "0,2,5" --index-separator "_"

# Work with Excel files
btools pre select_data data.xlsx --sheet "Sheet1" --output results.csv
```

### Configuration System

btools uses a flexible .env configuration system that allows you to customize display settings and behavior. Configuration files are loaded with the following priority order (higher priority overrides lower):

1. **Package defaults**: `<package_dir>/.env` (lowest priority)
2. **Project settings**: `<current_directory>/.env` (medium priority)  
3. **User settings**: `<home_directory>/.env` (highest priority)

#### Available Configuration Variables

**Display Mode Thresholds:**

```bash
# Auto mode column thresholds (when to switch display modes)
VIEW_AUTO_NORMAL_MAX_COLS=10      # Switch from normal to rotated mode
VIEW_AUTO_ROTATED_MAX_COLS=14     # Switch from rotated to wrapped mode
```

**Normal Mode Settings:**

```bash
VIEW_NORMAL_MAX_COL_WIDTH=15      # Maximum column width in characters
VIEW_NORMAL_MAX_CELL_LENGTH=10    # Maximum cell content length
```

**Rotated Headers Mode Settings:**

```bash
VIEW_ROTATED_MAX_COL_WIDTH=12       # Maximum column width
VIEW_ROTATED_MAX_CELL_LENGTH=14     # Maximum cell content length  
VIEW_ROTATED_HEADER_MAX_LENGTH=12   # Maximum header length (truncated with ellipsis)
```

**Wrapped Mode Settings:**

```bash
VIEW_WRAPPED_COLS_PER_SECTION=15  # Number of columns per section
VIEW_WRAPPED_MAX_COL_WIDTH=12     # Maximum column width
VIEW_WRAPPED_MAX_CELL_LENGTH=14   # Maximum cell content length
```

**General Display Settings:**

```bash
VIEW_DEFAULT_ROWS=10              # Default number of rows to display
VIEW_MAX_ROWS=1000               # Maximum allowed rows
VIEW_ELLIPSIS_STRING=...         # String used for truncation
VIEW_NULL_DISPLAY_STYLE=dim red  # Rich style for null values
```

**TOML Output Settings:**

```bash
VIEW_OUT_UNIQUE_MAX=20           # Max unique values to include in TOML output
                                # Columns with unique_count <= this value will
                                # include their unique values in the TOML report
```

#### Example Configuration

Create a `.env` file in your home directory to customize default behavior:

```bash
# ~/.env
VIEW_AUTO_NORMAL_MAX_COLS=8
VIEW_ELLIPSIS_STRING=***
VIEW_OUT_UNIQUE_MAX=50
VIEW_NULL_DISPLAY_STYLE=red
```

Or create a project-specific configuration:

```bash
# ./project/.env  
VIEW_WRAPPED_COLS_PER_SECTION=20
VIEW_OUT_UNIQUE_MAX=5
```

### Python API

#### Data Selection with PreSelectDataPolars

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

#### Data Viewing and Profiling with PreViewData

```python
from btools.scripts.pre.pre_view import PreViewData

# Basic dataset exploration
viewer = PreViewData(
    input_file="data.csv",
    rows=50,            # Show 50 rows
    cols=25,            # Show 25 columns
    display_mode="auto" # Auto-select display mode
)

# View and analyze the dataset
viewer.view()

# Generate detailed TOML report
profiler = PreViewData(
    input_file="data.csv",
    output_info="analysis.toml",  # Generate detailed report
    show_stats=True,              # Include statistical summary
    show_types=True,              # Show data types
    show_missing=True             # Show missing value analysis
)

profiler.view()
```

#### Advanced Data Viewing

```python
# Custom display settings and range selection
viewer = PreViewData(
    input_file="large_dataset.csv",
    rows="100,200",        # Show rows 100-200
    cols="5,15",           # Show columns 5-15
    display_mode="wrapped", # Force wrapped display
    sep="|",               # Custom separator
    show_stats=False       # Skip statistical summary
)

# Excel file with specific sheet
excel_viewer = PreViewData(
    input_file="data.xlsx",
    sheet="Sheet2",        # Specific sheet
    output_info="excel_analysis.toml"
)

# Get configuration information
info = viewer.get_info()
print(f"Viewer config: {info}")
```

#### Working with Different File Formats

```python
# Excel files
excel_selector = PreSelectDataPolars(
    input_file="data.xlsx",
    sheet="Sheet1",  # Specify sheet name
    sep=None         # Auto-detect separator
)

# Custom delimited files
custom_selector = PreSelectDataPolars(
    input_file="data.txt",
    sep="|"  # Pipe-delimited file
)

# TSV files
tsv_selector = PreSelectDataPolars(
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
│   │       ├── __init__.py
│   │       └── cli_pre.py      # CLI commands for pre-processing
│   └── scripts/               # Core functionality
│       ├── __init__.py
│       └── pre/
│           ├── __init__.py
│           ├── pre_select_data.py  # Data selection tools
│           └── pre_view.py         # Data viewing and profiling
├── tests/                     # Comprehensive test suite
│   ├── conftest.py
│   ├── data/                  # Test data files
│   └── scripts/pre/
│       └── test_pre_select_data.py
├── .env                       # Default configuration
├── pyproject.toml            # Project configuration
├── uv.lock                   # Dependency lock file
└── README.md
```

## API Reference

### PreViewData

A Polars-based data viewer and profiler for fast dataset exploration.

**Methods:**

- `view()`: Display dataset overview, column info, statistics, and data preview
- `get_info()`: Get information about the viewer configuration
- `_read_data()`: Read data from the input file using appropriate format detection
- `_generate_detailed_info()`: Generate comprehensive dataset metadata for TOML export

**Configuration Options:**

- `input_file`: Path to the input data file
- `rows`: Number of rows or range to display (e.g., 50 or "10,100")
- `cols`: Number of columns or range to display (e.g., 25 or "5,15")
- `output_info`: Path for TOML output file (optional)
- `sep`: Custom separator for delimited files
- `sheet`: Sheet name/number for Excel files
- `display_mode`: Display mode ("auto", "normal", "rotated", "wrapped")
- `show_stats`: Include statistical summary
- `show_types`: Show data types and missing value info
- `show_missing`: Show missing value analysis

### PreSelectDataPolars

A Polars-based data selector optimized for large datasets and high performance.

**Methods:**

- `process()`: Process the input file and return selected data
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

