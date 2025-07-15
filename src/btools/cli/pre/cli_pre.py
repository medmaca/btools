"""Defines all CLI commands for the template-cli application."""

import os

import click

from btools.scripts.pre.pre_select_data import PreSelectDataPolars
from btools.scripts.pre.pre_view import PreViewData


# Create a group called "pre"
@click.group("pre")
def pre_group():
    """Pre-processing commands for data exploration and subset selection.

    This group provides tools to explore, profile, and extract subsets from various
    data file formats including CSV, Excel, and TSV files. All commands use Polars
    for fast data processing and support both regular and gzip-compressed files.

    Available commands:
    - select_data: Extract subsets of data with flexible row/column selection
    - view: Quick data profiling and preview with beautiful terminal output
    """


# Data selection command
@pre_group.command("select_data")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output CSV file path (defaults to input file with '_subset.csv' suffix)")
@click.option(
    "--index-col",
    "-i",
    default="0",
    help="Column(s) to use as row index. Single integer (e.g., 0) or comma-separated integers "
    + "(e.g., '1,2,4') for concatenated index (default: 0)",
)
@click.option(
    "--col-start",
    "-c",
    default="1",
    help="Column selection (zero-based, default: 1). "
    "Supports single columns (e.g., '1'), ranges (e.g., '1:5' for columns 1-5), "
    "and multiple ranges (e.g., '1:3,5:8' for columns 1-3 and 5-8)",
)
@click.option("--row-index", "--ri", default=0, help="Row to use as column header (zero-based, default: 0)")
@click.option(
    "--row-start",
    "--rs",
    "-r",
    default="1",
    help="Row selection (zero-based, default: 1). "
    "Supports single rows (e.g., '1'), ranges (e.g., '1:5' for rows 1-5), "
    "and multiple ranges (e.g., '1:3,5:8' for rows 1-3 and 5-8)",
)
@click.option("--sep", "-s", help="Separator/delimiter to use when reading the file (overrides auto-detection)")
@click.option("--sheet", help="Sheet name or number to read from Excel files (default: first sheet)")
@click.option(
    "--transpose",
    "-t",
    is_flag=True,
    default=False,
    help="Transpose the data before applying row/column selections",
)
@click.option(
    "--index-separator",
    "--is",
    default="#",
    help="Separator to use when concatenating multiple index columns (default: '#')",
)
def select_data(
    input_file: str,
    output: str | None,
    index_col: str,
    col_start: str,
    row_index: int,
    row_start: str,
    sep: str | None,
    sheet: str | None,
    transpose: bool,
    index_separator: str,
):
    """Select a subset of data from an input file and save as CSV (or CSV.gz).

    This command reads data from various file formats (CSV, Excel, TSV) and allows you
    to select specific subsets based on column and row parameters. All indices are zero-based.
    This tool uses Polars for fast data processing.

    SUPPORTED FILE FORMATS:
    - CSV files (.csv, .csv.gz): Auto-detected or custom separator
    - Excel files (.xlsx, .xls): Sheet selection supported
    - TSV files (.tsv, .tsv.gz): Tab-separated values
    - Other delimited files: Custom separator via --sep

    OUTPUT BEHAVIOR:
    By default, the output file will be named with the input filename plus '_subset.csv' suffix.
    For example, 'data.xlsx' will output to 'data_subset.csv'. The output format is always CSV
    unless the GZIP_OUT environment variable is set, which creates .csv.gz files.

    MAIN OPTIONS:
    --output/-o: Specify custom output file path
    --index-col/-i: Column(s) to use as row index (default: 0)
      • Single column: --index-col 0
      • Multiple columns: --index-col "1,2,4" (concatenated with separator)
    --col-start/-c: Column selection starting point (default: 1)
      • Single column to end: --col-start 2
      • Range: --col-start "1:5" (columns 1-5)
      • Multiple ranges: --col-start "1:3,5:8" (columns 1-3 and 5-8)
    --row-index/--ri: Row to use as column header (default: 0)
    --row-start/--rs/-r: Row selection starting point (default: 1)
      • Single row to end: --row-start 2
      • Range: --row-start "1:10" (rows 1-10)
      • Multiple ranges: --row-start "1:5,10:15" (rows 1-5 and 10-15)

    ADVANCED OPTIONS:
    --transpose/-t: Transpose data (swap rows/columns) before applying selections
    --sep/-s: Custom separator for delimited files (e.g., "|", "\\t" for tab)
    --sheet: Excel sheet selection by name or index (0-based)
    --index-separator/--is: Separator for concatenating multiple index columns (default: "#")

    OPERATION ORDER:
    1. Read data from input file
    2. Apply transpose (if --transpose is specified)
    3. Apply row and column selections
    4. Write to output file

    Example usage:
    \b
    # Basic usage with defaults (outputs to input_subset.csv)
    btools pre select_data input.csv

    # Select data starting from column 2, using row 1 as headers
    btools pre select_data input.xlsx --col-start 2 --row-index 1

    # Transpose data before selection (rows become columns)
    btools pre select_data input.csv --transpose

    # Transpose and then select specific columns and rows
    btools pre select_data input.xlsx --transpose --col-start "1:5" --row-start "1:10"

    # Select columns 1-5 and rows 1-10 (range support)
    btools pre select_data input.xlsx --col-start "1:5" --row-start "1:10"

    # Select first 25 rows starting from column 3
    btools pre select_data input.csv --col-start 3 --row-start "1:25"

    # Select multiple column ranges (1-3 and 5-8) and row ranges (1-10 and 20-30)
    btools pre select_data input.xlsx --col-start "1:3,5:8" --row-start "1:10,20:30"

    # Use multiple columns (1,2,4) for index, separated by '#'
    btools pre select_data input.csv --index-col "1,2,4" --index-separator "#"

    # Specify a specific Excel sheet by name
    btools pre select_data workbook.xlsx --sheet "Data Sheet" --output results.csv

    # Specify a specific Excel sheet by index (0-based)
    btools pre select_data workbook.xlsx --sheet 1 --output results.csv

    # Specify custom output file and parameters
    btools pre select_data data.tsv --output subset.csv --index-col 1 --row-start 2

    # Use custom separator for delimited files
    btools pre select_data data.txt --sep "|" --output result.csv

    # Use custom index separator for multiple index columns
    btools pre select_data data.csv --index-col "0,1" --index-separator "_" --output result.csv
    """
    try:
        # Convert index_col to int if it's a single number, otherwise keep as string
        parsed_index_col = int(index_col) if "," not in index_col else index_col

        # Use Polars implementation
        processor = PreSelectDataPolars(
            input_file=input_file,
            output_file=output,
            index_col=parsed_index_col,
            col_start=col_start,
            row_index=row_index,
            row_start=row_start,
            sep=sep,
            sheet=sheet,
            transpose=transpose,
            index_separator=index_separator,
        )
        processor.process()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e


# Data viewing/profiling command
@pre_group.command("view")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--rows", "-r", default="50", help="Number of rows to display or range (e.g., '50' or '10,60') (default: 50)")
@click.option(
    "--cols", "-c", default="25", help="Number of columns to display or range (e.g., '25' or '5,15') (default: 25)"
)
@click.option("--output-info", "--oi", help="If set will output information to a TOML file", default=False, is_flag=True)
@click.option("--output-info-file", "--oif", help="Filename to output detailed info to TOML file", default="fileinfo.toml")
@click.option("--sep", "-s", help="Custom separator for CSV files")
@click.option("--sheet", help="Sheet name or number for Excel files")
@click.option(
    "--display-mode",
    type=click.Choice(["auto", "normal", "rotated", "wrapped"]),
    default="auto",
    help="Display mode for data preview (default: auto)",
)
@click.option("--no-stats", is_flag=True, default=False, help="Don't show statistical summary")
@click.option("--no-types", is_flag=True, default=False, help="Don't show data types")
@click.option("--no-missing", is_flag=True, default=False, help="Don't show missing value analysis")
def view_data(
    input_file: str,
    rows: str,
    cols: str,
    output_info: bool,
    output_info_file: str | None,
    sep: str | None,
    sheet: str | None,
    display_mode: str,
    no_stats: bool,
    no_types: bool,
    no_missing: bool,
):
    """Quickly view and profile datasets with beautiful terminal output.

    This command provides a fast way to explore datasets by displaying comprehensive
    information about your data files. It uses Polars for fast data processing and
    Rich for beautiful terminal output with syntax highlighting and formatted tables.

    INFORMATION DISPLAYED:
    - Dataset overview (rows, columns, file size, memory usage)
    - Column information (data types, missing values, unique counts)
    - Statistical summary for numeric columns (mean, std, min, max, quartiles)
    - Data preview with configurable row and column display
    - Missing value analysis and patterns

    SUPPORTED FILE FORMATS:
    - CSV files (.csv, .csv.gz): Auto-detected or custom separator
    - Excel files (.xlsx, .xls): Sheet selection supported
    - TSV files (.tsv, .tsv.gz): Tab-separated values
    - Other delimited files: Custom separator via --sep

    MAIN OPTIONS:
    --rows/-r: Number or range of rows to display (default: 50)
      • Number: --rows 100 (first 100 rows)
      • Range: --rows "10,60" (rows 10-60)
    --cols/-c: Number or range of columns to display (default: 25)
      • Number: --cols 15 (first 15 columns)
      • Range: --cols "5,15" (columns 5-15)

    OUTPUT OPTIONS:
    --output-info/--oi: Generate detailed TOML report file
    --output-info-file/--oif: Custom filename for TOML report (default: fileinfo.toml)

    FILE FORMAT OPTIONS:
    --sep/-s: Custom separator for delimited files (e.g., "|", "\\t" for tab)
    --sheet: Excel sheet selection by name or index (0-based)

    DISPLAY CUSTOMIZATION:
    --display-mode: Control table layout
      • auto: Automatically choose best mode based on column count (default)
      • normal: Standard horizontal table (best for ≤5 columns)
      • rotated: Shortened column headers (good for 6-10 columns)
      • wrapped: Multi-section tables (best for >10 columns)

    ANALYSIS TOGGLES:
    --no-stats: Skip statistical summary for numeric columns
    --no-types: Skip data type analysis
    --no-missing: Skip missing value analysis

    Example usage:
    \b
    # Basic usage - show first 50 rows and 25 columns (auto mode)
    btools pre view data.csv

    # Show specific range of rows and columns
    btools pre view data.csv --rows "10,100" --cols "5,15"

    # Force wrapped display mode for many columns
    btools pre view data.csv --cols "1,20" --display-mode wrapped

    # Use rotated headers for medium column count
    btools pre view data.csv --cols 10 --display-mode rotated

    # Generate detailed TOML report
    btools pre view data.csv --output-info --output-info-file analysis.toml

    # Excel file with specific sheet
    btools pre view data.xlsx --sheet "Sheet2"

    # Custom separator and minimal output
    btools pre view data.txt --sep "|" --no-stats

    # Show only data preview without types or stats, limited columns
    btools pre view large_data.csv --no-types --no-stats --rows 10 --cols 5

    # Quick peek at first 10 rows and 5 columns with normal display
    btools pre view data.csv --rows 10 --cols 5 --display-mode normal
    """
    try:
        output_info_path: str | None = None
        if output_info and output_info_file:
            output_info_path = os.path.join(os.path.dirname(input_file), output_info_file)

        viewer = PreViewData(
            input_file=input_file,
            rows=rows,
            cols=cols,
            output_info=output_info_path,
            sep=sep,
            sheet=sheet,
            show_stats=not no_stats,
            show_types=not no_types,
            show_missing=not no_missing,
            display_mode=display_mode,
        )
        viewer.view()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e
