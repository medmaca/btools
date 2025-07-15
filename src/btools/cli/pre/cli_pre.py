"""Defines all CLI commands for the template-cli application."""

import os

import click

from btools.scripts.pre.pre_select_data import PreSelectDataPolars
from btools.scripts.pre.pre_view import PreViewData


# Create a group called "pre"
@click.group("pre")
def pre_group():
    """Pre-processing commands."""


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
    default="1",
    help="Row selection (zero-based, default: 1). "
    "Supports single rows (e.g., '1'), ranges (e.g., '1:5' for rows 1-5), "
    "and multiple ranges (e.g., '1:3,5:8' for rows 1-3 and 5-8)",
)
@click.option("--sep", "-s", help="Separator/delimiter to use when reading the file (overrides auto-detection)")
@click.option("--sheet", help="Sheet name or number to read from Excel files (default: first sheet)")
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
    index_separator: str,
):
    """Select a subset of data from an input file and save as CSV.

    This command reads data from various file formats (CSV, Excel, TSV) and allows you
    to select specific subsets based on column and row parameters. All indices are zero-based.

    This tool uses Polars for fast data processing.

    By default, the output file will be named with the input filename plus '_subset.csv' suffix.
    For example, 'data.xlsx' will output to 'data_subset.csv'.

    For Excel files, you can specify which sheet to read using the --sheet option.
    This accepts either a sheet name (e.g., "Sheet2") or a sheet index (e.g., "1" for the second sheet).

    The --index-col option supports multiple columns for creating concatenated indices.
    Use comma-separated column numbers (e.g., "1,2,4") to combine values from multiple columns.

    Example usage:
    \b
    # Basic usage with defaults (outputs to input_subset.csv)
    btools pre select_data input.csv

    # Select data starting from column 2, using row 1 as headers
    btools pre select_data input.xlsx --col-start 2 --row-index 1

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

    This command provides a fast way to explore datasets by displaying:
    - Dataset overview (rows, columns, memory usage)
    - Column information (data types, missing values, unique counts)
    - Statistical summary for numeric columns
    - Data preview with configurable row and column display

    The tool supports multiple file formats including CSV, TSV, Excel, and more.
    It uses Polars for fast data processing and Rich for beautiful terminal output.

    Display modes:
    - auto: Automatically choose best display mode based on column count
    - normal: Standard horizontal table (best for ≤5 columns)
    - rotated: Shortened column headers (good for 6-10 columns)
    - wrapped: Multi-section tables (best for >10 columns)

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
