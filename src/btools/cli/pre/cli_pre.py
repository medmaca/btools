"""CLI commands for data pre-processing and exploration.

This module defines Click command groups and commands for data exploration,
profiling, and subset selection functionality in the btools package.
"""

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

    Commands:
        select_data: Extract subsets of data with flexible row/column selection.
        view: Quick data profiling and preview with beautiful terminal output.
    """


# Data selection command
@pre_group.command("select_data")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    help=("Output file path (CSV or Parquet; defaults to input file with '_subset.csv' or '.parquet' suffix)"),
)
@click.option(
    "--index-col",
    "-i",
    default="0",
    help=(
        "Column(s) to use as row index. Single integer (e.g., 0) or "
        "comma-separated integers (e.g., '1,2,4') for concatenated index (default: 0)"
    ),
)
@click.option(
    "--col-start",
    "-c",
    default="1",
    help=(
        "Column selection (zero-based, default: 1). Supports single columns (e.g., '1'), "
        "ranges (e.g., '1:5'), and multiple ranges (e.g., '1:3,5:8')"
    ),
)
@click.option("--row-index", "--ri", default=0, help="Row to use as column header (zero-based, default: 0)")
@click.option(
    "--row-start",
    "--rs",
    "-r",
    default="1",
    help=(
        "Row selection (zero-based, default: 1). Supports single rows (e.g., '1'), "
        "ranges (e.g., '1:5'), and multiple ranges (e.g., '1:3,5:8')"
    ),
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
@click.option(
    "--parquet",
    "-p",
    is_flag=True,
    default=False,
    help="Write output as Parquet file (default: CSV)",
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
    parquet: bool,
):
    """Extract subsets of data from input files and save as CSV or Parquet.

    Reads data from various file formats (CSV, Excel, TSV, Parquet) and allows selection of
    specific subsets based on flexible row and column parameters. All indices are zero-based.
    Uses Polars for fast data processing.

    Args:
        input_file (str): Path to the input data file (CSV, Excel, TSV, Parquet, etc.).
        output (str, optional): Custom output file path. If not provided, defaults to input filename
            with '_subset.csv' or '.parquet' suffix depending on --parquet flag.
        index_col (str): Column(s) to use as row index. Can be a single integer ("0") or
            comma-separated integers ("1,2,4") for concatenated index.
        col_start (str): Column selection starting point. Supports single columns ("1"),
            ranges ("1:5"), and multiple ranges ("1:3,5:8").
        row_index (int): Row to use as column header (zero-based).
        row_start (str): Row selection starting point. Supports single rows ("1"),
            ranges ("1:5"), and multiple ranges ("1:3,5:8").
        sep (str, optional): Custom separator/delimiter for reading files (overrides auto-detection).
        sheet (str, optional): Excel sheet name or number (0-based). If not specified, uses first sheet.
        transpose (bool): If True, transpose data before applying row/column selections.
        index_separator (str): Separator for concatenating multiple index columns.
        parquet (bool): If True, output will be written as a Parquet file (default: False).

    Raises:
        click.Abort: If file processing fails or invalid parameters are provided.

    Note:
        Output format is CSV unless --parquet is specified, in which case output is Parquet.
        GZIP compression is not applied to Parquet files.

    Examples:
        Basic usage with defaults:
            $ btools pre select_data input.csv

        Output as Parquet:
            $ btools pre select_data input.csv --parquet

        Select data starting from column 2, using row 1 as headers:
            $ btools pre select_data input.xlsx --col-start 2 --row-index 1

        Transpose data before selection:
            $ btools pre select_data input.csv --transpose

        Select specific column and row ranges:
            $ btools pre select_data input.xlsx --col-start "1:5" --row-start "1:10"

        Use multiple columns for index with custom separator:
            $ btools pre select_data input.csv --index-col "1,2,4" --index-separator "#"

        Process Excel file with specific sheet:
            $ btools pre select_data workbook.xlsx --sheet "Data Sheet" --output results.csv

        Use custom separator for delimited files:
            $ btools pre select_data data.txt --sep "|" --output result.csv
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
            parquet_out=parquet,
        )
        processor.process()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e


# Data viewing/profiling command
@pre_group.command("view")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--rows",
    "-r",
    default="10",
    help="Number of rows to display, range (e.g., '10:60'), or multiple ranges (e.g., '1:10,50:60') (default: 10)",
)
@click.option(
    "--cols",
    "-c",
    default="10",
    help="Number of columns to display, range (e.g., '5:15'), or multiple ranges (e.g., '1:5,10:15') (default: 25)",
)
@click.option("--output-info", "--oi", help="If set will output information to a TOML file", default=False, is_flag=True)
@click.option("--output-info-file", "--oif", help="Filename to output detailed info to TOML file", default=None)
@click.option("--sep", "-s", help="Custom separator for CSV files")
@click.option("--sheet", help="Sheet name or number for Excel files")
@click.option(
    "--dataset-overview",
    "--do",
    is_flag=True,
    default=False,
    help="Show dataset overview only",
)
@click.option(
    "--column-info",
    "--ci",
    is_flag=True,
    default=False,
    help="Show column information only",
)
@click.option(
    "--numeric-stats",
    "--ns",
    is_flag=True,
    default=False,
    help="Show numeric statistics only",
)
@click.option(
    "--show-source-numbers",
    "--snn",
    is_flag=True,
    default=False,
    help="Show original source row/column numbers instead of subset positions",
)
def view_data(
    input_file: str,
    rows: str,
    cols: str,
    output_info: bool,
    output_info_file: str | None,
    sep: str | None,
    sheet: str | None,
    dataset_overview: bool,
    column_info: bool,
    numeric_stats: bool,
    show_source_numbers: bool,
):
    """View and profile datasets with beautiful terminal output.

    Provides fast data exploration by displaying comprehensive information about data
    files. Uses Polars for fast data processing and Rich for beautiful terminal output
    with syntax highlighting and formatted tables.

    By default, shows only the data preview with both row and column numbers displayed
    to help identify data positions. Files are treated as raw data without headers,
    so row 0 corresponds to the actual first line of the file. Column numbers appear
    in the header row, and row numbers appear in the first column of the table.

    Args:
        input_file: Path to the input data file (CSV, Excel, TSV, etc.). Supports
            regular and gzip-compressed files.
        rows: Number of rows to display or range specification. Can be a number ("50"),
            single range ("10:60"), or multiple ranges ("1:10,50:60"). All ranges
            are zero-based and inclusive.
        cols: Number of columns to display or range specification. Can be a number ("25"),
            single range ("5:15"), or multiple ranges ("1:5,10:15"). All ranges
            are zero-based and inclusive.
        output_info: If True, generates a detailed TOML report file with comprehensive
            dataset analysis including column statistics and data quality metrics.
        output_info_file: Custom filename for TOML report output. Defaults to
            "fileinfo.toml" in the same directory as input file.
        sep: Custom separator for CSV/delimited files (overrides auto-detection).
            Common values include ",", ";", "\t", "|".
        sheet: Excel sheet name or number (0-based). If not specified, uses first sheet.
            Can be sheet name string or numeric index.
        dataset_overview: If True, shows only dataset overview section (file info,
            shape, memory usage). Suppresses data preview when used.
        column_info: If True, shows only column information section (data types,
            missing values, unique counts). Suppresses data preview when used.
        numeric_stats: If True, shows only numeric statistics section (mean, std,
            min, max, quartiles). Suppresses data preview when used.
        show_source_numbers: If True, shows original source row/column numbers instead
            of sequential subset positions. Useful for referencing back to original
            dataset positions when working with subsets.

    Raises:
        click.Abort: If file processing fails, file doesn't exist, invalid parameters
            are provided, or unsupported file format is encountered.

    Note:
        - Files are read as raw data without automatic header detection. Row 0 in the
          data preview corresponds to the actual first line of the file, allowing for
          accurate row numbering that matches file line positions.
        - Analysis section flags (--dataset-overview, --column-info, --numeric-stats)
          can be combined to show multiple sections. When any of these flags are used,
          the default data preview is suppressed.
        - Row and column numbering uses zero-based indexing consistent with most
          programming languages and data processing tools.
        - The adaptive display automatically handles varying numbers of columns by
          using single tables for few columns and multi-section display for many columns.
        - All file formats are auto-detected unless custom separator is specified.

    Examples:
        Basic usage - show data preview with row/column numbers (row 0 = first file line):
            $ btools pre view data.csv

        Show specific range of rows and columns (all zero-based, inclusive):
            $ btools pre view data.csv --rows "10:100" --cols "5:15"

        Show multiple ranges of rows and columns:
            $ btools pre view data.csv --rows "1:10,50:60" --cols "1:5,10:15"

        Show original source positions instead of subset positions:
            $ btools pre view data.csv --rows "1:3,7:9" --cols "0:2,5:7" --snn

        Show only dataset overview:
            $ btools pre view data.csv --dataset-overview

        Show dataset overview and column info (no data preview):
            $ btools pre view data.csv --do --ci

        Show all analysis sections (no data preview):
            $ btools pre view data.csv --do --ci --ns

        Generate detailed TOML report:
            $ btools pre view data.csv --output-info --output-info-file analysis.toml

        Excel file with specific sheet:
            $ btools pre view data.xlsx --sheet "Sheet2"

        Custom separator with data preview:
            $ btools pre view data.txt --sep "|" --rows 10 --cols 5

        Gzipped file with range selection:
            $ btools pre view data.csv.gz --rows "0:100" --cols "0:10"
    """
    try:
        output_info_path: str | None = None
        if output_info and output_info_file:
            output_info_path = os.path.join(os.path.dirname(input_file), output_info_file)
        elif output_info:
            output_info_path = os.path.join(os.path.dirname(input_file), input_file + "_info.toml")

        viewer = PreViewData(
            input_file=input_file,
            rows=rows,
            cols=cols,
            output_info=output_info_path,
            sep=sep,
            sheet=sheet,
            show_dataset_overview=dataset_overview,
            show_column_info=column_info,
            show_numeric_stats=numeric_stats,
            show_source_numbers=show_source_numbers,
        )
        viewer.view()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e
