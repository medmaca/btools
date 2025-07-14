"""Defines all CLI commands for the template-cli application."""

import click

from btools.scripts.pre.pre_select_data import PreSelectData, PreSelectDataPolars


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
@click.option("--col-start", "-c", default=1, help="Column from which to start outputting data (zero-based, default: 1)")
@click.option("--row-index", "--ri", default=0, help="Row to use as column header (zero-based, default: 0)")
@click.option("--row-start", "--rs", default=1, help="Row from which to start outputting data (zero-based, default: 1)")
@click.option("--sep", "-s", help="Separator/delimiter to use when reading the file (overrides auto-detection)")
@click.option("--sheet", help="Sheet name or number to read from Excel files (default: first sheet)")
@click.option(
    "--index-separator",
    "--is",
    default="#",
    help="Separator to use when concatenating multiple index columns (default: '#')",
)
@click.option(
    "--use-pandas",
    is_flag=True,
    default=False,
    help="Use pandas instead of Polars for data processing (legacy mode)",
)
def select_data(
    input_file: str,
    output: str | None,
    index_col: str,
    col_start: int,
    row_index: int,
    row_start: int,
    sep: str | None,
    sheet: str | None,
    index_separator: str,
    use_pandas: bool,
):
    """Select a subset of data from an input file and save as CSV.

    This command reads data from various file formats (CSV, Excel, TSV) and allows you
    to select specific subsets based on column and row parameters. All indices are zero-based.

    By default, this tool uses Polars for fast data processing. You can use the --use-pandas
    flag to switch to the legacy pandas implementation if needed.

    By default, the output file will be named with the input filename plus '_subset.csv' suffix.
    For example, 'data.xlsx' will output to 'data_subset.csv'.

    For Excel files, you can specify which sheet to read using the --sheet option.
    This accepts either a sheet name (e.g., "Sheet2") or a sheet index (e.g., "1" for the second sheet).

    The --index-col option supports multiple columns for creating concatenated indices.
    Use comma-separated column numbers (e.g., "1,2,4") to combine values from multiple columns.

    Example usage:
    \b
    # Basic usage with defaults (uses Polars, outputs to input_subset.csv)
    btools pre select_data input.csv

    # Select data starting from column 2, using row 1 as headers
    btools pre select_data input.xlsx --col-start 2 --row-index 1

    # Use multiple columns (1,2,4) for index, separated by '#'
    btools pre select_data input.csv --index-col "1,2,4" --index-separator "#"

    # Specify a specific Excel sheet by name
    btools pre select_data workbook.xlsx --sheet "Data Sheet" --output results.csv

    # Specify a specific Excel sheet by index (0-based)
    btools pre select_data workbook.xlsx --sheet 1 --output results.csv

    # Specify custom output file and parameters
    btools pre select_data data.tsv --output subset.csv --index-col 1 --row-start 2

    # Use pandas for legacy compatibility
    btools pre select_data large_data.csv --use-pandas

    # Use custom separator for delimited files
    btools pre select_data data.txt --sep "|" --output result.csv
    """
    try:
        # Convert index_col to int if it's a single number, otherwise keep as string
        parsed_index_col = int(index_col) if "," not in index_col else index_col

        # Choose implementation based on flag (Polars is default, pandas is legacy)
        processor_class = PreSelectData if use_pandas else PreSelectDataPolars

        processor = processor_class(
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
