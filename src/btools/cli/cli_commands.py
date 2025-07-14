"""Defines all CLI commands for the template-cli application."""

import click

from btools.scripts.pre.pre_select_data import PreSelectData
from btools.scripts.pre.rand_numbers import generate_random_numbers


# Create a group called "pre"
@click.group("pre")
def pre_group():
    """Pre-processing commands."""


# Random number generation command
@pre_group.command("random_num")
@click.option("--count", "-c", default=10, help="Number of random numbers to generate.")
def rand_gen(count: int):
    """Generate a list of random numbers using the generate_random_numbers function."""
    generate_random_numbers(count)
    generate_random_numbers(count)


# Data selection command
@pre_group.command("select_data")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output CSV file path (defaults to input file with '_subset.csv' suffix)")
@click.option("--index-col", default=0, help="Column to use as row index (zero-based, default: 0)")
@click.option("--data-start-col", default=1, help="Column from which to start outputting data (zero-based, default: 1)")
@click.option("--row-index", default=0, help="Row to use as column header (zero-based, default: 0)")
@click.option("--row-start", default=1, help="Row from which to start outputting data (zero-based, default: 1)")
@click.option("--sep", help="Separator/delimiter to use when reading the file (overrides auto-detection)")
def select_data(
    input_file: str,
    output: str | None,
    index_col: int,
    data_start_col: int,
    row_index: int,
    row_start: int,
    sep: str | None,
):
    """Select a subset of data from an input file and save as CSV.

    This command reads data from various file formats (CSV, Excel, TSV) and allows you
    to select specific subsets based on column and row parameters. All indices are zero-based.

    By default, the output file will be named with the input filename plus '_subset.csv' suffix.
    For example, 'data.xlsx' will output to 'data_subset.csv'.

    Example usage:
    \b
    # Basic usage with defaults (outputs to input_subset.csv)
    btools pre select_data input.csv

    # Select data starting from column 2, using row 1 as headers
    btools pre select_data input.xlsx --data-start-col 2 --row-index 1

    # Specify custom output file and parameters
    btools pre select_data data.tsv --output subset.csv --index-col 1 --row-start 2

    # Use custom separator for delimited files
    btools pre select_data data.txt --sep "|" --output result.csv
    """
    try:
        processor = PreSelectData(
            input_file=input_file,
            output_file=output,
            index_col=index_col,
            data_start_col=data_start_col,
            row_index=row_index,
            row_start=row_start,
            sep=sep,
        )
        processor.process()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e
