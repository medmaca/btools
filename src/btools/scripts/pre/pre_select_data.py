#!/usr/bin/env python3
"""Script for selecting a subset of data from input files using pandas."""

from pathlib import Path
from typing import Optional

import pandas as pd


class PreSelectData:
    """Class for selecting subsets of data from input files.

    This class provides functionality to read data files and select specific
    subsets based on column and row parameters. All indices are zero-based.
    """

    def __init__(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        index_col: int = 0,
        data_start_col: int = 1,
        row_index: int = 0,
        row_start: int = 1,
        sep: Optional[str] = None,
    ):
        """Initialize the PreSelectData class.

        Args:
            input_file: Path to the input data file
            output_file: Path to the output CSV file (defaults to input_file with "_subset.csv" suffix)
            index_col: Column to use as row index (default: 0)
            data_start_col: Column from which to start outputting data (default: 1)
            row_index: Row to use as column header (default: 0)
            row_start: Row from which to start outputting data (default: 1)
            sep: Separator/delimiter to use when reading the file (default: None, auto-detect based on file extension)
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file) if output_file else self._generate_output_filename()
        self.index_col = index_col
        self.data_start_col = data_start_col
        self.row_index = row_index
        self.row_start = row_start
        self.sep = sep

    def _generate_output_filename(self) -> Path:
        """Generate output filename based on input filename."""
        return self.input_file.with_stem(self.input_file.stem + "_subset").with_suffix(".csv")

    def _read_data(self) -> pd.DataFrame:
        """Read data from the input file.

        Returns:
            DataFrame containing the input data

        Raises:
            FileNotFoundError: If the input file doesn't exist
            Exception: If there's an error reading the file
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        try:
            # If a custom separator is provided, use it for CSV-like files
            if self.sep is not None:
                return pd.read_csv(self.input_file, sep=self.sep, header=None)

            # Otherwise, auto-detect format based on extension
            file_extension = self.input_file.suffix.lower()

            if file_extension == ".csv":
                return pd.read_csv(self.input_file, header=None)
            elif file_extension in [".xlsx", ".xls"]:
                return pd.read_excel(self.input_file, header=None)
            elif file_extension == ".tsv":
                return pd.read_csv(self.input_file, sep="\t", header=None)
            else:
                # Default to CSV format for unknown extensions
                return pd.read_csv(self.input_file, header=None)

        except Exception as e:
            raise Exception(f"Error reading file {self.input_file}: {str(e)}")

    def _select_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select the specified subset of data from the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame containing the selected subset
        """
        # Validate parameters
        if self.row_index >= len(df):
            raise ValueError(f"row_index ({self.row_index}) is out of bounds. DataFrame has {len(df)} rows.")

        if self.index_col >= len(df.columns):
            raise ValueError(f"index_col ({self.index_col}) is out of bounds. DataFrame has {len(df.columns)} columns.")

        if self.data_start_col >= len(df.columns):
            raise ValueError(
                f"data_start_col ({self.data_start_col}) is out of bounds. DataFrame has {len(df.columns)} columns."
            )

        if self.row_start >= len(df):
            raise ValueError(f"row_start ({self.row_start}) is out of bounds. DataFrame has {len(df)} rows.")

        # Extract column headers from the specified row
        column_headers = df.iloc[self.row_index, self.data_start_col :].tolist()
        index_name = df.iloc[self.row_index, self.index_col]

        # Extract row indices from the specified column
        row_indices = df.iloc[self.row_start :, self.index_col].tolist()

        # Extract the data subset
        data_subset = df.iloc[self.row_start :, self.data_start_col :]

        # Create the new DataFrame with proper headers and indices
        result_df = pd.DataFrame(data=data_subset.values, columns=column_headers, index=row_indices)
        result_df.index.name = index_name  # Set the index name for clarity

        return result_df

    def process(self) -> None:
        """Process the input file and save the selected subset to output file."""
        print(f"Reading data from: {self.input_file}")
        df = self._read_data()

        print(f"Original data shape: {df.shape}")
        print(f"Selecting subset with parameters:")
        print(f"  - Index column: {self.index_col}")
        print(f"  - Data start column: {self.data_start_col}")
        print(f"  - Row index (header): {self.row_index}")
        print(f"  - Row start: {self.row_start}")

        selected_df = self._select_subset(df)

        print(f"Selected data shape: {selected_df.shape}")
        print(f"Saving to: {self.output_file}")

        # Save to CSV file
        print(f"HERE: {selected_df}")
        selected_df.to_csv(self.output_file)

        print("Processing completed successfully!")

    def get_info(self) -> dict:
        """Get information about the data selection parameters.

        Returns:
            Dictionary containing the current configuration
        """
        return {
            "input_file": str(self.input_file),
            "output_file": str(self.output_file),
            "index_col": self.index_col,
            "data_start_col": self.data_start_col,
            "row_index": self.row_index,
            "row_start": self.row_start,
        }


def main():
    """Main function for command-line usage."""
    # Example usage - this can be replaced with click integration
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pre_select_data.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    processor = PreSelectData(input_file=input_file, output_file=output_file)
    processor.process()


if __name__ == "__main__":
    main()
