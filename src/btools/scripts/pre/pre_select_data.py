#!/usr/bin/env python3
"""Script for selecting a subset of data from input files using pandas."""

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]


class PreSelectData:
    """Class for selecting subsets of data from input files.

    This class provides functionality to read data files and select specific
    subsets based on column and row parameters. All indices are zero-based.
    """

    def __init__(
        self,
        input_file: str,
        output_file: str | None = None,
        index_col: int = 0,
        data_start_col: int = 1,
        row_index: int = 0,
        row_start: int = 1,
        sep: str | None = None,
        sheet: str | None = None,
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
            sheet: Sheet name or number to read from Excel files (default: None, uses first sheet)
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file) if output_file else self._generate_output_filename()
        self.index_col = index_col
        self.data_start_col = data_start_col
        self.row_index = row_index
        self.row_start = row_start
        self.sep = sep
        self.sheet = sheet

    def _generate_output_filename(self) -> Path:
        """Generate output filename based on input filename."""
        return self.input_file.with_stem(self.input_file.stem + "_subset").with_suffix(".csv")

    def _read_data(self) -> pd.DataFrame:  # type: ignore[type-arg]
        """Read data from the input file.

        Supports multiple file formats:
        - CSV files (.csv): Uses pandas.read_csv with auto-detected or custom separator
        - Excel files (.xlsx, .xls): Uses pandas.read_excel with optional sheet selection
        - TSV files (.tsv): Uses pandas.read_csv with tab separator
        - Other formats: Defaults to CSV format

        For Excel files, the sheet parameter controls which sheet to read:
        - If sheet is None: reads the first sheet (index 0)
        - If sheet is a string: reads the sheet with that name
        - If sheet is convertible to int: reads the sheet at that index

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
                if self.sep == "\\t":
                    print(f"\tReading TSV file: {self.input_file}")
                    return pd.read_csv(self.input_file, sep="\t", header=None)  # type: ignore[call-overload]
                print(f"\tReading file with custom separator:({self.sep})")
                return pd.read_csv(self.input_file, sep=self.sep, header=None)  # type: ignore[call-overload]

            # Otherwise, auto-detect format based on extension
            file_extension = self.input_file.suffix.lower()

            if file_extension == ".csv":
                return pd.read_csv(self.input_file, header=None)  # type: ignore[call-overload]
            elif file_extension in [".xlsx", ".xls"]:
                sheet_name = self.sheet if self.sheet is not None else 0
                if self.sheet is not None:
                    print(f"\tReading Excel file: {self.input_file}, sheet: {self.sheet}")
                else:
                    print(f"\tReading Excel file: {self.input_file}")
                return pd.read_excel(self.input_file, sheet_name=sheet_name, header=None)  # type: ignore[call-overload]
            elif file_extension == ".tsv":
                return pd.read_csv(self.input_file, sep="\t", header=None)  # type: ignore[call-overload]
            else:
                # Default to CSV format for unknown extensions
                return pd.read_csv(self.input_file, header=None)  # type: ignore[call-overload]

        except Exception as e:
            raise OSError(f"Error reading file {self.input_file}: {str(e)}") from e

    def _select_subset(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[type-arg]
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
        column_headers: list[Any] = df.iloc[self.row_index, self.data_start_col :].tolist()  # type: ignore[attr-defined]
        index_name: Any = df.iloc[self.row_index, self.index_col]  # type: ignore[assignment]

        # Extract row indices from the specified column
        row_indices: list[Any] = df.iloc[self.row_start :, self.index_col].tolist()  # type: ignore[attr-defined]

        # Extract the data subset
        data_subset: Any = df.iloc[self.row_start :, self.data_start_col :]  # type: ignore[assignment]

        # Create the new DataFrame with proper headers and indices
        result_df = pd.DataFrame(data=data_subset.values, columns=column_headers, index=row_indices)  # type: ignore[attr-defined,arg-type]
        result_df.index.name = index_name  # Set the index name for clarity

        return result_df

    def process(self) -> None:
        """Process the input file and save the selected subset to output file."""
        print(f"Reading data from: {self.input_file}")
        if self.sheet is not None:
            print(f"  Sheet: {self.sheet}")
        if self.sep is not None:
            print(f"  Separator: {repr(self.sep)}")

        df = self._read_data()

        print(f"Original data shape: {df.shape}")
        print("Selecting subset with parameters:")
        print(f"  - Index column: {self.index_col}")
        print(f"  - Data start column: {self.data_start_col}")
        print(f"  - Row index (header): {self.row_index}")
        print(f"  - Row start: {self.row_start}")

        selected_df = self._select_subset(df)

        print(f"Selected data shape: {selected_df.shape}")
        print(f"Saving to: {self.output_file}")

        # Save to CSV file
        selected_df.to_csv(self.output_file)  # type: ignore[call-overload]

        print("Processing completed successfully!")

    def get_info(self) -> dict[str, str | int | None]:
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
            "sep": self.sep,
            "sheet": self.sheet,
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
