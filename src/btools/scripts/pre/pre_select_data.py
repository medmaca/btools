#!/usr/bin/env python3
"""Script for selecting a subset of data from input files using Polars."""

import gzip
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _get_true_file_extension(file_path: Path) -> str:
    """Get the true file extension, handling .gz files properly.

    For example:
    - 'data.csv.gz' returns '.csv'
    - 'data.tsv.gz' returns '.tsv'
    - 'data.xlsx.gz' returns '.xlsx'
    - 'data.csv' returns '.csv'

    Args:
        file_path: Path to the file

    Returns:
        The true file extension (without .gz)
    """
    suffixes = file_path.suffixes
    if len(suffixes) >= 2 and suffixes[-1].lower() == ".gz":
        return suffixes[-2].lower()
    elif len(suffixes) >= 1:
        return suffixes[-1].lower()
    else:
        return ""


def _is_gzipped(file_path: Path) -> bool:
    """Check if a file is gzip compressed.

    Args:
        file_path: Path to the file

    Returns:
        True if the file is gzip compressed
    """
    return file_path.suffix.lower() == ".gz"


def _should_write_gzipped() -> bool:
    """Check if output files should be gzip compressed based on environment variable.

    Returns:
        True if GZIP_OUT environment variable is set to True/true/1
    """
    gzip_out = os.getenv("GZIP_OUT", "False").lower()
    return gzip_out in ("true", "1", "yes")


class PreSelectDataPolars:
    """Class for selecting subsets of data from input files using Polars for better performance.

    This class provides functionality to read data files and select specific
    subsets based on column and row parameters. All indices are zero-based.
    Uses Polars for faster data processing compared to pandas.
    """

    def __init__(
        self,
        input_file: str,
        output_file: str | None = None,
        index_col: int | str = 0,
        col_start: int | str = 1,
        row_index: int = 0,
        row_start: int | str = 1,
        sep: str | None = None,
        sheet: str | int | None = None,
        transpose: bool = False,
        index_separator: str = "#",
        parquet_out: bool = False,
    ):
        """Initialize the PreSelectDataPolars class.

        Args:
            input_file: Path to the input data file
            output_file: Path to the output CSV file (defaults to input_file with "_subset.csv" suffix)
            index_col: Column(s) to use as row index. Can be a single integer (e.g., 0) or
                      comma-separated string of integers (e.g., "1,2,4") for concatenated index (default: 0)
            col_start: Column from which to start outputting data. Can be single integer (e.g., 1),
                      single range string (e.g., "1:50") to output columns 1-50, or multiple ranges
                      (e.g., "1:4,8:10,12:24") to select and concatenate multiple column ranges (default: 1)
            row_index: Row to use as column header (default: 0)
            row_start: Row from which to start outputting data. Can be single integer (e.g., 1),
                      single range string (e.g., "1:100") to output rows 1-100, or multiple ranges
                      (e.g., "1:10,20:30") to select and concatenate multiple row ranges (default: 1)
            sep: Separator/delimiter to use when reading the file (default: None, auto-detect based on file extension)
            sheet: Sheet name (string) or sheet index (integer) to read from Excel files (default: None, uses first sheet)
            transpose: Whether to transpose the data before applying row/column selections (default: False)
            index_separator: Separator to use when concatenating multiple index columns (default: "#")
            parquet_out: If True, output will be written as a Parquet file (default: False)
        """
        self.input_file = Path(input_file)
        self.index_col = index_col
        self.col_start = col_start
        self.row_index = row_index
        self.row_start = row_start
        self.sep = sep
        self.sheet = sheet
        self.transpose = transpose
        self.index_separator = index_separator
        self.parquet_out = parquet_out
        self.output_file = Path(output_file) if output_file else self._generate_output_filename()

    def _parse_index_columns(self) -> list[int]:
        """Parse index_col parameter into a list of integers.

        Returns:
            List of column indices to use for row index
        """
        if isinstance(self.index_col, int):
            return [self.index_col]
        else:  # str case
            try:
                return [int(x.strip()) for x in self.index_col.split(",")]
            except ValueError as e:
                raise ValueError(
                    f"Invalid index_col format '{self.index_col}'. Use comma-separated integers like '1,2,4'"
                ) from e

    def _parse_multi_range_parameter(self, param: int | str, param_name: str) -> list[tuple[int, int | None]]:
        """Parse a multi-range parameter that can be an int, single range, or multiple ranges.

        Args:
            param: Either an integer, single range "start:end", or multiple ranges "1:4,8:10,12:24"
            param_name: Name of the parameter for error messages

        Returns:
            List of (start, end) tuples where end is None if only start is specified

        Raises:
            ValueError: If the parameter format is invalid
        """
        if isinstance(param, int):
            return [(param, None)]

        # param is str due to type union
        try:
            ranges: list[tuple[int, int | None]] = []
            # Split by comma to get individual range specifications
            range_specs = [spec.strip() for spec in param.split(",")]

            for spec in range_specs:
                if ":" in spec:
                    # Range format "start:end"
                    parts = [p.strip() for p in spec.split(":")]
                    if len(parts) != 2:
                        raise ValueError(f"Invalid {param_name} range format '{spec}'. Use 'start:end' format")
                    start, end = int(parts[0]), int(parts[1])
                    if start > end:
                        raise ValueError(f"Invalid {param_name} range '{spec}': start must be <= end")
                    # For polars, we return end + 1 to include the end in the slice
                    ranges.append((start, end + 1))
                else:
                    # Single number - start from this position to end
                    ranges.append((int(spec), None))

            return ranges
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(
                    f"Invalid {param_name} format '{param}'. Use integer, 'start:end', or 'start1:end1,start2:end2' format"
                ) from e
            raise

    def _parse_col_start(self) -> list[tuple[int, int | None]]:
        """Parse col_start parameter into list of start and end positions."""
        return self._parse_multi_range_parameter(self.col_start, "col_start")

    def _parse_row_start(self) -> list[tuple[int, int | None]]:
        """Parse row_start parameter into list of start and end positions."""
        return self._parse_multi_range_parameter(self.row_start, "row_start")

    def _generate_range_suffix(
        self, num_rows: int, num_cols: int, has_multi_col_ranges: bool, has_multi_row_ranges: bool
    ) -> str:
        """Generate suffix for filename based on the number of rows and columns in subset.

        Args:
            num_rows: Number of rows in the result
            num_cols: Number of columns in the result
            has_multi_col_ranges: Whether multiple column ranges were used
            has_multi_row_ranges: Whether multiple row ranges were used
        """
        if has_multi_col_ranges and has_multi_row_ranges:
            return f"_row{num_rows}_col{num_cols}_msc_msr"
        elif has_multi_col_ranges:
            return f"_row{num_rows}_col{num_cols}_msc"
        elif has_multi_row_ranges:
            return f"_row{num_rows}_col{num_cols}_msr"
        else:
            return f"_row{num_rows}_col{num_cols}"

    def _sanitize_sheet_name(self, sheet_name: str) -> str:
        """Sanitize sheet name for use in filename by replacing spaces with hyphens.

        Args:
            sheet_name: Original sheet name

        Returns:
            Sanitized sheet name suitable for filename
        """
        return sheet_name.replace(" ", "-")

    def _generate_output_filename(self) -> Path:
        """Generate output filename based on input filename.

        Note: The range suffix will be added later in the process method
        when we know the actual dimensions of the subset.

        If GZIP_OUT environment variable is True, adds .gz extension.
        For Excel files, appends sheet information after "_sheet_".
        """
        # Get the true base name, removing any file extensions (.csv, .tsv, .gz, etc.)
        # For example: "file.tsv.gz" -> "file", "file.csv" -> "file"
        if _is_gzipped(self.input_file):
            # For gzipped files, remove both .gz and the inner extension
            base_stem = self.input_file.name
            for suffix in self.input_file.suffixes:
                base_stem = base_stem.replace(suffix, "")
        else:
            # For non-gzipped files, just remove the extension
            base_stem = self.input_file.stem

        # Add sheet suffix for Excel files
        sheet_suffix = ""
        true_extension = _get_true_file_extension(self.input_file)
        if true_extension in [".xlsx", ".xls"]:
            if self.sheet is not None:
                # Use provided sheet name/number
                sanitized_sheet = self._sanitize_sheet_name(str(self.sheet))
                sheet_suffix = f"_sheet_{sanitized_sheet}"
            else:
                # Default to first sheet (index 0)
                sheet_suffix = "_sheet_0"

        # Add transpose suffix if transpose is enabled
        transpose_suffix = "_t" if self.transpose else ""
        if hasattr(self, "parquet_out") and self.parquet_out:
            base_name = self.input_file.with_name(base_stem + "_subset" + sheet_suffix + transpose_suffix).with_suffix(
                ".parquet"
            )
            return base_name
        base_name = self.input_file.with_name(base_stem + "_subset" + sheet_suffix + transpose_suffix).with_suffix(".csv")
        if _should_write_gzipped():
            return base_name.with_suffix(".csv.gz")
        return base_name

    def _update_output_filename_with_range(
        self, output_file: Path, num_rows: int, num_cols: int, has_multi_col_ranges: bool, has_multi_row_ranges: bool
    ) -> Path:
        """Update output filename to include range information."""
        range_suffix = self._generate_range_suffix(num_rows, num_cols, has_multi_col_ranges, has_multi_row_ranges)

        # Handle .csv.gz files specially
        if output_file.suffixes == [".csv", ".gz"]:
            # For .csv.gz files, we want: name_subset_rowX_colY.csv.gz
            # Current stem is: name_subset.csv
            # We want to insert range before .csv: name_subset_rowX_colY.csv
            base_stem = output_file.stem[:-4]  # Remove '.csv' from stem
            new_stem = base_stem + range_suffix + ".csv"
            return output_file.with_stem(new_stem)
        else:
            # For regular files, just append to stem
            stem = output_file.stem
            new_stem = stem + range_suffix
            return output_file.with_stem(new_stem)

    def _read_data(self) -> pl.DataFrame:
        """Read data from the input file using Polars.

        Supports multiple file formats:
        - CSV files (.csv, .csv.gz): Uses polars.read_csv with auto-detected or custom separator
        - Excel files (.xlsx, .xls): Uses polars.read_excel (note: .gz not supported for Excel)
        - TSV files (.tsv, .tsv.gz): Uses polars.read_csv with tab separator
        - Parquet files (.parquet): Treats first row as header, rest as data
        - Other formats: Defaults to CSV format

        Automatically handles gzip-compressed files by detecting .gz extension.

        For Excel files, the sheet parameter controls which sheet to read:
        - If sheet is None: reads the first sheet (index 0)
        - If sheet is a string: reads the sheet with that name
        - If sheet is an integer: reads the sheet at that index

        Returns:
            Polars DataFrame containing the input data

        Raises:
            FileNotFoundError: If the input file doesn't exist
            Exception: If there's an error reading the file
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        try:
            df: pl.DataFrame
            is_gzipped = _is_gzipped(self.input_file)
            true_extension = _get_true_file_extension(self.input_file)
            # Parquet input support
            if true_extension == ".parquet":
                print(f"\tReading Parquet file: {self.input_file}")
                df = pl.read_parquet(self.input_file)
                # Treat first row as header, rest as data
                if df.height < 1:
                    raise ValueError("Parquet file is empty or missing header row.")

                data_rows = df  # For Parquet, we treat the whole DataFrame as data
                # Rename columns to generic names for consistency (1-based indexing)
                num_cols = data_rows.width
                current_columns = data_rows.columns
                expected_columns = [f"column_{i + 1}" for i in range(num_cols)]
                column_mapping = {current_columns[i]: expected_columns[i] for i in range(num_cols)}
                data_rows = data_rows.rename(column_mapping)
                return data_rows

            # If a custom separator is provided, use it for CSV-like files
            if self.sep is not None:
                if self.sep == "\\t":
                    print(f"\tReading {'gzipped ' if is_gzipped else ''}TSV file: {self.input_file}")
                    df = pl.read_csv(self.input_file, separator="\t", has_header=False)
                else:
                    prefix = "gzipped " if is_gzipped else ""
                    print(f"\tReading {prefix}file with custom separator:({self.sep})")
                    df = pl.read_csv(self.input_file, separator=self.sep, has_header=False)
            else:
                # Auto-detect format based on true extension (ignoring .gz)
                if true_extension == ".csv":
                    df = pl.read_csv(self.input_file, has_header=False)
                elif true_extension in [".xlsx", ".xls"]:
                    if is_gzipped:
                        raise ValueError(f"Gzipped Excel files are not supported: {self.input_file}")

                    # Use Polars read_excel function
                    if self.sheet is not None:
                        print(f"\tReading Excel file: {self.input_file}, sheet: {self.sheet}")
                        # Handle both string and integer sheet references
                        if isinstance(self.sheet, int):
                            df = pl.read_excel(self.input_file, sheet_id=self.sheet, has_header=False)
                        else:
                            df = pl.read_excel(self.input_file, sheet_name=self.sheet, has_header=False)
                    else:
                        print(f"\tReading Excel file: {self.input_file}")
                        df = pl.read_excel(self.input_file, has_header=False)
                elif true_extension == ".tsv":
                    df = pl.read_csv(self.input_file, separator="\t", has_header=False)
                else:
                    # Default to CSV format for unknown extensions
                    df = pl.read_csv(self.input_file, has_header=False)

            # Rename columns to generic names for consistency (1-based indexing)
            # Check if columns are already properly named or need renaming
            num_cols = df.width
            current_columns = df.columns

            # If columns are already named column_1, column_2, etc., no need to rename
            expected_columns = [f"column_{i + 1}" for i in range(num_cols)]
            if current_columns != expected_columns:
                # Create mapping from current names to expected names
                column_mapping = {current_columns[i]: f"column_{i + 1}" for i in range(num_cols)}
                df = df.rename(column_mapping)

            return df

        except Exception as e:
            raise OSError(f"Error reading file {self.input_file}: {str(e)}") from e

    def _select_subset(self, df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
        """Select the specified subset of data from the DataFrame using Polars operations.

        Args:
            df: Input Polars DataFrame

        Returns:
            Tuple of (Polars DataFrame with generic column names, list of original headers)
        """
        # Parse range parameters
        index_columns_list = self._parse_index_columns()
        col_ranges = self._parse_col_start()
        row_ranges = self._parse_row_start()

        # Validate index columns
        for col_idx in index_columns_list:
            if col_idx >= df.width:
                raise ValueError(f"index_col ({col_idx}) is out of bounds. DataFrame has {df.width} columns.")

        if self.row_index >= df.height:
            raise ValueError(f"row_index ({self.row_index}) is out of bounds. DataFrame has {df.height} rows.")

        # Validate column ranges
        for col_start, col_end in col_ranges:
            if col_start >= df.width:
                raise ValueError(f"col_start ({col_start}) is out of bounds. DataFrame has {df.width} columns.")
            if col_end is not None and col_end > df.width:
                raise ValueError(f"col_end ({col_end}) is out of bounds. DataFrame has {df.width} columns.")

        # Validate row ranges
        for row_start, row_end in row_ranges:
            if row_start >= df.height:
                raise ValueError(f"row_start ({row_start}) is out of bounds. DataFrame has {df.height} rows.")
            if row_end is not None and row_end > df.height:
                raise ValueError(f"row_end ({row_end}) is out of bounds. DataFrame has {df.height} rows.")

        # Extract column headers from the specified row (optimized for large datasets)
        header_row = df.slice(self.row_index, 1)

        # Process multiple column ranges
        selected_columns: list[str] = []
        column_headers: list[str] = []

        for col_start, col_end in col_ranges:
            actual_col_end = col_end if col_end is not None else df.width

            # Collect column indices for this range
            range_columns = [f"column_{col_idx + 1}" for col_idx in range(col_start, actual_col_end)]
            selected_columns.extend(range_columns)

            # Extract all headers for this range in one operation (much faster)
            if range_columns:
                header_values = header_row.select(range_columns).row(0)
                column_headers.extend(str(val) for val in header_values)

        # Extract index name(s) - concatenate if multiple columns (optimized)
        if len(index_columns_list) == 1:
            index_name = str(header_row.select(pl.col(f"column_{index_columns_list[0] + 1}")).item())
        else:
            index_cols = [f"column_{col + 1}" for col in index_columns_list]
            index_values = header_row.select(index_cols).row(0)
            index_name = self.index_separator.join(str(val) for val in index_values)

        # Process multiple row ranges and concatenate them
        all_data_rows: list[pl.DataFrame] = []
        all_row_indices: list[str] = []

        for row_start, row_end in row_ranges:
            actual_row_end = row_end if row_end is not None else df.height
            data_length = actual_row_end - row_start
            data_rows = df.slice(row_start, data_length)

            # Extract row indices from the specified column(s) - optimized for large datasets
            if len(index_columns_list) == 1:
                index_col_name = f"column_{index_columns_list[0] + 1}"
                # Use vectorized operation instead of converting to list
                row_indices = [str(val) for val in data_rows.select(pl.col(index_col_name)).to_series()]
            else:
                # Create concatenated index from multiple columns - vectorized approach
                index_cols = [f"column_{col + 1}" for col in index_columns_list]
                # Get all rows at once instead of iterating
                index_data = data_rows.select(index_cols)
                row_indices = [self.index_separator.join(str(val) for val in row_tuple) for row_tuple in index_data.rows()]

            # Select the columns for this row range
            data_subset = data_rows.select(selected_columns)

            all_data_rows.append(data_subset)
            all_row_indices.extend(row_indices)

        # Concatenate all row ranges
        final_data: pl.DataFrame = all_data_rows[0] if len(all_data_rows) == 1 else pl.concat(all_data_rows, how="vertical")

        # Keep generic column names for the DataFrame, but track original headers separately
        # This avoids Polars transpose issues with duplicate column names
        generic_column_names = [f"col_{i + 1}" for i in range(len(selected_columns))]
        column_mapping: dict[str, str] = {
            old_name: new_name for old_name, new_name in zip(selected_columns, generic_column_names, strict=True)
        }
        final_data = final_data.rename(column_mapping)

        # Add the index as a column with generic name
        result_df: pl.DataFrame = final_data.with_columns(pl.Series(name="index_col", values=all_row_indices))

        # Move the index column to the front
        cols = ["index_col"] + [col for col in result_df.columns if col != "index_col"]
        result_df = result_df.select(cols)

        # Prepare the original headers list (index name + column headers)
        original_headers = [index_name] + column_headers

        return result_df, original_headers

    def _transpose_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transpose the DataFrame using Polars' built-in transpose method.

        This converts the DataFrame so that rows become columns and columns become rows.
        After transposition, we ensure the column names follow the standard generic naming
        convention used by the rest of the code.

        Args:
            df: Input Polars DataFrame

        Returns:
            Transposed Polars DataFrame with generic column names
        """
        try:
            print(f"  Starting transpose of shape {df.shape}...")

            # Use Polars' transpose method with include_header=False to avoid duplicate column name issues
            # This ensures we get generic column names instead of using potentially duplicate data values
            transposed_df = df.transpose(include_header=False)

            print(f"  Transpose completed. New shape: {transposed_df.shape}")

            # Rename columns to follow the generic naming convention (column_1, column_2, etc.)
            # This ensures consistency with the rest of the code that expects 1-based indexing
            # Use lazy evaluation for better performance with large datasets
            num_cols = transposed_df.width
            if num_cols > 1000:
                print(f"  Renaming {num_cols} columns (this may take a moment for large datasets)...")

            column_mapping = {f"column_{i}": f"column_{i + 1}" for i in range(num_cols)}
            transposed_df = transposed_df.rename(column_mapping)

            return transposed_df

        except Exception as e:
            raise ValueError(f"Error during transpose operation: {e}") from e

    def process(self) -> None:
        """Process the input file and save the selected subset to output file."""
        print(f"Reading data from: {self.input_file}")
        if self.sheet is not None:
            print(f"  Sheet: {self.sheet}")
        if self.sep is not None:
            print(f"  Separator: {repr(self.sep)}")

        df = self._read_data()

        # Apply transpose if requested
        if self.transpose:
            print("Transposing data...")
            df = self._transpose_data(df)

        print(f"Original data shape: {df.shape}")
        print("Selecting subset with parameters:")
        print(f"  - Index column: {self.index_col}")
        print(f"  - Column start: {self.col_start}")
        print(f"  - Row index (header): {self.row_index}")
        print(f"  - Row start: {self.row_start}")
        if self.transpose:
            print("  - Data was transposed before selection")

        # Add progress indication for large datasets
        if df.shape[0] > 10000 or df.shape[1] > 1000:
            print("Processing large dataset - this may take a moment...")

        selected_df, original_headers = self._select_subset(df)
        shape_a, shape_b = selected_df.shape

        # Update output filename to include range information
        # Subtract 1 from columns because the index column is included
        actual_data_cols = shape_b - 1
        col_ranges = self._parse_col_start()
        row_ranges = self._parse_row_start()

        has_multi_col_ranges = len(col_ranges) > 1
        has_multi_row_ranges = len(row_ranges) > 1

        # Check if any ranges are specified (not just single start positions)
        has_range_specified = any(col_end is not None for _, col_end in col_ranges) or any(
            row_end is not None for _, row_end in row_ranges
        )

        if has_range_specified or has_multi_col_ranges or has_multi_row_ranges:
            updated_output_file = self._update_output_filename_with_range(
                self.output_file, shape_a, actual_data_cols, has_multi_col_ranges, has_multi_row_ranges
            )
        else:
            updated_output_file = self.output_file

        print(f"Selected data shape: {shape_a},{actual_data_cols}")
        print(f"Saving to: {updated_output_file}")

        # Save to Parquet or CSV file
        if self.parquet_out or str(updated_output_file).endswith(".parquet"):
            self._write_parquet_file(selected_df, updated_output_file, original_headers)
        else:
            self._write_csv_file(selected_df, updated_output_file, original_headers)

        print("Processing completed successfully!")

    def _write_parquet_file(self, df: pl.DataFrame, output_file: Path, original_headers: list[str]) -> None:
        """Write DataFrame to Parquet file, with header row as first row and generic column names.

        Args:
            df: DataFrame to write (with generic column names)
            output_file: Path where to write the file
            original_headers: List of original column headers to write as first row
        """
        # Create a header row DataFrame with the original headers
        header_row = pl.DataFrame([original_headers], schema=df.schema, orient="row")
        # Concatenate header row with data
        df_with_headers = pl.concat([header_row, df], how="vertical")
        # Write with generic column names
        df_with_headers.write_parquet(str(output_file), compression="snappy")

    def _write_csv_file(self, df: pl.DataFrame, output_file: Path, original_headers: list[str]) -> None:
        """Write DataFrame to CSV file, with optional gzip compression.

        Inserts the original headers as the first row and uses Polars' native CSV writing.
        For gzip files, writes directly to gzip binary stream.

        Args:
            df: DataFrame to write (with generic column names)
            output_file: Path where to write the file
            original_headers: List of original column headers to write as first row
        """
        # Create a header row DataFrame with the original headers
        header_row = pl.DataFrame([original_headers], schema=df.schema, orient="row")

        # Concatenate header row with data
        df_with_headers = pl.concat([header_row, df], how="vertical")

        if output_file.suffix.lower() == ".gz":
            # For gzip files, write directly to gzip binary stream

            with gzip.open(output_file, "wb") as f:
                df_with_headers.write_csv(f, include_header=False)  # type: ignore
        else:
            # Regular CSV file - Polars handles this efficiently
            df_with_headers.write_csv(str(output_file), include_header=False)

    def get_info(self) -> dict[str, str | int | None | bool]:
        """Get information about the data selection parameters.

        Returns:
            Dictionary containing the current configuration
        """
        return {
            "input_file": str(self.input_file),
            "output_file": str(self.output_file),
            "index_col": self.index_col,
            "col_start": self.col_start,
            "row_index": self.row_index,
            "row_start": self.row_start,
            "sep": self.sep,
            "sheet": self.sheet,
            "transpose": self.transpose,
            "index_separator": self.index_separator,
        }


def main():
    """Main function for command-line usage."""
    # Example usage - this can be replaced with click integration
    import argparse

    parser = argparse.ArgumentParser(description="Select subset of data from input files using Polars.")
    parser.add_argument("input_file", help="Path to input data file")
    parser.add_argument("output_file", nargs="?", default=None, help="Path to output file")
    parser.add_argument("-p", "--parquet", action="store_true", help="Write output as Parquet file")
    # ...existing code for other arguments...
    args = parser.parse_args()

    processor = PreSelectDataPolars(
        input_file=args.input_file,
        output_file=args.output_file,
        parquet_out=args.parquet,
    )
    processor.process()


if __name__ == "__main__":
    main()
