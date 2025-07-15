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
        sheet: str | None = None,
        transpose: bool = False,
        index_separator: str = "#",
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
            sheet: Sheet name or number to read from Excel files (default: None, uses first sheet)
            transpose: Whether to transpose the data before applying row/column selections (default: False)
            index_separator: Separator to use when concatenating multiple index columns (default: "#")
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file) if output_file else self._generate_output_filename()
        self.index_col = index_col
        self.col_start = col_start
        self.row_index = row_index
        self.row_start = row_start
        self.sep = sep
        self.sheet = sheet
        self.transpose = transpose
        self.index_separator = index_separator

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

    def _generate_output_filename(self) -> Path:
        """Generate output filename based on input filename.

        Note: The range suffix will be added later in the process method
        when we know the actual dimensions of the subset.

        If GZIP_OUT environment variable is True, adds .gz extension.
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

        base_name = self.input_file.with_name(base_stem + "_subset").with_suffix(".csv")
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
        - Other formats: Defaults to CSV format

        Automatically handles gzip-compressed files by detecting .gz extension.

        For Excel files, the sheet parameter controls which sheet to read:
        - If sheet is None: reads the first sheet (index 0)
        - If sheet is a string: reads the sheet with that name
        - If sheet is convertible to int: reads the sheet at that index

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

    def _select_subset(self, df: pl.DataFrame) -> pl.DataFrame:
        """Select the specified subset of data from the DataFrame using Polars operations.

        Args:
            df: Input Polars DataFrame

        Returns:
            Polars DataFrame containing the selected subset
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

        # Extract column headers from the specified row
        header_row = df.slice(self.row_index, 1)

        # Process multiple column ranges
        selected_columns: list[str] = []
        column_headers: list[str] = []

        for col_start, col_end in col_ranges:
            actual_col_end = col_end if col_end is not None else df.width

            # Collect column indices for this range
            for col_idx in range(col_start, actual_col_end):
                selected_columns.append(f"column_{col_idx + 1}")
                # Get header name for this column
                header_name = str(header_row.select(pl.col(f"column_{col_idx + 1}")).item())
                column_headers.append(header_name)

        # Extract index name(s) - concatenate if multiple columns
        if len(index_columns_list) == 1:
            index_name = str(header_row.select(pl.col(f"column_{index_columns_list[0] + 1}")).item())
        else:
            index_parts = [str(header_row.select(pl.col(f"column_{col + 1}")).item()) for col in index_columns_list]
            index_name = self.index_separator.join(index_parts)

        # Process multiple row ranges and concatenate them
        all_data_rows: list[pl.DataFrame] = []
        all_row_indices: list[str] = []

        for row_start, row_end in row_ranges:
            actual_row_end = row_end if row_end is not None else df.height
            data_length = actual_row_end - row_start
            data_rows = df.slice(row_start, data_length)

            # Extract row indices from the specified column(s)
            if len(index_columns_list) == 1:
                index_col_name = f"column_{index_columns_list[0] + 1}"
                row_indices = [str(val) for val in data_rows.select(pl.col(index_col_name)).to_series().to_list()]
            else:
                # Create concatenated index from multiple columns
                index_cols = [f"column_{col + 1}" for col in index_columns_list]
                row_indices: list[str] = []
                for row in data_rows.select(index_cols).iter_rows():
                    row_indices.append(self.index_separator.join(str(val) for val in row))

            # Select the columns for this row range
            data_subset = data_rows.select(selected_columns)

            all_data_rows.append(data_subset)
            all_row_indices.extend(row_indices)

        # Concatenate all row ranges
        final_data: pl.DataFrame = all_data_rows[0] if len(all_data_rows) == 1 else pl.concat(all_data_rows, how="vertical")

        # Rename columns to use the headers we extracted, making sure column names are unique
        if len(selected_columns) == len(column_headers):
            # Make column headers unique by adding suffixes to duplicates
            unique_headers = self._make_column_names_unique(column_headers)
            column_mapping: dict[str, str] = {
                old_name: new_name for old_name, new_name in zip(selected_columns, unique_headers, strict=True)
            }
            final_data = final_data.rename(column_mapping)

        # Add the index as a column
        result_df: pl.DataFrame = final_data.with_columns(pl.Series(name=index_name, values=all_row_indices))

        # Move the index column to the front
        cols = [index_name] + [col for col in result_df.columns if col != index_name]
        result_df = result_df.select(cols)

        return result_df

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
            # Use Polars' transpose method with include_header=False to avoid duplicate column name issues
            # This ensures we get generic column names instead of using potentially duplicate data values
            transposed_df = df.transpose(include_header=False)

            # Rename columns to follow the generic naming convention (column_1, column_2, etc.)
            # This ensures consistency with the rest of the code that expects 1-based indexing
            column_mapping = {old_name: f"column_{i + 1}" for i, old_name in enumerate(transposed_df.columns)}
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

        selected_df = self._select_subset(df)
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

        # Save to CSV file using Polars (with gzip support)
        self._write_csv_file(selected_df, updated_output_file)

        print("Processing completed successfully!")

    def _write_csv_file(self, df: pl.DataFrame, output_file: Path) -> None:
        """Write DataFrame to CSV file, with optional gzip compression.

        Args:
            df: DataFrame to write
            output_file: Path where to write the file
        """
        if output_file.suffix.lower() == ".gz":
            # For gzip files, we need to write to a temporary file first, then compress
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_file:
                temp_path = Path(temp_file.name)

            try:
                # Write to temporary file
                df.write_csv(str(temp_path))

                # Compress to final destination
                with open(temp_path, "rb") as f_in, gzip.open(output_file, "wb") as f_out:
                    f_out.write(f_in.read())
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()
        else:
            # Regular CSV file
            df.write_csv(str(output_file))

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

    def _make_column_names_unique(self, column_headers: list[str]) -> list[str]:
        """Make column names unique by adding suffixes to duplicates.

        Args:
            column_headers: List of column header names that may contain duplicates

        Returns:
            List of unique column names with suffixes added to duplicates
        """
        unique_headers: list[str] = []
        seen_names: dict[str, int] = {}

        for header in column_headers:
            if header not in seen_names:
                seen_names[header] = 0
                unique_headers.append(header)
            else:
                seen_names[header] += 1
                unique_headers.append(f"{header}_{seen_names[header]}")

        return unique_headers


def main():
    """Main function for command-line usage."""
    # Example usage - this can be replaced with click integration
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pre_select_data.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    processor = PreSelectDataPolars(input_file=input_file, output_file=output_file)
    processor.process()


if __name__ == "__main__":
    main()
