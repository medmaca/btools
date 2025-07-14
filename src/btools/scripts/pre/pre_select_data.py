#!/usr/bin/env python3
"""Script for selecting a subset of data from input files using pandas."""

import gzip
import os
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
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
        index_separator: str = "#",
    ):
        """Initialize the PreSelectDataPolars class.

        Args:
            input_file: Path to the input data file
            output_file: Path to the output CSV file (defaults to input_file with "_subset.csv" suffix)
            index_col: Column(s) to use as row index. Can be a single integer (e.g., 0) or
                      comma-separated string of integers (e.g., "1,2,4") for concatenated index (default: 0)
            col_start: Column from which to start outputting data. Can be single integer (e.g., 1) or
                      range string (e.g., "1,50") to output columns 1-50 (default: 1)
            row_index: Row to use as column header (default: 0)
            row_start: Row from which to start outputting data. Can be single integer (e.g., 1) or
                      range string (e.g., "1,100") to output rows 1-100 (default: 1)
            sep: Separator/delimiter to use when reading the file (default: None, auto-detect based on file extension)
            sheet: Sheet name or number to read from Excel files (default: None, uses first sheet)
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

    def _parse_range_parameter(self, param: int | str, param_name: str) -> tuple[int, int | None]:
        """Parse a range parameter that can be either an int or a range string.

        Args:
            param: Either an integer or a string in format "start,end"
            param_name: Name of the parameter for error messages

        Returns:
            Tuple of (start, end) where end is None if only start is specified

        Raises:
            ValueError: If the parameter format is invalid
        """
        if isinstance(param, int):
            return (param, None)
        # param is str due to type union
        try:
            if "," in param:
                parts = [p.strip() for p in param.split(",")]
                if len(parts) != 2:
                    raise ValueError(f"Invalid {param_name} range format '{param}'. Use 'start,end' format")
                start, end = int(parts[0]), int(parts[1])
                if start > end:
                    raise ValueError(f"Invalid {param_name} range '{param}': start must be <= end")
                # For polars, we return end + 1 to include the end in the slice
                return (start, end + 1)
            else:
                return (int(param), None)
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid {param_name} format '{param}'. Use integer or 'start,end' format") from e
            raise

    def _parse_col_start(self) -> tuple[int, int | None]:
        """Parse col_start parameter into start and end positions."""
        return self._parse_range_parameter(self.col_start, "col_start")

    def _parse_row_start(self) -> tuple[int, int | None]:
        """Parse row_start parameter into start and end positions."""
        return self._parse_range_parameter(self.row_start, "row_start")

    def _generate_range_suffix(self, num_rows: int, num_cols: int) -> str:
        """Generate suffix for filename based on the number of rows and columns in subset."""
        return f"_row{num_rows}_col{num_cols}"

    def _generate_output_filename(self) -> Path:
        """Generate output filename based on input filename.

        Note: The range suffix will be added later in the process method
        when we know the actual dimensions of the subset.

        If GZIP_OUT environment variable is True, adds .gz extension.
        """
        base_name = self.input_file.with_stem(self.input_file.stem + "_subset").with_suffix(".csv")
        if _should_write_gzipped():
            return base_name.with_suffix(".csv.gz")
        return base_name

    def _update_output_filename_with_range(self, output_file: Path, num_rows: int, num_cols: int) -> Path:
        """Update output filename to include range information."""
        stem = output_file.stem
        range_suffix = self._generate_range_suffix(num_rows, num_cols)

        # Insert range suffix before any existing suffix
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

            # Rename columns to generic names for consistency
            num_cols = df.width
            df = df.rename({f"column_{i}": f"column_{i}" for i in range(num_cols)})

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
        col_start, col_end = self._parse_col_start()
        row_start, row_end = self._parse_row_start()

        # Validate parameters
        # Check all index columns are within bounds
        for col_idx in index_columns_list:
            if col_idx >= df.width:
                raise ValueError(f"index_col ({col_idx}) is out of bounds. DataFrame has {df.width} columns.")

        if self.row_index >= df.height:
            raise ValueError(f"row_index ({self.row_index}) is out of bounds. DataFrame has {df.height} rows.")

        if col_start >= df.width:
            raise ValueError(f"col_start ({col_start}) is out of bounds. DataFrame has {df.width} columns.")

        if row_start >= df.height:
            raise ValueError(f"row_start ({row_start}) is out of bounds. DataFrame has {df.height} rows.")

        # Validate ranges
        if col_end is not None and col_end >= df.width:
            raise ValueError(f"col_end ({col_end}) is out of bounds. DataFrame has {df.width} columns.")

        if row_end is not None and row_end >= df.height:
            raise ValueError(f"row_end ({row_end}) is out of bounds. DataFrame has {df.height} rows.")

        # Calculate actual end positions
        actual_col_end = col_end if col_end is not None else df.width
        actual_row_end = row_end if row_end is not None else df.height

        # Extract column headers from the specified row
        header_row = df.slice(self.row_index, 1)
        # Polars uses column_1, column_2, etc. when has_header=False (1-indexed)
        header_cols = [pl.col(f"column_{i + 1}") for i in range(col_start, actual_col_end)]
        column_headers = [str(val) for val in header_row.select(header_cols).row(0)]

        # Extract index name(s) - concatenate if multiple columns
        if len(index_columns_list) == 1:
            index_name = str(header_row.select(pl.col(f"column_{index_columns_list[0] + 1}")).item())
        else:
            index_parts = [str(header_row.select(pl.col(f"column_{col + 1}")).item()) for col in index_columns_list]
            index_name = self.index_separator.join(index_parts)

        # Extract the data subset and row indices (with range support)
        data_length = actual_row_end - row_start
        data_rows = df.slice(row_start, data_length)

        # Extract row indices from the specified column(s) - concatenate if multiple columns
        if len(index_columns_list) == 1:
            index_col_name = f"column_{index_columns_list[0] + 1}"
            row_indices = [str(val) for val in data_rows.select(pl.col(index_col_name)).to_series().to_list()]
        else:
            # Create concatenated index from multiple columns
            index_cols = [f"column_{col + 1}" for col in index_columns_list]
            row_indices: list[str] = []
            for row in data_rows.select(index_cols).iter_rows():
                row_indices.append(self.index_separator.join(str(val) for val in row))

        # Extract the data subset (columns from col_start to col_end)
        data_columns = [f"column_{i + 1}" for i in range(col_start, actual_col_end)]
        data_subset = data_rows.select(data_columns)

        # Rename columns to use the headers we extracted
        column_mapping = {old_name: new_name for old_name, new_name in zip(data_subset.columns, column_headers, strict=True)}
        data_subset = data_subset.rename(column_mapping)

        # Add the index as a column (Polars doesn't have a traditional row index like pandas)
        result_df = data_subset.with_columns(pl.Series(name=index_name, values=row_indices))

        # Move the index column to the front
        cols = [index_name] + [col for col in result_df.columns if col != index_name]
        result_df = result_df.select(cols)

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
        print(f"  - Column start: {self.col_start}")
        print(f"  - Row index (header): {self.row_index}")
        print(f"  - Row start: {self.row_start}")

        selected_df = self._select_subset(df)
        shape_a, shape_b = selected_df.shape

        # Update output filename to include range information
        # Subtract 1 from columns because the index column is included
        actual_data_cols = shape_b - 1
        _, col_end = self._parse_col_start()
        _, row_end = self._parse_row_start()
        if (col_end is not None) or (row_end is not None):
            updated_output_file = self._update_output_filename_with_range(self.output_file, shape_a, actual_data_cols)
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

    def get_info(self) -> dict[str, str | int | None]:
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
            "index_separator": self.index_separator,
        }


class PreSelectData:
    """Class for selecting subsets of data from input files.

    This class provides functionality to read data files and select specific
    subsets based on column and row parameters. All indices are zero-based.
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
        index_separator: str = "#",
    ):
        """Initialize the PreSelectData class.

        Args:
            input_file: Path to the input data file
            output_file: Path to the output CSV file (defaults to input_file with "_subset.csv" suffix)
            index_col: Column(s) to use as row index. Can be a single integer (e.g., 0) or
                      comma-separated string of integers (e.g., "1,2,4") for concatenated index (default: 0)
            col_start: Column from which to start outputting data. Can be single integer (e.g., 1) or
                      range string (e.g., "1,50") to output columns 1-50 (default: 1)
            row_index: Row to use as column header (default: 0)
            row_start: Row from which to start outputting data. Can be single integer (e.g., 1) or
                      range string (e.g., "1,100") to output rows 1-100 (default: 1)
            sep: Separator/delimiter to use when reading the file (default: None, auto-detect based on file extension)
            sheet: Sheet name or number to read from Excel files (default: None, uses first sheet)
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

    def _parse_range_parameter(self, param: int | str, param_name: str) -> tuple[int, int | None]:
        """Parse a range parameter that can be either an int or a range string.

        Args:
            param: Either an integer or a string in format "start,end"
            param_name: Name of the parameter for error messages

        Returns:
            Tuple of (start, end) where end is None if only start is specified

        Raises:
            ValueError: If the parameter format is invalid
        """
        if isinstance(param, int):
            return (param, None)
        # param is str due to type union
        try:
            if "," in param:
                parts = [p.strip() for p in param.split(",")]
                if len(parts) != 2:
                    raise ValueError(f"Invalid {param_name} range format '{param}'. Use 'start,end' format")
                start, end = int(parts[0]), int(parts[1])
                if start > end:
                    raise ValueError(f"Invalid {param_name} range '{param}': start must be <= end")
                return (start, end)
            else:
                return (int(param), None)
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid {param_name} format '{param}'. Use integer or 'start,end' format") from e
            raise

    def _parse_col_start(self) -> tuple[int, int | None]:
        """Parse col_start parameter into start and end positions."""
        return self._parse_range_parameter(self.col_start, "col_start")

    def _parse_row_start(self) -> tuple[int, int | None]:
        """Parse row_start parameter into start and end positions."""
        return self._parse_range_parameter(self.row_start, "row_start")

    def _generate_range_suffix(self, num_rows: int, num_cols: int) -> str:
        """Generate suffix for filename based on the number of rows and columns in subset."""
        return f"_row{num_rows}_col{num_cols}"

    def _generate_output_filename(self) -> Path:
        """Generate output filename based on input filename.

        Note: The range suffix will be added later in the process method
        when we know the actual dimensions of the subset.

        If GZIP_OUT environment variable is True, adds .gz extension.
        """
        base_name = self.input_file.with_stem(self.input_file.stem + "_subset").with_suffix(".csv")
        if _should_write_gzipped():
            return base_name.with_suffix(".csv.gz")
        return base_name

    def _update_output_filename_with_range(self, output_file: Path, num_rows: int, num_cols: int) -> Path:
        """Update output filename to include range information."""
        stem = output_file.stem
        range_suffix = self._generate_range_suffix(num_rows, num_cols)

        # Insert range suffix before any existing suffix
        new_stem = stem + range_suffix
        return output_file.with_stem(new_stem)

    def _read_data(self) -> pd.DataFrame:  # type: ignore[type-arg]
        """Read data from the input file.

        Supports multiple file formats:
        - CSV files (.csv, .csv.gz): Uses pandas.read_csv with auto-detected or custom separator
        - Excel files (.xlsx, .xls): Uses pandas.read_excel with optional sheet selection (note: .gz not supported)
        - TSV files (.tsv, .tsv.gz): Uses pandas.read_csv with tab separator
        - Other formats: Defaults to CSV format

        Automatically handles gzip-compressed files by detecting .gz extension.

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
            is_gzipped = _is_gzipped(self.input_file)
            true_extension = _get_true_file_extension(self.input_file)

            # If a custom separator is provided, use it for CSV-like files
            if self.sep is not None:
                if self.sep == "\\t":
                    prefix = "gzipped " if is_gzipped else ""
                    print(f"\tReading {prefix}TSV file: {self.input_file}")
                    return pd.read_csv(self.input_file, sep="\t", header=None, low_memory=False)  # type: ignore[call-overload]
                prefix = "gzipped " if is_gzipped else ""
                print(f"\tReading {prefix}file with custom separator:({self.sep})")
                return pd.read_csv(self.input_file, sep=self.sep, header=None, low_memory=False)  # type: ignore[call-overload]

            # Auto-detect format based on true extension (ignoring .gz)
            if true_extension == ".csv":
                return pd.read_csv(self.input_file, header=None, low_memory=False)  # type: ignore[call-overload]
            elif true_extension in [".xlsx", ".xls"]:
                if is_gzipped:
                    raise ValueError(f"Gzipped Excel files are not supported: {self.input_file}")

                sheet_name = self.sheet if self.sheet is not None else 0
                if self.sheet is not None:
                    print(f"\tReading Excel file: {self.input_file}, sheet: {self.sheet}")
                else:
                    print(f"\tReading Excel file: {self.input_file}")
                return pd.read_excel(self.input_file, sheet_name=sheet_name, header=None)  # type: ignore[call-overload]
            elif true_extension == ".tsv":
                return pd.read_csv(self.input_file, sep="\t", header=None, low_memory=False)  # type: ignore[call-overload]
            else:
                # Default to CSV format for unknown extensions
                return pd.read_csv(self.input_file, header=None, low_memory=False)  # type: ignore[call-overload]

        except Exception as e:
            raise OSError(f"Error reading file {self.input_file}: {str(e)}") from e

    def _select_subset(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[type-arg]
        """Select the specified subset of data from the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame containing the selected subset
        """
        # Parse range parameters
        index_columns_list = self._parse_index_columns()
        col_start, col_end = self._parse_col_start()
        row_start, row_end = self._parse_row_start()

        # Validate parameters
        # Check all index columns are within bounds
        for col_idx in index_columns_list:
            if col_idx >= len(df.columns):
                raise ValueError(f"index_col ({col_idx}) is out of bounds. DataFrame has {len(df.columns)} columns.")

        if self.row_index >= len(df):
            raise ValueError(f"row_index ({self.row_index}) is out of bounds. DataFrame has {len(df)} rows.")

        if col_start >= len(df.columns):
            raise ValueError(f"col_start ({col_start}) is out of bounds. DataFrame has {len(df.columns)} columns.")

        if row_start >= len(df):
            raise ValueError(f"row_start ({row_start}) is out of bounds. DataFrame has {len(df)} rows.")

        # Validate ranges
        if col_end is not None and col_end >= len(df.columns):
            raise ValueError(f"col_end ({col_end}) is out of bounds. DataFrame has {len(df.columns)} columns.")

        if row_end is not None and row_end >= len(df):
            raise ValueError(f"row_end ({row_end}) is out of bounds. DataFrame has {len(df)} rows.")

        # Calculate actual end positions (pandas uses exclusive end)
        actual_col_end = col_end + 1 if col_end is not None else len(df.columns)
        actual_row_end = row_end + 1 if row_end is not None else len(df)

        # Extract column headers from the specified row
        column_headers: list[Any] = df.iloc[self.row_index, col_start:actual_col_end].tolist()  # type: ignore[attr-defined]

        # Extract index name(s) - concatenate if multiple columns
        if len(index_columns_list) == 1:
            index_name: Any = df.iloc[self.row_index, index_columns_list[0]]  # type: ignore[assignment]
        else:
            index_parts = [str(df.iloc[self.row_index, col]) for col in index_columns_list]  # type: ignore[index]
            index_name = self.index_separator.join(index_parts)

        # Extract row indices from the specified column(s) - concatenate if multiple columns
        if len(index_columns_list) == 1:
            row_indices: list[Any] = df.iloc[row_start:actual_row_end, index_columns_list[0]].tolist()  # type: ignore[attr-defined]
        else:
            # Create concatenated index from multiple columns
            row_data = df.iloc[row_start:actual_row_end, index_columns_list]  # type: ignore[assignment]
            row_indices = [
                self.index_separator.join(str(val) for val in row)  # type: ignore[var-annotated]
                for row in row_data.values  # type: ignore[attr-defined]
            ]

        # Extract the data subset
        data_subset: Any = df.iloc[row_start:actual_row_end, col_start:actual_col_end]  # type: ignore[assignment]

        # Create the new DataFrame with proper headers and indices
        result_df = pd.DataFrame(data=data_subset.values, columns=column_headers, index=row_indices)  # type: ignore[attr-defined,arg-type]
        result_df.index.name = index_name  # Set the index name for clarity

        return result_df

    def _write_csv_file(self, df: pd.DataFrame, file_path: Path) -> None:  # type: ignore[type-arg]
        """Write DataFrame to CSV file, with optional gzip compression.

        Args:
            df: DataFrame to write
            file_path: Output file path
        """
        if file_path.suffix == ".gz":
            # Use pandas' built-in compression support for gzipped output
            df.to_csv(file_path, compression="gzip")  # type: ignore[call-overload]
        else:
            df.to_csv(file_path)  # type: ignore[call-overload]

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
        print(f"  - Column start: {self.col_start}")
        print(f"  - Row index (header): {self.row_index}")
        print(f"  - Row start: {self.row_start}")

        selected_df = self._select_subset(df)

        # Update output filename to include range information
        # The shape includes the index, so we use the actual data columns
        num_rows, num_cols = selected_df.shape
        _, col_end = self._parse_col_start()
        _, row_end = self._parse_row_start()
        if (col_end is not None) or (row_end is not None):
            updated_output_file = self._update_output_filename_with_range(self.output_file, num_rows, num_cols)
        else:
            updated_output_file = self.output_file

        print(f"Selected data shape: {selected_df.shape}")
        print(f"Saving to: {updated_output_file}")

        # Save to CSV file (with optional gzip compression)
        self._write_csv_file(selected_df, updated_output_file)

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
            "col_start": self.col_start,
            "row_index": self.row_index,
            "row_start": self.row_start,
            "sep": self.sep,
            "sheet": self.sheet,
            "index_separator": self.index_separator,
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
