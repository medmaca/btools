#!/usr/bin/env python3
"""Script for quickly viewing and profiling datasets using Polars with enhanced syntax support.

This module provides functionality to quickly explore and profile datasets with:
- Multi-range row and column selection with colon syntax (e.g., "1:5,10:15")
- Configurable lazy vs eager loading for performance optimization
- Rich terminal output with beautiful formatting
- Optional detailed analysis and TOML output
- Original display layout and behavior restored

Supports CSV, TSV, Excel files (including gzipped files where applicable).
Uses Polars for high-performance data operations.
"""

import contextlib
import os
from pathlib import Path
from typing import Any

import polars as pl
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Load environment variables from .env files with priority order:
# 1. Package directory (for defaults)
# 2. Current working directory
# 3. Home directory (highest priority, overrides others)

# Get the package directory (where this script is located)
package_dir = Path(__file__).parent.parent.parent.parent

# Load .env files in order (later loads override earlier ones)
load_dotenv(dotenv_path=package_dir / ".env", override=False)  # Package defaults
load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)  # Current directory
load_dotenv(dotenv_path=Path.home() / ".env", override=True)  # User home directory (highest priority)

# Environment variable to enable test conditions
LOAD_ENV = os.getenv("BTOOLS_TEST_ENV") == "true"


def _apply_environment_conditions(df: pl.DataFrame) -> pl.DataFrame:
    """Apply test environment conditions if enabled.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with test conditions applied (if enabled)
    """
    if not LOAD_ENV:
        return df

    # Add test environment modifications here if needed
    return df


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
    """Check if a file is gzip-compressed by examining the file extension.

    Args:
        file_path: Path to the file to check

    Returns:
        True if the file appears to be gzip-compressed, False otherwise
    """
    return str(file_path).endswith(".gz")


def _parse_multi_range_parameter(param: int | str, param_name: str) -> list[tuple[int, int]]:
    """Parse multi-range parameter into list of (start, end) tuples.

    Supports:
    - Single number: 10 -> [(0, 9)]
    - Single range: "5:15" -> [(5, 15)]
    - Multiple ranges: "1:5,10:15" -> [(1, 5), (10, 15)]

    Args:
        param: Parameter value (int or str)
        param_name: Parameter name for error messages

    Returns:
        List of (start, end) tuples (both inclusive)

    Raises:
        ValueError: If the parameter format is invalid
    """
    if isinstance(param, int):
        return [(0, param - 1)]

    # param is str
    try:
        ranges: list[tuple[int, int]] = []
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
                ranges.append((start, end))
            else:
                # Single number - if it's the only spec, treat as count from beginning
                # If it's one of multiple specs, treat as individual index
                num = int(spec)
                if len(range_specs) == 1:
                    # Single spec, treat as count from beginning
                    ranges.append((0, num - 1))
                else:
                    # Multiple specs, treat as individual index
                    ranges.append((num, num))

        return ranges
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(
                f"Invalid {param_name} format '{param}'. Use integer, 'start:end', or 'start1:end1,start2:end2' format"
            ) from e
        raise


class ViewConfig:
    """Configuration settings for unified adaptive display mode."""

    def __init__(self):
        """Initialize view configuration with minimal environment variables."""
        # Performance settings
        self.use_lazy_loading = os.getenv("VIEW_LAZY_LOAD", "False").lower() == "true"

        # Unified display settings (single adaptive mode) - with error handling
        try:
            self.cols_per_section = int(os.getenv("VIEW_COLS_PER_SECTION", "12"))
        except ValueError:
            self.cols_per_section = 12

        try:
            self.max_col_width = int(os.getenv("VIEW_MAX_COL_WIDTH", "20"))
        except ValueError:
            self.max_col_width = 20

        self.ellipsis = os.getenv("VIEW_ELLIPSIS", "...")

        # Output settings
        try:
            self.out_unique_max = int(os.getenv("VIEW_OUT_UNIQUE_MAX", "20"))
        except ValueError:
            self.out_unique_max = 20

        # Display style (fixed, no need for env var)
        self.null_display_style = "dim red"


class PreViewData:
    """Class for viewing and profiling datasets using Polars with unified adaptive display.

    This class provides functionality to explore datasets with:
    - Multi-range row and column selection (e.g., "1:5,10:15")
    - Configurable lazy vs eager loading
    - Unified adaptive display that handles any number of columns
    - Optional detailed analysis and TOML output
    """

    def __init__(
        self,
        input_file: str,
        rows: int | str = 10,
        cols: int | str = 25,
        output_info: str | None = None,
        sep: str | None = None,
        sheet: str | None = None,
        show_dataset_overview: bool = False,
        show_column_info: bool = False,
        show_numeric_stats: bool = False,
        show_source_numbers: bool = False,
    ):
        """Initialize PreViewData with unified adaptive display.

        Args:
            input_file: Path to the input file
            rows: Row specification - number, range ("10:60"), or multiple ranges ("1:10,50:60")
            cols: Column specification - number, range ("5:15"), or multiple ranges ("1:5,10:15")
            output_info: Path for detailed TOML output file
            sep: Custom separator for CSV files
            sheet: Sheet name or number for Excel files
            show_dataset_overview: Show only dataset overview section
            show_column_info: Show only column information section
            show_numeric_stats: Show only numeric statistics section
            show_source_numbers: Show original source row/column numbers instead of subset positions
        """
        self.input_file = Path(input_file)
        self.rows = rows
        self.cols = cols
        self.output_info = Path(output_info) if output_info else None
        self.sep = sep
        self.sheet = sheet
        self.show_dataset_overview = show_dataset_overview
        self.show_column_info = show_column_info
        self.show_numeric_stats = show_numeric_stats
        self.show_source_numbers = show_source_numbers

        # Initialize console and config
        self.console = Console()
        self.config = ViewConfig()

    def _read_data(self) -> pl.DataFrame:
        """Read data from the input file using Polars with headers for analysis.

        Uses lazy or eager loading based on VIEW_LAZY_LOAD environment variable.
        This method reads with headers for proper analysis functionality.

        Returns:
            Polars DataFrame containing the input data with headers

        Raises:
            FileNotFoundError: If the input file doesn't exist
            Exception: If there's an error reading the file
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        try:
            is_gzipped = _is_gzipped(self.input_file)
            true_extension = _get_true_file_extension(self.input_file)

            # Custom separator handling - read with headers for analysis
            if self.sep is not None:
                if self.sep == "\\t":
                    if self.config.use_lazy_loading:
                        return pl.scan_csv(self.input_file, separator="\t").collect()
                    else:
                        return pl.read_csv(self.input_file, separator="\t")
                else:
                    if self.config.use_lazy_loading:
                        return pl.scan_csv(self.input_file, separator=self.sep).collect()
                    else:
                        return pl.read_csv(self.input_file, separator=self.sep)

            # Auto-detect format based on true extension
            # Read with headers for proper analysis functionality
            if true_extension == ".csv":
                if self.config.use_lazy_loading:
                    return pl.scan_csv(self.input_file).collect()
                else:
                    return pl.read_csv(self.input_file)
            elif true_extension in [".xlsx", ".xls"]:
                if is_gzipped:
                    raise ValueError(f"Gzipped Excel files are not supported: {self.input_file}")

                # Excel files must be read eagerly
                if self.sheet is not None:
                    return pl.read_excel(self.input_file, sheet_name=self.sheet)
                else:
                    return pl.read_excel(self.input_file)
            elif true_extension == ".tsv":
                if self.config.use_lazy_loading:
                    return pl.scan_csv(self.input_file, separator="\t").collect()
                else:
                    return pl.read_csv(self.input_file, separator="\t")
            else:
                # Default to CSV format
                if self.config.use_lazy_loading:
                    return pl.scan_csv(self.input_file).collect()
                else:
                    return pl.read_csv(self.input_file)

        except Exception as e:
            raise OSError(f"Error reading file {self.input_file}: {str(e)}") from e

    def _read_data_for_preview(self) -> pl.DataFrame:
        """Read data from the input file without headers for Data Preview.

        Uses lazy or eager loading based on VIEW_LAZY_LOAD environment variable.
        This method reads without headers to show raw file content with accurate row numbering.

        Returns:
            Polars DataFrame containing the raw input data without headers

        Raises:
            FileNotFoundError: If the input file doesn't exist
            Exception: If there's an error reading the file
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        try:
            is_gzipped = _is_gzipped(self.input_file)
            true_extension = _get_true_file_extension(self.input_file)

            # Custom separator handling - read without headers for raw data
            if self.sep is not None:
                if self.sep == "\\t":
                    if self.config.use_lazy_loading:
                        return pl.scan_csv(self.input_file, separator="\t", has_header=False).collect()
                    else:
                        return pl.read_csv(self.input_file, separator="\t", has_header=False)
                else:
                    if self.config.use_lazy_loading:
                        return pl.scan_csv(self.input_file, separator=self.sep, has_header=False).collect()
                    else:
                        return pl.read_csv(self.input_file, separator=self.sep, has_header=False)

            # Auto-detect format based on true extension
            # Read without headers to treat all rows as raw data
            if true_extension == ".csv":
                if self.config.use_lazy_loading:
                    return pl.scan_csv(self.input_file, has_header=False).collect()
                else:
                    return pl.read_csv(self.input_file, has_header=False)
            elif true_extension in [".xlsx", ".xls"]:
                if is_gzipped:
                    raise ValueError(f"Gzipped Excel files are not supported: {self.input_file}")

                # Excel files must be read eagerly
                if self.sheet is not None:
                    return pl.read_excel(self.input_file, sheet_name=self.sheet, has_header=False)
                else:
                    return pl.read_excel(self.input_file, has_header=False)
            elif true_extension == ".tsv":
                if self.config.use_lazy_loading:
                    return pl.scan_csv(self.input_file, separator="\t", has_header=False).collect()
                else:
                    return pl.read_csv(self.input_file, separator="\t", has_header=False)
            else:
                # Default to CSV format
                if self.config.use_lazy_loading:
                    return pl.scan_csv(self.input_file, has_header=False).collect()
                else:
                    return pl.read_csv(self.input_file, has_header=False)

        except Exception as e:
            raise OSError(f"Error reading file {self.input_file}: {str(e)}") from e

    def _create_data_overview_table(self, df: pl.DataFrame) -> Table:
        """Create a Rich table with basic dataset overview (original layout)."""
        table = Table(title="📊 Dataset Overview", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        # Basic info
        shape = df.shape
        table.add_row("File", str(self.input_file.name))
        table.add_row("Rows", f"{shape[0]:,}")
        table.add_row("Columns", f"{shape[1]:,}")

        # Memory usage (approximate)
        try:
            memory_mb = df.estimated_size("mb")
            table.add_row("Est. Memory", f"{memory_mb:.2f} MB")
        except (AttributeError, ValueError):
            table.add_row("Est. Memory", "N/A")

        return table

    def _create_column_info_table(self, df: pl.DataFrame) -> Table:
        """Create a Rich table with column information (original layout)."""
        table = Table(title="📋 Column Information", show_header=True, header_style="bold green")
        table.add_column("Column", style="cyan", no_wrap=True)
        table.add_column("Type", style="yellow")
        table.add_column("Non-Null", style="white")
        table.add_column("Missing", style="red")
        table.add_column("Missing %", style="red")
        table.add_column("Unique", style="blue")

        total_rows = df.height

        for col_name in df.columns:
            col_data = df[col_name]
            dtype = str(col_data.dtype)

            # Count nulls
            null_count = col_data.null_count()
            non_null_count = total_rows - null_count
            missing_pct = (null_count / total_rows * 100) if total_rows > 0 else 0

            # Count unique values
            try:
                unique_count = col_data.n_unique()
            except (AttributeError, ValueError):
                unique_count = "N/A"

            table.add_row(
                col_name,
                dtype,
                f"{non_null_count:,}",
                f"{null_count:,}",
                f"{missing_pct:.1f}%",
                f"{unique_count:,}" if isinstance(unique_count, int) else str(unique_count),
            )

        return table

    def _create_statistics_table(self, df: pl.DataFrame) -> Table:
        """Create a Rich table with basic statistics for numeric columns (original layout)."""
        table = Table(title="📈 Numeric Statistics", show_header=True, header_style="bold blue")
        table.add_column("Column", style="cyan", no_wrap=True)
        table.add_column("Mean", style="white")
        table.add_column("Std", style="white")
        table.add_column("Min", style="green")
        table.add_column("25%", style="white")
        table.add_column("50%", style="yellow")
        table.add_column("75%", style="white")
        table.add_column("Max", style="red")

        numeric_cols = df.select(
            pl.col(pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)
        ).columns

        if not numeric_cols:
            table.add_row("No numeric columns found", "", "", "", "", "", "", "")
            return table

        for col_name in numeric_cols:
            try:
                col_data = df[col_name]

                # Extract statistics (handle potential None values)
                mean_val = col_data.mean()
                std_val = col_data.std()
                min_val = col_data.min()
                max_val = col_data.max()

                # Calculate percentiles
                try:
                    q25 = col_data.quantile(0.25, interpolation="midpoint")
                    median = col_data.quantile(0.5, interpolation="midpoint")
                    q75 = col_data.quantile(0.75, interpolation="midpoint")
                except (AttributeError, ValueError):
                    q25, median, q75 = None, None, None

                def format_num(val: Any) -> str:
                    if val is None:
                        return "N/A"
                    if isinstance(val, int | float):
                        if abs(val) < 1000:
                            return f"{val:.2f}"
                        else:
                            return f"{val:.2e}"
                    return str(val)

                table.add_row(
                    col_name,
                    format_num(mean_val),
                    format_num(std_val),
                    format_num(min_val),
                    format_num(q25),
                    format_num(median),
                    format_num(q75),
                    format_num(max_val),
                )

            except (ValueError, AttributeError, TypeError) as e:
                table.add_row(col_name, "Error", str(e)[:20], "", "", "", "", "")

        return table

    def _get_source_indices(self) -> tuple[list[int], list[int]]:
        """Get the original source row and column indices for the current selection.

        Returns:
            Tuple of (source_row_indices, source_col_indices)
        """
        # Parse row and column ranges to get original indices
        row_ranges = _parse_multi_range_parameter(self.rows, "rows")
        col_ranges = _parse_multi_range_parameter(self.cols, "cols")

        # Collect all selected row indices (original positions)
        source_rows: list[int] = []
        for start, end in row_ranges:
            source_rows.extend(range(start, end + 1))

        # Collect all selected column indices (original positions)
        source_cols: list[int] = []
        for start, end in col_ranges:
            source_cols.extend(range(start, end + 1))

        # Remove duplicates and sort
        source_rows = sorted(set(source_rows))
        source_cols = sorted(set(source_cols))

        return source_rows, source_cols

    def _get_selected_data(self, df: pl.DataFrame) -> tuple[pl.DataFrame, int, int, int, int]:
        """Get data subset based on row and column selections.

        Args:
            df: Input DataFrame to select from

        Returns:
            Tuple of (selected_df, start_row, end_row, start_col, end_col)
        """
        # Parse row and column ranges
        row_ranges = _parse_multi_range_parameter(self.rows, "rows")
        col_ranges = _parse_multi_range_parameter(self.cols, "cols")

        # Collect all selected row indices
        selected_rows: list[int] = []
        for start, end in row_ranges:
            selected_rows.extend(range(start, min(end + 1, df.height)))

        # Collect all selected column indices
        selected_cols: list[int] = []
        for start, end in col_ranges:
            selected_cols.extend(range(start, min(end + 1, df.width)))

        # Remove duplicates and sort
        selected_rows = sorted(set(selected_rows))
        selected_cols = sorted(set(selected_cols))

        # Get actual start/end for display purposes
        start_row = selected_rows[0] if selected_rows else 0
        end_row = selected_rows[-1] if selected_rows else 0
        start_col = selected_cols[0] if selected_cols else 0
        end_col = selected_cols[-1] if selected_cols else 0

        # Create subset DataFrame
        if selected_rows and selected_cols:
            # Select rows first
            row_subset = df.slice(0, 0)  # Empty DataFrame with same schema
            for row_idx in selected_rows:
                if row_idx < df.height:
                    row_subset = pl.concat([row_subset, df.slice(row_idx, 1)])

            # Then select columns
            column_names = df.columns
            valid_col_names = [column_names[i] for i in selected_cols if i < len(column_names)]
            subset_df = row_subset.select(valid_col_names) if valid_col_names else row_subset
        else:
            subset_df = df

        return subset_df, start_row, end_row, start_col, end_col

    def _create_data_preview_table(
        self, df: pl.DataFrame, start_row: int, end_row: int, start_col: int, end_col: int
    ) -> Table:
        """Create a Rich table showing the actual data preview with unified adaptive layout."""
        # Get source indices if needed
        if self.show_source_numbers:
            source_rows, source_cols = self._get_source_indices()
        else:
            source_rows, source_cols = [], []

        return self._create_adaptive_table(df, start_row, end_row, start_col, end_col, source_rows, source_cols)

    def _create_adaptive_table(
        self,
        display_df: pl.DataFrame,
        start_row: int,
        end_row: int,
        start_col: int,
        end_col: int,
        source_rows: list[int],
        source_cols: list[int],
    ) -> Table:
        """Create an adaptive table that intelligently handles any number of columns.

        This unified approach:
        - Few columns: Shows them with full width (like normal mode)
        - Many columns: Uses condensed width to fit more
        - Too many columns: Shows in multiple sections
        """
        total_cols = len(display_df.columns)
        cols_per_section = self.config.cols_per_section

        # Calculate optimal column width based on number of columns
        if total_cols <= 6:
            # Few columns - use full width
            col_width = self.config.max_col_width
            title_suffix = ""
        elif total_cols <= cols_per_section:
            # Medium number - use reasonable condensed width (but respect user setting)
            col_width = max(12, min(self.config.max_col_width, self.config.max_col_width // 2 + 8))
            title_suffix = " [Condensed]"
        else:
            # Many columns - use user's setting (they know what they want)
            col_width = self.config.max_col_width
            title_suffix = f" [Showing all {total_cols} columns]"

        # Create the main table
        title_text = (
            f"🔍 Data Preview (Rows {start_row}-"
            f"{min(end_row, start_row + display_df.height - 1)}, "
            f"Cols {start_col}-{end_col}){title_suffix}"
        )

        # If we can fit all columns in one table, do that
        if total_cols <= cols_per_section:
            return self._create_single_table(
                display_df, title_text, col_width, start_col, start_row, source_rows, source_cols
            )
        else:
            # Create multiple sections
            return self._create_multi_section_table(
                display_df, title_text, col_width, start_col, cols_per_section, start_row, source_rows, source_cols
            )

    def _create_single_table(
        self,
        display_df: pl.DataFrame,
        title: str,
        col_width: int,
        start_col: int,
        start_row: int,
        source_rows: list[int],
        source_cols: list[int],
    ) -> Table:
        """Create a single table for all columns."""
        table = Table(
            title=title,
            show_header=True,
            header_style="bold cyan",
        )

        # Add row number column first with "Row" as header
        table.add_column("Row", style="dim yellow", max_width=8, overflow="ellipsis")

        # Add data columns using column numbers as headers
        if self.show_source_numbers and source_cols:
            # Use original source column indices as headers
            for i, _col_name in enumerate(display_df.columns):
                col_header = f"#{source_cols[i]}" if i < len(source_cols) else f"#{start_col + i}"
                table.add_column(col_header, style="white", max_width=col_width, overflow="ellipsis")
        else:
            # Use sequential indices starting from start_col as headers
            for i, _col_name in enumerate(display_df.columns):
                col_header = f"#{start_col + i}"
                table.add_column(col_header, style="white", max_width=col_width, overflow="ellipsis")

        # Add data rows with correct row numbers (starting from 0 for first data row)
        max_cell_len = max(6, col_width - 2)
        for row_idx, row in enumerate(display_df.iter_rows()):
            formatted_row: list[str] = []
            # Add row number as first column (0-based for actual data rows)
            if self.show_source_numbers and source_rows:
                actual_row_num = source_rows[row_idx] if row_idx < len(source_rows) else start_row + row_idx
            else:
                actual_row_num = start_row + row_idx
            formatted_row.append(f"[dim yellow]#{actual_row_num}[/dim yellow]")

            # Add data values
            for val in row:
                if val is None:
                    formatted_row.append(f"[{self.config.null_display_style}]null[/{self.config.null_display_style}]")
                else:
                    str_val = str(val)
                    if len(str_val) > max_cell_len:
                        truncate_len = max_cell_len - len(self.config.ellipsis)
                        str_val = str_val[:truncate_len] + self.config.ellipsis
                    formatted_row.append(str_val)
            table.add_row(*formatted_row)

        return table

    def _create_multi_section_table(
        self,
        display_df: pl.DataFrame,
        base_title: str,
        col_width: int,
        start_col: int,
        cols_per_section: int,
        start_row: int,
        source_rows: list[int],
        source_cols: list[int],
    ) -> Table:
        """Create all sections and print them in correct order."""
        total_cols = len(display_df.columns)
        num_sections = (total_cols + cols_per_section - 1) // cols_per_section

        # Print all sections in correct order (1, 2, 3, ...)
        for section_idx in range(num_sections):
            section_start = section_idx * cols_per_section
            section_end = min(section_start + cols_per_section, total_cols)
            section_df = display_df.select(display_df.columns[section_start:section_end])

            section_title = (
                f"🔍 Data Preview - Section {section_idx + 1}/{num_sections} "
                f"(Cols {start_col + section_start}-{start_col + section_end - 1})"
            )
            # Get source columns for this section if needed
            section_source_cols = []
            if self.show_source_numbers and source_cols:
                section_source_cols = source_cols[section_start:section_end]

            section_table = self._create_single_table(
                section_df, section_title, col_width, start_col + section_start, start_row, source_rows, section_source_cols
            )

            if section_idx > 0:
                self.console.print("\n")
            self.console.print(section_table)

        # Return a summary table as the "main" result
        summary_table = Table(title=f"📊 Summary: {total_cols} columns shown in {num_sections} sections")
        summary_table.add_column("Info", style="cyan")
        summary_table.add_column("Value", style="white")
        summary_table.add_row("Total Columns", str(total_cols))
        summary_table.add_row("Sections", str(num_sections))
        summary_table.add_row("Columns per Section", str(cols_per_section))

        return summary_table

    def _generate_detailed_info(self, df: pl.DataFrame) -> dict[str, Any]:
        """Generate detailed information about the dataset for file output (original format)."""
        info: dict[str, Any] = {
            "file_info": {
                "file_path": str(self.input_file),
                "file_size_bytes": self.input_file.stat().st_size if self.input_file.exists() else None,
                "rows": df.height,
                "columns": df.width,
            },
            "column_details": {},
            "summary_statistics": {},
            "data_quality": {
                "total_missing_values": 0,
                "columns_with_missing": [],
                "memory_usage_mb": None,
            },
        }

        with contextlib.suppress(AttributeError, ValueError):
            info["data_quality"]["memory_usage_mb"] = df.estimated_size("mb")

        total_missing = 0
        columns_with_missing: list[str] = []

        # Analyze each column
        for col_idx, col_name in enumerate(df.columns):
            col_data = df[col_name]
            null_count = col_data.null_count()

            if null_count > 0:
                columns_with_missing.append(col_name)
                total_missing += null_count

            col_info: dict[str, Any] = {
                "dtype": str(col_data.dtype),
                "column_number": col_idx,
                "non_null_count": df.height - null_count,
                "null_count": null_count,
                "null_percentage": (null_count / df.height * 100) if df.height > 0 else 0,
                "unique_count": None,
                "memory_usage": None,
                "unique_values": None,
            }

            with contextlib.suppress(AttributeError, ValueError):
                unique_count = col_data.n_unique()
                col_info["unique_count"] = unique_count

                # Include unique values if count is within threshold
                if unique_count <= self.config.out_unique_max:
                    try:
                        # Get unique values, excluding nulls, and convert to Python types
                        unique_vals = col_data.drop_nulls().unique().to_list()
                        # Sort the values if possible for consistent output
                        with contextlib.suppress(TypeError):
                            unique_vals.sort()
                        col_info["unique_values"] = unique_vals
                    except (AttributeError, ValueError, TypeError):
                        # If we can't get unique values for some reason, leave as None
                        pass

            # Add statistics for numeric columns
            numeric_dtypes = {
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
                pl.Float32,
                pl.Float64,
            }
            if str(col_data.dtype) in [str(dt) for dt in numeric_dtypes]:
                try:
                    stats = {
                        "mean": col_data.mean(),
                        "std": col_data.std(),
                        "min": col_data.min(),
                        "max": col_data.max(),
                        "median": col_data.median(),
                    }
                    # Add percentiles
                    try:
                        q25 = col_data.quantile(0.25, interpolation="midpoint")
                        q75 = col_data.quantile(0.75, interpolation="midpoint")
                        stats["q25"] = q25
                        stats["q75"] = q75
                    except (AttributeError, ValueError):
                        pass

                    info["summary_statistics"][col_name] = stats
                except (AttributeError, ValueError, TypeError):
                    pass

            info["column_details"][col_name] = col_info

        info["data_quality"]["total_missing_values"] = total_missing
        info["data_quality"]["columns_with_missing"] = columns_with_missing

        return info

    def _clean_for_toml(self, data: Any) -> Any:
        """Clean data for TOML serialization by replacing None with appropriate values."""
        if data is None:
            return "null"
        elif isinstance(data, dict):
            return {str(k): self._clean_for_toml(v) for k, v in data.items()}  # type: ignore[misc]
        elif isinstance(data, list):
            return [self._clean_for_toml(item) for item in data]  # type: ignore[misc]
        else:
            return data

    def view(self) -> None:
        """Main method to view and analyze the dataset with dual reading strategy."""
        self.console.print(f"\n[bold green]📁 Loading dataset: {self.input_file.name}[/bold green]")

        # Determine what to show based on flags
        show_any_section = self.show_dataset_overview or self.show_column_info or self.show_numeric_stats
        show_data_preview = not show_any_section  # Show data preview by default or when no analysis flags are set

        # Read data with appropriate strategy based on what we're displaying
        analysis_df = None
        preview_df = None

        if show_any_section or self.output_info:
            # Need header-aware data for analysis
            analysis_df = self._read_data()
            if LOAD_ENV:
                analysis_df = _apply_environment_conditions(analysis_df)

        if show_data_preview:
            # Need raw data for accurate Data Preview
            preview_df = self._read_data_for_preview()
            if LOAD_ENV:
                preview_df = _apply_environment_conditions(preview_df)

        # Show analysis sections if requested
        if show_any_section and analysis_df is not None:
            if self.show_dataset_overview:
                overview_table = self._create_data_overview_table(analysis_df)
                self.console.print(overview_table)

            if self.show_column_info:
                column_table = self._create_column_info_table(analysis_df)
                self.console.print("\n", column_table)

            if self.show_numeric_stats:
                stats_table = self._create_statistics_table(analysis_df)
                self.console.print("\n", stats_table)

        # Show data preview if requested (default behavior)
        if show_data_preview and preview_df is not None:
            # Get selected data subset from preview data
            display_df, start_row, end_row, start_col, end_col = self._get_selected_data(preview_df)
            preview_table = self._create_data_preview_table(display_df, start_row, end_row, start_col, end_col)
            self.console.print("\n", preview_table)

        # Generate detailed info file if requested (uses analysis data)
        if self.output_info and analysis_df is not None:
            self.console.print(f"\n[dim]Generating detailed info file: {self.output_info}[/dim]")
            detailed_info = self._generate_detailed_info(analysis_df)
            cleaned_info = self._clean_for_toml(detailed_info)

            try:
                import tomli_w

                with open(self.output_info, "wb") as f:
                    tomli_w.dump(cleaned_info, f)
                self.console.print(f"[green]✓ Detailed info written to: {self.output_info}[/green]")
            except ImportError:
                self.console.print(
                    "[yellow]Warning: tomli_w not available. Install with 'pip install tomli_w' for TOML output.[/yellow]"
                )
            except Exception as e:
                self.console.print(f"[red]Error writing output file: {str(e)}[/red]")


def main():
    """Main function for command-line usage."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="View and profile datasets with enhanced syntax support")
    parser.add_argument("input_file", help="Path to the input data file")
    parser.add_argument("--rows", "-r", default=10, help="Number of rows or range (e.g., '10' or '10:60' or '1:5,10:15')")
    parser.add_argument("--cols", "-c", default=25, help="Number of columns or range (e.g., '25' or '5:15' or '1:3,8:12')")
    parser.add_argument("--output-info", "-o", help="Path to output detailed info file (TOML format)")
    parser.add_argument("--sep", "-s", help="Separator/delimiter for CSV files")
    parser.add_argument("--sheet", help="Sheet name or number for Excel files")
    parser.add_argument("--dataset-overview", action="store_true", help="Show only dataset overview")
    parser.add_argument("--column-info", action="store_true", help="Show only column information")
    parser.add_argument("--numeric-stats", action="store_true", help="Show only numeric statistics")

    args = parser.parse_args()

    # Convert string arguments to appropriate types
    rows = args.rows
    if isinstance(rows, str) and rows.isdigit():
        rows = int(rows)

    cols = args.cols
    if isinstance(cols, str) and cols.isdigit():
        cols = int(cols)

    try:
        viewer = PreViewData(
            input_file=args.input_file,
            rows=rows,
            cols=cols,
            output_info=args.output_info,
            sep=args.sep,
            sheet=args.sheet,
            show_dataset_overview=args.dataset_overview,
            show_column_info=args.column_info,
            show_numeric_stats=args.numeric_stats,
        )
        viewer.view()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
