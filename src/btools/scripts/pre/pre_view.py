#!/usr/bin/env python3
"""Script for quickly viewing and profiling datasets using Polars."""

import contextlib
import os
from pathlib import Path
from typing import Any

import polars as pl
import tomli_w
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


class ViewConfig:
    """Configuration class for pre_view display modes."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # Auto mode thresholds
        self.auto_normal_max_cols = int(os.getenv("VIEW_AUTO_NORMAL_MAX_COLS", "5"))
        self.auto_rotated_max_cols = int(os.getenv("VIEW_AUTO_ROTATED_MAX_COLS", "10"))

        # Normal mode settings
        self.normal_max_col_width = int(os.getenv("VIEW_NORMAL_MAX_COL_WIDTH", "20"))
        self.normal_max_cell_length = int(os.getenv("VIEW_NORMAL_MAX_CELL_LENGTH", "18"))

        # Rotated headers mode settings
        self.rotated_max_col_width = int(os.getenv("VIEW_ROTATED_MAX_COL_WIDTH", "12"))
        self.rotated_max_cell_length = int(os.getenv("VIEW_ROTATED_MAX_CELL_LENGTH", "10"))
        self.rotated_header_max_length = int(os.getenv("VIEW_ROTATED_HEADER_MAX_LENGTH", "8"))

        # Wrapped mode settings
        self.wrapped_cols_per_section = int(os.getenv("VIEW_WRAPPED_COLS_PER_SECTION", "5"))
        self.wrapped_max_col_width = int(os.getenv("VIEW_WRAPPED_MAX_COL_WIDTH", "15"))
        self.wrapped_max_cell_length = int(os.getenv("VIEW_WRAPPED_MAX_CELL_LENGTH", "13"))

        # General display settings
        self.default_rows = int(os.getenv("VIEW_DEFAULT_ROWS", "10"))
        self.max_rows = int(os.getenv("VIEW_MAX_ROWS", "1000"))
        self.ellipsis_string = os.getenv("VIEW_ELLIPSIS_STRING", "...")
        self.null_display_style = os.getenv("VIEW_NULL_DISPLAY_STYLE", "dim red")

        # TOML output settings
        self.out_unique_max = int(os.getenv("VIEW_OUT_UNIQUE_MAX", "20"))


class PreViewData:
    """Class for quickly viewing and profiling datasets using Polars.

    This class provides functionality to quickly explore datasets with:
    - Configurable row display (single number or range)
    - Data type analysis
    - Missing value analysis
    - Basic statistics
    - Optional detailed output file
    """

    def __init__(
        self,
        input_file: str,
        rows: int | str = 50,
        cols: int | str = 25,
        output_info: str | None = None,
        sep: str | None = None,
        sheet: str | None = None,
        show_dataset_overview: bool = False,
        show_column_info: bool = False,
        show_numeric_stats: bool = False,
        display_mode: str = "auto",
    ):
        """Initialize the PreViewData class.

        Args:
            input_file: Path to the input data file
            rows: Number of rows to display. Can be single integer (e.g., 50) or
                 range string (e.g., "50,100") to display rows 50-100 (default: 50)
            cols: Number of columns to display. Can be single integer (e.g., 25) or
                 range string (e.g., "5,15") to display columns 5-15 (default: 25)
            output_info: Path to output detailed info file (TOML format)
            sep: Separator/delimiter to use when reading the file (default: None, auto-detect)
            sheet: Sheet name or number to read from Excel files (default: None, uses first sheet)
            show_dataset_overview: Whether to show dataset overview only (default: False)
            show_column_info: Whether to show column information only (default: False)
            show_numeric_stats: Whether to show numeric statistics only (default: False)
            display_mode: Display mode for data preview. Options: "auto", "normal", "rotated", "wrapped" (default: "auto")
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
        self.display_mode = display_mode
        self.console = Console()
        self.config = ViewConfig()

    def _parse_rows_parameter(self) -> tuple[int, int | None]:
        """Parse rows parameter into start and end positions.

        Returns:
            Tuple of (start, end) where end is None if only start is specified

        Raises:
            ValueError: If the parameter format is invalid
        """
        if isinstance(self.rows, int):
            return (0, self.rows)

        # self.rows is str
        try:
            if "," in self.rows:
                parts = [p.strip() for p in self.rows.split(",")]
                if len(parts) != 2:
                    raise ValueError(f"Invalid rows range format '{self.rows}'. Use 'start,end' format")
                start, end = int(parts[0]), int(parts[1])
                if start > end:
                    raise ValueError(f"Invalid rows range '{self.rows}': start must be <= end")
                return (start, end)
            else:
                return (0, int(self.rows))
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid rows format '{self.rows}'. Use integer or 'start,end' format") from e
            raise

    def _parse_cols_parameter(self) -> tuple[int, int | None]:
        """Parse cols parameter into start and end positions.

        Returns:
            Tuple of (start, end) where end is None if only start is specified

        Raises:
            ValueError: If the parameter format is invalid
        """
        if isinstance(self.cols, int):
            return (0, self.cols)

        # self.cols is str
        try:
            if "," in self.cols:
                parts = [p.strip() for p in self.cols.split(",")]
                if len(parts) != 2:
                    raise ValueError(f"Invalid cols range format '{self.cols}'. Use 'start,end' format")
                start, end = int(parts[0]), int(parts[1])
                if start > end:
                    raise ValueError(f"Invalid cols range '{self.cols}': start must be <= end")
                return (start, end)
            else:
                return (0, int(self.cols))
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid cols format '{self.cols}'. Use integer or 'start,end' format") from e
            raise

    def _read_data(self) -> pl.DataFrame:
        """Read data from the input file using Polars.

        Supports multiple file formats:
        - CSV files (.csv, .csv.gz): Uses polars.read_csv with auto-detected or custom separator
        - Excel files (.xlsx, .xls): Uses polars.read_excel (note: .gz not supported for Excel)
        - TSV files (.tsv, .tsv.gz): Uses polars.read_csv with tab separator
        - Other formats: Defaults to CSV format

        Automatically handles gzip-compressed files by detecting .gz extension.

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
                    self.console.print(f"[dim]Reading {'gzipped ' if is_gzipped else ''}TSV file: {self.input_file}[/dim]")
                    df = pl.read_csv(self.input_file, separator="\t")
                else:
                    prefix = "gzipped " if is_gzipped else ""
                    self.console.print(f"[dim]Reading {prefix}file with custom separator: {repr(self.sep)}[/dim]")
                    df = pl.read_csv(self.input_file, separator=self.sep)
            else:
                # Auto-detect format based on true extension (ignoring .gz)
                if true_extension == ".csv":
                    df = pl.read_csv(self.input_file)
                elif true_extension in [".xlsx", ".xls"]:
                    if is_gzipped:
                        raise ValueError(f"Gzipped Excel files are not supported: {self.input_file}")

                    if self.sheet is not None:
                        self.console.print(f"[dim]Reading Excel file: {self.input_file}, sheet: {self.sheet}[/dim]")
                        df = pl.read_excel(self.input_file, sheet_name=self.sheet)
                    else:
                        self.console.print(f"[dim]Reading Excel file: {self.input_file}[/dim]")
                        df = pl.read_excel(self.input_file)
                elif true_extension == ".tsv":
                    df = pl.read_csv(self.input_file, separator="\t")
                else:
                    # Default to CSV format for unknown extensions
                    df = pl.read_csv(self.input_file)

            return df

        except Exception as e:
            raise OSError(f"Error reading file {self.input_file}: {str(e)}") from e

    def _create_data_overview_table(self, df: pl.DataFrame) -> Table:
        """Create a Rich table with basic dataset overview."""
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
        """Create a Rich table with column information."""
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
        """Create a Rich table with basic statistics for numeric columns."""
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

    def _create_data_preview_table(
        self, df: pl.DataFrame, start_row: int, end_row: int, start_col: int, end_col: int
    ) -> Table:
        """Create a Rich table showing the actual data preview with adaptive layout."""
        # Select row subset
        display_df = df.slice(start_row, end_row - start_row)

        # Select column subset
        all_columns = display_df.columns
        if start_col >= len(all_columns):
            raise ValueError(f"start_col ({start_col}) is out of bounds. DataFrame has {len(all_columns)} columns.")

        end_col = min(end_col, len(all_columns))
        selected_columns = all_columns[start_col:end_col]
        display_df = display_df.select(selected_columns)

        num_cols = len(selected_columns)

        # Determine display mode
        if self.display_mode == "auto":
            # Adaptive display based on number of columns using config values
            if num_cols <= self.config.auto_normal_max_cols:
                return self._create_standard_table(display_df, start_row, end_row, start_col, end_col)
            elif num_cols <= self.config.auto_rotated_max_cols:
                return self._create_rotated_header_table(display_df, start_row, end_row, start_col, end_col)
            else:
                return self._create_wrapped_table(display_df, start_row, end_row, start_col, end_col)
        elif self.display_mode == "normal":
            return self._create_standard_table(display_df, start_row, end_row, start_col, end_col)
        elif self.display_mode == "rotated":
            return self._create_rotated_header_table(display_df, start_row, end_row, start_col, end_col)
        elif self.display_mode == "wrapped":
            return self._create_wrapped_table(display_df, start_row, end_row, start_col, end_col)
        else:
            # Default to auto mode for unknown display modes
            if num_cols <= 5:
                return self._create_standard_table(display_df, start_row, end_row, start_col, end_col)
            elif num_cols <= 10:
                return self._create_rotated_header_table(display_df, start_row, end_row, start_col, end_col)
            else:
                return self._create_wrapped_table(display_df, start_row, end_row, start_col, end_col)

    def _create_standard_table(
        self, display_df: pl.DataFrame, start_row: int, end_row: int, start_col: int, end_col: int
    ) -> Table:
        """Create a standard horizontal table for small number of columns."""
        title_text = (
            f"🔍 Data Preview (Rows {start_row}-"
            f"{min(end_row - 1, start_row + display_df.height - 1)}, "
            f"Cols {start_col}-{end_col - 1})"
        )
        table = Table(
            title=title_text,
            show_header=True,
            header_style="bold cyan",
        )

        # Add columns with normal headers
        for col_name in display_df.columns:
            table.add_column(col_name, style="white", max_width=self.config.normal_max_col_width, overflow="ellipsis")

        # Add column number row
        col_numbers = [str(start_col + i) for i in range(len(display_df.columns))]
        table.add_row(*[f"[dim blue]#{num}[/dim blue]" for num in col_numbers])

        # Add rows
        for row in display_df.iter_rows():
            formatted_row: list[str] = []
            for val in row:
                if val is None:
                    formatted_row.append(f"[{self.config.null_display_style}]null[/{self.config.null_display_style}]")
                else:
                    str_val = str(val)
                    if len(str_val) > self.config.normal_max_cell_length:
                        truncate_len = self.config.normal_max_cell_length - len(self.config.ellipsis_string)
                        str_val = str_val[:truncate_len] + self.config.ellipsis_string
                    formatted_row.append(str_val)
            table.add_row(*formatted_row)

        return table

    def _create_rotated_header_table(
        self, display_df: pl.DataFrame, start_row: int, end_row: int, start_col: int, end_col: int
    ) -> Table:
        """Create a table with rotated column headers for medium number of columns."""
        title_text = (
            f"🔍 Data Preview (Rows {start_row}-"
            f"{min(end_row - 1, start_row + display_df.height - 1)}, "
            f"Cols {start_col}-{end_col - 1}) [Rotated Headers]"
        )
        table = Table(
            title=title_text,
            show_header=True,
            header_style="bold cyan",
        )

        # Add columns with rotated headers (using vertical text approximation)
        for col_name in display_df.columns:
            # Create a vertical-style header by putting each character on a new "line" using spaces
            max_header_len = self.config.rotated_header_max_length
            rotated_header = (
                col_name[: max_header_len - len(self.config.ellipsis_string[:2])] + self.config.ellipsis_string[:2]
                if len(col_name) > max_header_len
                else col_name
            )

            table.add_column(
                rotated_header,
                style="white",
                max_width=self.config.rotated_max_col_width,
                overflow="ellipsis",
                justify="center",
            )

        # Add column number row
        col_numbers = [str(start_col + i) for i in range(len(display_df.columns))]
        table.add_row(*[f"[dim blue]#{num}[/dim blue]" for num in col_numbers])

        # Add rows
        for row in display_df.iter_rows():
            formatted_row: list[str] = []
            for val in row:
                if val is None:
                    formatted_row.append(f"[{self.config.null_display_style}]null[/{self.config.null_display_style}]")
                else:
                    str_val = str(val)
                    if len(str_val) > self.config.rotated_max_cell_length:
                        truncate_len = self.config.rotated_max_cell_length - len(self.config.ellipsis_string[:2])
                        str_val = str_val[:truncate_len] + self.config.ellipsis_string[:2]
                    formatted_row.append(str_val)
            table.add_row(*formatted_row)

        return table

    def _create_wrapped_table(
        self, display_df: pl.DataFrame, start_row: int, end_row: int, start_col: int, end_col: int
    ) -> Table:
        """Create multiple tables wrapping columns across sections."""
        tables: list[Table] = []
        cols_per_section = self.config.wrapped_cols_per_section
        total_cols = len(display_df.columns)

        for section_start in range(0, total_cols, cols_per_section):
            section_end = min(section_start + cols_per_section, total_cols)
            section_columns = display_df.columns[section_start:section_end]
            section_df = display_df.select(section_columns)

            table = Table(
                title=f"Cols {start_col + section_start}-{start_col + section_end - 1}",
                show_header=True,
                header_style="bold cyan",
                title_style="bold blue",
            )

            # Add columns for this section
            for col_name in section_df.columns:
                table.add_column(col_name, style="white", max_width=self.config.wrapped_max_col_width, overflow="ellipsis")

            # Add column number row
            section_col_numbers = [str(start_col + section_start + i) for i in range(len(section_df.columns))]
            table.add_row(*[f"[dim blue]#{num}[/dim blue]" for num in section_col_numbers])

            # Add rows for this section
            for row in section_df.iter_rows():
                formatted_row: list[str] = []
                for val in row:
                    if val is None:
                        formatted_row.append(f"[{self.config.null_display_style}]null[/{self.config.null_display_style}]")
                    else:
                        str_val = str(val)
                        if len(str_val) > self.config.wrapped_max_cell_length:
                            truncate_len = self.config.wrapped_max_cell_length - len(self.config.ellipsis_string)
                            str_val = str_val[:truncate_len] + self.config.ellipsis_string
                        formatted_row.append(str_val)
                table.add_row(*formatted_row)

            tables.append(table)

        # Create a combined display
        main_table = Table.grid()
        main_table.add_column()

        # Add title
        title_text = (
            f"[bold cyan]🔍 Data Preview (Rows {start_row}-"
            f"{min(end_row - 1, start_row + display_df.height - 1)}, "
            f"Cols {start_col}-{end_col - 1}) [Wrapped Layout][/bold cyan]"
        )
        main_table.add_row(title_text)

        # Add each table section
        for i, table in enumerate(tables):
            if i > 0:
                main_table.add_row("")  # Add spacing between sections
            main_table.add_row(table)

        return main_table

    def _generate_detailed_info(self, df: pl.DataFrame) -> dict[str, Any]:
        """Generate detailed information about the dataset for file output."""
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
        """Main method to view and analyze the dataset."""
        self.console.print(f"\n[bold green]📁 Loading dataset: {self.input_file.name}[/bold green]")

        # Read the data
        df = self._read_data()

        # Parse row parameters
        start_row, end_row = self._parse_rows_parameter()
        if end_row is None:
            end_row = min(start_row + 50, df.height)  # Default to showing 50 rows from start

        # Parse column parameters
        start_col, end_col = self._parse_cols_parameter()
        if end_col is None:
            end_col = min(start_col + 25, df.width)  # Default to showing 25 columns from start

        # Validate row range
        if start_row >= df.height:
            raise ValueError(f"start_row ({start_row}) is out of bounds. DataFrame has {df.height} rows.")
        if end_row > df.height:
            end_row = df.height

        # Validate column range
        if start_col >= df.width:
            raise ValueError(f"start_col ({start_col}) is out of bounds. DataFrame has {df.width} columns.")
        if end_col > df.width:
            end_col = df.width

        # Determine what to show based on flags
        show_any_section = self.show_dataset_overview or self.show_column_info or self.show_numeric_stats

        if show_any_section:
            # Show specific sections only
            if self.show_dataset_overview:
                overview_table = self._create_data_overview_table(df)
                self.console.print(overview_table)

            if self.show_column_info:
                column_table = self._create_column_info_table(df)
                self.console.print("\n", column_table)

            if self.show_numeric_stats:
                stats_table = self._create_statistics_table(df)
                self.console.print("\n", stats_table)
        else:
            # Default: show only data preview
            preview_table = self._create_data_preview_table(df, start_row, end_row, start_col, end_col)
            self.console.print("\n", preview_table)

        # Generate detailed info file if requested
        if self.output_info:
            self.console.print(f"\n[dim]Generating detailed info file: {self.output_info}[/dim]")
            detailed_info = self._generate_detailed_info(df)
            cleaned_info = self._clean_for_toml(detailed_info)

            with open(self.output_info, "wb") as f:
                tomli_w.dump(cleaned_info, f)

            self.console.print(f"[green]✓[/green] Detailed info saved to: {self.output_info}")

        self.console.print("\n[bold green]✅ Analysis complete![/bold green]")

    def get_info(self) -> dict[str, Any]:
        """Get information about the view configuration.

        Returns:
            Dictionary containing the current configuration
        """
        return {
            "input_file": str(self.input_file),
            "rows": self.rows,
            "cols": self.cols,
            "output_info": str(self.output_info) if self.output_info else None,
            "sep": self.sep,
            "sheet": self.sheet,
            "show_dataset_overview": self.show_dataset_overview,
            "show_column_info": self.show_column_info,
            "show_numeric_stats": self.show_numeric_stats,
            "display_mode": self.display_mode,
        }


def main():
    """Main function for command-line usage."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Quickly view and profile datasets")
    parser.add_argument("input_file", help="Path to the input data file")
    parser.add_argument("--rows", "-r", default=50, help="Number of rows to display or range (e.g., '50' or '10,60')")
    parser.add_argument("--cols", "-c", default=25, help="Number of columns to display or range (e.g., '25' or '5,15')")
    parser.add_argument("--output-info", "--oi", help="Output detailed info to TOML file")
    parser.add_argument("--sep", help="Custom separator for CSV files")
    parser.add_argument("--sheet", help="Sheet name or number for Excel files")
    parser.add_argument(
        "--display-mode",
        choices=["auto", "normal", "rotated", "wrapped"],
        default="auto",
        help="Display mode for data preview (default: auto)",
    )

    # New display section flags
    parser.add_argument("--dataset-overview", "--do", action="store_true", help="Show dataset overview only")
    parser.add_argument("--column-info", "--ci", action="store_true", help="Show column information only")
    parser.add_argument("--numeric-stats", "--ns", action="store_true", help="Show numeric statistics only")

    args = parser.parse_args()

    try:
        viewer = PreViewData(
            input_file=args.input_file,
            rows=args.rows,
            cols=args.cols,
            output_info=args.output_info,
            sep=args.sep,
            sheet=args.sheet,
            show_dataset_overview=args.dataset_overview,
            show_column_info=args.column_info,
            show_numeric_stats=args.numeric_stats,
            display_mode=args.display_mode,
        )
        viewer.view()

    except (FileNotFoundError, ValueError, OSError) as e:
        console = Console()
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
