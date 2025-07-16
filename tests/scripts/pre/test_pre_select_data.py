"""Unit tests for pre_select_data.py module.

This module contains comprehensive tests for the PreSelectDataPolars class,
covering functionality for data subset selection, file I/O, parameter parsing,
and error handling scenarios.
"""
# pyright: reportPrivateUsage=false

import gzip
import os
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from btools.scripts.pre.pre_select_data import PreSelectDataPolars


class TestPreSelectDataPolars:
    """Test suite for PreSelectDataPolars class."""

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        """Create sample test data as Polars DataFrame."""
        return pl.DataFrame(
            {
                "column_1": ["ID", "A", "B", "C", "D"],
                "column_2": ["Name", "Alice", "Bob", "Charlie", "David"],
                "column_3": ["Age", "25", "30", "35", "40"],
                "column_4": ["City", "NY", "LA", "Chicago", "Miami"],
                "column_5": ["Score", "85", "92", "78", "88"],
            }
        )

    @pytest.fixture
    def csv_file(self, sample_data: pl.DataFrame, tmp_path: Path) -> Path:
        """Create a temporary CSV file with sample data."""
        file_path = tmp_path / "test_data.csv"
        # Write without header to match how the PreSelectDataPolars expects data
        sample_data.write_csv(str(file_path), include_header=False)
        return file_path

    @pytest.fixture
    def gzipped_csv_file(self, sample_data: pl.DataFrame, tmp_path: Path) -> Path:
        """Create a temporary gzipped CSV file with sample data."""
        file_path = tmp_path / "test_data.csv.gz"
        # Write to temp CSV first (without header to match the gzipped version)
        temp_csv = tmp_path / "temp.csv"
        sample_data.write_csv(str(temp_csv), include_header=False)

        # Compress to .gz
        with open(temp_csv, "rb") as f_in, gzip.open(file_path, "wb") as f_out:
            f_out.write(f_in.read())

        temp_csv.unlink()  # Clean up temp file
        return file_path

    def test_initialization_default_params(self, csv_file: Path) -> None:
        """Test initialization with default parameters."""
        processor = PreSelectDataPolars(input_file=str(csv_file))

        assert processor.input_file == csv_file
        assert processor.index_col == 0
        assert processor.col_start == 1
        assert processor.row_index == 0
        assert processor.row_start == 1
        assert processor.sep is None
        assert processor.sheet is None
        assert processor.index_separator == "#"

    def test_initialization_custom_params(self, csv_file: Path) -> None:
        """Test initialization with custom parameters."""
        output_file = "custom_output.csv"
        processor = PreSelectDataPolars(
            input_file=str(csv_file),
            output_file=output_file,
            index_col="0,1",
            col_start="2:4",
            row_index=0,
            row_start="1:3",
            sep=",",
            sheet="Sheet1",
            index_separator="|",
        )

        assert str(processor.output_file) == output_file
        assert processor.index_col == "0,1"
        assert processor.col_start == "2:4"
        assert processor.row_start == "1:3"
        assert processor.sep == ","
        assert processor.sheet == "Sheet1"
        assert processor.index_separator == "|"

    def test_parse_index_columns_single(self, csv_file: Path) -> None:
        """Test parsing single index column."""
        processor = PreSelectDataPolars(input_file=str(csv_file), index_col=2)
        result = processor._parse_index_columns()  # type: ignore
        assert result == [2]

    def test_parse_index_columns_multiple(self, csv_file: Path) -> None:
        """Test parsing multiple index columns."""
        processor = PreSelectDataPolars(input_file=str(csv_file), index_col="0,2,4")
        result = processor._parse_index_columns()  # type: ignore
        assert result == [0, 2, 4]

    def test_parse_multi_range_parameter_single_int(self, csv_file: Path) -> None:
        """Test parsing multi-range parameter with single integer."""
        processor = PreSelectDataPolars(input_file=str(csv_file))
        result = processor._parse_multi_range_parameter(5, "test_param")  # type: ignore
        assert result == [(5, None)]

    def test_parse_multi_range_parameter_single_range(self, csv_file: Path) -> None:
        """Test parsing multi-range parameter with single range string."""
        processor = PreSelectDataPolars(input_file=str(csv_file))
        result = processor._parse_multi_range_parameter("2:5", "test_param")  # type: ignore
        assert result == [(2, 6)]  # end is exclusive, so 5+1=6

    def test_parse_multi_range_parameter_multiple_ranges(self, csv_file: Path) -> None:
        """Test parsing multi-range parameter with multiple ranges."""
        processor = PreSelectDataPolars(input_file=str(csv_file))
        result = processor._parse_multi_range_parameter("1:3,5:7", "test_param")  # type: ignore
        assert result == [(1, 4), (5, 8)]  # Both ranges converted to exclusive end

    def test_read_csv_file(self, csv_file: Path) -> None:
        """Test reading CSV file."""
        processor = PreSelectDataPolars(input_file=str(csv_file))
        df = processor._read_data()  # type: ignore

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (5, 5)  # 5 rows, 5 columns as per sample_data

    def test_read_gzipped_csv_file(self, gzipped_csv_file: Path) -> None:
        """Test reading gzipped CSV file."""
        processor = PreSelectDataPolars(input_file=str(gzipped_csv_file))
        df = processor._read_data()  # type: ignore

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (5, 5)  # 5 rows, 5 columns as per sample_data

    def test_select_subset_basic(self, csv_file: Path) -> None:
        """Test basic subset selection."""
        processor = PreSelectDataPolars(input_file=str(csv_file), index_col=0, col_start=1, row_index=0, row_start=1)
        df = processor._read_data()  # type: ignore
        subset, original_headers = processor._select_subset(df)  # type: ignore

        # Should have 4 rows (excluding header) and 5 columns (index + 4 data columns)
        assert subset.shape == (4, 5)  # 4 data rows after excluding header row

        # Check that the original headers are preserved
        assert original_headers[0] == "ID"

    def test_select_subset_with_ranges(self, csv_file: Path) -> None:
        """Test subset selection with column and row ranges."""
        processor = PreSelectDataPolars(
            input_file=str(csv_file),
            index_col=0,
            col_start="1:3",  # columns 1-3
            row_index=0,
            row_start="1:2",  # rows 1-2
        )
        df = processor._read_data()  # type: ignore
        subset, _original_headers = processor._select_subset(df)  # type: ignore

        # Should have 2 rows and 4 columns (index + 3 data columns)
        assert subset.shape == (2, 4)

    def test_select_subset_multiple_index_cols(self, csv_file: Path) -> None:
        """Test subset selection with multiple index columns."""
        processor = PreSelectDataPolars(
            input_file=str(csv_file),
            index_col="0,1",  # Use both ID and Name as index
            col_start=2,
            row_index=0,
            row_start=1,
            index_separator="|",
        )
        df = processor._read_data()  # type: ignore
        subset, _original_headers = processor._select_subset(df)  # type: ignore

        # Check that index values are concatenated with separator
        index_values = subset.select(pl.col("index_col")).to_series().to_list()
        assert "|" in index_values[0]  # Should contain separator

    def test_process_end_to_end(self, csv_file: Path, tmp_path: Path) -> None:
        """Test complete processing pipeline."""
        output_file = tmp_path / "output.csv"
        processor = PreSelectDataPolars(
            input_file=str(csv_file), output_file=str(output_file), index_col=0, col_start=1, row_index=0, row_start=1
        )

        processor.process()

        # Check that output file was created
        assert output_file.exists()

        # Check that we can read the output
        result_df = pl.read_csv(output_file)
        assert result_df.shape[0] == 4  # 4 data rows after excluding header

    def test_get_info(self, csv_file: Path) -> None:
        """Test get_info method."""
        processor = PreSelectDataPolars(input_file=str(csv_file), index_col=0, col_start=1)

        info = processor.get_info()

        assert "input_file" in info
        assert "output_file" in info
        assert "index_col" in info
        assert info["index_col"] == 0
        assert info["col_start"] == 1

    def test_transpose_data_functionality(self, csv_file: Path) -> None:
        """Test the transpose functionality."""
        processor = PreSelectDataPolars(input_file=str(csv_file))
        df = processor._read_data()  # type: ignore

        # Original shape: 5 rows × 5 columns
        assert df.shape == (5, 5)

        # Test transpose
        transposed_df = processor._transpose_data(df)  # type: ignore

        # After transpose: 5 columns × 5 rows (dimensions swapped)
        assert transposed_df.shape == (5, 5)

        # Check that column names follow the expected pattern
        expected_columns = ["column_1", "column_2", "column_3", "column_4", "column_5"]
        assert transposed_df.columns == expected_columns

        # Check that the first row of transposed data matches the first column of original
        original_first_col = df.select(pl.col("column_1")).to_series().to_list()
        transposed_first_row = transposed_df.slice(0, 1).select(pl.all()).row(0)
        assert list(transposed_first_row) == original_first_col

    def test_transpose_with_process_end_to_end(self, csv_file: Path, tmp_path: Path) -> None:
        """Test complete processing pipeline with transpose enabled."""
        output_file = tmp_path / "transposed_output.csv"
        processor = PreSelectDataPolars(
            input_file=str(csv_file),
            output_file=str(output_file),
            index_col=0,
            col_start=1,
            row_index=0,
            row_start=1,
            transpose=True,
        )

        processor.process()

        # Check that output file was created
        assert output_file.exists()

        # Check that we can read the output
        result_df = pl.read_csv(output_file)

        # With transpose, original 5×5 becomes 5×5, then after selection should be 4×4
        # (excluding 1 row for header and 1 column for index)
        assert result_df.shape[0] == 4  # 4 data rows after excluding header

    def test_transpose_initialization_parameter(self, csv_file: Path) -> None:
        """Test that transpose parameter is properly initialized and stored."""
        # Test default value (False)
        processor_default = PreSelectDataPolars(input_file=str(csv_file))
        assert processor_default.transpose is False

        # Test explicit False
        processor_false = PreSelectDataPolars(input_file=str(csv_file), transpose=False)
        assert processor_false.transpose is False

        # Test explicit True
        processor_true = PreSelectDataPolars(input_file=str(csv_file), transpose=True)
        assert processor_true.transpose is True

    def test_transpose_get_info_includes_parameter(self, csv_file: Path) -> None:
        """Test that get_info method includes transpose parameter."""
        processor = PreSelectDataPolars(input_file=str(csv_file), transpose=True)

        info = processor.get_info()

        assert "transpose" in info
        assert info["transpose"] is True

        # Test with False value
        processor_false = PreSelectDataPolars(input_file=str(csv_file), transpose=False)
        info_false = processor_false.get_info()
        assert info_false["transpose"] is False

    def test_transpose_empty_dataframe(self, tmp_path: Path) -> None:
        """Test transpose with empty DataFrame."""
        # Create an empty CSV file
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")

        processor = PreSelectDataPolars(input_file=str(empty_file), transpose=True)

        # This should handle the empty case gracefully
        try:
            df = processor._read_data()  # type: ignore
            if df.height > 0:  # Only test transpose if we have data
                transposed = processor._transpose_data(df)  # type: ignore
                assert isinstance(transposed, pl.DataFrame)
        except (OSError, ValueError, pl.exceptions.ComputeError):
            # If reading empty file fails, that's acceptable for this test
            pass

    def test_transpose_rectangular_data(self, tmp_path: Path) -> None:
        """Test transpose with non-square (rectangular) data."""
        # Create rectangular data (3 rows × 4 columns)
        rectangular_data = pl.DataFrame(
            {
                "column_1": ["A", "D", "G"],
                "column_2": ["B", "E", "H"],
                "column_3": ["C", "F", "I"],
                "column_4": ["X", "Y", "Z"],
            }
        )

        rect_file = tmp_path / "rectangular.csv"
        rectangular_data.write_csv(str(rect_file), include_header=False)

        processor = PreSelectDataPolars(input_file=str(rect_file), transpose=True)
        df = processor._read_data()  # type: ignore

        # Original: 3 rows × 4 columns
        assert df.shape == (3, 4)

        transposed = processor._transpose_data(df)  # type: ignore

        # After transpose: 4 rows × 3 columns
        assert transposed.shape == (4, 3)

        # Check column naming
        expected_columns = ["column_1", "column_2", "column_3"]
        assert transposed.columns == expected_columns

    @patch.dict(os.environ, {"GZIP_OUT": "True"})
    def test_excel_sheet_name_in_filename_string_sheet(self, tmp_path: Path) -> None:
        """Test that string sheet names are included in output filename for Excel files."""
        # Create a mock Excel file path (doesn't need to exist for filename generation test)
        excel_file = tmp_path / "test_data.xlsx"

        processor = PreSelectDataPolars(
            input_file=str(excel_file),
            sheet="Data Sheet",  # Sheet name with space
        )

        output_filename = str(processor.output_file)
        assert "_sheet_Data-Sheet" in output_filename
        assert output_filename.endswith(".csv.gz")  # Should have .gz due to mocked environment

    @patch.dict(os.environ, {}, clear=True)  # Clear environment to ensure default behavior
    def test_excel_sheet_name_in_filename_string_sheet_no_gzip(self, tmp_path: Path) -> None:
        """Test that string sheet names are included in output filename without gzip compression."""
        # Create a mock Excel file path (doesn't need to exist for filename generation test)
        excel_file = tmp_path / "test_data.xlsx"

        processor = PreSelectDataPolars(
            input_file=str(excel_file),
            sheet="Data Sheet",  # Sheet name with space
        )

        output_filename = str(processor.output_file)
        assert "_sheet_Data-Sheet" in output_filename
        assert output_filename.endswith(".csv")  # Should NOT have .gz due to default behavior

    def test_excel_sheet_name_in_filename_integer_sheet(self, tmp_path: Path) -> None:
        """Test that integer sheet indices are included in output filename for Excel files."""
        excel_file = tmp_path / "test_data.xlsx"

        processor = PreSelectDataPolars(
            input_file=str(excel_file),
            sheet=2,  # Sheet index
        )

        output_filename = str(processor.output_file)
        assert "_sheet_2" in output_filename

    def test_excel_sheet_name_in_filename_default_sheet(self, tmp_path: Path) -> None:
        """Test that default sheet (None) results in _sheet_0 in filename for Excel files."""
        excel_file = tmp_path / "test_data.xlsx"

        processor = PreSelectDataPolars(
            input_file=str(excel_file),
            sheet=None,  # Default sheet
        )

        output_filename = str(processor.output_file)
        assert "_sheet_0" in output_filename

    def test_non_excel_files_no_sheet_suffix(self, tmp_path: Path) -> None:
        """Test that non-Excel files don't get sheet suffixes even when sheet is specified."""
        csv_file = tmp_path / "test_data.csv"

        processor = PreSelectDataPolars(
            input_file=str(csv_file),
            sheet="SomeSheet",  # This should be ignored for CSV files
        )

        output_filename = processor.output_file.name  # Just the filename, not the full path
        assert "_sheet_" not in output_filename

    def test_sanitize_sheet_name(self, tmp_path: Path) -> None:
        """Test that sheet names are properly sanitized for filename use."""
        excel_file = tmp_path / "test_data.xlsx"

        processor = PreSelectDataPolars(
            input_file=str(excel_file),
            sheet="My Data Sheet",  # Multiple spaces
        )

        # Test the sanitize method directly
        sanitized = processor._sanitize_sheet_name("My Data Sheet")  # type: ignore
        assert sanitized == "My-Data-Sheet"

        # Test in filename
        output_filename = str(processor.output_file)
        assert "_sheet_My-Data-Sheet" in output_filename

    def test_write_and_read_parquet(self, csv_file: Path, tmp_path: Path) -> None:
        """Test writing to Parquet and reading back preserves header row as first row."""
        parquet_file = tmp_path / "output.parquet"
        processor = PreSelectDataPolars(
            input_file=str(csv_file),
            output_file=str(parquet_file),
            index_col=0,
            col_start=1,
            row_index=0,
            row_start=1,
            parquet_out=True,
        )
        processor.process()
        assert parquet_file.exists()

        # Read back the Parquet file
        df = pl.read_parquet(parquet_file)
        # First row should be the original header
        header_row = df.slice(0, 1)
        # Remaining rows are data
        data_rows = df.slice(1, df.height - 1)
        # Check header values
        expected_header = ["ID", "Name", "Age", "City", "Score"]
        # The header row includes the index column as the first value
        assert list(header_row.row(0)) == expected_header
        # Check data shape
        assert data_rows.shape[0] == 4
        assert data_rows.shape[1] == 5

    def test_parquet_to_csv_roundtrip(self, csv_file: Path, tmp_path: Path) -> None:
        """Test Parquet to CSV roundtrip preserves header row and data shape."""
        parquet_file = tmp_path / "output.parquet"
        csv_out_file = tmp_path / "output_from_parquet.csv"
        # Write Parquet
        processor = PreSelectDataPolars(
            input_file=str(csv_file),
            output_file=str(parquet_file),
            index_col=0,
            col_start=1,
            row_index=0,
            row_start=1,
            parquet_out=True,
        )
        processor.process()
        # Read Parquet and write to CSV
        processor2 = PreSelectDataPolars(
            input_file=str(parquet_file),
            output_file=str(csv_out_file),
            index_col=0,
            col_start=1,
            row_index=0,
            row_start=1,
            parquet_out=False,
        )
        processor2.process()
        # Check CSV output
        assert csv_out_file.exists()
        df = pl.read_csv(csv_out_file)
        # First row should be the first data row, not the header
        first_row = df.slice(0, 1)
        expected_first_data_row = ["A", "Alice", "25", "NY", "85"]
        assert [str(x) for x in first_row.row(0)] == expected_first_data_row
        # Data shape: 4 rows (data), 5 columns
        assert df.shape[0] == 4
        assert df.shape[1] == 5
