"""Unit tests for pre_select_data.py module.

This module contains comprehensive tests for the PreSelectDataPolars class,
covering functionality for data subset selection, file I/O, parameter parsing,
and error handling scenarios.
"""

import gzip
from pathlib import Path

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
        result = processor._parse_index_columns()  # type: ignore[reportPrivateUsage]
        assert result == [2]

    def test_parse_index_columns_multiple(self, csv_file: Path) -> None:
        """Test parsing multiple index columns."""
        processor = PreSelectDataPolars(input_file=str(csv_file), index_col="0,2,4")
        result = processor._parse_index_columns()  # type: ignore[reportPrivateUsage]
        assert result == [0, 2, 4]

    def test_parse_multi_range_parameter_single_int(self, csv_file: Path) -> None:
        """Test parsing multi-range parameter with single integer."""
        processor = PreSelectDataPolars(input_file=str(csv_file))
        result = processor._parse_multi_range_parameter(5, "test_param")  # type: ignore[reportPrivateUsage]
        assert result == [(5, None)]

    def test_parse_multi_range_parameter_single_range(self, csv_file: Path) -> None:
        """Test parsing multi-range parameter with single range string."""
        processor = PreSelectDataPolars(input_file=str(csv_file))
        result = processor._parse_multi_range_parameter("2:5", "test_param")  # type: ignore[reportPrivateUsage]
        assert result == [(2, 6)]  # end is exclusive, so 5+1=6

    def test_parse_multi_range_parameter_multiple_ranges(self, csv_file: Path) -> None:
        """Test parsing multi-range parameter with multiple ranges."""
        processor = PreSelectDataPolars(input_file=str(csv_file))
        result = processor._parse_multi_range_parameter("1:3,5:7", "test_param")  # type: ignore[reportPrivateUsage]
        assert result == [(1, 4), (5, 8)]  # Both ranges converted to exclusive end

    def test_read_csv_file(self, csv_file: Path) -> None:
        """Test reading CSV file."""
        processor = PreSelectDataPolars(input_file=str(csv_file))
        df = processor._read_data()  # type: ignore[reportPrivateUsage]

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (5, 5)  # 5 rows, 5 columns as per sample_data

    def test_read_gzipped_csv_file(self, gzipped_csv_file: Path) -> None:
        """Test reading gzipped CSV file."""
        processor = PreSelectDataPolars(input_file=str(gzipped_csv_file))
        df = processor._read_data()  # type: ignore[reportPrivateUsage]

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (5, 5)  # 5 rows, 5 columns as per sample_data

    def test_select_subset_basic(self, csv_file: Path) -> None:
        """Test basic subset selection."""
        processor = PreSelectDataPolars(input_file=str(csv_file), index_col=0, col_start=1, row_index=0, row_start=1)
        df = processor._read_data()  # type: ignore[reportPrivateUsage]
        subset = processor._select_subset(df)  # type: ignore[reportPrivateUsage]

        # Should have 4 rows (excluding header) and 5 columns (index + 4 data columns)
        assert subset.shape == (4, 5)  # 4 data rows after excluding header row

        # Check that the index column is first
        assert subset.columns[0] == "ID"

    def test_select_subset_with_ranges(self, csv_file: Path) -> None:
        """Test subset selection with column and row ranges."""
        processor = PreSelectDataPolars(
            input_file=str(csv_file),
            index_col=0,
            col_start="1:3",  # columns 1-3
            row_index=0,
            row_start="1:2",  # rows 1-2
        )
        df = processor._read_data()  # type: ignore[reportPrivateUsage]
        subset = processor._select_subset(df)  # type: ignore[reportPrivateUsage]

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
        df = processor._read_data()  # type: ignore[reportPrivateUsage]
        subset = processor._select_subset(df)  # type: ignore[reportPrivateUsage]

        # Check that index values are concatenated with separator
        index_values = subset.select(pl.col(subset.columns[0])).to_series().to_list()
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


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_file_not_found(self) -> None:
        """Test error handling for non-existent file."""
        processor = PreSelectDataPolars(input_file="nonexistent.csv")

        with pytest.raises(FileNotFoundError):
            processor._read_data()  # type: ignore[reportPrivateUsage]

    def test_invalid_index_col_format(self, tmp_path: Path) -> None:
        """Test error handling for invalid index_col format."""
        csv_file = tmp_path / "test.csv"
        # Create minimal test file
        pl.DataFrame({"col1": [1, 2], "col2": [3, 4]}).write_csv(str(csv_file))

        processor = PreSelectDataPolars(input_file=str(csv_file), index_col="invalid,format,")

        with pytest.raises(ValueError, match="Invalid index_col format"):
            processor._parse_index_columns()  # type: ignore[reportPrivateUsage]

    def test_invalid_range_parameter(self, tmp_path: Path) -> None:
        """Test error handling for invalid range parameters."""
        csv_file = tmp_path / "test.csv"
        pl.DataFrame({"col1": [1, 2], "col2": [3, 4]}).write_csv(str(csv_file))

        processor = PreSelectDataPolars(input_file=str(csv_file))

        with pytest.raises(ValueError, match="Invalid.*range"):
            processor._parse_multi_range_parameter("5:3", "test_param")  # type: ignore[reportPrivateUsage]  # start > end

    def test_out_of_bounds_parameters(self, tmp_path: Path) -> None:
        """Test error handling for out-of-bounds parameters."""
        csv_file = tmp_path / "test.csv"
        # Create small test file (2x2)
        pl.DataFrame({"col1": ["A", "B"], "col2": ["C", "D"]}).write_csv(str(csv_file))

        processor = PreSelectDataPolars(
            input_file=str(csv_file),
            index_col=5,  # Out of bounds
            col_start=1,
            row_index=0,
            row_start=1,
        )

        df = processor._read_data()  # type: ignore[reportPrivateUsage]
        with pytest.raises(ValueError, match="index_col.*out of bounds"):
            processor._select_subset(df)  # type: ignore[reportPrivateUsage]
