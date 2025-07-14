"""Comprehensive unit tests for pre_select_data.py module.

This module contains tests for both PreSelectData (pandas) and PreSelectDataPolars classes,
testing all options and edge cases.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from btools.scripts.pre.pre_select_data import PreSelectData, PreSelectDataPolars

test_data_path = "data/data_pre"


class TestPreSelectDataPolars:
    """Test suite for PreSelectDataPolars class."""

    @pytest.fixture
    def test_data_dir(self) -> Path:
        """Get the test data directory."""
        return Path(__file__).parent.parent.parent / test_data_path

    @pytest.fixture
    def csv_file(self, test_data_dir: Path) -> Path:
        """Path to test CSV file."""
        return test_data_dir / "test_data.csv"

    @pytest.fixture
    def tsv_file(self, test_data_dir: Path) -> Path:
        """Path to test TSV file."""
        return test_data_dir / "test_data.tsv"

    @pytest.fixture
    def excel_file(self, test_data_dir: Path) -> Path:
        """Path to test Excel file."""
        return test_data_dir / "test_data.xlsx"

    @pytest.fixture
    def pipe_file(self, test_data_dir: Path) -> Path:
        """Path to test pipe-delimited file."""
        return test_data_dir / "test_data_pipe.txt"

    @pytest.fixture
    def temp_output_dir(self) -> Generator[Path, None, None]:
        """Create a temporary directory for output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_init_default_params(self, csv_file: Path) -> None:
        """Test initialization with default parameters."""
        processor = PreSelectDataPolars(str(csv_file))

        assert processor.input_file == csv_file
        # Output file depends on GZIP_OUT environment variable
        expected_output = csv_file.with_stem(csv_file.stem + "_subset").with_suffix(".csv.gz")
        assert processor.output_file == expected_output
        assert processor.index_col == 0
        assert processor.col_start == 1
        assert processor.row_index == 0
        assert processor.row_start == 1
        assert processor.sep is None
        assert processor.sheet is None
        assert processor.index_separator == "#"

    def test_init_custom_params(self, csv_file: Path, temp_output_dir: Path) -> None:
        """Test initialization with custom parameters."""
        output_file = temp_output_dir / "custom_output.csv"
        processor = PreSelectDataPolars(
            input_file=str(csv_file),
            output_file=str(output_file),
            index_col="1,2",
            col_start="2,4",
            row_index=1,
            row_start="2,5",
            sep=",",
            sheet="Sheet1",
            index_separator="|",
        )

        assert processor.input_file == csv_file
        assert processor.output_file == output_file
        assert processor.index_col == "1,2"
        assert processor.col_start == "2,4"
        assert processor.row_index == 1
        assert processor.row_start == "2,5"
        assert processor.sep == ","
        assert processor.sheet == "Sheet1"
        assert processor.index_separator == "|"

    def test_parse_index_columns_single(self, csv_file: Path) -> None:
        """Test parsing single index column."""
        processor = PreSelectDataPolars(str(csv_file), index_col=2)
        result = processor._parse_index_columns()  # type: ignore[attr-defined]
        assert result == [2]

    def test_parse_index_columns_multiple(self, csv_file: Path) -> None:
        """Test parsing multiple index columns."""
        processor = PreSelectDataPolars(str(csv_file), index_col="1,2,4")
        result = processor._parse_index_columns()  # type: ignore[attr-defined]
        assert result == [1, 2, 4]

    def test_parse_index_columns_invalid(self, csv_file: Path) -> None:
        """Test parsing invalid index columns."""
        processor = PreSelectDataPolars(str(csv_file), index_col="1,abc,4")
        with pytest.raises(ValueError, match="Invalid index_col format"):
            processor._parse_index_columns()  # type: ignore[attr-defined]

    def test_parse_range_parameter_single_int(self, csv_file: Path) -> None:
        """Test parsing single integer range parameter."""
        processor = PreSelectDataPolars(str(csv_file))
        result = processor._parse_range_parameter(5, "test_param")  # type: ignore[attr-defined]
        assert result == (5, None)

    def test_parse_range_parameter_range_string(self, csv_file: Path) -> None:
        """Test parsing range string parameter."""
        processor = PreSelectDataPolars(str(csv_file))
        result = processor._parse_range_parameter("1,5", "test_param")  # type: ignore[attr-defined]
        assert result == (1, 6)  # Polars adds 1 to end for inclusive range

    def test_parse_range_parameter_invalid_format(self, csv_file: Path) -> None:
        """Test parsing invalid range parameter format."""
        processor = PreSelectDataPolars(str(csv_file))
        with pytest.raises(ValueError, match="Invalid test_param range format"):
            processor._parse_range_parameter("1,2,3", "test_param")  # type: ignore[attr-defined]

    def test_parse_range_parameter_invalid_order(self, csv_file: Path) -> None:
        """Test parsing range parameter with start > end."""
        processor = PreSelectDataPolars(str(csv_file))
        with pytest.raises(ValueError, match="start must be <= end"):
            processor._parse_range_parameter("5,2", "test_param")  # type: ignore[attr-defined]

    def test_parse_range_parameter_invalid_values(self, csv_file: Path) -> None:
        """Test parsing range parameter with invalid values."""
        processor = PreSelectDataPolars(str(csv_file))
        with pytest.raises(ValueError, match="Invalid test_param format"):
            processor._parse_range_parameter("abc", "test_param")  # type: ignore[attr-defined]

    def test_generate_range_suffix(self, csv_file: Path) -> None:
        """Test generating range suffix for filename."""
        processor = PreSelectDataPolars(str(csv_file))
        result = processor._generate_range_suffix(10, 5)  # type: ignore[attr-defined]
        assert result == "_row10_col5"

    def test_generate_output_filename(self, csv_file: Path) -> None:
        """Test generating default output filename."""
        processor = PreSelectDataPolars(str(csv_file))
        # Output file depends on GZIP_OUT environment variable
        expected = csv_file.with_stem(csv_file.stem + "_subset").with_suffix(".csv.gz")
        assert processor._generate_output_filename() == expected  # type: ignore[attr-defined]

    def test_update_output_filename_with_range(self, csv_file: Path, temp_output_dir: Path) -> None:
        """Test updating output filename with range information."""
        processor = PreSelectDataPolars(str(csv_file))
        output_file = temp_output_dir / "test_output.csv"
        result = processor._update_output_filename_with_range(output_file, 10, 5)  # type: ignore[attr-defined]
        expected = temp_output_dir / "test_output_row10_col5.csv"
        assert result == expected

    def test_read_data_csv(self, csv_file: Path) -> None:
        """Test reading CSV file."""
        processor = PreSelectDataPolars(str(csv_file))
        df = processor._read_data()  # type: ignore[attr-defined]

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (11, 5)  # 10 data rows + 1 header row

    def test_read_data_tsv(self, tsv_file: Path) -> None:
        """Test reading TSV file."""
        processor = PreSelectDataPolars(str(tsv_file))
        df = processor._read_data()  # type: ignore[attr-defined]

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (6, 5)  # 5 data rows + 1 header row

    def test_read_data_excel(self, excel_file: Path) -> None:
        """Test reading Excel file."""
        processor = PreSelectDataPolars(str(excel_file))
        df = processor._read_data()  # type: ignore[attr-defined]

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (11, 5)  # 10 data rows + 1 header row

    def test_read_data_excel_with_sheet(self, excel_file: Path) -> None:
        """Test reading Excel file with specific sheet."""
        processor = PreSelectDataPolars(str(excel_file), sheet="Products")
        df = processor._read_data()  # type: ignore[attr-defined]

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (4, 3)  # 3 data rows + 1 header row

    def test_read_data_custom_separator(self, pipe_file: Path) -> None:
        """Test reading file with custom separator."""
        processor = PreSelectDataPolars(str(pipe_file), sep="|")
        df = processor._read_data()  # type: ignore[attr-defined]

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (6, 5)  # 5 data rows + 1 header row

    def test_read_data_tab_separator(self, tsv_file: Path) -> None:
        """Test reading file with tab separator specified as string."""
        processor = PreSelectDataPolars(str(tsv_file), sep="\\t")
        df = processor._read_data()  # type: ignore[attr-defined]

        assert isinstance(df, pl.DataFrame)
        assert df.shape == (6, 5)

    def test_read_data_file_not_found(self, temp_output_dir: Path) -> None:
        """Test reading non-existent file."""
        non_existent_file = temp_output_dir / "non_existent.csv"
        processor = PreSelectDataPolars(str(non_existent_file))

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            processor._read_data()  # type: ignore[attr-defined]

    def test_select_subset_basic(self, csv_file: Path) -> None:
        """Test basic subset selection."""
        processor = PreSelectDataPolars(str(csv_file), index_col=0, col_start=1, row_start=1)
        df = processor._read_data()  # type: ignore[attr-defined]
        result = processor._select_subset(df)  # type: ignore[attr-defined]

        assert isinstance(result, pl.DataFrame)
        # Should have ID column + 4 data columns (Name, Age, Score, Category)
        assert result.shape == (10, 5)

    def test_select_subset_with_ranges(self, csv_file: Path) -> None:
        """Test subset selection with column and row ranges."""
        processor = PreSelectDataPolars(str(csv_file), index_col=0, col_start="1,2", row_start="1,3")
        df = processor._read_data()  # type: ignore[attr-defined]
        result = processor._select_subset(df)  # type: ignore[attr-defined]

        assert isinstance(result, pl.DataFrame)
        # Should have ID column + 2 data columns (Name, Age) and 3 rows
        assert result.shape == (3, 3)

    def test_select_subset_multiple_index_columns(self, csv_file: Path) -> None:
        """Test subset selection with multiple index columns."""
        processor = PreSelectDataPolars(str(csv_file), index_col="0,1", col_start=2, index_separator="|")
        df = processor._read_data()  # type: ignore[attr-defined]
        result = processor._select_subset(df)  # type: ignore[attr-defined]

        assert isinstance(result, pl.DataFrame)
        # Should have concatenated index + remaining columns
        assert result.shape == (10, 4)
        # Check that index names are concatenated
        assert "|" in result.columns[0]

    def test_select_subset_out_of_bounds_index_col(self, csv_file: Path) -> None:
        """Test subset selection with out-of-bounds index column."""
        processor = PreSelectDataPolars(str(csv_file), index_col=10)  # Only 5 columns (0-4)
        df = processor._read_data()  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="index_col .* is out of bounds"):
            processor._select_subset(df)  # type: ignore[attr-defined]

    def test_select_subset_out_of_bounds_row_index(self, csv_file: Path) -> None:
        """Test subset selection with out-of-bounds row index."""
        processor = PreSelectDataPolars(str(csv_file), row_index=20)  # Only 11 rows (0-10)
        df = processor._read_data()  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="row_index .* is out of bounds"):
            processor._select_subset(df)  # type: ignore[attr-defined]

    def test_select_subset_out_of_bounds_col_start(self, csv_file: Path) -> None:
        """Test subset selection with out-of-bounds col_start."""
        processor = PreSelectDataPolars(str(csv_file), col_start=10)  # Only 5 columns (0-4)
        df = processor._read_data()  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="col_start .* is out of bounds"):
            processor._select_subset(df)  # type: ignore[attr-defined]

    def test_select_subset_out_of_bounds_row_start(self, csv_file: Path) -> None:
        """Test subset selection with out-of-bounds row_start."""
        processor = PreSelectDataPolars(str(csv_file), row_start=20)  # Only 11 rows (0-10)
        df = processor._read_data()  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="row_start .* is out of bounds"):
            processor._select_subset(df)  # type: ignore[attr-defined]

    @patch("builtins.print")
    def test_process_basic(self, mock_print: Any, csv_file: Path, temp_output_dir: Path) -> None:
        """Test basic processing workflow."""
        output_file = temp_output_dir / "output.csv"
        processor = PreSelectDataPolars(str(csv_file), str(output_file))

        processor.process()

        assert output_file.exists()
        # Read the output file to verify content
        result_df = pl.read_csv(output_file)
        assert result_df.shape == (10, 5)  # 10 rows, 5 columns (ID + 4 data columns)

    @patch("builtins.print")
    def test_process_with_ranges(self, mock_print: Any, csv_file: Path, temp_output_dir: Path) -> None:
        """Test processing with range parameters."""
        output_file = temp_output_dir / "output.csv"
        processor = PreSelectDataPolars(str(csv_file), str(output_file), col_start="1,2", row_start="1,5")

        processor.process()

        # Should generate filename with range suffix
        expected_files = list(temp_output_dir.glob("output_row5_col*.csv"))
        assert len(expected_files) == 1

        result_df = pl.read_csv(expected_files[0])
        assert result_df.shape == (5, 3)  # 5 rows, 3 columns (ID + 2 data columns)

    def test_get_info(self, csv_file: Path) -> None:
        """Test getting processor information."""
        processor = PreSelectDataPolars(str(csv_file), index_col="1,2", col_start="2,4", sep=",")

        info = processor.get_info()

        assert info["input_file"] == str(csv_file)
        assert info["index_col"] == "1,2"
        assert info["col_start"] == "2,4"
        assert info["sep"] == ","

    def test_gzipped_csv_input(self, test_data_dir: Path, temp_output_dir: Path) -> None:
        """Test reading from gzipped CSV files."""
        csv_gz_file = test_data_dir / "test_data.csv.gz"
        output_file = temp_output_dir / "output.csv"

        processor = PreSelectDataPolars(str(csv_gz_file), str(output_file))
        # Test that process completes without errors - this indirectly tests _read_data
        processor.process()

        # Output file should exist
        assert output_file.exists()

    def test_gzipped_tsv_input(self, test_data_dir: Path, temp_output_dir: Path) -> None:
        """Test reading from gzipped TSV files."""
        tsv_gz_file = test_data_dir / "test_data.tsv.gz"
        output_file = temp_output_dir / "output.csv"

        processor = PreSelectDataPolars(str(tsv_gz_file), str(output_file))
        # Test that process completes without errors - this indirectly tests _read_data
        processor.process()

        # Output file should exist
        assert output_file.exists()

    @patch.dict("os.environ", {"GZIP_OUT": "True"})
    def test_gzipped_output_when_env_set(self, csv_file: Path, temp_output_dir: Path) -> None:
        """Test that output files are gzipped when GZIP_OUT=True."""
        processor = PreSelectDataPolars(str(csv_file))

        # Process and check that output file has .gz extension when GZIP_OUT is True
        processor.process()
        assert processor.output_file.suffix == ".gz"

    @patch.dict("os.environ", {"GZIP_OUT": "False"})
    def test_uncompressed_output_when_env_false(self, csv_file: Path, temp_output_dir: Path) -> None:
        """Test that output files are not gzipped when GZIP_OUT=False."""
        processor = PreSelectDataPolars(str(csv_file))

        # Process and check that output file doesn't have .gz extension
        processor.process()
        assert processor.output_file.suffix == ".csv"

    @patch.dict("os.environ", {"GZIP_OUT": "True"})
    def test_process_with_gzipped_output(self, csv_file: Path, temp_output_dir: Path) -> None:
        """Test full processing with gzipped output."""
        processor = PreSelectDataPolars(str(csv_file))
        processor.output_file = temp_output_dir / "test_output.csv.gz"

        # Process should complete without errors
        processor.process()

        # Output file should exist and be readable
        assert processor.output_file.exists()

        # Should be able to read the gzipped output back
        import gzip

        with gzip.open(processor.output_file, "rt") as f:
            content = f.read()
            assert len(content) > 0
            assert "," in content  # Should be CSV format

    def test_file_extension_handling_integration(self, test_data_dir: Path, temp_output_dir: Path) -> None:
        """Test that different file extensions (including .gz) are handled correctly via integration."""
        # Test that .csv.gz files can be processed
        csv_gz_file = test_data_dir / "test_data.csv.gz"
        output_file = temp_output_dir / "test_csv_gz.csv"

        processor = PreSelectDataPolars(str(csv_gz_file), str(output_file))
        processor.process()
        assert output_file.exists()

        # Test that .tsv.gz files can be processed
        tsv_gz_file = test_data_dir / "test_data.tsv.gz"
        output_file2 = temp_output_dir / "test_tsv_gz.csv"

        processor2 = PreSelectDataPolars(str(tsv_gz_file), str(output_file2))
        processor2.process()
        assert output_file2.exists()


class TestPreSelectData:
    """Test suite for PreSelectData class (pandas implementation)."""

    @pytest.fixture
    def test_data_dir(self) -> Path:
        """Get the test data directory."""
        return Path(__file__).parent.parent.parent / test_data_path

    @pytest.fixture
    def csv_file(self, test_data_dir: Path) -> Path:
        """Path to test CSV file."""
        return test_data_dir / "test_data.csv"

    @pytest.fixture
    def excel_file(self, test_data_dir: Path) -> Path:
        """Path to test Excel file."""
        return test_data_dir / "test_data.xlsx"

    @pytest.fixture
    def temp_output_dir(self) -> Generator[Path, None, None]:
        """Create a temporary directory for output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_init_default_params(self, csv_file: Path) -> None:
        """Test initialization with default parameters."""
        processor = PreSelectData(str(csv_file))

        assert processor.input_file == csv_file
        # Output file depends on GZIP_OUT environment variable
        expected_output = csv_file.with_stem(csv_file.stem + "_subset").with_suffix(".csv.gz")
        assert processor.output_file == expected_output
        assert processor.index_col == 0
        assert processor.col_start == 1
        assert processor.row_index == 0
        assert processor.row_start == 1
        assert processor.sep is None
        assert processor.sheet is None
        assert processor.index_separator == "#"

    def test_parse_index_columns_single(self, csv_file: Path) -> None:
        """Test parsing single index column."""
        processor = PreSelectData(str(csv_file), index_col=2)
        result = processor._parse_index_columns()  # type: ignore[attr-defined]
        assert result == [2]

    def test_parse_index_columns_multiple(self, csv_file: Path) -> None:
        """Test parsing multiple index columns."""
        processor = PreSelectData(str(csv_file), index_col="1,2,4")
        result = processor._parse_index_columns()  # type: ignore[attr-defined]
        assert result == [1, 2, 4]

    def test_parse_range_parameter_single_int(self, csv_file: Path) -> None:
        """Test parsing single integer range parameter."""
        processor = PreSelectData(str(csv_file))
        result = processor._parse_range_parameter(5, "test_param")  # type: ignore[attr-defined]
        assert result == (5, None)

    def test_parse_range_parameter_range_string(self, csv_file: Path) -> None:
        """Test parsing range string parameter."""
        processor = PreSelectData(str(csv_file))
        result = processor._parse_range_parameter("1,5", "test_param")  # type: ignore[attr-defined]
        assert result == (1, 5)  # Pandas uses inclusive end

    def test_read_data_csv(self, csv_file: Path) -> None:
        """Test reading CSV file."""
        processor = PreSelectData(str(csv_file))
        df = processor._read_data()  # type: ignore[attr-defined]

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (11, 5)  # 10 data rows + 1 header row

    def test_read_data_excel(self, excel_file: Path) -> None:
        """Test reading Excel file."""
        processor = PreSelectData(str(excel_file))
        df = processor._read_data()  # type: ignore[attr-defined]

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (11, 5)  # 10 data rows + 1 header row

    def test_read_data_excel_with_sheet_name(self, excel_file: Path) -> None:
        """Test reading Excel file with specific sheet name."""
        processor = PreSelectData(str(excel_file), sheet="Products")
        df = processor._read_data()  # type: ignore[attr-defined]

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (4, 3)  # 3 data rows + 1 header row

    def test_select_subset_basic(self, csv_file: Path) -> None:
        """Test basic subset selection."""
        processor = PreSelectData(str(csv_file), index_col=0, col_start=1, row_start=1)
        df = processor._read_data()  # type: ignore[attr-defined]
        result = processor._select_subset(df)  # type: ignore[attr-defined]

        assert isinstance(result, pd.DataFrame)
        # Should have 10 rows and 4 data columns (Name, Age, Score, Category)
        assert result.shape == (10, 4)

    def test_select_subset_with_ranges(self, csv_file: Path) -> None:
        """Test subset selection with column and row ranges."""
        processor = PreSelectData(str(csv_file), index_col=0, col_start="1,2", row_start="1,3")
        df = processor._read_data()  # type: ignore[attr-defined]
        result = processor._select_subset(df)  # type: ignore[attr-defined]

        assert isinstance(result, pd.DataFrame)
        # Should have 3 rows and 2 data columns (Name, Age)
        assert result.shape == (3, 2)

    def test_select_subset_multiple_index_columns(self, csv_file: Path) -> None:
        """Test subset selection with multiple index columns."""
        processor = PreSelectData(str(csv_file), index_col="0,1", col_start=2, index_separator="|")
        df = processor._read_data()  # type: ignore[attr-defined]
        result = processor._select_subset(df)  # type: ignore[attr-defined]

        assert isinstance(result, pd.DataFrame)
        # Should have 10 rows and 3 columns (Age, Score, Category)
        assert result.shape == (10, 3)
        # Check that index names are concatenated
        assert "|" in str(result.index.name)

    def test_select_subset_out_of_bounds_index_col(self, csv_file: Path) -> None:
        """Test subset selection with out-of-bounds index column."""
        processor = PreSelectData(str(csv_file), index_col=10)  # Only 5 columns (0-4)
        df = processor._read_data()  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="index_col .* is out of bounds"):
            processor._select_subset(df)  # type: ignore[attr-defined]

    @patch("builtins.print")
    def test_process_basic(self, mock_print: Any, csv_file: Path, temp_output_dir: Path) -> None:
        """Test basic processing workflow."""
        output_file = temp_output_dir / "output.csv"
        processor = PreSelectData(str(csv_file), str(output_file))

        processor.process()

        assert output_file.exists()
        # Read the output file to verify content
        result_df = pd.read_csv(output_file, index_col=0)  # type: ignore[misc]
        assert result_df.shape == (10, 4)  # 10 rows, 4 data columns

    @patch("builtins.print")
    def test_process_with_ranges(self, mock_print: Any, csv_file: Path, temp_output_dir: Path) -> None:
        """Test processing with range parameters."""
        output_file = temp_output_dir / "output.csv"
        processor = PreSelectData(str(csv_file), str(output_file), col_start="1,2", row_start="1,5")

        processor.process()

        # Should generate filename with range suffix
        expected_files = list(temp_output_dir.glob("output_row5_col*.csv"))
        assert len(expected_files) == 1

        result_df = pd.read_csv(expected_files[0], index_col=0)  # type: ignore[misc]
        assert result_df.shape == (5, 2)  # 5 rows, 2 data columns

    def test_get_info(self, csv_file: Path) -> None:
        """Test getting processor information."""
        processor = PreSelectData(str(csv_file), index_col="1,2", col_start="2,4", sep=",")

        info = processor.get_info()

        assert info["input_file"] == str(csv_file)
        assert info["index_col"] == "1,2"
        assert info["col_start"] == "2,4"
        assert info["sep"] == ","

    def test_gzipped_csv_input_pandas(self, test_data_dir: Path, temp_output_dir: Path) -> None:
        """Test reading from gzipped CSV files with Pandas implementation."""
        csv_gz_file = test_data_dir / "test_data.csv.gz"
        output_file = temp_output_dir / "output_pandas.csv"

        processor = PreSelectData(str(csv_gz_file), str(output_file))
        # Test that process completes without errors
        processor.process()

        # Output file should exist
        assert output_file.exists()

    def test_gzipped_tsv_input_pandas(self, test_data_dir: Path, temp_output_dir: Path) -> None:
        """Test reading from gzipped TSV files with Pandas implementation."""
        tsv_gz_file = test_data_dir / "test_data.tsv.gz"
        output_file = temp_output_dir / "output_pandas.csv"

        processor = PreSelectData(str(tsv_gz_file), str(output_file))
        # Test that process completes without errors
        processor.process()

        # Output file should exist
        assert output_file.exists()

    @patch.dict("os.environ", {"GZIP_OUT": "True"})
    def test_gzipped_output_when_env_set_pandas(self, csv_file: Path, temp_output_dir: Path) -> None:
        """Test that output files are gzipped when GZIP_OUT=True with Pandas."""
        processor = PreSelectData(str(csv_file))

        # Process and check that output file has .gz extension
        processor.process()
        assert processor.output_file.suffix == ".gz"

    @patch.dict("os.environ", {"GZIP_OUT": "False"})
    def test_uncompressed_output_when_env_false_pandas(self, csv_file: Path, temp_output_dir: Path) -> None:
        """Test that output files are not gzipped when GZIP_OUT=False with Pandas."""
        processor = PreSelectData(str(csv_file))

        # Process and check that output file doesn't have .gz extension
        processor.process()
        assert processor.output_file.suffix == ".csv"

    @patch.dict("os.environ", {"GZIP_OUT": "True"})
    def test_process_with_gzipped_output_pandas(self, csv_file: Path, temp_output_dir: Path) -> None:
        """Test full processing with gzipped output using Pandas."""
        processor = PreSelectData(str(csv_file))
        processor.output_file = temp_output_dir / "test_output_pandas.csv.gz"

        # Process should complete without errors
        processor.process()

        # Output file should exist and be readable
        assert processor.output_file.exists()

        # Should be able to read the gzipped output back
        import gzip

        with gzip.open(processor.output_file, "rt") as f:
            content = f.read()
            assert len(content) > 0
            assert "," in content  # Should be CSV format


class TestComparisonBetweenImplementations:
    """Test that both implementations produce comparable results."""

    @pytest.fixture
    def test_data_dir(self) -> Path:
        """Get the test data directory."""
        return Path(__file__).parent.parent.parent / test_data_path

    @pytest.fixture
    def csv_file(self, test_data_dir: Path) -> Path:
        """Path to test CSV file."""
        return test_data_dir / "test_data.csv"

    @pytest.fixture
    def temp_output_dir(self) -> Generator[Path, None, None]:
        """Create a temporary directory for output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @patch("builtins.print")
    def test_both_implementations_produce_same_data_shape(
        self, mock_print: Any, csv_file: Path, temp_output_dir: Path
    ) -> None:
        """Test that both pandas and Polars implementations produce same data shape."""
        pandas_output = temp_output_dir / "pandas_output.csv"
        polars_output = temp_output_dir / "polars_output.csv"

        # Process with pandas
        pandas_processor = PreSelectData(str(csv_file), str(pandas_output))
        pandas_processor.process()

        # Process with Polars
        polars_processor = PreSelectDataPolars(str(csv_file), str(polars_output))
        polars_processor.process()

        # Read both outputs and compare shapes
        pandas_result = pd.read_csv(pandas_output, index_col=0)  # type: ignore[misc]
        polars_result = pl.read_csv(polars_output)

        # Remove index column from Polars result for fair comparison
        polars_data_cols = polars_result.shape[1] - 1

        assert pandas_result.shape == (10, 4)  # 10 rows, 4 data columns
        assert polars_result.shape[0] == 10  # 10 rows
        assert polars_data_cols == 4  # 4 data columns (excluding index)

    @patch("builtins.print")
    def test_both_implementations_with_ranges(self, mock_print: Any, csv_file: Path, temp_output_dir: Path) -> None:
        """Test that both implementations handle ranges correctly."""
        pandas_output = temp_output_dir / "pandas_range.csv"
        polars_output = temp_output_dir / "polars_range.csv"

        # Process with pandas
        pandas_processor = PreSelectData(
            str(csv_file), str(pandas_output), index_col=0, col_start="1,2", row_start="1,3", row_index=0
        )
        pandas_processor.process()

        # Process with Polars
        polars_processor = PreSelectDataPolars(
            str(csv_file), str(polars_output), index_col=0, col_start="1,2", row_start="1,3", row_index=0
        )
        polars_processor.process()

        # Both should produce files with range suffix
        pandas_files = list(temp_output_dir.glob("pandas_range_row3_col*.csv"))
        polars_files = list(temp_output_dir.glob("polars_range_row3_col*.csv"))

        assert len(pandas_files) == 1
        assert len(polars_files) == 1

        # Read and compare shapes
        pandas_result = pd.read_csv(pandas_files[0], index_col=0)  # type: ignore[misc]
        polars_result = pl.read_csv(polars_files[0])

        assert pandas_result.shape == (3, 2)  # 3 rows, 2 data columns
        assert polars_result.shape[0] == 3  # 3 rows
        assert polars_result.shape[1] == 3  # 3 total columns (index + 2 data)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    @pytest.fixture
    def test_data_dir(self) -> Path:
        """Get the test data directory."""
        return Path(__file__).parent.parent.parent / test_data_path

    @pytest.fixture
    def csv_file(self, test_data_dir: Path) -> Path:
        """Path to test CSV file."""
        return test_data_dir / "test_data.csv"

    @pytest.fixture
    def temp_output_dir(self) -> Generator[Path, None, None]:
        """Create a temporary directory for output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_empty_file_handling(self, temp_output_dir: Path) -> None:
        """Test handling of empty files."""
        empty_file = temp_output_dir / "empty.csv"
        empty_file.write_text("")

        processor = PreSelectDataPolars(str(empty_file))
        with pytest.raises((OSError, pl.exceptions.ComputeError)):  # Should raise some kind of error
            processor._read_data()  # type: ignore[attr-defined]

    def test_file_with_single_row(self, temp_output_dir: Path) -> None:
        """Test handling of file with single row."""
        single_row_file = temp_output_dir / "single_row.csv"
        single_row_file.write_text("A,B,C\n")

        processor = PreSelectDataPolars(str(single_row_file))
        df = processor._read_data()  # type: ignore[attr-defined]
        assert df.shape == (1, 3)

    def test_file_with_single_column(self, temp_output_dir: Path) -> None:
        """Test handling of file with single column."""
        single_col_file = temp_output_dir / "single_col.csv"
        single_col_file.write_text("A\n1\n2\n3\n")

        processor = PreSelectDataPolars(str(single_col_file), col_start=0)
        df = processor._read_data()  # type: ignore[attr-defined]
        result = processor._select_subset(df)  # type: ignore[attr-defined]
        assert result.shape == (3, 1)

    def test_unicode_content(self, temp_output_dir: Path) -> None:
        """Test handling of files with Unicode content."""
        unicode_file = temp_output_dir / "unicode.csv"
        unicode_file.write_text("名前,年齢,スコア\n太郎,25,85\n花子,30,92\n", encoding="utf-8")

        processor = PreSelectDataPolars(str(unicode_file))
        df = processor._read_data()  # type: ignore[attr-defined]
        assert df.shape == (3, 3)

    def test_very_large_numbers(self, temp_output_dir: Path) -> None:
        """Test handling of very large numbers."""
        large_numbers_file = temp_output_dir / "large_numbers.csv"
        large_numbers_file.write_text("ID,Value\n1,999999999999999999\n2,1.23456789012345e+50\n")

        processor = PreSelectDataPolars(str(large_numbers_file))
        df = processor._read_data()  # type: ignore[attr-defined]
        assert df.shape == (3, 2)

    def test_mixed_data_types(self, temp_output_dir: Path) -> None:
        """Test handling of mixed data types in columns."""
        mixed_file = temp_output_dir / "mixed.csv"
        mixed_file.write_text("ID,Mixed\n1,Text\n2,123\n3,45.67\n4,True\n")

        processor = PreSelectDataPolars(str(mixed_file))
        df = processor._read_data()  # type: ignore[attr-defined]
        result = processor._select_subset(df)  # type: ignore[attr-defined]
        assert result.shape == (4, 2)
