"""Comprehensive unit tests for pre_view.py module.

This module contains tests for PreViewData class and ViewConfig,
testing all options, display modes, and edge cases.
"""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest
from rich.console import Console
from rich.table import Table

from btools.scripts.pre.pre_view import PreViewData, ViewConfig

test_data_path = "data/data_pre"


class TestViewConfig:
    """Test suite for ViewConfig class."""

    def test_default_config(self) -> None:
        """Test ViewConfig with default environment variables."""
        # Clear any environment variables that might affect the test
        env_vars_to_clear = [
            "VIEW_AUTO_NORMAL_MAX_COLS",
            "VIEW_AUTO_ROTATED_MAX_COLS",
            "VIEW_NORMAL_MAX_COL_WIDTH",
            "VIEW_NORMAL_MAX_CELL_LENGTH",
            "VIEW_ROTATED_MAX_COL_WIDTH",
            "VIEW_ROTATED_MAX_CELL_LENGTH",
            "VIEW_ROTATED_HEADER_MAX_LENGTH",
            "VIEW_WRAPPED_COLS_PER_SECTION",
            "VIEW_WRAPPED_MAX_COL_WIDTH",
            "VIEW_WRAPPED_MAX_CELL_LENGTH",
            "VIEW_DEFAULT_ROWS",
            "VIEW_MAX_ROWS",
            "VIEW_ELLIPSIS_STRING",
            "VIEW_NULL_DISPLAY_STYLE",
            "VIEW_OUT_UNIQUE_MAX",
        ]

        # Store original values
        original_values = {}
        for var in env_vars_to_clear:
            if var in os.environ:
                original_values[var] = os.environ[var]
                del os.environ[var]

        try:
            config = ViewConfig()

            # Check default values
            assert config.auto_normal_max_cols == 5
            assert config.auto_rotated_max_cols == 10
            assert config.normal_max_col_width == 20
            assert config.normal_max_cell_length == 18
            assert config.rotated_max_col_width == 12
            assert config.rotated_max_cell_length == 10
            assert config.rotated_header_max_length == 8
            assert config.wrapped_cols_per_section == 5
            assert config.wrapped_max_col_width == 15
            assert config.wrapped_max_cell_length == 13
            assert config.default_rows == 10
            assert config.max_rows == 1000
            assert config.ellipsis_string == "..."
            assert config.null_display_style == "dim red"
            assert config.out_unique_max == 20

        finally:
            # Restore original values
            for var, value in original_values.items():  # type: ignore[reportUnknownVariableType]
                os.environ[var] = value

    @patch.dict(
        os.environ,
        {
            "VIEW_AUTO_NORMAL_MAX_COLS": "3",
            "VIEW_AUTO_ROTATED_MAX_COLS": "8",
            "VIEW_NORMAL_MAX_COL_WIDTH": "25",
            "VIEW_ELLIPSIS_STRING": "...",
            "VIEW_OUT_UNIQUE_MAX": "15",
        },
    )
    def test_custom_config(self) -> None:
        """Test ViewConfig with custom environment variables."""
        config = ViewConfig()

        assert config.auto_normal_max_cols == 3
        assert config.auto_rotated_max_cols == 8
        assert config.normal_max_col_width == 25
        assert config.ellipsis_string == "..."
        assert config.out_unique_max == 15


class TestPreViewData:
    """Test suite for PreViewData class."""

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
    def temp_output_file(self) -> Generator[Path, None, None]:
        """Create a temporary output file."""
        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
            temp_path = Path(f.name)
        yield temp_path
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def sample_dataframe(self) -> pl.DataFrame:
        """Create a sample DataFrame for testing."""
        return pl.DataFrame(
            {
                "ID": [1, 2, 3, 4, 5],
                "Name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                "Age": [25, 30, 35, 28, 32],
                "Score": [85.5, 92.0, 78.5, 88.0, 91.5],
                "Category": ["A", "B", "A", "C", "B"],
                "Active": [True, False, True, True, False],
                "Notes": ["Good", None, "Excellent", "Average", "Good"],
            }
        )

    def test_init_default_parameters(self, csv_file: Path) -> None:
        """Test PreViewData initialization with default parameters."""
        viewer = PreViewData(str(csv_file))

        assert viewer.input_file == csv_file
        assert viewer.rows == 50
        assert viewer.cols == 25
        assert viewer.output_info is None
        assert viewer.sep is None
        assert viewer.sheet is None
        assert viewer.show_stats is True
        assert viewer.show_types is True
        assert viewer.show_missing is True
        assert viewer.display_mode == "auto"
        assert isinstance(viewer.console, Console)
        assert isinstance(viewer.config, ViewConfig)

    def test_init_custom_parameters(self, csv_file: Path, temp_output_file: Path) -> None:
        """Test PreViewData initialization with custom parameters."""
        viewer = PreViewData(
            input_file=str(csv_file),
            rows="10,20",
            cols="2,8",
            output_info=str(temp_output_file),
            sep=",",
            sheet="Sheet1",
            show_stats=False,
            show_types=False,
            show_missing=False,
            display_mode="normal",
        )

        assert viewer.input_file == csv_file
        assert viewer.rows == "10,20"
        assert viewer.cols == "2,8"
        assert viewer.output_info == temp_output_file
        assert viewer.sep == ","
        assert viewer.sheet == "Sheet1"
        assert viewer.show_stats is False
        assert viewer.show_types is False
        assert viewer.show_missing is False
        assert viewer.display_mode == "normal"

    def test_parse_rows_parameter_single_int(self, csv_file: Path) -> None:
        """Test parsing rows parameter with single integer."""
        viewer = PreViewData(str(csv_file), rows=25)
        start, end = viewer._parse_rows_parameter()  # type: ignore[attr-defined]
        assert start == 0
        assert end == 25

    def test_parse_rows_parameter_single_str(self, csv_file: Path) -> None:
        """Test parsing rows parameter with single string."""
        viewer = PreViewData(str(csv_file), rows="30")
        start, end = viewer._parse_rows_parameter()  # type: ignore[attr-defined]
        assert start == 0
        assert end == 30

    def test_parse_rows_parameter_range(self, csv_file: Path) -> None:
        """Test parsing rows parameter with range."""
        viewer = PreViewData(str(csv_file), rows="10,50")
        start, end = viewer._parse_rows_parameter()  # type: ignore[attr-defined]
        assert start == 10
        assert end == 50

    def test_parse_rows_parameter_invalid_range(self, csv_file: Path) -> None:
        """Test parsing rows parameter with invalid range."""
        viewer = PreViewData(str(csv_file), rows="50,10")
        with pytest.raises(ValueError, match="start must be <= end"):
            viewer._parse_rows_parameter()  # type: ignore[attr-defined]

    def test_parse_rows_parameter_invalid_format(self, csv_file: Path) -> None:
        """Test parsing rows parameter with invalid format."""
        viewer = PreViewData(str(csv_file), rows="10,20,30")
        with pytest.raises(ValueError, match="Invalid rows range format"):
            viewer._parse_rows_parameter()  # type: ignore[attr-defined]

    def test_parse_rows_parameter_non_numeric(self, csv_file: Path) -> None:
        """Test parsing rows parameter with non-numeric values."""
        viewer = PreViewData(str(csv_file), rows="abc")
        with pytest.raises(ValueError, match="Invalid rows format"):
            viewer._parse_rows_parameter()  # type: ignore[attr-defined]

    def test_parse_cols_parameter_single_int(self, csv_file: Path) -> None:
        """Test parsing cols parameter with single integer."""
        viewer = PreViewData(str(csv_file), cols=15)
        start, end = viewer._parse_cols_parameter()  # type: ignore[attr-defined]
        assert start == 0
        assert end == 15

    def test_parse_cols_parameter_single_str(self, csv_file: Path) -> None:
        """Test parsing cols parameter with single string."""
        viewer = PreViewData(str(csv_file), cols="20")
        start, end = viewer._parse_cols_parameter()  # type: ignore[attr-defined]
        assert start == 0
        assert end == 20

    def test_parse_cols_parameter_range(self, csv_file: Path) -> None:
        """Test parsing cols parameter with range."""
        viewer = PreViewData(str(csv_file), cols="2,8")
        start, end = viewer._parse_cols_parameter()  # type: ignore[attr-defined]
        assert start == 2
        assert end == 8

    def test_parse_cols_parameter_invalid_range(self, csv_file: Path) -> None:
        """Test parsing cols parameter with invalid range."""
        viewer = PreViewData(str(csv_file), cols="8,2")
        with pytest.raises(ValueError, match="start must be <= end"):
            viewer._parse_cols_parameter()  # type: ignore[attr-defined]

    def test_read_data_csv(self, csv_file: Path) -> None:
        """Test reading CSV data."""
        viewer = PreViewData(str(csv_file))
        df = viewer._read_data()  # type: ignore[attr-defined]

        assert isinstance(df, pl.DataFrame)
        assert df.height > 0
        assert df.width > 0
        assert "ID" in df.columns
        assert "Name" in df.columns

    def test_read_data_tsv(self, tsv_file: Path) -> None:
        """Test reading TSV data."""
        viewer = PreViewData(str(tsv_file))
        df = viewer._read_data()  # type: ignore[attr-defined]

        assert isinstance(df, pl.DataFrame)
        assert df.height > 0
        assert df.width > 0

    def test_read_data_excel(self, excel_file: Path) -> None:
        """Test reading Excel data."""
        viewer = PreViewData(str(excel_file))
        df = viewer._read_data()  # type: ignore[attr-defined]

        assert isinstance(df, pl.DataFrame)
        assert df.height > 0
        assert df.width > 0

    def test_read_data_custom_separator(self, pipe_file: Path) -> None:
        """Test reading data with custom separator."""
        viewer = PreViewData(str(pipe_file), sep="|")
        df = viewer._read_data()  # type: ignore[attr-defined]

        assert isinstance(df, pl.DataFrame)
        assert df.height > 0
        assert df.width > 0

    def test_read_data_tab_separator(self, csv_file: Path) -> None:
        """Test reading data with tab separator."""
        viewer = PreViewData(str(csv_file), sep="\\t")
        # This should work even if it's not a TSV file (polars will handle it)
        df = viewer._read_data()  # type: ignore[attr-defined]
        assert isinstance(df, pl.DataFrame)

    def test_read_data_excel_with_sheet(self, excel_file: Path) -> None:
        """Test reading Excel data with specific sheet."""
        viewer = PreViewData(str(excel_file), sheet="Sheet1")
        df = viewer._read_data()  # type: ignore[attr-defined]

        assert isinstance(df, pl.DataFrame)
        assert df.height > 0

    def test_read_data_nonexistent_file(self) -> None:
        """Test reading data from nonexistent file."""
        viewer = PreViewData("nonexistent_file.csv")
        with pytest.raises(FileNotFoundError):
            viewer._read_data()  # type: ignore[attr-defined]

    def test_create_data_overview_table(self, csv_file: Path) -> None:
        """Test creating data overview table."""
        viewer = PreViewData(str(csv_file))
        df = viewer._read_data()  # type: ignore[attr-defined]
        table = viewer._create_data_overview_table(df)  # type: ignore[attr-defined]

        assert isinstance(table, Table)
        assert table.title == "📊 Dataset Overview"

    def test_create_column_info_table(self, csv_file: Path) -> None:
        """Test creating column information table."""
        viewer = PreViewData(str(csv_file))
        df = viewer._read_data()  # type: ignore[attr-defined]
        table = viewer._create_column_info_table(df)  # type: ignore[attr-defined]

        assert isinstance(table, Table)
        assert table.title == "📋 Column Information"

    def test_create_statistics_table(self, csv_file: Path) -> None:
        """Test creating statistics table."""
        viewer = PreViewData(str(csv_file))
        df = viewer._read_data()  # type: ignore[attr-defined]
        table = viewer._create_statistics_table(df)  # type: ignore[attr-defined]

        assert isinstance(table, Table)
        assert table.title == "📈 Numeric Statistics"

    def test_create_statistics_table_no_numeric_columns(self) -> None:
        """Test creating statistics table with no numeric columns."""
        # Create a DataFrame with only string columns
        df = pl.DataFrame(
            {"Name": ["Alice", "Bob", "Charlie"], "Category": ["A", "B", "C"], "Notes": ["Good", "Fair", "Excellent"]}
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_file = Path(f.name)

        try:
            viewer = PreViewData(str(temp_file))
            # Mock the _read_data method to return our test DataFrame
            viewer._read_data = Mock(return_value=df)  # type: ignore[method-assign]

            table = viewer._create_statistics_table(df)  # type: ignore[attr-defined]
            assert isinstance(table, Table)
            assert table.title == "📈 Numeric Statistics"
        finally:
            temp_file.unlink()

    def test_create_data_preview_table_auto_mode(self, csv_file: Path) -> None:
        """Test creating data preview table in auto mode."""
        viewer = PreViewData(str(csv_file), display_mode="auto")
        df = viewer._read_data()  # type: ignore[attr-defined]

        # Test with small number of columns (should use standard)
        table = viewer._create_data_preview_table(df, 0, 5, 0, 3)  # type: ignore[attr-defined]
        assert isinstance(table, Table)

    def test_create_data_preview_table_normal_mode(self, csv_file: Path) -> None:
        """Test creating data preview table in normal mode."""
        viewer = PreViewData(str(csv_file), display_mode="normal")
        df = viewer._read_data()  # type: ignore[attr-defined]
        table = viewer._create_data_preview_table(df, 0, 5, 0, 5)  # type: ignore[attr-defined]

        assert isinstance(table, Table)

    def test_create_data_preview_table_rotated_mode(self, csv_file: Path) -> None:
        """Test creating data preview table in rotated mode."""
        viewer = PreViewData(str(csv_file), display_mode="rotated")
        df = viewer._read_data()  # type: ignore[attr-defined]
        table = viewer._create_data_preview_table(df, 0, 5, 0, 5)  # type: ignore[attr-defined]

        assert isinstance(table, Table)

    def test_create_data_preview_table_wrapped_mode(self, csv_file: Path) -> None:
        """Test creating data preview table in wrapped mode."""
        viewer = PreViewData(str(csv_file), display_mode="wrapped")
        df = viewer._read_data()  # type: ignore[attr-defined]
        table = viewer._create_data_preview_table(df, 0, 5, 0, 5)  # type: ignore[attr-defined]

        assert isinstance(table, Table)

    def test_create_data_preview_table_invalid_start_col(self, csv_file: Path) -> None:
        """Test creating data preview table with invalid start column."""
        viewer = PreViewData(str(csv_file))
        df = viewer._read_data()  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="start_col .* is out of bounds"):
            viewer._create_data_preview_table(df, 0, 5, 100, 105)  # type: ignore[attr-defined]

    def test_create_standard_table(self, sample_dataframe: pl.DataFrame) -> None:
        """Test creating standard table display."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_file = Path(f.name)

        try:
            viewer = PreViewData(str(temp_file))
            table = viewer._create_standard_table(sample_dataframe, 0, 5, 0, 5)  # type: ignore[attr-defined]

            assert isinstance(table, Table)
            assert "Data Preview" in str(table.title)
        finally:
            temp_file.unlink()

    def test_create_rotated_header_table(self, sample_dataframe: pl.DataFrame) -> None:
        """Test creating rotated header table display."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_file = Path(f.name)

        try:
            viewer = PreViewData(str(temp_file))
            table = viewer._create_rotated_header_table(sample_dataframe, 0, 5, 0, 5)  # type: ignore[attr-defined]

            assert isinstance(table, Table)
            assert "Rotated Headers" in str(table.title)
        finally:
            temp_file.unlink()

    def test_create_wrapped_table(self, sample_dataframe: pl.DataFrame) -> None:
        """Test creating wrapped table display."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_file = Path(f.name)

        try:
            viewer = PreViewData(str(temp_file))
            table = viewer._create_wrapped_table(sample_dataframe, 0, 5, 0, 7)  # type: ignore[attr-defined]

            assert isinstance(table, Table)
        finally:
            temp_file.unlink()

    def test_generate_detailed_info(self, csv_file: Path) -> None:
        """Test generating detailed information."""
        viewer = PreViewData(str(csv_file))
        df = viewer._read_data()  # type: ignore[attr-defined]
        info = viewer._generate_detailed_info(df)  # type: ignore[attr-defined]

        assert isinstance(info, dict)
        assert "file_info" in info
        assert "column_details" in info
        assert "summary_statistics" in info
        assert "data_quality" in info

        # Check file_info structure
        file_info = info["file_info"]
        assert "file_path" in file_info
        assert "file_size_bytes" in file_info
        assert "rows" in file_info
        assert "columns" in file_info

        # Check data_quality structure
        data_quality = info["data_quality"]
        assert "total_missing_values" in data_quality
        assert "columns_with_missing" in data_quality
        assert isinstance(data_quality["columns_with_missing"], list)

    def test_clean_for_toml(self, csv_file: Path) -> None:
        """Test cleaning data for TOML serialization."""
        viewer = PreViewData(str(csv_file))

        # Test with None values
        data = {"key1": None, "key2": "value", "key3": [None, "item"]}
        cleaned = viewer._clean_for_toml(data)  # type: ignore[attr-defined]

        assert cleaned["key1"] == "null"
        assert cleaned["key2"] == "value"
        assert cleaned["key3"] == ["null", "item"]

    def test_clean_for_toml_nested(self, csv_file: Path) -> None:
        """Test cleaning nested data structures for TOML."""
        viewer = PreViewData(str(csv_file))

        data = {"level1": {"level2": {"null_value": None, "list_with_nulls": [1, None, 3]}}}
        cleaned = viewer._clean_for_toml(data)  # type: ignore[attr-defined]

        assert cleaned["level1"]["level2"]["null_value"] == "null"
        assert cleaned["level1"]["level2"]["list_with_nulls"] == [1, "null", 3]

    def test_view_method_basic(self, csv_file: Path, temp_output_file: Path) -> None:
        """Test the main view method with basic parameters."""
        viewer = PreViewData(str(csv_file), output_info=str(temp_output_file))

        # Mock console.print to avoid actual output during testing
        with patch.object(viewer.console, "print"):
            viewer.view()

        # Check that output file was created
        assert temp_output_file.exists()

        # Verify the TOML file can be read
        with open(temp_output_file, "rb") as f:
            try:
                import tomllib  # type: ignore[import-not-found]

                data = tomllib.load(f)  # type: ignore[reportUnknownMemberType]
            except ImportError:
                # Fallback for older Python versions
                import tomli  # type: ignore[import-not-found]

                data = tomli.load(f)  # type: ignore[reportUnknownMemberType]

        assert isinstance(data, dict)
        assert "file_info" in data
        assert "column_details" in data

    def test_view_method_no_output(self, csv_file: Path) -> None:
        """Test the view method without output file."""
        viewer = PreViewData(str(csv_file))

        # Mock console.print to avoid actual output during testing
        with patch.object(viewer.console, "print"):
            viewer.view()

        # Should complete without errors

    def test_view_method_with_ranges(self, csv_file: Path) -> None:
        """Test the view method with row and column ranges."""
        viewer = PreViewData(str(csv_file), rows="1,5", cols="1,3")

        # Mock console.print to avoid actual output during testing
        with patch.object(viewer.console, "print"):
            viewer.view()

    def test_view_method_row_out_of_bounds(self, csv_file: Path) -> None:
        """Test view method with row start out of bounds."""
        viewer = PreViewData(str(csv_file), rows="1000,1100")

        with pytest.raises(ValueError, match="start_row .* is out of bounds"):
            viewer.view()

    def test_view_method_col_out_of_bounds(self, csv_file: Path) -> None:
        """Test view method with column start out of bounds."""
        viewer = PreViewData(str(csv_file), cols="100,110")

        with pytest.raises(ValueError, match="start_col .* is out of bounds"):
            viewer.view()

    def test_view_method_display_modes(self, csv_file: Path) -> None:
        """Test view method with different display modes."""
        display_modes = ["auto", "normal", "rotated", "wrapped"]

        for mode in display_modes:
            viewer = PreViewData(str(csv_file), display_mode=mode)

            # Mock console.print to avoid actual output during testing
            with patch.object(viewer.console, "print"):
                viewer.view()

    def test_view_method_disabled_features(self, csv_file: Path) -> None:
        """Test view method with disabled features."""
        viewer = PreViewData(str(csv_file), show_stats=False, show_types=False, show_missing=False)

        # Mock console.print to avoid actual output during testing
        with patch.object(viewer.console, "print"):
            viewer.view()

    def test_get_info(self, csv_file: Path, temp_output_file: Path) -> None:
        """Test getting configuration information."""
        viewer = PreViewData(
            str(csv_file),
            rows="10,20",
            cols="2,8",
            output_info=str(temp_output_file),
            sep=",",
            sheet="Sheet1",
            show_stats=False,
            show_types=True,
            show_missing=False,
            display_mode="rotated",
        )

        info = viewer.get_info()

        assert isinstance(info, dict)
        assert info["input_file"] == str(csv_file)
        assert info["rows"] == "10,20"
        assert info["cols"] == "2,8"
        assert info["output_info"] == str(temp_output_file)
        assert info["sep"] == ","
        assert info["sheet"] == "Sheet1"
        assert info["show_stats"] is False
        assert info["show_types"] is True
        assert info["show_missing"] is False
        assert info["display_mode"] == "rotated"

    def test_get_info_no_output(self, csv_file: Path) -> None:
        """Test getting configuration info with no output file."""
        viewer = PreViewData(str(csv_file))
        info = viewer.get_info()

        assert info["output_info"] is None

    @pytest.mark.slow
    def test_large_dataset_handling(self, csv_file: Path) -> None:
        """Test handling of large dataset parameters."""
        viewer = PreViewData(str(csv_file), rows=1000, cols=50)
        # Just read the data to ensure no errors
        viewer._read_data()  # type: ignore[attr-defined,reportPrivateUsage]

        # Should handle gracefully even if dataset is smaller
        # Mock console.print to avoid actual output during testing
        with patch.object(viewer.console, "print"):
            viewer.view()

    def test_error_handling_invalid_file_format(self) -> None:
        """Test error handling for invalid file format."""
        # Create a file with invalid content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("invalid,csv,content\nwith,malformed\nrows")
            temp_file = Path(f.name)

        try:
            viewer = PreViewData(str(temp_file))
            # This might succeed or fail depending on polars' error tolerance
            # The test ensures the code handles it gracefully
            try:
                df = viewer._read_data()  # type: ignore[attr-defined,reportPrivateUsage]
                assert isinstance(df, pl.DataFrame)
            except Exception as e:  # noqa: BLE001
                assert isinstance(e, OSError)
        finally:
            temp_file.unlink()

    def test_memory_usage_estimation(self, csv_file: Path) -> None:
        """Test memory usage estimation in overview."""
        viewer = PreViewData(str(csv_file))
        df = viewer._read_data()  # type: ignore[attr-defined,reportPrivateUsage]

        # Create overview table and check it doesn't crash
        table = viewer._create_data_overview_table(df)  # type: ignore[attr-defined,reportPrivateUsage]
        assert isinstance(table, Table)

    @patch.dict(os.environ, {"VIEW_ELLIPSIS_STRING": ">>>"})
    def test_custom_ellipsis_string(self, csv_file: Path) -> None:
        """Test custom ellipsis string from environment."""
        viewer = PreViewData(str(csv_file))
        assert viewer.config.ellipsis_string == ">>>"

    @patch.dict(os.environ, {"VIEW_NULL_DISPLAY_STYLE": "bold blue"})
    def test_custom_null_display_style(self, csv_file: Path) -> None:
        """Test custom null display style from environment."""
        viewer = PreViewData(str(csv_file))
        assert viewer.config.null_display_style == "bold blue"

    def test_unique_values_in_detailed_info(self) -> None:
        """Test unique values inclusion in detailed info."""
        # Create a DataFrame with few unique values
        df = pl.DataFrame(
            {
                "Category": ["A", "B", "A", "C", "B", "A"],
                "Status": ["Active", "Inactive", "Active", "Active", "Inactive", "Active"],
                "ID": [1, 2, 3, 4, 5, 6],  # More unique values than threshold
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_file = Path(f.name)

        try:
            viewer = PreViewData(str(temp_file))
            # Mock the _read_data method
            viewer._read_data = Mock(return_value=df)  # type: ignore[method-assign,reportPrivateUsage]

            info = viewer._generate_detailed_info(df)  # type: ignore[attr-defined,reportPrivateUsage]

            # Category should have unique values (only 3 unique values)
            category_details = info["column_details"]["Category"]
            assert "unique_values" in category_details
            assert category_details["unique_values"] is not None
            assert set(category_details["unique_values"]) == {"A", "B", "C"}

            # Status should have unique values (only 2 unique values)
            status_details = info["column_details"]["Status"]
            assert "unique_values" in status_details
            assert status_details["unique_values"] is not None

            # ID should not have unique values listed (6 unique values, might exceed threshold)
            id_details = info["column_details"]["ID"]
            assert "unique_values" in id_details
            # Could be None if exceeds threshold

        finally:
            temp_file.unlink()

    def test_handle_missing_values_display(self) -> None:
        """Test handling and display of missing values."""
        # Create DataFrame with missing values
        df = pl.DataFrame(
            {"Name": ["Alice", "Bob", None, "Diana"], "Age": [25, None, 35, 28], "Score": [85.5, 92.0, None, 88.0]}
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_file = Path(f.name)

        try:
            viewer = PreViewData(str(temp_file))
            viewer._read_data = Mock(return_value=df)  # type: ignore[method-assign,reportPrivateUsage]

            # Test column info table creation
            table = viewer._create_column_info_table(df)  # type: ignore[attr-defined,reportPrivateUsage]
            assert isinstance(table, Table)

            # Test detailed info generation
            info = viewer._generate_detailed_info(df)  # type: ignore[attr-defined,reportPrivateUsage]
            assert info["data_quality"]["total_missing_values"] > 0
            assert len(info["data_quality"]["columns_with_missing"]) > 0

        finally:
            temp_file.unlink()
