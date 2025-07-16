"""Comprehensive unit tests for the new unified pre_view.py module.

This module tests the simplified ViewConfig and PreViewData classes,
focusing on the unified adaptive display mode and colon syntax support.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from btools.scripts.pre.pre_view import PreViewData, ViewConfig


class TestViewConfig:
    """Test suite for the simplified ViewConfig class."""

    def test_default_config(self) -> None:
        """Test ViewConfig with default environment variables."""
        # Clear any environment variables that might affect the test
        env_vars_to_clear = [
            "VIEW_COLS_PER_SECTION",
            "VIEW_MAX_COL_WIDTH",
            "VIEW_ELLIPSIS",
            "VIEW_OUT_UNIQUE_MAX",
            "VIEW_LAZY_LOAD",
        ]

        # Store original values
        original_values: dict[str, str] = {}
        for var in env_vars_to_clear:
            if var in os.environ:
                original_values[var] = os.environ[var]
                del os.environ[var]

        try:
            config = ViewConfig()

            # Test default values match the expected defaults
            assert config.cols_per_section == 12
            assert config.max_col_width == 20
            assert config.ellipsis == "..."
            assert config.out_unique_max == 20
            assert config.use_lazy_loading is False

        finally:
            # Restore original values
            for var, value in original_values.items():
                os.environ[var] = value

    def test_custom_config(self) -> None:
        """Test ViewConfig with custom environment variables."""
        with patch.dict(
            os.environ,
            {
                "VIEW_COLS_PER_SECTION": "6",
                "VIEW_MAX_COL_WIDTH": "30",
                "VIEW_ELLIPSIS": "...",
                "VIEW_OUT_UNIQUE_MAX": "15",
                "VIEW_LAZY_LOAD": "True",
            },
        ):
            config = ViewConfig()

            assert config.cols_per_section == 6
            assert config.max_col_width == 30
            assert config.ellipsis == "..."
            assert config.out_unique_max == 15
            assert config.use_lazy_loading is True

    def test_invalid_config_values(self) -> None:
        """Test ViewConfig handles invalid environment variable values gracefully."""
        with patch.dict(
            os.environ,
            {
                "VIEW_COLS_PER_SECTION": "invalid",
                "VIEW_MAX_COL_WIDTH": "not_a_number",
                "VIEW_LAZY_LOAD": "maybe",
            },
        ):
            # Should not raise an exception, should use defaults
            config = ViewConfig()
            assert config.cols_per_section == 12  # default for invalid
            assert config.max_col_width == 20  # default for invalid
            assert config.use_lazy_loading is False  # default for invalid boolean


class TestPreViewData:
    """Test suite for PreViewData class with the unified display system."""

    @pytest.fixture
    def sample_csv_file(self) -> Path:
        """Use our test data that already exists."""
        return Path("tests/data/data_pre/test_data.csv")

    def test_initialization(self, sample_csv_file: Path) -> None:
        """Test PreViewData initialization."""
        viewer = PreViewData(input_file=str(sample_csv_file), rows=10, cols="0:5")

        assert viewer.input_file == sample_csv_file
        assert viewer.rows == 10
        assert viewer.cols == "0:5"
        assert isinstance(viewer.config, ViewConfig)

    def test_file_format_support(self) -> None:
        """Test support for different file formats."""
        # Test CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("A,B\n1,2\n3,4\n")
            f.flush()
            csv_viewer = PreViewData(f.name, 5, 10)
            # Test that viewer can be created and configured properly
            assert csv_viewer.input_file.suffix == ".csv"
            os.unlink(f.name)

        # Test TSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("A\tB\n1\t2\n3\t4\n")
            f.flush()
            tsv_viewer = PreViewData(f.name, 5, 10)
            # Test that viewer can be created and configured properly
            assert tsv_viewer.input_file.suffix == ".tsv"
            os.unlink(f.name)

    def test_lazy_loading_config(self, sample_csv_file: Path) -> None:
        """Test lazy loading configuration."""
        # Test with lazy loading enabled
        with patch.dict(os.environ, {"VIEW_LAZY_LOAD": "True"}):
            viewer = PreViewData(str(sample_csv_file), 5, 10)
            assert viewer.config.use_lazy_loading is True

        # Test with lazy loading disabled
        with patch.dict(os.environ, {"VIEW_LAZY_LOAD": "False"}):
            viewer = PreViewData(str(sample_csv_file), 5, 10)
            assert viewer.config.use_lazy_loading is False

    def test_column_width_settings(self) -> None:
        """Test that column width settings are respected."""
        # Create data with long column names and content
        df = pl.DataFrame({"Very_Long_Column_Name": ["Very long content"], "Short": ["OK"]})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)

            try:
                # Test with custom width settings
                with patch.dict(os.environ, {"VIEW_MAX_COL_WIDTH": "15"}):
                    viewer = PreViewData(f.name, 2, 10)
                    assert viewer.config.max_col_width == 15
            finally:
                os.unlink(f.name)

    def test_error_handling(self) -> None:
        """Test error handling for invalid files."""
        # Test with non-existent file
        non_existent = Path("does_not_exist.csv")
        viewer = PreViewData(str(non_existent), 5, 10)

        # Test that it raises an error when trying to view non-existent file
        with pytest.raises((OSError, FileNotFoundError)):
            viewer.view()

    def test_gzipped_files(self) -> None:
        """Test support for gzipped CSV files."""
        # Use the existing gzipped test data
        gz_file = Path("tests/data/data_pre/test_data.csv.gz")
        if gz_file.exists():
            viewer = PreViewData(str(gz_file), 5, 10)
            # Test that gzipped files can be processed through the view() method
            try:
                viewer.view()
                success = True
            except Exception:
                success = False
            assert success, "Should be able to process gzipped files"

    def test_integration_with_dataset(self) -> None:
        """Integration test with test dataset."""
        csv_file = Path("tests/data/data_pre/test_data.csv")
        if csv_file.exists():
            viewer = PreViewData(str(csv_file), 5, "1:3")

            # Test that it processes without errors
            try:
                viewer.view()
                success = True
            except Exception:
                success = False

            assert success, "Should be able to view dataset without errors"

    def test_colon_syntax_integration(self) -> None:
        """Integration test for colon syntax with actual data."""
        # Create test data with enough columns
        data = {f"Col_{i}": [f"val_{i}_{j}" for j in range(3)] for i in range(10)}
        df = pl.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.write_csv(f.name)

            try:
                # Test colon syntax
                viewer = PreViewData(f.name, 3, "1:3,5:7")

                # Should not raise an error
                try:
                    viewer.view()
                    success = True
                except Exception:
                    success = False

                assert success, "Should handle colon syntax without errors"
            finally:
                os.unlink(f.name)
