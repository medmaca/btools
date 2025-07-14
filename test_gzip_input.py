#!/usr/bin/env python3
"""Test script for reading gzipped input files."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

from btools.scripts.pre.pre_select_data import PreSelectDataPolars

load_dotenv(Path(__file__).parent / ".env")

# Test reading gzipped CSV input
test_data_gz = Path(__file__).parent / "tests" / "data" / "data_pre" / "test_data.csv.gz"
temp_output = Path(__file__).parent / "temp_test_gz_input.csv"

print(f"Testing gzipped input: {test_data_gz}")
print(f"GZIP_OUT environment: {os.getenv('GZIP_OUT')}")

# Temporarily set GZIP_OUT to False to test regular output with gz input
os.environ["GZIP_OUT"] = "False"

processor = PreSelectDataPolars(input_file=str(test_data_gz), output_file=str(temp_output), col_start=1, row_start=1)

print(f"Processing gzipped input to: {processor.output_file}")

try:
    processor.process()
    print("✓ Processing gzipped input completed successfully!")

    # Check if the file exists
    if temp_output.exists():
        print(f"✓ Output file created: {temp_output}")
        print(f"File size: {temp_output.stat().st_size} bytes")

        # Try to read it
        import polars as pl

        df = pl.read_csv(temp_output)
        print(f"✓ Successfully processed .gz input: {df.shape}")
        print(f"First few rows:\n{df.head(3)}")

        # Clean up
        temp_output.unlink()
        print("✓ Temporary file cleaned up")
    else:
        print("✗ Output file was not created")

except Exception as e:
    print(f"✗ Error during processing: {e}")
    if temp_output.exists():
        temp_output.unlink()
