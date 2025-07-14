#!/usr/bin/env python3
"""Test script for full GZIP functionality including processing."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

from btools.scripts.pre.pre_select_data import PreSelectDataPolars

load_dotenv(Path(__file__).parent / ".env")

# Create a test using the CSV file
test_data_path = Path(__file__).parent / "tests" / "data" / "data_pre" / "test_data.csv"
temp_output = Path(__file__).parent / "temp_test_output.csv.gz"

print(f"Testing with: {test_data_path}")
print(f"GZIP_OUT environment: {os.getenv('GZIP_OUT')}")

# Test with explicit output file
processor = PreSelectDataPolars(input_file=str(test_data_path), output_file=str(temp_output), col_start=1, row_start=1)

print(f"Processing with output: {processor.output_file}")

# Process the data
try:
    processor.process()
    print("✓ Processing completed successfully!")

    # Check if the file exists and is gzipped
    if temp_output.exists():
        print(f"✓ Output file created: {temp_output}")
        print(f"File size: {temp_output.stat().st_size} bytes")

        # Try to read it to verify it's a valid gzipped CSV
        import polars as pl

        df = pl.read_csv(temp_output)
        print(f"✓ Successfully read gzipped CSV: {df.shape}")
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
