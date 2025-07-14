#!/usr/bin/env python3
"""Test script for GZIP_OUT environment variable functionality."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

from btools.scripts.pre.pre_select_data import PreSelectDataPolars

# Test environment variable directly
print("Direct GZIP_OUT check:", repr(os.getenv("GZIP_OUT")))

# Test with dotenv
load_dotenv(Path(__file__).parent / ".env")
print("After loading .env GZIP_OUT:", repr(os.getenv("GZIP_OUT")))

# Test the functionality indirectly by checking if output files have .gz extension

# Test actual functionality

# Create a test using the CSV file
test_data_path = Path(__file__).parent / "tests" / "data" / "data_pre" / "test_data.csv"
print(f"Testing with: {test_data_path}")

# Test with explicit output file
processor = PreSelectDataPolars(
    input_file=str(test_data_path),
    output_file=None,  # Should trigger _generate_output_filename
    col_start=1,
    row_start=1,
)

print(f"Generated output file: {processor.output_file}")
print(f"Output file suffix: {processor.output_file.suffix}")
