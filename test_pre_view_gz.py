#!/usr/bin/env python3
"""Test script for pre_view.py gzipped file support."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from btools.scripts.pre.pre_view import PreViewData

# Test reading gzipped CSV input
test_data_gz = Path(__file__).parent / "tests" / "data" / "data_pre" / "test_data.csv.gz"

print(f"Testing pre_view with gzipped input: {test_data_gz}")

try:
    print("Creating PreViewData instance...")
    viewer = PreViewData(str(test_data_gz), rows=5)
    print("✓ Successfully created PreViewData instance")

    # Process and display
    print("Starting view process...")
    viewer.view()
    print("✓ Successfully viewed gzipped CSV file!")

except Exception as e:
    print(f"✗ Error during viewing: {e}")
    import traceback

    traceback.print_exc()
