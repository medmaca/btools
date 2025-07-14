"""Configuration file for pytest test runs."""

from typing import Any

import pytest


def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


def pytest_collection_modifyitems(config: Any, items: Any) -> None:  # noqa: ARG001
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark tests that involve file I/O as potentially slow
        if "process" in item.name or "read_data" in item.name:
            item.add_marker(pytest.mark.slow)

        # Mark comparison tests as integration tests
        if "comparison" in item.name.lower():
            item.add_marker(pytest.mark.integration)
