"""Pytest configuration for calibration integration tests.

Applies the ``calibration`` marker to every test collected from this directory.
"""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Add calibration marker to all tests under tests/integration/calibration/."""
    this_dir = str(__file__).rsplit("/conftest.py", 1)[0]
    for item in items:
        if str(item.fspath).startswith(this_dir):
            item.add_marker(pytest.mark.calibration)
