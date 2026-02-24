"""Pytest configuration for integration tests.

Applies the ``integration`` marker to every test collected from this directory.
"""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Add integration marker to all tests under tests/integration/."""
    this_dir = str(__file__).rsplit("/conftest.py", 1)[0]
    for item in items:
        if str(item.fspath).startswith(this_dir):
            item.add_marker(pytest.mark.integration)
