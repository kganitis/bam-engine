"""Pytest configuration for validation tests.

Applies the ``validation`` marker to every test collected from this directory.
"""

import pytest


class StabilityWarning(UserWarning):
    """Emitted when stability pass rate is marginal (90-95%)."""


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Add validation marker to all tests under tests/validation/."""
    this_dir = str(__file__).rsplit("/conftest.py", 1)[0]
    for item in items:
        if str(item.fspath).startswith(this_dir):
            item.add_marker(pytest.mark.validation)
