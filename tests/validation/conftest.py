"""Conftest for validation tests — isolates extension event hooks.

Validation scenario tests import extension modules which register pipeline
hooks globally via ``@event(replace=...)``. Without cleanup, hooks from one
scenario (e.g., buffer-stock) persist into the next (e.g., growth+),
causing KeyError when the replacement events access roles that weren't
attached.

This fixture saves and restores the global hook registry around each test
module so scenarios don't interfere with each other.
"""

from __future__ import annotations

import pytest

from bamengine.core.registry import _EVENT_HOOKS


@pytest.fixture(autouse=True, scope="module")
def _isolate_event_hooks():
    """Save event hooks before module, restore after."""
    saved = dict(_EVENT_HOOKS)
    yield
    _EVENT_HOOKS.clear()
    _EVENT_HOOKS.update(saved)
