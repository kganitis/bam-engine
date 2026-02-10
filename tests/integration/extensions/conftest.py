"""Conftest for extension integration tests — manages event hook lifecycle.

Extension modules register pipeline hooks at import time via
``@event(replace=...)``. These hooks persist in the global ``_EVENT_HOOKS``
registry. Since Python caches imports in ``sys.modules``, the decorator
code only runs once per process — so if another test module's conftest
cleans up hooks, they won't be re-registered on subsequent imports.

This fixture explicitly re-registers hooks at module start and cleans up
after all tests in the module complete.
"""

from __future__ import annotations

import pytest

from bamengine.core.registry import _EVENT_HOOKS, register_event_hook


@pytest.fixture(autouse=True, scope="module")
def _ensure_buffer_stock_hooks():
    """Register buffer-stock hooks for this module, clean up after."""
    saved = dict(_EVENT_HOOKS)

    # Import extension events (may be cached, but we need them registered)
    from extensions.buffer_stock import (  # noqa: F401
        ConsumersCalcBufferStockPropensity,
        ConsumersDecideBufferStockSpending,
    )

    # Explicitly register hooks (idempotent — overwrites if already present)
    register_event_hook(
        "consumers_calc_buffer_stock_propensity",
        replace="consumers_calc_propensity",
    )
    register_event_hook(
        "consumers_decide_buffer_stock_spending",
        replace="consumers_decide_income_to_spend",
    )

    yield

    _EVENT_HOOKS.clear()
    _EVENT_HOOKS.update(saved)
