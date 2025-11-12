"""Unit tests for revenue events.

These tests verify that revenue events can execute without crashing

Internal logic is tested in tests/unit/events/internal/test_revenue.py.
Event registration is verified implicitly by successful execution.
"""

import bamengine.events  # noqa: F401 - register all events for Simulation.init()
from bamengine.core import get_event
from bamengine.simulation import Simulation

# ============================================================================
# Execution Tests
# ============================================================================


def test_firms_collect_revenue_executes():
    """FirmsCollectRevenue executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_collect_revenue")()
    event.execute(sim)  # Should not crash


def test_firms_validate_debt_commitments_executes():
    """FirmsValidateDebtCommitments executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_validate_debt_commitments")()
    event.execute(sim)  # Should not crash


def test_firms_pay_dividends_executes():
    """FirmsPayDividends executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_pay_dividends")()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_revenue_event_chain():
    """Revenue events can execute in sequence."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Execute in sequence
    events = [
        "firms_collect_revenue",
        "firms_validate_debt_commitments",
        "firms_pay_dividends",
    ]

    for e in events:
        get_event(e)().execute(sim)  # Should not crash
