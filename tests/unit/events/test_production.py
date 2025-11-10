"""Unit tests for production events.

These tests verify that production events can execute without crashing.

Internal logic is tested in tests/unit/events/internal/test_production.py.
Event registration is verified implicitly by successful execution.
"""

import bamengine.events  # noqa: F401 - register all events for Simulation.init()
from bamengine.core import get_event
from bamengine.simulation import Simulation


# ============================================================================
# Execution Tests
# ============================================================================


def test_firms_pay_wages_executes():
    """FirmsPayWages executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_pay_wages")()
    event.execute(sim)  # Should not crash


def test_workers_receive_wage_executes():
    """WorkersReceiveWage executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("workers_receive_wage")()
    event.execute(sim)  # Should not crash


def test_firms_run_production_executes():
    """FirmsRunProduction executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_run_production")()
    event.execute(sim)  # Should not crash


def test_workers_update_contracts_executes():
    """WorkersUpdateContracts executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("workers_update_contracts")()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_production_event_chain():
    """Production events can execute in sequence."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Execute in sequence
    events = [
        "firms_pay_wages",
        "workers_receive_wage",
        "firms_run_production",
        "workers_update_contracts",
    ]

    for e in events:
        get_event(e)().execute(sim)  # Should not crash
