"""Unit tests for economy statistics events.

These tests verify that economy stats events can execute without crashing

Internal logic is tested in tests/unit/systems/test_production.py.
Event registration is verified implicitly by successful execution.
"""

import bamengine.events  # noqa: F401 - register all events for Simulation.init()
from bamengine.core import get_event
from bamengine.simulation import Simulation


# ============================================================================
# Execution Tests
# ============================================================================


def test_update_avg_mkt_price_executes():
    """UpdateAvgMktPrice executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("update_avg_mkt_price")()
    event.execute(sim)  # Should not crash


def test_calc_unemployment_rate_executes():
    """CalcUnemploymentRate executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("calc_unemployment_rate")()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_economy_stats_event_chain():
    """Economy stats events can execute in sequence."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Execute in sequence
    events = [
        "update_avg_mkt_price",
        "calc_unemployment_rate",
    ]

    for e in events:
        get_event(e)().execute(sim)  # Should not crash
