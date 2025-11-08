"""Unit tests for planning events.

These tests verify that planning events can execute without crashing.

Internal logic is tested in tests/unit/systems/test_planning.py.
Event registration is verified implicitly by successful execution.
"""

import bamengine.events  # noqa: F401 - register all events for Simulation.init()
from bamengine.core import get_event
from bamengine.simulation import Simulation


# ============================================================================
# Execution Tests
# ============================================================================


def test_firms_decide_desired_production_executes():
    """FirmsDecideDesiredProduction executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_decide_desired_production")()
    event.execute(sim)  # Should not crash


def test_firms_calc_breakeven_price_executes():
    """FirmsCalcBreakevenPrice executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_calc_breakeven_price")()
    event.execute(sim)  # Should not crash


def test_firms_adjust_price_executes():
    """FirmsAdjustPrice executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_adjust_price")()
    event.execute(sim)  # Should not crash


def test_firms_decide_desired_labor_executes():
    """FirmsDecideDesiredLabor executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_decide_desired_labor")()
    event.execute(sim)  # Should not crash


def test_firms_decide_vacancies_executes():
    """FirmsDecideVacancies executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_decide_vacancies")()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_planning_event_chain_executes():
    """Planning events can execute in sequence."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Execute in sequence
    events = [
        "firms_decide_desired_production",
        "firms_calc_breakeven_price",
        "firms_adjust_price",
        "firms_decide_desired_labor",
        "firms_decide_vacancies",
    ]

    for e in events:
        get_event(e)().execute(sim)  # Should not crash
