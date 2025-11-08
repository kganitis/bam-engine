"""Unit tests for labor market events.

These tests verify that labor market events can execute without crashing.

Internal logic is tested in tests/unit/systems/test_labor_market.py.
Event registration is verified implicitly by successful execution.
"""

import bamengine.events  # noqa: F401 - register all events for Simulation.init()
from bamengine.core import get_event
from bamengine.simulation import Simulation


# ============================================================================
# Execution Tests
# ============================================================================


def test_calc_annual_inflation_rate_executes():
    """CalcAnnualInflationRate executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("calc_annual_inflation_rate")()
    event.execute(sim)  # Should not crash


def test_adjust_minimum_wage_executes():
    """AdjustMinimumWage executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("adjust_minimum_wage")()
    event.execute(sim)  # Should not crash


def test_firms_decide_wage_offer_executes():
    """FirmsDecideWageOffer executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_decide_wage_offer")()
    event.execute(sim)  # Should not crash


def test_workers_decide_firms_to_apply_executes():
    """WorkersDecideFirmsToApply executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("workers_decide_firms_to_apply")()
    event.execute(sim)  # Should not crash


def test_workers_send_one_round_executes():
    """WorkersSendOneRound executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("workers_send_one_round")()
    event.execute(sim)  # Should not crash


def test_firms_hire_workers_executes():
    """FirmsHireWorkers executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_hire_workers")()
    event.execute(sim)  # Should not crash


def test_firms_calc_wage_bill_executes():
    """FirmsCalcWageBill executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_calc_wage_bill")()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_labor_market_event_chain_executes():
    """Labor market events can execute in sequence."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Execute in sequence
    events = [
        "calc_annual_inflation_rate",
        "adjust_minimum_wage",
        "firms_decide_wage_offer",
        "workers_decide_firms_to_apply",
        "workers_send_one_round",
        "firms_hire_workers",
        "firms_calc_wage_bill",
    ]

    for e in events:
        get_event(e)().execute(sim)  # Should not crash
