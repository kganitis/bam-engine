"""Unit tests for credit market events.

These tests verify that credit market events can execute without crashing.

Internal logic is tested in tests/unit/events/internal/test_credit_market.py.
Event registration is verified implicitly by successful execution.
"""

import bamengine.events  # noqa: F401 - register all events for Simulation.init()
from bamengine.core import get_event
from bamengine.simulation import Simulation

# ============================================================================
# Execution Tests
# ============================================================================


def test_banks_decide_credit_supply_executes():
    """BanksDecideCreditSupply executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("banks_decide_credit_supply")()
    event.execute(sim)  # Should not crash


def test_banks_decide_interest_rate_executes():
    """BanksDecideInterestRate executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("banks_decide_interest_rate")()
    event.execute(sim)  # Should not crash


def test_firms_decide_credit_demand_executes():
    """FirmsDecideCreditDemand executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_decide_credit_demand")()
    event.execute(sim)  # Should not crash


def test_firms_calc_financial_fragility_executes():
    """FirmsCalcFinancialFragility executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_calc_financial_fragility")()
    event.execute(sim)  # Should not crash


def test_firms_prepare_loan_applications_executes():
    """FirmsPrepareLoanApplications executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_prepare_loan_applications")()
    event.execute(sim)  # Should not crash


def test_credit_market_round_executes():
    """CreditMarketRound executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    # opex_shock must be set before credit_market_round (pipeline guarantees this)
    get_event("banks_decide_interest_rate")().execute(sim)
    event = get_event("credit_market_round")()
    event.execute(sim)  # Should not crash


def test_firms_fire_workers_executes():
    """FirmsFireWorkers executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_fire_workers")()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_credit_market_event_chain():
    """Credit market events can execute in sequence."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Execute in sequence
    events = [
        "banks_decide_credit_supply",
        "banks_decide_interest_rate",
        "firms_decide_credit_demand",
        "firms_calc_financial_fragility",
        "firms_prepare_loan_applications",
        "credit_market_round",
        "firms_fire_workers",
    ]

    for e in events:
        get_event(e)().execute(sim)  # Should not crash
