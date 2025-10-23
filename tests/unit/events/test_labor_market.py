"""Unit tests for labor market events.

These tests verify that labor market events:
1. Declare correct dependencies
2. Can execute without crashing

Internal logic is tested in tests/unit/systems/test_labor_market.py.
Event registration is verified implicitly by successful execution.
"""

from bamengine.simulation import Simulation

# Import events to ensure they register
from bamengine.events.labor_market import (  # noqa: E402
    AdjustMinimumWage,
    CalcAnnualInflationRate,
    FirmsCalcWageBill,
    FirmsDecideWageOffer,
    FirmsHireWorkers,
    WorkersDecideFirmsToApply,
    WorkersSendOneRound,
)


# ============================================================================
# Dependency Tests
# ============================================================================


def test_calc_annual_inflation_rate_no_dependencies():
    """CalcAnnualInflationRate has no dependencies."""
    event = CalcAnnualInflationRate()
    assert event.dependencies == ()


def test_adjust_minimum_wage_dependencies():
    """AdjustMinimumWage declares correct dependency."""
    event = AdjustMinimumWage()
    assert event.dependencies == ("calc_annual_inflation_rate",)


def test_firms_decide_wage_offer_dependencies():
    """FirmsDecideWageOffer declares correct dependency."""
    event = FirmsDecideWageOffer()
    assert event.dependencies == ("firms_decide_vacancies",)


def test_workers_decide_firms_to_apply_dependencies():
    """WorkersDecideFirmsToApply declares correct dependency."""
    event = WorkersDecideFirmsToApply()
    assert event.dependencies == ("firms_decide_wage_offer",)


def test_workers_send_one_round_dependencies():
    """WorkersSendOneRound declares correct dependency."""
    event = WorkersSendOneRound()
    assert event.dependencies == ("workers_decide_firms_to_apply",)


def test_firms_hire_workers_dependencies():
    """FirmsHireWorkers declares correct dependency."""
    event = FirmsHireWorkers()
    assert event.dependencies == ("workers_send_one_round",)


def test_firms_calc_wage_bill_dependencies():
    """FirmsCalcWageBill declares correct dependency."""
    event = FirmsCalcWageBill()
    assert event.dependencies == ("firms_hire_workers",)


# ============================================================================
# Execution Tests
# ============================================================================


def test_calc_annual_inflation_rate_executes():
    """CalcAnnualInflationRate executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = CalcAnnualInflationRate()
    event.execute(sim)  # Should not crash


def test_adjust_minimum_wage_executes():
    """AdjustMinimumWage executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = AdjustMinimumWage()
    event.execute(sim)  # Should not crash


def test_firms_decide_wage_offer_executes():
    """FirmsDecideWageOffer executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsDecideWageOffer()
    event.execute(sim)  # Should not crash


def test_workers_decide_firms_to_apply_executes():
    """WorkersDecideFirmsToApply executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = WorkersDecideFirmsToApply()
    event.execute(sim)  # Should not crash


def test_workers_send_one_round_executes():
    """WorkersSendOneRound executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = WorkersSendOneRound()
    event.execute(sim)  # Should not crash


def test_firms_hire_workers_executes():
    """FirmsHireWorkers executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsHireWorkers()
    event.execute(sim)  # Should not crash


def test_firms_calc_wage_bill_executes():
    """FirmsCalcWageBill executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsCalcWageBill()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_labor_market_event_chain_executes():
    """Labor market events can execute in dependency order."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Execute in dependency order
    events = [
        CalcAnnualInflationRate(),
        AdjustMinimumWage(),
        FirmsDecideWageOffer(),
        WorkersDecideFirmsToApply(),
        WorkersSendOneRound(),
        FirmsHireWorkers(),
        FirmsCalcWageBill(),
    ]

    for event in events:
        event.execute(sim)  # Should not crash
