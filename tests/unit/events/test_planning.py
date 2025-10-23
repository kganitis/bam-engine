"""Unit tests for planning events.

These tests verify that planning events:
1. Declare correct dependencies
2. Can execute without crashing

Internal logic is tested in tests/unit/systems/test_planning.py.
Event registration is verified implicitly by successful execution.
"""

from bamengine.simulation import Simulation

# Import events to ensure they register
from bamengine.events.planning import (  # noqa: E402
    FirmsAdjustPrice,
    FirmsCalcBreakevenPrice,
    FirmsDecideDesiredLabor,
    FirmsDecideDesiredProduction,
    FirmsDecideVacancies,
)


# ============================================================================
# Dependency Tests
# ============================================================================


def test_firms_decide_desired_production_no_dependencies():
    """FirmsDecideDesiredProduction has no dependencies."""
    event = FirmsDecideDesiredProduction()
    assert event.dependencies == ()


def test_firms_calc_breakeven_price_dependencies():
    """FirmsCalcBreakevenPrice declares correct dependency."""
    event = FirmsCalcBreakevenPrice()
    assert event.dependencies == ("firms_decide_desired_production",)


def test_firms_adjust_price_dependencies():
    """FirmsAdjustPrice declares correct dependency."""
    event = FirmsAdjustPrice()
    assert event.dependencies == ("firms_calc_breakeven_price",)


def test_firms_decide_desired_labor_dependencies():
    """FirmsDecideDesiredLabor declares correct dependency."""
    event = FirmsDecideDesiredLabor()
    assert event.dependencies == ("firms_decide_desired_production",)


def test_firms_decide_vacancies_dependencies():
    """FirmsDecideVacancies declares correct dependency."""
    event = FirmsDecideVacancies()
    assert event.dependencies == ("firms_decide_desired_labor",)


# ============================================================================
# Execution Tests
# ============================================================================


def test_firms_decide_desired_production_executes():
    """FirmsDecideDesiredProduction executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsDecideDesiredProduction()
    event.execute(sim)  # Should not crash


def test_firms_calc_breakeven_price_executes():
    """FirmsCalcBreakevenPrice executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsCalcBreakevenPrice()
    event.execute(sim)  # Should not crash


def test_firms_adjust_price_executes():
    """FirmsAdjustPrice executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsAdjustPrice()
    event.execute(sim)  # Should not crash


def test_firms_decide_desired_labor_executes():
    """FirmsDecideDesiredLabor executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsDecideDesiredLabor()
    event.execute(sim)  # Should not crash


def test_firms_decide_vacancies_executes():
    """FirmsDecideVacancies executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsDecideVacancies()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_planning_event_chain_executes():
    """Planning events can execute in dependency order."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Execute in dependency order
    events = [
        FirmsDecideDesiredProduction(),
        FirmsCalcBreakevenPrice(),
        FirmsAdjustPrice(),
        FirmsDecideDesiredLabor(),
        FirmsDecideVacancies(),
    ]

    for event in events:
        event.execute(sim)  # Should not crash
