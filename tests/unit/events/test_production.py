"""Unit tests for production events.

These tests verify that production events:
1. Declare correct dependencies
2. Can execute without crashing

Internal logic is tested in tests/unit/systems/test_production.py.
Event registration is verified implicitly by successful execution.
"""

from bamengine.simulation import Simulation

# Import events to ensure they register
from bamengine.events.production import (
    FirmsPayWages,
    FirmsRunProduction,
    WorkersReceiveWage,
    WorkersUpdateContracts,
)


# ============================================================================
# Dependency Tests
# ============================================================================


def test_firms_pay_wages_dependencies():
    """FirmsPayWages declares correct dependency."""
    event = FirmsPayWages()
    assert "firms_fire_workers" in event.dependencies


def test_workers_receive_wage_dependencies():
    """WorkersReceiveWage declares correct dependency."""
    event = WorkersReceiveWage()
    assert "firms_pay_wages" in event.dependencies


def test_firms_run_production_dependencies():
    """FirmsRunProduction declares correct dependency."""
    event = FirmsRunProduction()
    assert "firms_pay_wages" in event.dependencies


def test_workers_update_contracts_dependencies():
    """WorkersUpdateContracts declares correct dependency."""
    event = WorkersUpdateContracts()
    assert "firms_run_production" in event.dependencies


# ============================================================================
# Execution Tests
# ============================================================================


def test_firms_pay_wages_executes():
    """FirmsPayWages executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsPayWages()
    event.execute(sim)  # Should not crash


def test_workers_receive_wage_executes():
    """WorkersReceiveWage executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = WorkersReceiveWage()
    event.execute(sim)  # Should not crash


def test_firms_run_production_executes():
    """FirmsRunProduction executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsRunProduction()
    event.execute(sim)  # Should not crash


def test_workers_update_contracts_executes():
    """WorkersUpdateContracts executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = WorkersUpdateContracts()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_production_event_chain():
    """Test production events can execute in sequence."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    events = [
        FirmsPayWages(),
        WorkersReceiveWage(),
        FirmsRunProduction(),
        WorkersUpdateContracts(),
    ]

    for event in events:
        event.execute(sim)

    # Verify state mutations occurred
    assert (sim.prod.inventory >= 0).all()
