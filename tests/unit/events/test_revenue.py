"""Unit tests for revenue events.

These tests verify that revenue events:
1. Declare correct dependencies
2. Can execute without crashing

Internal logic is tested in tests/unit/systems/test_revenue.py.
Event registration is verified implicitly by successful execution.
"""

from bamengine.simulation import Simulation

# Import events to ensure they register
from bamengine.events.revenue import (
    FirmsCollectRevenue,
    FirmsPayDividends,
    FirmsValidateDebtCommitments,
)


# ============================================================================
# Dependency Tests
# ============================================================================


def test_firms_collect_revenue_dependencies():
    """FirmsCollectRevenue declares correct dependency."""
    event = FirmsCollectRevenue()
    assert "consumers_finalize_purchases" in event.dependencies


def test_firms_validate_debt_commitments_dependencies():
    """FirmsValidateDebtCommitments declares correct dependency."""
    event = FirmsValidateDebtCommitments()
    assert "firms_collect_revenue" in event.dependencies


def test_firms_pay_dividends_dependencies():
    """FirmsPayDividends declares correct dependency."""
    event = FirmsPayDividends()
    assert "firms_validate_debt_commitments" in event.dependencies


# ============================================================================
# Execution Tests
# ============================================================================


def test_firms_collect_revenue_executes():
    """FirmsCollectRevenue executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsCollectRevenue()
    event.execute(sim)  # Should not crash


def test_firms_validate_debt_commitments_executes():
    """FirmsValidateDebtCommitments executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsValidateDebtCommitments()
    event.execute(sim)  # Should not crash


def test_firms_pay_dividends_executes():
    """FirmsPayDividends executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsPayDividends()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_revenue_event_chain():
    """Test revenue events can execute in sequence."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    events = [
        FirmsCollectRevenue(),
        FirmsValidateDebtCommitments(),
        FirmsPayDividends(),
    ]

    for event in events:
        event.execute(sim)

    # Should complete without error
    # TODO Verify state mutations occurred if needed
