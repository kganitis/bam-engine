"""Unit tests for economy statistics events.

These tests verify that economy stats events:
1. Declare correct dependencies
2. Can execute without crashing

Internal logic is tested in tests/unit/systems/test_production.py.
Event registration is verified implicitly by successful execution.
"""

from bamengine.simulation import Simulation

# Import events to ensure they register
from bamengine.events.economy_stats import CalcUnemploymentRate, UpdateAvgMktPrice


# ============================================================================
# Dependency Tests
# ============================================================================


def test_update_avg_mkt_price_dependencies():
    """UpdateAvgMktPrice declares correct dependency."""
    event = UpdateAvgMktPrice()
    assert "firms_adjust_price" in event.dependencies


def test_calc_unemployment_rate_dependencies():
    """CalcUnemploymentRate declares correct dependency."""
    event = CalcUnemploymentRate()
    assert "spawn_replacement_banks" in event.dependencies


# ============================================================================
# Execution Tests
# ============================================================================


def test_update_avg_mkt_price_executes():
    """UpdateAvgMktPrice executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = UpdateAvgMktPrice()
    event.execute(sim)  # Should not crash


def test_calc_unemployment_rate_executes():
    """CalcUnemploymentRate executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = CalcUnemploymentRate()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_economy_stats_event_chain():
    """Test economy stats events can execute in sequence."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    events = [
        UpdateAvgMktPrice(),
        CalcUnemploymentRate(),
    ]

    for event in events:
        event.execute(sim)

    # Verify state mutations occurred
    assert sim.ec.avg_mkt_price > 0
    # Unemployment rate is stored in history
    assert len(sim.ec.unemp_rate_history) > 0
