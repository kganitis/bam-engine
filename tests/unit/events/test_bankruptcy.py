"""Unit tests for bankruptcy events.

These tests verify that bankruptcy events:
1. Declare correct dependencies
2. Can execute without crashing

Internal logic is tested in tests/unit/systems/test_bankruptcy.py.
Event registration is verified implicitly by successful execution.
"""

from bamengine.simulation import Simulation

# Import events to ensure they register
from bamengine.events.bankruptcy import (
    FirmsUpdateNetWorth,
    MarkBankruptBanks,
    MarkBankruptFirms,
    SpawnReplacementBanks,
    SpawnReplacementFirms,
)


# ============================================================================
# Dependency Tests
# ============================================================================


def test_firms_update_net_worth_dependencies():
    """FirmsUpdateNetWorth declares correct dependency."""
    event = FirmsUpdateNetWorth()
    assert "firms_pay_dividends" in event.dependencies


def test_mark_bankrupt_firms_dependencies():
    """MarkBankruptFirms declares correct dependency."""
    event = MarkBankruptFirms()
    assert "firms_update_net_worth" in event.dependencies


def test_mark_bankrupt_banks_dependencies():
    """MarkBankruptBanks declares correct dependency."""
    event = MarkBankruptBanks()
    assert "mark_bankrupt_firms" in event.dependencies


def test_spawn_replacement_firms_dependencies():
    """SpawnReplacementFirms declares correct dependency."""
    event = SpawnReplacementFirms()
    assert "mark_bankrupt_firms" in event.dependencies


def test_spawn_replacement_banks_dependencies():
    """SpawnReplacementBanks declares correct dependency."""
    event = SpawnReplacementBanks()
    assert "mark_bankrupt_banks" in event.dependencies


# ============================================================================
# Execution Tests
# ============================================================================


def test_firms_update_net_worth_executes():
    """FirmsUpdateNetWorth executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = FirmsUpdateNetWorth()
    event.execute(sim)  # Should not crash


def test_mark_bankrupt_firms_executes():
    """MarkBankruptFirms executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = MarkBankruptFirms()
    event.execute(sim)  # Should not crash


def test_mark_bankrupt_banks_executes():
    """MarkBankruptBanks executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = MarkBankruptBanks()
    event.execute(sim)  # Should not crash


def test_spawn_replacement_firms_executes():
    """SpawnReplacementFirms executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = SpawnReplacementFirms()
    event.execute(sim)  # Should not crash


def test_spawn_replacement_banks_executes():
    """SpawnReplacementBanks executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = SpawnReplacementBanks()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_bankruptcy_event_chain():
    """Test bankruptcy events can execute in sequence."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    events = [
        FirmsUpdateNetWorth(),
        MarkBankruptFirms(),
        MarkBankruptBanks(),
        SpawnReplacementFirms(),
        SpawnReplacementBanks(),
    ]

    for event in events:
        event.execute(sim)

    # Should complete without error
