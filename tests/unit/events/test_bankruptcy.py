"""Unit tests for bankruptcy events.

These tests verify that bankruptcy events can execute without crashing

Internal logic is tested in tests/unit/events/internal/test_bankruptcy.py.
Event registration is verified implicitly by successful execution.
"""

import bamengine.events  # noqa: F401 - register all events for Simulation.init()
from bamengine.core import get_event
from bamengine.simulation import Simulation


# ============================================================================
# Execution Tests
# ============================================================================


def test_firms_update_net_worth_executes():
    """FirmsUpdateNetWorth executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("firms_update_net_worth")()
    event.execute(sim)  # Should not crash


def test_mark_bankrupt_firms_executes():
    """MarkBankruptFirms executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("mark_bankrupt_firms")()
    event.execute(sim)  # Should not crash


def test_mark_bankrupt_banks_executes():
    """MarkBankruptBanks executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("mark_bankrupt_banks")()
    event.execute(sim)  # Should not crash


def test_spawn_replacement_firms_executes():
    """SpawnReplacementFirms executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("spawn_replacement_firms")()
    event.execute(sim)  # Should not crash


def test_spawn_replacement_banks_executes():
    """SpawnReplacementBanks executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("spawn_replacement_banks")()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_bankruptcy_event_chain():
    """Bankruptcy events can execute in sequence."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Execute in sequence
    events = [
        "firms_update_net_worth",
        "mark_bankrupt_firms",
        "mark_bankrupt_banks",
        "spawn_replacement_firms",
        "spawn_replacement_banks",
    ]

    for e in events:
        get_event(e)().execute(sim)  # Should not crash
