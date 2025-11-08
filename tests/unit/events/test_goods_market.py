"""Unit tests for goods market events.

These tests verify that goods market events can execute without crashing

Internal logic is tested in tests/unit/systems/test_goods_market.py.
Event registration is verified implicitly by successful execution.
"""

import bamengine.events  # noqa: F401 - register all events for Simulation.init()
from bamengine.core import get_event
from bamengine.simulation import Simulation


# ============================================================================
# Execution Tests
# ============================================================================


def test_consumers_calc_propensity_executes():
    """ConsumersCalcPropensity executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("consumers_calc_propensity")()
    event.execute(sim)  # Should not crash


def test_consumers_decide_income_to_spend_executes():
    """ConsumersDecideIncomeToSpend executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("consumers_decide_income_to_spend")()
    event.execute(sim)  # Should not crash


def test_consumers_decide_firms_to_visit_executes():
    """ConsumersDecideFirmsToVisit executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("consumers_decide_firms_to_visit")()
    event.execute(sim)  # Should not crash


def test_consumers_shop_one_round_executes():
    """ConsumersShopOneRound executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("consumers_shop_one_round")()
    event.execute(sim)  # Should not crash


def test_consumers_finalize_purchases_executes():
    """ConsumersFinalizePurchases executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = get_event("consumers_finalize_purchases")()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_goods_market_event_chain():
    """Goods market events can execute in sequence."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    # Execute in sequence
    events = [
        "consumers_calc_propensity",
        "consumers_decide_income_to_spend",
        "consumers_decide_firms_to_visit",
    ]

    for e in events:
        get_event(e)().execute(sim)  # Should not crash
