"""Unit tests for goods market events.

These tests verify that goods market events:
1. Declare correct dependencies
2. Can execute without crashing

Internal logic is tested in tests/unit/systems/test_goods_market.py.
Event registration is verified implicitly by successful execution.
"""

from bamengine.simulation import Simulation

# Import events to ensure they register
from bamengine.events.goods_market import (
    ConsumersCalcPropensity,
    ConsumersDecideFirmsToVisit,
    ConsumersDecideIncomeToSpend,
    ConsumersFinalizePurchases,
    ConsumersShopOneRound,
)


# ============================================================================
# Dependency Tests
# ============================================================================


def test_consumers_calc_propensity_dependencies():
    """ConsumersCalcPropensity declares correct dependency."""
    event = ConsumersCalcPropensity()
    assert "workers_receive_wage" in event.dependencies


def test_consumers_decide_income_to_spend_dependencies():
    """ConsumersDecideIncomeToSpend declares correct dependency."""
    event = ConsumersDecideIncomeToSpend()
    assert "consumers_calc_propensity" in event.dependencies


def test_consumers_decide_firms_to_visit_dependencies():
    """ConsumersDecideFirmsToVisit declares correct dependencies."""
    event = ConsumersDecideFirmsToVisit()
    deps = event.dependencies
    assert "consumers_decide_income_to_spend" in deps
    assert "firms_run_production" in deps


def test_consumers_shop_one_round_dependencies():
    """ConsumersShopOneRound declares correct dependency."""
    event = ConsumersShopOneRound()
    assert "consumers_decide_firms_to_visit" in event.dependencies


def test_consumers_finalize_purchases_dependencies():
    """ConsumersFinalizePurchases declares correct dependency."""
    event = ConsumersFinalizePurchases()
    assert "consumers_shop_one_round" in event.dependencies


# ============================================================================
# Execution Tests
# ============================================================================


def test_consumers_calc_propensity_executes():
    """ConsumersCalcPropensity executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = ConsumersCalcPropensity()
    event.execute(sim)  # Should not crash


def test_consumers_decide_income_to_spend_executes():
    """ConsumersDecideIncomeToSpend executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = ConsumersDecideIncomeToSpend()
    event.execute(sim)  # Should not crash


def test_consumers_decide_firms_to_visit_executes():
    """ConsumersDecideFirmsToVisit executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = ConsumersDecideFirmsToVisit()
    event.execute(sim)  # Should not crash


def test_consumers_shop_one_round_executes():
    """ConsumersShopOneRound executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = ConsumersShopOneRound()
    event.execute(sim)  # Should not crash


def test_consumers_finalize_purchases_executes():
    """ConsumersFinalizePurchases executes without error."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)
    event = ConsumersFinalizePurchases()
    event.execute(sim)  # Should not crash


# ============================================================================
# Event Chain Test
# ============================================================================


def test_goods_market_event_chain():
    """Test goods market events can execute in sequence."""
    sim = Simulation.init(n_firms=10, n_households=50, seed=42)

    events = [
        ConsumersCalcPropensity(),
        ConsumersDecideIncomeToSpend(),
        ConsumersDecideFirmsToVisit(),
    ]

    for event in events:
        event.execute(sim)

    # Verify state mutations occurred
    assert (sim.con.propensity >= 0).all()
    assert (sim.con.propensity <= 1).all()
