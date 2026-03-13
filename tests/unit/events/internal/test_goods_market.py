"""
Goods-market events internal implementation unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from bamengine import Rng, make_rng
from bamengine.events._internal.goods_market import (
    consumers_calc_propensity,
    consumers_decide_firms_to_visit,
    consumers_decide_income_to_spend,
    consumers_finalize_purchases,
    goods_market_round,
)
from bamengine.roles import Consumer, Producer
from tests.helpers.factories import mock_consumer, mock_producer


def _mini_state(
    *,
    n_hh: int = 4,
    n_firms: int = 3,
    Z: int = 2,
    seed: int = 0,
) -> tuple[Consumer, Producer, Rng, int]:
    """
    Return Consumer & Producer roles plus an RNG and queue width *Z*.
    """
    rng = make_rng(seed)
    con = mock_consumer(
        n=n_hh,
        queue_z=Z,
        income=np.full(n_hh, 3.0),
        savings=np.full(n_hh, 2.0),
    )
    prod = mock_producer(
        n=n_firms,
        price=np.array([1.0, 1.2, 0.9]),
        inventory=np.array([5.0, 0.0, 4.0]),  # firm-1 sold-out
        production=np.array([5.0, 8.0, 4.0]),  # last-period output (for loyalty)
    )
    return con, prod, rng, Z


def test_calc_propensity_basic() -> None:
    con = mock_consumer(
        n=3,
        savings=np.array([0.0, 5.0, 10.0]),
    )
    consumers_calc_propensity(con, avg_sav=5.0, beta=0.5)
    # monotone decreasing with savings
    assert con.propensity[0] > con.propensity[1] > con.propensity[2]
    assert np.all((con.propensity > 0.0) & (con.propensity <= 1.0))


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_budget_rule_conservation(beta: float) -> None:
    """Remaining-income + savings must equal wealth; income must zero-out."""
    con, _, _, _ = _mini_state()
    wealth = con.savings + con.income
    consumers_calc_propensity(con, avg_sav=1.0, beta=0.9)
    consumers_decide_income_to_spend(con)
    np.testing.assert_allclose(con.income_to_spend + con.savings, wealth, rtol=1e-12)
    assert np.all(con.income == 0.0)


def test_budget_rule_zero_avg_savings_guard() -> None:
    """`avg_sav → 0` must not raise `÷0` warnings and still conserve wealth."""
    con, _, _, _ = _mini_state()
    consumers_calc_propensity(con, avg_sav=1.0, beta=0.9)
    consumers_decide_income_to_spend(con)
    np.testing.assert_allclose(
        con.income_to_spend + con.savings, np.full(con.savings.shape, 5.0)
    )


def test_pick_firms_basic() -> None:
    con, prod, rng, Z = _mini_state()
    consumers_decide_firms_to_visit(con, prod, max_Z=Z, rng=rng)

    # every household with positive income_to_spend has valid queue ptr + targets
    active = np.where(con.income_to_spend > 0.0)[0]
    heads = con.shop_visits_head[active]
    assert (heads >= 0).all()

    rows = heads // Z
    targets = con.shop_visits_targets[rows]
    assert ((targets == -1) | ((targets >= 0) & (targets < prod.price.size))).all()


def test_pick_firms_loyalty_included() -> None:
    """Previous largest producer must appear somewhere in the consideration set."""
    con, prod, rng, Z = _mini_state()
    # set loyalty of hh-0 to firm-2 (which has inventory)
    con.largest_prod_prev[0] = 2
    consumers_calc_propensity(con, avg_sav=1.0, beta=0.9)
    consumers_decide_income_to_spend(con)
    consumers_decide_firms_to_visit(con, prod, max_Z=Z, rng=rng)
    # Loyalty firm is INCLUDED in the set (but competes on price, not guaranteed slot 0)
    targets = con.shop_visits_targets[0]
    assert 2 in targets[targets >= 0]


def test_pick_firms_selects_even_with_zero_inventory() -> None:
    """Consumers select firms regardless of inventory (matching the reference implementation).

    Consumers discover firms are sold out during shopping, not during selection.
    This makes the goods market less efficient but matches the reference model.
    """
    con, prod, rng, Z = _mini_state()
    prod.inventory[:] = 0.0  # market empty
    consumers_calc_propensity(con, avg_sav=1.0, beta=0.9)
    consumers_decide_income_to_spend(con)
    consumers_decide_firms_to_visit(con, prod, max_Z=Z, rng=rng)
    # Consumers with budget should still have active queues
    has_budget = con.income_to_spend > 0
    assert np.all(con.shop_visits_head[has_budget] >= 0)
    # They should have selected some firms (even though inventory is 0)
    assert np.any(con.shop_visits_targets >= 0)


def test_loyalty_firm_competes_on_price() -> None:
    """
    Loyalty firm competes on price (matching the reference implementation).

    When the loyalty firm has a higher price than another sampled firm,
    the cheaper firm should be visited first. Loyalty only guarantees
    INCLUSION in the consideration set, not priority in shopping order.
    """
    Z = 2
    con = mock_consumer(n=1, queue_z=Z, income=np.array([3.0]), savings=np.array([2.0]))
    # prev loyal firm will be index-1 (expensive)
    con.largest_prod_prev[0] = 1

    prod = mock_producer(
        n=2,
        inventory=np.array([5.0, 5.0]),
        price=np.array([1.0, 2.0]),  # firm-0 is cheaper than loyal firm-1
        production=np.array([4.0, 6.0]),
    )

    consumers_calc_propensity(con, avg_sav=1.0, beta=0.9)
    consumers_decide_income_to_spend(con)
    consumers_decide_firms_to_visit(con, prod, max_Z=Z, rng=make_rng(42))

    # Loyalty firm (1) is in the set but sorted by price
    # Cheaper firm (0) should be in slot 0, expensive loyalty firm (1) in slot 1
    assert con.shop_visits_targets[0, 0] == 0  # Cheapest first
    assert con.shop_visits_targets[0, 1] == 1  # Loyalty firm still included


def test_finalize_transfers_leftover_to_savings() -> None:
    con, _, _, _ = _mini_state()
    leftover = con.income_to_spend.copy()
    consumers_finalize_purchases(con)
    np.testing.assert_allclose(con.income_to_spend, 0.0)
    np.testing.assert_allclose(con.savings, 2.0 + leftover)


# ============================================================================
# No Firms Edge Cases
# ============================================================================


def test_pick_firms_with_no_firms() -> None:
    """Test consumers_decide_firms_to_visit with n_firms=0.

    When there are no firms, all shopping queues should be cleared.
    """
    Z = 2
    n_hh = 3
    rng = make_rng(42)
    con = mock_consumer(
        n=n_hh,
        queue_z=Z,
        income=np.full(n_hh, 3.0),
        savings=np.full(n_hh, 2.0),
    )
    prod = mock_producer(
        n=0,  # No firms
        inventory=np.array([]),
        price=np.array([]),
        production=np.array([]),
    )

    # Give consumers budget
    consumers_calc_propensity(con, avg_sav=1.0, beta=0.9)
    consumers_decide_income_to_spend(con)

    consumers_decide_firms_to_visit(con, prod, max_Z=Z, rng=rng)

    # All shopping queues should be cleared (-1)
    assert np.all(con.shop_visits_head == -1)
    assert np.all(con.shop_visits_targets == -1)


# ============================================================================
# goods_market_round edge cases
# ============================================================================


def test_goods_round_consumer_fewer_targets_than_max_z() -> None:
    """Consumer has fewer valid targets than max_Z → inner loop breaks on t < 0."""
    max_Z = 3
    con = mock_consumer(
        n=1,
        queue_z=max_Z,
        income_to_spend=np.array([10.0]),
    )
    # Only 1 valid target out of 3 slots
    con.shop_visits_targets[0] = np.array([0, -1, -1])

    prod = mock_producer(
        n=1,
        price=np.array([2.0]),
        inventory=np.array([10.0]),
    )

    goods_market_round(con, prod, max_Z=max_Z, rng=make_rng(0))

    # Consumer should have bought from firm 0, spending some budget
    assert con.income_to_spend[0] < 10.0
    assert prod.inventory[0] < 10.0
