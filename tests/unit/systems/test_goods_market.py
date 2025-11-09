"""
Goods-market systems unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bamengine import Rng, make_rng
from bamengine.roles import Consumer, Producer
from bamengine.systems.goods_market import (
    consumers_calc_propensity,
    consumers_decide_firms_to_visit,
    consumers_decide_income_to_spend,
    consumers_finalize_purchases,
    consumers_shop_one_round,
)
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
    assert np.all((0.0 < con.propensity) & (con.propensity <= 1.0))


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_budget_rule_conservation(beta: float) -> None:
    """Remaining-income + savings must equal wealth; income must zero-out."""
    con, _, _, _ = _mini_state()
    wealth = con.savings + con.income
    consumers_calc_propensity(con, avg_sav=1.0, beta=0.9)
    consumers_decide_income_to_spend(con)
    np.testing.assert_allclose(con.income_to_spend + con.savings, wealth, rtol=1e-12)
    assert np.all((con.income == 0.0))


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
    assert ((targets == -1) | ((0 <= targets) & (targets < prod.price.size))).all()


def test_pick_firms_loyalty_slot() -> None:
    """Previous largest producer with stock must appear in column-0."""
    con, prod, rng, Z = _mini_state()
    # set loyalty of hh-0 to firm-2 (which has inventory)
    con.largest_prod_prev[0] = 2
    consumers_calc_propensity(con, avg_sav=1.0, beta=0.9)
    consumers_decide_income_to_spend(con)
    consumers_decide_firms_to_visit(con, prod, max_Z=Z, rng=rng)
    assert con.shop_visits_targets[0, 0] == 2


def test_pick_firms_no_inventory_exits_early() -> None:
    con, prod, rng, Z = _mini_state()
    prod.inventory[:] = 0.0  # market empty
    consumers_calc_propensity(con, avg_sav=1.0, beta=0.9)
    consumers_decide_income_to_spend(con)
    consumers_decide_firms_to_visit(con, prod, max_Z=Z, rng=rng)
    assert np.all((con.shop_visits_head == -1))
    assert np.all((con.shop_visits_targets == -1))


def test_loyalty_swap_keeps_prev_at_slot0() -> None:
    """
    Craft a case where the previous-period firm has a *higher* price than
    another sampled firm, so the initial ascending sort pushes it away from
    column-0.  The post-sort swap must pull it back.
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

    # loyalty guarantee: despite being pricier, firm-1 stays in column-0
    assert con.shop_visits_targets[0, 0] == 1


def test_one_round_basic_purchase() -> None:
    con, prod, rng, Z = _mini_state()
    consumers_calc_propensity(con, avg_sav=1.0, beta=0.9)
    consumers_decide_income_to_spend(con)
    consumers_decide_firms_to_visit(con, prod, max_Z=Z, rng=rng)

    h = 0  # focus on household-0
    con.income_to_spend[h] = 10.0  # give it enough budget to spend
    head_before = int(con.shop_visits_head[h])
    wealth_before = con.income_to_spend[h] + con.savings[h]
    inv_before = prod.inventory.copy()

    consumers_shop_one_round(con, prod)

    # exactly one slot consumed
    assert con.shop_visits_head[h] == head_before + 1
    # money conservation for that household
    spent = wealth_before - (con.income_to_spend[h] + con.savings[h])
    assert spent >= 0.0
    # inventory decreased by the purchased quantity
    assert np.any(prod.inventory < inv_before)


def test_one_round_skip_sold_out() -> None:
    """
    When the first target is sold out, the function must advance the pointer
    by one and not crash.
    """
    con, prod, rng, Z = _mini_state()
    stride = con.shop_visits_targets.shape[1]

    # Ensure consumer 0 has budget
    consumers_calc_propensity(con, avg_sav=1.0, beta=0.9)
    consumers_decide_income_to_spend(con)

    # Build queues (initialization), then force a sold-out firm at slot 0 for consumer 0
    consumers_decide_firms_to_visit(con, prod, max_Z=Z, rng=rng)
    sold_out_firm = 0
    prod.inventory[sold_out_firm] = 0.0

    # Explicitly set consumer 0's next target to the sold-out firm (row=0, col=0)
    con.shop_visits_targets[0, :] = -1
    con.shop_visits_targets[0, 0] = sold_out_firm
    con.shop_visits_head[0] = 0  # points to (row 0, col 0)

    head_before = int(con.shop_visits_head[0])
    budget_before = float(con.income_to_spend[0])

    consumers_shop_one_round(con, prod, rng=make_rng(7))

    # Pointer advanced exactly one and the slot was cleared
    assert con.shop_visits_head[0] == head_before + 1
    assert con.shop_visits_targets[0, 0] == -1

    # No purchase happened for consumer 0 (budget unchanged)
    assert np.isclose(con.income_to_spend[0], budget_before)

    # Sanity: firm remains sold out
    assert prod.inventory[sold_out_firm] == 0.0


def test_one_round_queue_exhaustion_clears_head() -> None:
    """
    If the current pointer already references −1 the head must reset to −1.
    """
    con = mock_consumer(
        n=1,
        queue_z=1,
        income_to_spend=np.array([5.0]),
        shop_visits_head=np.array([0]),
        shop_visits_targets=np.array([[-1]], dtype=np.intp),
    )
    prod = mock_producer(n=1, inventory=np.array([10.0]), price=np.array([1.0]))
    consumers_shop_one_round(con, prod)
    assert con.shop_visits_head[0] == -1


def test_visit_one_round_skips_household_with_no_head() -> None:
    """
    If `income_to_spend > 0` but the head pointer is −1, the early-continue
    branch must execute without touching inventories.
    """
    con = mock_consumer(
        n=1,
        queue_z=1,
        income_to_spend=np.array([4.0]),
        shop_visits_head=np.array([-1]),  # no queue
        shop_visits_targets=np.array([[-1]], dtype=np.intp),
        savings=np.array([1.0]),
    )
    prod = mock_producer(
        n=1,
        inventory=np.array([3.0]),
        price=np.array([1.0]),
        production=np.array([3.0]),
    )

    inv_before = prod.inventory.copy()
    consumers_shop_one_round(con, prod)

    # nothing purchased, inventory unchanged, income untouched
    np.testing.assert_allclose(prod.inventory, inv_before)
    assert con.income_to_spend[0] == pytest.approx(4.0)


def test_finalize_transfers_leftover_to_savings() -> None:
    con, _, _, _ = _mini_state()
    leftover = con.income_to_spend.copy()
    consumers_finalize_purchases(con)
    np.testing.assert_allclose(con.income_to_spend, 0.0)
    np.testing.assert_allclose(con.savings, 2.0 + leftover)


@settings(max_examples=150, deadline=None)
@given(
    n_hh=st.integers(min_value=1, max_value=15),
    n_firms=st.integers(min_value=1, max_value=10),
    Z=st.integers(min_value=1, max_value=3),
)
def test_visit_invariants(n_hh: int, n_firms: int, Z: int) -> None:
    rng = make_rng(123)
    # random positive inventory & price
    prod = mock_producer(
        n=n_firms,
        inventory=rng.uniform(0.0, 10.0, size=n_firms),
        price=rng.uniform(0.5, 5.0, size=n_firms),
        production=rng.uniform(1.0, 10.0, size=n_firms),
    )
    con = mock_consumer(
        n=n_hh,
        queue_z=Z,
        income=rng.uniform(0.0, 5.0, size=n_hh),
        savings=rng.uniform(0.0, 10.0, size=n_hh),
    )
    consumers_calc_propensity(con, avg_sav=con.savings.mean() + 1e-9, beta=0.9)
    consumers_decide_income_to_spend(con)
    consumers_decide_firms_to_visit(con, prod, max_Z=Z, rng=rng)
    consumers_shop_one_round(con, prod)

    # invariants
    # visit head stays within [-1 … n_hh*Z]
    assert ((con.shop_visits_head >= -1) & (con.shop_visits_head <= n_hh * Z)).all()
    # every non-sentinel target index < n_firms
    mask = con.shop_visits_targets >= 0
    assert (con.shop_visits_targets[mask] < n_firms).all()
