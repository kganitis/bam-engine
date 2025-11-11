"""
Event-5 integration tests  ⸺  goods-market round
=================================================

Two test-cases run the *entire* goods-market sequence on a tiny `Simulation`.

1. test_event_goods_market_basic
   Regression-style money-flow & stock-flow checks.

2. test_goods_market_post_state_consistency
   Deeper invariants that need a full-system lens.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bamengine.simulation import Simulation
from bamengine.events._internal.goods_market import (  # systems under test
    consumers_calc_propensity,
    consumers_decide_firms_to_visit,
    consumers_decide_income_to_spend,
    consumers_finalize_purchases,
    consumers_shop_one_round,
)


def _run_goods_event(sch: Simulation) -> dict[str, NDArray[np.float64]]:
    """
    Drive the full Event-5 pipeline **once** and return *snapshots* of the
    vectors that callers need for delta checks.

    Returns
    -------
    dict with keys
      inc_before        : disposable income at *t-0*
      sav_before        : savings            at *t-0*
      sav_after_split   : savings right **after** the budget split
      inv_before        : firm inventories   at *t-0*
      unspent_income    : household income *not* spent after shopping
      inv_after         : firm inventories   at *t-1*
      sav_final         : savings            at *t-1* (after finalise)
    """
    snap: dict[str, NDArray[np.float64]] = {
        "inc_before": sch.con.income.copy(),
        "sav_before": sch.con.savings.copy(),
        "inv_before": sch.prod.inventory.copy(),
    }

    # 0. propensity + budget split
    avg_sav = float(sch.con.savings.mean() + 1e-12)
    consumers_calc_propensity(sch.con, avg_sav=avg_sav, beta=sch.beta)
    consumers_decide_income_to_spend(sch.con)
    snap["sav_after_split"] = sch.con.savings.copy()

    # shopping
    consumers_decide_firms_to_visit(sch.con, sch.prod, max_Z=sch.max_Z, rng=sch.rng)
    for _ in range(sch.max_Z):
        consumers_shop_one_round(sch.con, sch.prod)

    # final bookkeeping
    snap["inv_after"] = sch.prod.inventory.copy()
    snap["unspent_income"] = sch.con.income_to_spend.copy()
    consumers_finalize_purchases(sch.con)
    snap["sav_final"] = sch.con.savings.copy()

    return snap


def test_event_goods_market_basic(tiny_sched: Simulation) -> None:
    """
    Happy-path regression:

    * Σ household-spending == Σ firm-sales value
    * no inventory < 0
    * savings non-negative and income_to_spend flushed to zero
    """
    sch = tiny_sched

    # craft a non-degenerate initial state
    sch.prod.inventory[:] = sch.rng.uniform(5.0, 12.0, sch.prod.inventory.size)
    sch.prod.production[:] = sch.prod.inventory  # keep consistent
    sch.con.income[:] = sch.rng.uniform(2.0, 6.0, sch.con.income.size)
    sch.con.savings[:] = sch.rng.uniform(5.0, 15.0, sch.con.savings.size)

    snap = _run_goods_event(sch)

    # money-flow identity
    wealth0 = snap["inc_before"] + snap["sav_before"]
    wealth1 = snap["sav_final"]
    spent = wealth0 - wealth1  # ≥ 0

    qty_sold = snap["inv_before"] - snap["inv_after"]
    sales_value = (qty_sold * sch.prod.price).sum()

    np.testing.assert_allclose(spent.sum(), sales_value, rtol=1e-12)

    # stock-flow guards
    assert (snap["inv_after"] >= -1e-12).all()
    assert (snap["sav_final"] >= -1e-12).all()
    assert np.all(sch.con.income_to_spend == 0.0)


def test_goods_market_post_state_consistency(tiny_sched: Simulation) -> None:
    """
    Invariants checked after the round:

    1. `largest_prod_prev` either −1 or a valid firm id.
    2. `shop_visits_head` ∈ [−1 … n_households × Z].
    3. Budget identity per household:
         sav_after_split + spent_income == sav_final
       → equivalently   sav_final − unspent_income == sav_after_split
    """
    sch = tiny_sched

    # richer starting state
    sch.prod.inventory[:] = sch.rng.uniform(1.0, 8.0, sch.prod.inventory.size)
    sch.prod.production[:] = sch.prod.inventory
    sch.con.income[:] = sch.rng.uniform(1.0, 4.0, sch.con.income.size)
    sch.con.savings[:] = sch.rng.uniform(0.0, 10.0, sch.con.savings.size)

    snap = _run_goods_event(sch)

    n_hh, Z = sch.con.savings.size, sch.max_Z

    # largest_prod_prev bounds
    mask = sch.con.largest_prod_prev >= 0
    if mask.any():
        assert (sch.con.largest_prod_prev[mask] < sch.prod.price.size).all()

    # head-pointer domain
    heads = sch.con.shop_visits_head
    assert ((heads >= -1) & (heads <= n_hh * Z)).all()

    # household budget identity
    np.testing.assert_allclose(
        snap["sav_final"] - snap["unspent_income"],
        snap["sav_after_split"],
        rtol=1e-9,
    )
