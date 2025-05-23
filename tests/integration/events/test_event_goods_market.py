"""
Event-5 integration tests (goods market)

Two tests exercise the *entire* goods-market sequence on the tiny scheduler:

1. test_event_goods_market_basic
   – regression-style money-flow & stock-flow checks.

2. test_goods_market_post_state_consistency
   – deeper invariants that require seeing all components together.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bamengine.scheduler import Scheduler
from bamengine.systems.goods_market import (  # systems under test
    consumers_calc_propensity,
    consumers_decide_firms_to_visit,
    consumers_decide_income_to_spend,
    consumers_finalize_purchases,
    consumers_visit_one_round,
)


# --------------------------------------------------------------------------- #
# helper – run ONE goods-market event                                         #
# --------------------------------------------------------------------------- #
def _run_goods_event(
    sch: Scheduler,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Execute the complete Event-5 logic once.

    Returns
    -------
    tuple(inc_before, sav_before, unspent_income, inv_before, inv_after)
    The caller can derive deltas from these snapshots.
    """
    inc0 = sch.con.income.copy()
    inv1 = sch.prod.inventory.copy()
    sav0 = sch.con.savings.copy()

    # ----- propensity & budget split ------------------------------------
    avg_sav = float(sch.con.savings.mean() + 1e-12)
    consumers_calc_propensity(sch.con, avg_sav=avg_sav, beta=sch.beta)
    consumers_decide_income_to_spend(sch.con)

    sav1 = sch.con.savings.copy()

    # ---- shopping ------------------------------------------------------
    consumers_decide_firms_to_visit(sch.con, sch.prod, max_Z=sch.max_Z, rng=sch.rng)
    for _ in range(sch.max_Z):
        consumers_visit_one_round(sch.con, sch.prod)

    # ---- final bookkeeping ---------------------------------------------
    inv2 = sch.prod.inventory.copy()
    uinc = sch.con.income_to_spend.copy()  # unspent income
    consumers_finalize_purchases(sch.con)
    sav2 = sch.con.savings.copy()

    return inc0, sav0, sav1, inv1, uinc, inv2, sav2


# --------------------------------------------------------------------------- #
# 1. Regression-style basic test                                              #
# --------------------------------------------------------------------------- #
def test_event_goods_market_basic(tiny_sched: Scheduler) -> None:
    """
    Happy-path regression:

    • total household spending equals total value of goods sold
    • no firm inventory drops below zero
    • every household ends with non-negative savings and zero income_to_spend
    """
    sch = tiny_sched

    # -------- craft a non-degenerate starting state ----------------------
    sch.prod.inventory[:] = sch.rng.uniform(5.0, 12.0, sch.prod.inventory.size)
    sch.prod.production[:] = sch.prod.inventory  # keep self-consistent
    sch.con.income[:] = sch.rng.uniform(2.0, 6.0, sch.con.income.size)
    sch.con.savings[:] = sch.rng.uniform(5.0, 15.0, sch.con.savings.size)

    inc0, sav0, _, inv1, _, inv2, sav2 = _run_goods_event(sch)

    # ----- 1.  money-flow identity --------------------------------------
    wealth0 = inc0 + sav0
    wealth2 = sav2  # after finalise
    spent = wealth0 - wealth2  # ≥ 0 by construction

    qty_sold = inv1 - inv2  # ≥ 0
    sales_value = (qty_sold * sch.prod.price).sum()

    # households spend exactly what firms earned
    np.testing.assert_allclose(spent.sum(), sales_value, rtol=1e-12)

    # ----- 2.  stock-flow guards ----------------------------------------
    assert np.all((inv2 >= -1e-12))  # never negative
    assert np.all((sav2 >= -1e-12))
    assert np.all((sch.con.income_to_spend == 0.0))


# --------------------------------------------------------------------------- #
# 2. Post-event state consistency                                             #
# --------------------------------------------------------------------------- #
def test_goods_market_post_state_consistency(tiny_sched: Scheduler) -> None:
    """
    Deeper invariants:

      1. every household’s `largest_prod_prev` is either −1
         *or* points to a firm that still exists.

      2. shop_visits_head pointers remain in the valid range
         [−1 … n_hh * Z).

      3. for each household  h
           savings_h(t)  =  savings_h(t−1) + unspent_income_h(t)
    """
    sch = tiny_sched

    # -------- richer initial state --------------------------------------
    sch.prod.inventory[:] = sch.rng.uniform(1.0, 8.0, sch.prod.inventory.size)
    sch.prod.production[:] = sch.prod.inventory
    sch.con.income[:] = sch.rng.uniform(1.0, 4.0, sch.con.income.size)
    sch.con.savings[:] = sch.rng.uniform(0.0, 10.0, sch.con.savings.size)

    inc0, sav0, sav1, inv1, uinc, inv2, sav2 = _run_goods_event(sch)

    n_hh, Z = sav2.size, sch.max_Z

    # ---- 1.  largest_prod_prev in bounds -------------------------------
    mask = sch.con.largest_prod_prev >= 0
    if mask.any():  # guard empty slice
        assert np.all((sch.con.largest_prod_prev[mask] < sch.prod.price.size))

    # ---- 2.  head pointers within valid domain -------------------------
    heads = sch.con.shop_visits_head
    assert ((heads >= -1) & (heads <= n_hh * Z)).all()

    # ---- 3.  household budget identity ---------------------------------
    # unspent income was transferred back to savings inside finalise
    assert np.allclose(sav2 - uinc, sav1, rtol=1e-9)
