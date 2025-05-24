"""
Event-3 integration tests (credit market)

Two tests exercise the full credit-market round on the tiny scheduler:

1. test_event_credit_market_basic
   – happy-path regression + cross-component money-flow checks.

2. test_credit_market_post_state_consistency
   – deeper invariants that require seeing *all* components together.

They complement the unit tests by checking ledger arithmetic, borrower ↔ lender
balance symmetry, queue flushing, and scheduler helpers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bamengine.scheduler import Scheduler
from bamengine.systems.credit_market import (
    banks_decide_credit_supply,
    banks_decide_interest_rate,
    banks_provide_loans,
    firms_calc_credit_metrics,
    firms_decide_credit_demand,
    firms_fire_workers,
    firms_prepare_loan_applications,
    firms_send_one_loan_app,
)


# --------------------------------------------------------------------------- #
# helper – run ONE credit event                                               #
# --------------------------------------------------------------------------- #
def _run_credit_event(sch: Scheduler) -> NDArray[np.float64]:
    """
    Execute the complete Event-3 logic once and return the snapshot of
    credit_demand *before* loans are granted (for later delta checks).
    """
    # The default LoanBook has capacity=128 but zero-length arrays; allocate
    # them now so the first append cannot hit a broadcast error.
    if sch.lb.borrower.size == 0:
        sch.lb.capacity = 0  # forces a resize on the first _ensure_capacity()

    banks_decide_credit_supply(sch.lend, v=sch.ec.v)
    banks_decide_interest_rate(
        sch.lend, r_bar=sch.ec.r_bar, h_phi=sch.h_phi, rng=sch.rng
    )

    # --- demand + fragility ------------------------------------------------
    firms_decide_credit_demand(sch.bor)
    demand_before = sch.bor.credit_demand.copy()
    firms_calc_credit_metrics(sch.bor)

    # --- application cycle -------------------------------------------------
    firms_prepare_loan_applications(sch.bor, sch.lend, max_H=sch.max_H, rng=sch.rng)
    for _ in range(sch.max_H):
        firms_send_one_loan_app(sch.bor, sch.lend)
        banks_provide_loans(sch.bor, sch.lb, sch.lend, r_bar=sch.ec.r_bar)

    # --- layoffs triggered by unmet wage bill ------------------------------
    firms_fire_workers(sch.emp, sch.wrk, rng=sch.rng)
    return demand_before


# --------------------------------------------------------------------------- #
# 1. Regression-style basic test                                              #
# --------------------------------------------------------------------------- #
def test_event_credit_market_basic(tiny_sched: Scheduler) -> None:
    sch = tiny_sched

    # ensure every firm wants credit and every bank can supply some
    sch.bor.wage_bill[:] = sch.rng.uniform(15.0, 25.0, sch.bor.wage_bill.size)
    sch.lend.credit_supply[:] = sch.rng.uniform(
        2000.0, 6000.0, sch.lend.credit_supply.size
    )

    total_funds_before = sch.bor.total_funds.copy()
    demand_before = _run_credit_event(sch)  # run Event-3

    lb = sch.lb
    assert lb.size > 0

    # ledger arithmetic columns consistent
    p = lb.principal[: lb.size]
    r = lb.rate[: lb.size]
    np.testing.assert_allclose(lb.interest[: lb.size], p * r)
    np.testing.assert_allclose(lb.debt[: lb.size], p * (1.0 + r))

    # borrower funds increased exactly by granted principal
    inc_by_borrower = np.bincount(
        lb.borrower[: lb.size], weights=p, minlength=sch.bor.total_funds.size
    )
    gained = sch.bor.total_funds - total_funds_before
    np.testing.assert_allclose(gained, inc_by_borrower)
    assert np.all(gained <= demand_before + 1e-9)

    # lender supplies decreased by the same principal
    dec_by_lender = np.bincount(
        lb.lender[: lb.size], weights=p, minlength=sch.lend.credit_supply.size
    )

    # pots were reset inside the event: C_k0 = equity_base * v
    pot_at_start = sch.lend.equity_base * sch.ec.v  # what each bank could lend
    supply_drop = pot_at_start - sch.lend.credit_supply
    np.testing.assert_allclose(dec_by_lender, supply_drop, atol=1e-9)

    # remaining demand is non-negative
    assert (sch.bor.credit_demand >= -1e-9).all()
    # regulatory lending-cap holds at the *end* of the event
    cap = sch.lend.equity_base * sch.ec.v
    assert (sch.lend.credit_supply <= cap + 1e-9).all()
    # exhausted banks cannot have negative supply
    assert (sch.lend.credit_supply >= -1e-9).all()


# --------------------------------------------------------------------------- #
# 2. Post-event state consistency                                             #
# --------------------------------------------------------------------------- #
def test_credit_market_post_state_consistency(tiny_sched: Scheduler) -> None:
    sch = tiny_sched

    # create a wage-bill gap so layoffs branch is surely taken
    sch.emp.current_labor[:] = sch.rng.integers(3, 6, sch.emp.current_labor.size)
    sch.emp.wage_offer[:] = sch.rng.uniform(1.0, 1.5, sch.emp.wage_offer.size)
    sch.emp.wage_bill[:] = sch.emp.current_labor * sch.emp.wage_offer + 5.0
    sch.bor.wage_bill = sch.emp.wage_bill  # same ndarray reference

    _run_credit_event(sch)

    # 1. wage bill now covered (after possible layoffs)
    assert (sch.emp.wage_bill <= sch.emp.total_funds + 1e-9).all()

    # 2. exhausted banks have flushed queues
    mask_exhausted = sch.lend.credit_supply < 1e-9
    assert np.all((sch.lend.recv_apps_head[mask_exhausted] == -1))

    # 3. LoanBook capacity >= size
    assert sch.lb.capacity >= sch.lb.size
