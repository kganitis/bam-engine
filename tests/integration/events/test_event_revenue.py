# tests/integration/events/test_event_revenue.py
"""
Event-6 integration tests  ⸺  revenue, debt-service & dividends
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bamengine.simulation import Simulation
from bamengine.systems.revenue import (
    firms_collect_revenue,
    firms_pay_dividends,
    firms_validate_debt_commitments,
)

_EPS = 1e-9


def _run_revenue_event(
    sch: Simulation,
    *,
    delta: float = 0.15,
) -> dict[str, NDArray[np.float64]]:
    """
    Drive the full Event-6 pipeline once and return *float* snapshots that
    callers need for delta checks.
    """
    n_firms = sch.bor.total_funds.size

    snap: dict[str, NDArray[np.float64]] = {
        "funds_before": sch.bor.total_funds.copy(),
        "equity_before": sch.lend.equity_base.copy(),
        "net_w_before": sch.bor.net_worth.copy(),
        "debt_tot_before": sch.lb.debt_per_borrower(n_firms),
        "interest_tot": sch.lb.interest_per_borrower(n_firms),
    }

    # revenue & gross-profit
    firms_collect_revenue(sch.prod, sch.bor)
    snap["funds_after_revenue"] = sch.bor.total_funds.copy()

    # debt service
    firms_validate_debt_commitments(sch.bor, sch.lend, sch.lb)

    # dividends
    firms_pay_dividends(sch.bor, delta=delta)

    snap["funds_after"] = sch.bor.total_funds.copy()
    snap["equity_after"] = sch.lend.equity_base.copy()
    snap["net_w_after"] = sch.bor.net_worth.copy()
    snap["debt_tot_after"] = sch.lb.debt_per_borrower(n_firms)
    return snap


def test_event_revenue_basic(tiny_sched: Simulation) -> None:
    """
    Crafted micro-scenario:

      • Firm-0 – high profit → full repayment, pays dividends
      • Firm-1 – low profit  → full repayment, negative net profits, no dividends
      • Firm-2 – fragile     → proportional write-off
    """
    sch = tiny_sched
    delta = 0.40

    # tailor the tiny simulation
    sch.prod.production[:] = 20.0
    sch.prod.inventory[:] = 20.0
    sch.prod.price[:] = 2.0
    sch.prod.inventory[0] = 0.0  # sells everything
    sch.prod.inventory[1] = 10.0  # sells half
    sch.prod.inventory[2] = 20.0  # sells nothing

    sch.bor.wage_bill[:] = 20.0

    sch.bor.total_funds[:] = 0.0
    sch.bor.net_worth[:] = 10.0

    sch.lend.equity_base[:] = 100.0

    # build a tiny ledger with exactly two rows
    m = 4
    sch.lb.capacity = m
    sch.lb.size = 3
    sch.lb.borrower = np.full(m, -1, np.int64)
    sch.lb.lender = np.full(m, -1, np.int64)
    sch.lb.principal = np.zeros(m, np.float64)
    sch.lb.rate = np.zeros(m, np.float64)
    sch.lb.interest = np.zeros(m, np.float64)
    sch.lb.debt = np.zeros(m, np.float64)

    sch.lb.borrower[:3] = [0, 1, 2]
    sch.lb.lender[:3] = [0, 0, 0]
    sch.lb.principal[:3] = [10.0, 10.0, 10.0]
    sch.lb.rate[:3] = [0.11, 0.11, 0.11]
    sch.lb.interest[:3] = sch.lb.principal[:3] * sch.lb.rate[:3]
    sch.lb.debt[:3] = sch.lb.principal[:3] * (1.0 + sch.lb.rate[:3])

    # run the event
    snap = _run_revenue_event(sch, delta=delta)

    sold = sch.prod.production - sch.prod.inventory
    revenue = sch.prod.price * sold

    debt_before = snap["debt_tot_before"]
    repay_mask = snap["funds_after_revenue"] >= debt_before - _EPS
    debt_paid = np.where(repay_mask, debt_before, 0.0)
    interest_paid = np.where(repay_mask, snap["interest_tot"], 0.0)
    bad_debt = np.where(~repay_mask, snap["net_w_before"], 0.0)

    # borrower cash
    cash_expected = (
        snap["funds_before"]
        + revenue
        - debt_paid
        - np.where(sch.bor.net_profit > 0.0, sch.bor.net_profit * delta, 0.0)
    )
    np.testing.assert_allclose(sch.bor.total_funds, cash_expected, rtol=_EPS)

    # lender equity
    equity_expected = snap["equity_before"].copy()
    equity_expected[0] += interest_paid.sum() - bad_debt.sum()

    np.testing.assert_allclose(sch.lend.equity_base, equity_expected, rtol=_EPS)

    # ledger size & rows
    assert 0 not in sch.lb.borrower[: sch.lb.size]  # firm-0 repaid
    assert 1 not in sch.lb.borrower[: sch.lb.size]  # firm-1 repaid
    assert sch.lb.size == 1


def test_revenue_post_state_consistency(tiny_sched: Simulation) -> None:
    sch = tiny_sched
    rng, delta = sch.rng, 0.15

    # mild randomisation
    sch.prod.production[:] = rng.uniform(5.0, 15.0, sch.prod.production.size)
    sch.prod.inventory[:] = sch.prod.production * rng.uniform(
        0.0, 0.8, sch.prod.production.size
    )
    sch.prod.price[:] = rng.uniform(1.0, 3.0, sch.prod.price.size)

    sch.bor.wage_bill[:] = rng.uniform(0.0, 8.0, sch.bor.wage_bill.size)
    sch.bor.total_funds[:] = rng.uniform(5.0, 40.0, sch.bor.total_funds.size)
    sch.bor.net_worth[:] = rng.uniform(5.0, 25.0, sch.bor.net_worth.size)

    sch.lend.equity_base[:] = rng.uniform(5_000.0, 12_000.0, sch.lend.equity_base.size)
    sch.lb.size = 0

    _run_revenue_event(sch, delta=delta)

    # Verify Borrower's Retained Profit Calculation
    expected_retained_profit = np.where(
        sch.bor.net_profit > 0.0,
        sch.bor.net_profit * (1.0 - delta),
        sch.bor.net_profit,
    )
    np.testing.assert_allclose(
        sch.bor.retained_profit,
        expected_retained_profit,
        rtol=_EPS,
    )

    # Lender equity non-negative & within cap
    assert (sch.lend.equity_base >= -_EPS).all()
    cap = sch.lend.equity_base * sch.ec.v
    assert (sch.lend.credit_supply <= cap + _EPS).all()

    # LoanBook structural guards
    assert sch.lb.size <= sch.lb.capacity
    assert (sch.lb.borrower[: sch.lb.size] < sch.bor.total_funds.size).all()
    assert (sch.lb.lender[: sch.lb.size] < sch.lend.equity_base.size).all()
