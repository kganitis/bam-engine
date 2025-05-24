# tests/helpers/invariants.py
"""
High-level invariants that must hold after *every* ``Scheduler.step``.
They deliberately stay coarse-grained so they remain valid even when the
micro-rules evolve.
"""
from __future__ import annotations

import numpy as np

from bamengine.scheduler import Scheduler


def assert_basic_invariants(sch: Scheduler) -> None:  # noqa: C901  (long but flat)
    # ------------------------------------------------------------------ #
    # 1. Planning & labour-market                                        #
    # ------------------------------------------------------------------ #
    # Desired production – finite & non-negative
    assert not np.isnan(sch.prod.desired_production).any()
    assert (sch.prod.desired_production >= 0).all()

    # Vacancies & wage offers
    assert (sch.emp.n_vacancies >= 0).all()
    assert (sch.emp.wage_offer >= sch.ec.min_wage).all()

    # Employment flags mirror firm labour counts
    assert set(np.unique(sch.wrk.employed)).issubset({0, 1})
    assert sch.wrk.employed.sum() == sch.emp.current_labor.sum()

    # Wage-bill consistency
    np.testing.assert_allclose(
        sch.emp.wage_bill,
        sch.emp.current_labor * sch.emp.wage_offer,
        rtol=1e-8,
    )

    # ------------------------------------------------------------------ #
    # 2. Credit-market                                                   #
    # ------------------------------------------------------------------ #
    # Non-negative balances
    assert (sch.bor.credit_demand >= -1e-9).all()
    assert (sch.lend.credit_supply >= -1e-9).all()
    assert (sch.emp.total_funds >= -1e-9).all()

    # Ledger bounds & arithmetic
    assert sch.lb.size <= sch.lb.capacity
    nb, nl = sch.bor.net_worth.size, sch.lend.credit_supply.size
    assert (sch.lb.borrower[: sch.lb.size] < nb).all()
    assert (sch.lb.lender[: sch.lb.size] < nl).all()

    if sch.lb.size:
        p = sch.lb.principal[: sch.lb.size]
        r = sch.lb.rate[: sch.lb.size]
        np.testing.assert_allclose(sch.lb.interest[: sch.lb.size], p * r, rtol=1e-8)
        np.testing.assert_allclose(sch.lb.debt[: sch.lb.size], p * (1 + r), rtol=1e-8)

    # ------------------------------------------------------------------ #
    # 3. Production & Goods-market                                       #
    # ------------------------------------------------------------------ #
    # Inventories & prices
    assert (sch.prod.inventory >= -1e-9).all()
    assert (sch.prod.price > 0).all() and not np.isnan(sch.prod.price).any()

    # Savings & income never negative
    assert (sch.con.savings >= -1e-9).all()
    assert (sch.con.income >= -1e-9).all()

    # Propensity bounded (0,1] whenever present
    if hasattr(sch.con, "propensity"):
        assert ((0 < sch.con.propensity) & (sch.con.propensity <= 1)).all()

    # ──────────────────────────────────────────────────────────────────────
    # 4. Revenue event bookkeeping
    # ──────────────────────────────────────────────────────────────────────
    # gross_profit = revenue – wage_bill  (checked indirectly: retained/net set)
    # retained ≤ net_profit   and   dividends ≥ 0
    pos = sch.bor.net_profit > 0
    assert (sch.bor.retained_profit[~pos] == sch.bor.net_profit[~pos]).all()
    assert (sch.bor.retained_profit[pos] < sch.bor.net_profit[pos] + 1e-12).all()
