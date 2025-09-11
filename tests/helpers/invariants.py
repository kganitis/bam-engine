# tests/helpers/invariants.py
"""
High-level invariants that must hold after *every* ``Simulation.step``.
They deliberately stay coarse-grained so they remain valid even when the
micro-rules evolve.
"""
from __future__ import annotations

import numpy as np

from bamengine.simulation import Simulation


def assert_basic_invariants(sch: Simulation) -> None:  # noqa: C901  (flat, long)
    """
    Raise ``AssertionError`` if *any* fundamental cross-component relationship
    is violated after a full period.

    The assertions are grouped by event “domain” so failures give a useful hint
    where the bug lives.
    """

    # Planning & labour-market
    # ------------------------
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
    employed_mask = sch.wrk.employed == 1
    np.testing.assert_allclose(
        sch.emp.wage_bill,
        np.bincount(
            sch.wrk.employer[employed_mask],
            weights=sch.wrk.wage[employed_mask],
            minlength=sch.n_firms,
        ),
        rtol=1e-8,
    )

    # Credit-market & finance
    # -----------------------
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

    # Production & Goods-market
    # -------------------------
    # Inventories & prices
    assert (sch.prod.inventory >= -1e-9).all()
    assert (sch.prod.price > 0).all() and not np.isnan(sch.prod.price).any()

    # Savings & income never negative
    assert (sch.con.savings >= -1e-9).all()
    assert (sch.con.income >= -1e-9).all()

    # Propensity bounded (0,1] whenever present
    if hasattr(sch.con, "propensity"):
        assert ((0 < sch.con.propensity) & (sch.con.propensity <= 1)).all()

    # Revenue event bookkeeping
    # -------------------------
    # gross_profit = revenue – wage_bill  (checked indirectly: retained/net set)
    # retained ≤ net_profit   and   dividends ≥ 0
    pos = sch.bor.net_profit > 0
    assert np.all((sch.bor.retained_profit[~pos] == sch.bor.net_profit[~pos]))
    assert np.all((sch.bor.retained_profit[pos] < sch.bor.net_profit[pos] + 1e-12))

    # Exit & entry
    # ------------
    # After replacement, ALL equities must be non-negative again
    assert (sch.bor.net_worth >= -1e-9).all()
    assert (sch.lend.equity_base >= -1e-9).all()

    # LoanBook rows may only reference healthy agents
    if sch.lb.size:
        assert not np.isin(
            sch.lb.borrower[: sch.lb.size], sch.ec.exiting_firms
        ).any(), "ledger row still references exited firm"
        assert not np.isin(
            sch.lb.lender[: sch.lb.size], sch.ec.exiting_banks
        ).any(), "ledger row still references exited bank"
