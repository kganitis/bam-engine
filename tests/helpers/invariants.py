# tests/helpers/invariants.py
import numpy as np

from bamengine.scheduler import Scheduler


def assert_basic_invariants(sch: Scheduler) -> None:
    """
    Core invariants that must hold after *any* `Scheduler.step()`.

    They deliberately cover only high-level algebra and cross-component
    consistency—detailed event logic is unit-tested elsewhere.
    """

    # ------------------------------------------------------------------ #
    # 1. Planning & labour-market                                        #
    # ------------------------------------------------------------------ #
    # Desired production never negative / NaN
    assert not np.isnan(sch.prod.desired_production).any()
    assert (sch.prod.desired_production >= 0).all()

    # Vacancies & wage offers
    assert (sch.emp.n_vacancies >= 0).all()
    assert (sch.emp.wage_offer >= sch.ec.min_wage).all()

    # Employment flags are boolean and mirror firm labour counts
    assert set(np.unique(sch.wrk.employed)).issubset({0, 1})
    assert sch.wrk.employed.sum() == sch.emp.current_labor.sum()

    # Wage-bill must equal L_i · w_i  **after** production event
    np.testing.assert_allclose(
        sch.emp.wage_bill,
        sch.emp.current_labor * sch.emp.wage_offer,
        rtol=1e-8,
    )

    # ------------------------------------------------------------------ #
    # 2. Credit-market                                                   #
    # ------------------------------------------------------------------ #
    # No negative balances anywhere
    assert (sch.bor.credit_demand >= -1e-9).all()
    assert (sch.lend.credit_supply >= -1e-9).all()
    assert (sch.emp.total_funds >= -1e-9).all()

    # Ledger capacity never exceeded & indices within bounds
    assert sch.lb.size <= sch.lb.capacity
    n_borrowers = sch.bor.net_worth.size
    n_lenders = sch.lend.credit_supply.size
    assert np.all((sch.lb.borrower[: sch.lb.size] < n_borrowers))
    assert np.all((sch.lb.lender[: sch.lb.size] < n_lenders))

    # Ledger arithmetic:  interest = p · r    debt = p · (1+r)
    if sch.lb.size:
        p = sch.lb.principal[: sch.lb.size]
        r = sch.lb.rate[: sch.lb.size]
        np.testing.assert_allclose(sch.lb.interest[: sch.lb.size], p * r, rtol=1e-8)
        np.testing.assert_allclose(sch.lb.debt[: sch.lb.size], p * (1.0 + r), rtol=1e-8)

    # Regulatory lending cap respected
    cap_per_bank = sch.lend.equity_base * sch.ec.v
    assert (sch.lend.credit_supply <= cap_per_bank + 1e-9).all()

    # ------------------------------------------------------------------ #
    # 3. Production & households                                         #
    # ------------------------------------------------------------------ #
    # Household income must stay non-negative
    assert (sch.con.income >= -1e-9).all()
