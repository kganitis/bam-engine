import numpy as np

from bamengine.scheduler import Scheduler


def assert_basic_invariants(sch: Scheduler) -> None:
    """
    Assertions that must hold after *any* ``Scheduler.step()`` call,
    regardless of economy size or number of periods.
    """
    # ------------------------------------------------------------------ #
    #  Planning & labor-market                                          #
    # ------------------------------------------------------------------ #
    # Production plans non-negative & finite
    assert not np.isnan(sch.prod.desired_production).any()
    assert (sch.prod.desired_production >= 0).all()

    # Vacancies never negative
    assert (sch.emp.n_vacancies >= 0).all()

    # Wage offers respect current minimum wage
    assert (sch.emp.wage_offer >= sch.ec.min_wage).all()

    # Employment flags are boolean
    assert set(np.unique(sch.wrk.employed)).issubset({0, 1})

    # Every employed worker counted in current_labor, and vice-versa
    assert sch.wrk.employed.sum() == sch.emp.current_labor.sum()

    # ------------------------------------------------------------------ #
    #  Credit-market                                                     #
    # ------------------------------------------------------------------ #
    # Borrower demand and lender supply never negative
    assert (sch.bor.credit_demand >= -1e-9).all()
    assert (sch.lend.credit_supply >= -1e-9).all()

    # Ledger capacity never exceeded
    assert sch.lb.size <= sch.lb.capacity

    # Ledger indices within component bounds
    n_borrowers = sch.bor.net_worth.size
    n_lenders = sch.lend.credit_supply.size
    assert (sch.lb.borrower[: sch.lb.size] < n_borrowers).all()
    assert (sch.lb.lender[: sch.lb.size] < n_lenders).all()

    # Basic ledger algebra:  interest = principal * rate,
    #                        debt     = principal * (1 + rate)
    if sch.lb.size:  # skip empty ledger
        principal = sch.lb.principal[: sch.lb.size]
        rate = sch.lb.rate[: sch.lb.size]
        np.testing.assert_allclose(
            sch.lb.interest[: sch.lb.size], principal * rate, rtol=1e-8
        )
        np.testing.assert_allclose(
            sch.lb.debt[: sch.lb.size], principal * (1.0 + rate), rtol=1e-8
        )
