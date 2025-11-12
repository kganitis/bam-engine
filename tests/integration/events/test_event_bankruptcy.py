"""
Event-7 / Event-8 integration tests  ⸺  bankruptcy & entry
==========================================================

Two test-cases drive the complete “exit-and-replace” sequence:

1. test_event_bankruptcy_entry_basic
   – micro-scenario that triggers all branches (firm & bank exits,
     worker layoffs, loan-purge, replacement spawn).

2. test_bankruptcy_entry_post_state_consistency
   – fuzzier run ensuring invariants hold for arbitrary mixes
     of bankrupt and healthy agents.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bamengine.events._internal.bankruptcy import (  # systems under test
    firms_update_net_worth,
    mark_bankrupt_banks,
    mark_bankrupt_firms,
    spawn_replacement_banks,
    spawn_replacement_firms,
)
from bamengine.simulation import Simulation


def _run_bankruptcy_entry_event(sch: Simulation) -> dict[str, Any]:
    """
    Drive Event-7 (bankruptcy) + Event-8 (entry) once and return
    the pieces that the regression tests need.
    """
    snap: dict[str, Any] = {
        #  loan-book
        "lb_borrower_before": sch.lb.borrower[: sch.lb.size].copy(),
        "lb_lender_before": sch.lb.lender[: sch.lb.size].copy(),
        "lb_borrower_after": None,  # filled later
        "lb_lender_after": None,
        #  workforce
        "employed_before": sch.wrk.employed.copy(),
        "employer_before": sch.wrk.employer.copy(),
        "employed_after": None,  # filled later
    }

    # Event-7
    firms_update_net_worth(sch.bor)
    mark_bankrupt_firms(sch.ec, sch.emp, sch.bor, sch.prod, sch.wrk, sch.lb)
    mark_bankrupt_banks(sch.ec, sch.lend, sch.lb)

    # Event-8
    spawn_replacement_firms(sch.ec, sch.prod, sch.emp, sch.bor, rng=sch.rng)
    spawn_replacement_banks(sch.ec, sch.lend, rng=sch.rng)

    # fill the “after” snapshots
    snap["lb_borrower_after"] = sch.lb.borrower[: sch.lb.size].copy()
    snap["lb_lender_after"] = sch.lb.lender[: sch.lb.size].copy()
    snap["employed_after"] = sch.wrk.employed.copy()
    return snap


def test_event_bankruptcy_entry_basic(tiny_sched: Simulation) -> None:
    """
    Crafted micro-scenario:

      • Firms 0 & 2 start with **negative** equity -→ exit & replacement.
      • Bank 1 starts with **negative** equity -→ exit & replacement.
      • Both failing firms hold loans; some loans are with failing bank.
      • Workers at firms 0/2 are laid off.

    We assert cash-flows, loan-book purge, worker status and that
    replacement agents come back with *positive* equity.
    """
    sch = tiny_sched
    rng = sch.rng

    # tailor the tiny simulation
    # firms
    sch.bor.net_worth[:] = rng.uniform(5.0, 15.0, sch.bor.net_worth.size)
    sch.bor.net_worth[[0, 2]] = [-4.0, -1.5]  # bankrupt ones
    sch.bor.total_funds[:] = sch.bor.net_worth
    sch.bor.retained_profit[:] = 0.0  # no further changes

    # give firms some production/inventory so replacement math is meaningful
    sch.prod.production[:] = rng.uniform(6.0, 12.0, sch.prod.production.size)
    sch.prod.inventory[:] = sch.prod.production
    sch.emp.wage_offer[:] = rng.uniform(1.0, 2.0, sch.emp.wage_offer.size)

    # Workers: put three on firm-0 and two on firm-2
    victims_f0 = np.where(sch.wrk.employed == 1)[0][:3]
    victims_f2 = np.where(sch.wrk.employed == 0)[0][:2]
    sch.wrk.employed[victims_f0] = 1
    sch.wrk.employer[victims_f0] = 0
    sch.wrk.employed[victims_f2] = 1
    sch.wrk.employer[victims_f2] = 2
    sch.emp.current_labor[0] = 3
    sch.emp.current_labor[2] = 2

    # banks
    sch.lend.equity_base[:] = 100.0
    sch.lend.equity_base[1] = -20.0  # bankrupt bank

    # loan-book: three rows
    sch.lb.capacity = 8
    sch.lb.size = 3
    sch.lb.borrower = np.full(8, -1, np.int64)
    sch.lb.lender = np.full(8, -1, np.int64)
    sch.lb.debt = np.zeros(8)
    sch.lb.borrower[:3] = [0, 1, 2]  # rows 0 & 2 go away
    sch.lb.lender[:3] = [0, 1, 1]  # row 1 goes away (bad bank)
    sch.lb.debt[:3] = [5.0, 7.5, 4.0]

    # run the double event
    snap = _run_bankruptcy_entry_event(sch)

    #  ledger: no row with old bankrupt ids
    assert not np.isin([0, 2], snap["lb_borrower_after"]).any()
    assert not np.isin([1], snap["lb_lender_after"]).any()

    # size shrank by exactly the bad rows
    bad_rows = (
        (snap["lb_borrower_before"] == 0)
        | (snap["lb_borrower_before"] == 2)
        | (snap["lb_lender_before"] == 1)
    )
    assert sch.lb.size == snap["lb_borrower_before"].size - np.sum(bad_rows)

    #  all workers who had employer 0 or 2 became unemployed
    was_with_bad_firm = (
        np.isin(snap["employer_before"], [0, 2]) & snap["employed_before"]
    )

    # every worker who *was* with a bad firm must now be unemployed
    assert (~snap["employed_after"][was_with_bad_firm]).all()

    #  replacements
    for i in (0, 2):
        assert sch.bor.net_worth[i] > 0
        np.testing.assert_allclose(sch.bor.net_worth[i], sch.bor.total_funds[i])
        assert sch.emp.current_labor[i] == 0
        assert sch.prod.inventory[i] == 0

    #  bank replacement
    assert sch.lend.equity_base[1] > 0
    assert sch.lend.credit_supply[1] == 0.0


def test_bankruptcy_entry_post_state_consistency(tiny_sched: Simulation) -> None:
    """
    Random mix of healthy/bankrupt agents  ➜  after the round:

      • all equities ≥ 0
      • ledger borrower / lender indices reference *existing*, healthy agents
      • labour counts and worker flags are in-sync.
    """
    sch = tiny_sched
    rng = sch.rng

    # randomise equities (≈30 % negative)
    bor_n = sch.bor.net_worth.size
    bad_firms = rng.choice(bor_n, size=int(0.3 * bor_n), replace=False)
    sch.bor.net_worth[:] = rng.uniform(2.0, 20.0, bor_n)
    sch.bor.net_worth[bad_firms] *= -0.5
    sch.bor.total_funds[:] = sch.bor.net_worth

    lend_n = sch.lend.equity_base.size
    # pick ≈30 % of the banks (at least one) to be “bad”
    n_bad_banks = max(1, int(0.3 * lend_n))
    bad_banks = rng.choice(lend_n, size=n_bad_banks, replace=False)
    sch.lend.equity_base[:] = rng.uniform(50.0, 120.0, lend_n)
    sch.lend.equity_base[bad_banks] *= -0.3

    # random loan-book
    sch.lb.capacity = 32
    sch.lb.size = 10
    sch.lb.borrower = rng.integers(0, bor_n, sch.lb.capacity, dtype=np.int64)
    sch.lb.lender = rng.integers(0, lend_n, sch.lb.capacity, dtype=np.int64)
    sch.lb.debt = rng.uniform(1.0, 5.0, sch.lb.capacity)

    # run
    _run_bankruptcy_entry_event(sch)

    # all equities non-negative
    assert (sch.bor.net_worth >= -1e-9).all()
    assert (sch.lend.equity_base >= -1e-9).all()

    # ledger indices within bounds & reference healthy entities
    assert sch.lb.size <= sch.lb.capacity
    assert (sch.lb.borrower[: sch.lb.size] < bor_n).all()
    assert (sch.lb.lender[: sch.lb.size] < lend_n).all()
    # none of the rows reference an exiting agent
    assert not np.isin(sch.lb.borrower[: sch.lb.size], sch.ec.exiting_firms).any()
    assert not np.isin(sch.lb.lender[: sch.lb.size], sch.ec.exiting_banks).any()

    # labour tables in-sync
    counts = np.bincount(
        sch.wrk.employer[sch.wrk.employed == 1],
        minlength=sch.emp.current_labor.size,
    )
    np.testing.assert_array_equal(counts, sch.emp.current_labor)
