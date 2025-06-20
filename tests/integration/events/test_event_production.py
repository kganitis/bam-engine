# tests/integration/events/test_event_production.py
"""
Event-4 integration tests (production)

The two tests below exercise the full production round on the *tiny* simulation:

1. test_event_production_basic
   – regression-style money-flow and material-balance checks.

2. test_production_post_state_consistency
   – deeper invariants that involve contract expiration and labour
     book-keeping across Worker ↔ Employer components.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bamengine.simulation import Simulation
from bamengine.systems.labor_market import firms_calc_wage_bill
from bamengine.systems.production import (
    # firms_decide_price,
    firms_pay_wages,
    firms_run_production,
    update_avg_mkt_price,
    workers_receive_wage,
    workers_update_contracts,
)


# def _run_production_event(
#     sch: Simulation, *, with_expiration: bool = False
# ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#     """
#     Execute the complete Event-4 logic once.
#
#     Parameters
#     ----------
#     with_expiration
#         If ``True`` also run the contract-expiration rule.
#
#     Returns
#     -------
#     tuple(total_funds_before, income_before)
#         Snapshots taken *before* paying wages – used by callers for
#         delta checks.
#     """
#     # make sure the wage-bill is up-to-date with current_labor
#     firms_calc_wage_bill(sch.emp)
#
#     # price rule & market price
#     firms_decide_price(
#         sch.prod,
#         sch.emp,
#         sch.lb,
#         p_avg=sch.ec.avg_mkt_price,
#         h_eta=sch.h_eta,
#         rng=sch.rng,
#     )
#     update_avg_mkt_price(sch.ec, sch.prod)
#
#     funds_before = sch.emp.total_funds.copy()
#     income_before = sch.con.income.copy()
#
#     # wage payment & receipt
#     firms_pay_wages(sch.emp)
#     workers_receive_wage(sch.con, sch.wrk)
#
#     # physical production
#     firms_run_production(sch.prod, sch.emp)
#
#     # contract expiration (optional)
#     if with_expiration:
#         workers_update_contracts(sch.wrk, sch.emp)
#
#     return funds_before, income_before


# def test_event_production_basic(tiny_sched: Simulation) -> None:
#     """
#     Happy-path regression:
#
#     • firm cash decreases exactly by the wage-bill paid
#     • household income increases by the same amount
#     • physical output and inventories are consistent with labour
#     • no firm is driven into negative cash
#     """
#     sch = tiny_sched
#
#     funds_before, income_before = _run_production_event(sch)
#
#     # cash-flow identity
#     np.testing.assert_allclose(
#         sch.emp.total_funds, funds_before - sch.emp.wage_bill, rtol=1e-12
#     )
#
#     # wages actually paid by each worker (only employed receive)
#     wages_paid = sch.wrk.wage * sch.wrk.employed
#     income_delta = sch.con.income - income_before
#     np.testing.assert_allclose(income_delta, wages_paid)
#
#     # material balance
#     expected_output = sch.prod.labor_productivity * sch.emp.current_labor
#     np.testing.assert_allclose(sch.prod.production, expected_output)
#     np.testing.assert_allclose(sch.prod.inventory, expected_output)
#
#     # avg market price appended & consistent
#     assert sch.ec.avg_mkt_price_history[-1] == sch.ec.avg_mkt_price
#     np.testing.assert_allclose(sch.ec.avg_mkt_price, sch.prod.price.mean())
#
#     # non-negativity guard
#     assert (sch.emp.total_funds >= -1e-9).all()


# def test_production_post_state_consistency(tiny_sched: Simulation) -> None:
#     """
#     Force many contracts to expire in the same step and verify that
#
#         • firm ↔ worker labour counts stay consistent
#         • expired workers are flagged correctly
#         • wage-bill is recomputed from the updated labour vector
#     """
#     sch = tiny_sched
#
#     # Craft a *self-consistent* starting roster
#     # every worker is employed
#     sch.wrk.employed[:] = 1
#
#     # assign workers round-robin to firms
#     sch.wrk.employer[:] = np.arange(sch.wrk.employed.size) % sch.emp.current_labor.size
#
#     # derive true labour counts from the worker table
#     counts = np.bincount(sch.wrk.employer, minlength=sch.emp.current_labor.size)
#     sch.emp.current_labor[:] = counts
#
#     # every contract will expire this period
#     sch.wrk.periods_left[:] = 1
#     sch.wrk.wage[:] = 1.25
#     firms_calc_wage_bill(sch.emp)  # sync wage-bill with roster
#
#     # Run Event-4 systems (with expirations enabled)
#     _run_production_event(sch, with_expiration=True)
#
#     # Post-event invariants
#     # ---------------------
#     # firm labour vector equals counted employed workers
#     counts_after = np.bincount(
#         sch.wrk.employer[sch.wrk.employed == 1],
#         minlength=sch.emp.current_labor.size,
#     )
#     np.testing.assert_array_equal(counts_after, sch.emp.current_labor)
#
#     # workers that left the roster are flagged as contract_expired
#     left_roster = sch.wrk.employed == 0
#     assert np.all((sch.wrk.contract_expired[left_roster] == 1))
#
#     # wage-bill rebuild: W_i == L_i · w_i
#     np.testing.assert_allclose(
#         sch.emp.wage_bill,
#         sch.emp.current_labor * sch.emp.wage_offer,
#         rtol=1e-12,
#     )
