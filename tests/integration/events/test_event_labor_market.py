# tests/integration/events/test_event_labor_market.py
"""
Event-2 integration tests (labor market)

The first test reproduces the original regression checks.
The second test adds deeper cross-component consistency assertions that
unit tests cannot see in isolation.
"""
from __future__ import annotations

import numpy as np
import pytest

from bamengine.simulation import Simulation
from bamengine.events._internal.labor_market import (
    adjust_minimum_wage,
    calc_annual_inflation_rate,
    firms_decide_wage_offer,
    firms_hire_workers,
    workers_decide_firms_to_apply,
    workers_send_one_round,
)


def _run_planning(sch: Simulation) -> None:
    from bamengine.events._internal.planning import (
        firms_decide_desired_labor,
        firms_decide_desired_production,
        firms_decide_vacancies,
    )

    p_avg = float(sch.prod.price.mean())
    firms_decide_desired_production(sch.prod, p_avg=p_avg, h_rho=sch.h_rho, rng=sch.rng)
    firms_decide_desired_labor(sch.prod, sch.emp)
    firms_decide_vacancies(sch.emp)


def test_event_labor_market(tiny_sched: Simulation) -> None:
    sch = tiny_sched

    # Event-1
    _run_planning(sch)

    original_vacancies = sch.emp.n_vacancies.copy()
    original_employed = sch.wrk.employed.copy()
    original_labor = sch.emp.current_labor.copy()

    # Event-2
    prev_floor = sch.ec.min_wage
    calc_annual_inflation_rate(sch.ec)
    adjust_minimum_wage(sch.ec)

    firms_decide_wage_offer(
        sch.emp,
        w_min=sch.ec.min_wage,
        h_xi=sch.h_xi,
        rng=sch.rng,
    )

    workers_decide_firms_to_apply(sch.wrk, sch.emp, max_M=sch.max_M, rng=sch.rng)

    for _ in range(sch.max_M):
        workers_send_one_round(sch.wrk, sch.emp)
        firms_hire_workers(sch.wrk, sch.emp, theta=sch.theta)

    # invariants
    m = sch.ec.min_wage_rev_period
    hist = sch.ec.avg_mkt_price_history
    if hist.size > m and (hist.size - 1) % m == 0:
        p_now, p_prev = hist[-2], hist[-m - 1]
        expected = prev_floor * (1 + (p_now - p_prev) / p_prev)
        assert sch.ec.min_wage == pytest.approx(expected)
    else:
        assert sch.ec.min_wage == pytest.approx(prev_floor)

    assert (sch.emp.wage_offer >= sch.ec.min_wage).all()

    assert set(np.unique(sch.wrk.employed)).issubset({0, 1})
    assert (sch.wrk.employed >= original_employed).all()
    hires = sch.wrk.employed.sum() - original_employed.sum()
    assert hires == (sch.emp.current_labor.sum() - original_labor.sum())

    assert (sch.emp.n_vacancies >= 0).all()
    vac_reduction = original_vacancies.sum() - sch.emp.n_vacancies.sum()
    assert hires == vac_reduction

    assert np.all((sch.wrk.job_apps_head[sch.wrk.employed == 1] == -1))


def test_labor_market_post_state_consistency(tiny_sched: Simulation) -> None:
    sch = tiny_sched
    _run_planning(sch)

    calc_annual_inflation_rate(sch.ec)
    adjust_minimum_wage(sch.ec)
    firms_decide_wage_offer(sch.emp, w_min=sch.ec.min_wage, h_xi=sch.h_xi, rng=sch.rng)
    workers_decide_firms_to_apply(sch.wrk, sch.emp, max_M=sch.max_M, rng=sch.rng)
    for _ in range(sch.max_M):
        workers_send_one_round(sch.wrk, sch.emp)
        firms_hire_workers(sch.wrk, sch.emp, theta=sch.theta)

    # worker ↔ firm labor counts are consistent
    counts = np.bincount(
        sch.wrk.employer[sch.wrk.employed == 1],
        minlength=sch.emp.current_labor.size,
    )
    np.testing.assert_array_equal(counts, sch.emp.current_labor)

    # employed workers paid the posted (≥ floor) wage
    employed_idx = np.where(sch.wrk.employed == 1)[0]
    worker_wages = sch.wrk.wage[employed_idx]
    firm_wages = sch.emp.wage_offer[sch.wrk.employer[employed_idx]]
    np.testing.assert_allclose(worker_wages, firm_wages)
    assert (worker_wages >= sch.ec.min_wage).all()

    # contract duration set correctly
    durations = sch.wrk.periods_left[employed_idx]
    assert np.issubdtype(durations.dtype, np.integer)
    assert (durations >= sch.theta).all()

    # inbound queues: any firm that still has vacancies must have
    # an empty queue; firms with zero vacancies may retain stale pointers
    mask_vac = sch.emp.n_vacancies > 0
    assert np.all((sch.emp.recv_job_apps_head[mask_vac] == -1))
