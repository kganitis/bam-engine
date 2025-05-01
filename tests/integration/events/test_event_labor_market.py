"""
Event-2 integration test

Runs the labor-market sequence on a fresh tiny scheduler:

   1. adjust_minimum_wage
   2. workers_prepare_applications
   3. up to M rounds of workers_send_one_round + firms_hire

and asserts cross-component invariants that no single unit test can see.
"""

import numpy as np

from bamengine.scheduler import Scheduler
from bamengine.systems.labor_market import (
    adjust_minimum_wage,
    firms_decide_wage_offer,
    firms_hire_workers,
    workers_prepare_applications,
    workers_send_one_round,
)


def test_event_labor_market(tiny_sched: Scheduler) -> None:
    sch = tiny_sched

    # ------------------------------------------------------------------ #
    # Run Event-1 manually to create vacancies & wage offers
    # ------------------------------------------------------------------ #
    from bamengine.systems.planning import (
        firms_decide_desired_labor,
        firms_decide_desired_production,
        firms_decide_vacancies,
    )

    p_avg = float(sch.prod.price.mean())
    firms_decide_desired_production(sch.prod, p_avg, sch.h_rho, sch.rng)
    firms_decide_desired_labor(sch.lab)
    firms_decide_vacancies(sch.vac)

    original_vacancies = sch.vac.n_vacancies.copy()
    original_employed = sch.ws.employed.copy()

    # ------------------------------------------------------------------ #
    # --------------------  EVENT-2  ----------------------------------- #
    # ------------------------------------------------------------------ #
    prev_floor = sch.ec.min_wage
    adjust_minimum_wage(sch.ec)

    firms_decide_wage_offer(
        sch.fw,
        w_min=sch.ec.min_wage,
        h_xi=sch.h_xi,
        rng=sch.rng,
    )

    workers_prepare_applications(sch.ws, sch.fw, max_M=sch.max_M, rng=sch.rng)

    for _ in range(sch.max_M):
        workers_send_one_round(sch.ws, sch.fh)
        firms_hire_workers(sch.ws, sch.fh, contract_theta=sch.theta)

    # ------------------------------------------------------------------ #
    # Assertions – cross-component invariants
    # ------------------------------------------------------------------ #

    # 1. Minimum wage never decreases
    assert sch.ec.min_wage >= prev_floor

    # 2. Every wage offer respects the (possibly revised) floor
    assert (sch.fw.wage_offer >= sch.ec.min_wage).all()

    # 3. Employment flags are boolean (0/1) and monotone non-decreasing
    assert set(np.unique(sch.ws.employed)).issubset({0, 1})
    assert (sch.ws.employed >= original_employed).all()

    # 4. Vacancies never go negative and decrease by ≤ hires
    assert (sch.fh.n_vacancies >= 0).all()
    hires = sch.ws.employed.sum() - original_employed.sum()
    vac_reduction = original_vacancies.sum() - sch.fh.n_vacancies.sum()
    assert hires == vac_reduction

    # 5. No hired worker keeps an active application pointer
    assert (sch.ws.apps_head[sch.ws.employed == 1] == -1).all()
