# tests/integration/events/test_event_planning.py
import numpy as np
import pytest

from bamengine.scheduler import Scheduler
from bamengine.systems.planning import (
    firms_decide_desired_labor,
    firms_decide_desired_production,
    firms_decide_vacancies,
)


def test_event_planning(tiny_sched: Scheduler) -> None:
    sch = tiny_sched

    # ---------- GIVEN -----------------------------------------------------
    prev_Y_mean = sch.prod.production.mean()

    # ---------- WHEN  – run Event-1 systems *manually* --------------------
    p_avg = float(sch.prod.price.mean())
    firms_decide_desired_production(sch.prod, p_avg=p_avg, h_rho=sch.h_rho, rng=sch.rng)
    firms_decide_desired_labor(sch.prod, sch.emp)
    firms_decide_vacancies(sch.emp)

    # ---------- THEN  – cross-component invariants ------------------------

    # 1. Non-negativity guards
    # Guards against silent underflow/overflow bugs.
    assert (sch.prod.desired_production >= 0.0).all()
    assert (sch.emp.desired_labor >= 0).all()
    assert (sch.emp.n_vacancies >= 0).all()

    # 2. Labor–productivity consistency:  Ld_i = ceil(Yd_i / a_i)
    # Confirms decide_desired_labor honouring productivity.
    expected_Ld = np.ceil(
        sch.prod.desired_production / sch.prod.labor_productivity
    ).astype(np.int64)
    np.testing.assert_array_equal(sch.emp.desired_labor, expected_Ld)

    # 3. Vacancies equation: V_i = max(Ld_i − L_i , 0)
    # Confirms coupling between labor & vacancies.
    expected_V = np.maximum(sch.emp.desired_labor - sch.emp.current_labor, 0).astype(
        np.int64
    )
    np.testing.assert_array_equal(sch.emp.n_vacancies, expected_V)

    # 4. Mean desired production changed (shock or inventory rule triggered)
    # Ensures shocks/inventory rules actually mutate state.
    assert sch.prod.desired_production.mean() != pytest.approx(prev_Y_mean)

    # 5. At least one firm posts vacancies whenever some desired labor > current
    # Detects mis-ordering where decide_vacancies ran before decide_desired_labor.
    if (sch.emp.desired_labor > sch.emp.current_labor).any():
        assert (sch.emp.n_vacancies > 0).any()

    # 6. Scheduler aggregation helpers must be in sync
    # Ensures the scheduler-level aggregates are consistent with the raw component state
    assert np.isclose(sch.mean_Yd, sch.prod.desired_production.mean())
    assert sch.mean_Ld == pytest.approx(sch.emp.desired_labor.mean())
