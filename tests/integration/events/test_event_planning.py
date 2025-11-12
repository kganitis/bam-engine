import numpy as np
import pytest

from bamengine.events._internal.planning import (
    firms_decide_desired_labor,
    firms_decide_desired_production,
    firms_decide_vacancies,
)
from bamengine.simulation import Simulation


def test_event_planning(tiny_sched: Simulation) -> None:
    sch = tiny_sched

    # GIVEN
    prev_Y_mean = sch.prod.production.mean()
    p_avg = float(sch.prod.price.mean())

    # Run planning systems manually
    firms_decide_desired_production(sch.prod, p_avg=p_avg, h_rho=sch.h_rho, rng=sch.rng)
    firms_decide_desired_labor(sch.prod, sch.emp)
    firms_decide_vacancies(sch.emp)

    # Cross-component invariants
    # --------------------------
    # Non-negativity guards
    # Guards against silent underflow/overflow bugs.
    assert (sch.prod.desired_production >= 0.0).all()
    assert (sch.emp.desired_labor >= 0).all()
    assert (sch.emp.n_vacancies >= 0).all()

    # Labor–productivity consistency:  Ld_i = ceil(Yd_i / a_i)
    # Confirms decide_desired_labor honouring productivity.
    expected_Ld = np.ceil(
        sch.prod.desired_production / sch.prod.labor_productivity
    ).astype(np.int64)
    np.testing.assert_array_equal(sch.emp.desired_labor, expected_Ld)

    # Vacancies equation: V_i = max(Ld_i − L_i , 0)
    # Confirms coupling between labor & vacancies.
    expected_V = np.maximum(sch.emp.desired_labor - sch.emp.current_labor, 0).astype(
        np.int64
    )
    np.testing.assert_array_equal(sch.emp.n_vacancies, expected_V)

    # Mean desired production changed (shock or inventory rule triggered)
    # Ensures shocks/inventory rules actually mutate state.
    assert sch.prod.desired_production.mean() != pytest.approx(prev_Y_mean)

    # At least one firm posts vacancies whenever some desired labor > current
    # Detects mis-ordering where decide_vacancies ran before decide_desired_labor.
    if (sch.emp.desired_labor > sch.emp.current_labor).any():
        assert (sch.emp.n_vacancies > 0).any()
