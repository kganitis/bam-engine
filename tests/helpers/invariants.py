import numpy as np

from bamengine.scheduler import Scheduler


def assert_basic_invariants(sch: Scheduler) -> None:
    """
    Assertions that must hold after *any* Scheduler.step() call,
    regardless of economy size or number of periods.
    """
    # Production plans non‑negative & finite
    assert not np.isnan(sch.prod.desired_production).any()
    assert (sch.prod.desired_production >= 0).all()

    # Vacancies never negative
    assert (sch.fh.n_vacancies >= 0).all()

    # Wage offers respect current minimum wage
    assert (sch.fw.wage_offer >= sch.ec.min_wage).all()

    # Employment flags are boolean
    assert set(np.unique(sch.ws.employed)).issubset({0, 1})

    # Every employed worker counted in current_labor, and vice‑versa
    assert sch.ws.employed.sum() == sch.fh.current_labor.sum()
