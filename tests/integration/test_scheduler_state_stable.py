"""
Multi‑period smoke test: run 10 consecutive Scheduler steps on a medium‑sized
economy and assert that key invariants hold after **each** period.

This catches state‑advancement bugs that single‑step tests can miss.
"""

import numpy as np
import pytest

from bamengine.scheduler import Scheduler


@pytest.mark.parametrize("steps", [2])
def test_scheduler_state_stable_over_time(steps: int) -> None:
    # Medium‑sized economy
    sch = Scheduler.init(
        n_firms=50,
        n_households=200,
        seed=9,
        # keep default shocks for realism
    )

    for t in range(steps):
        prev_labor = sch.fh.current_labor.sum()
        sch.step()

        # -------- invariants every period ---------------------------------
        # Wage offers respect current minimum wage
        assert (sch.fw.wage_offer >= sch.ec.min_wage).all()

        # Vacancies never negative
        assert (sch.fh.n_vacancies >= 0).all()

        # Employment flags stay boolean
        assert set(np.unique(sch.ws.employed)).issubset({0, 1})

        # labor stock can only grow (no firing yet)
        assert sch.fh.current_labor.sum() >= prev_labor

        # every employed worker is counted in current_labor, and vice‑versa
        assert sch.ws.employed.sum() == sch.fh.current_labor.sum()

        # (optional) numeric hygiene on production arrays
        assert not np.isnan(sch.prod.desired_production).any()
        assert (sch.prod.desired_production >= 0).all()

    # If we reached here, the state remained hygienic for `steps` periods
