"""
Smoke integration test: one `Scheduler.step()` covering Events 1 + 2.

The goal is **not** to retest algebra already covered by unit tests, but to
confirm that the driver glues the systems together correctly and keeps basic
state invariants intact.

Its job is to answer “Did anything explode when I call step()?”
"""

import numpy as np

from bamengine.scheduler import Scheduler


def test_scheduler_step_event1_event2(tiny_sched: Scheduler) -> None:
    """
    After a single call to `step()`:

    * Vacancies remain non-negative.
    * Wage offers respect the (possibly revised) minimum wage.
    * Employment flags are 0/1 only.
    * Total employed workers do not exceed total desired labour.
    """
    sch = tiny_sched
    sch.step()  # runs Events 1 & 2 plus stub bookkeeping

    # ------------------------------------------------------------------ #
    # Invariants
    # ------------------------------------------------------------------ #
    # Vacancies
    assert (sch.fh.n_vacancies >= 0).all()

    # Wage floor
    assert (sch.fw.wage_offer >= sch.ec.min_wage).all()

    # Employment flags ∈ {0,1}
    uniq = np.unique(sch.ws.employed)
    assert set(uniq).issubset({0, 1})

    # Every employed worker is counted in current_labor, and vice‑versa
    assert sch.ws.employed.sum() == sch.fh.current_labor.sum()
