# tests/integration/scheduler/test_scheduler.py
"""
Scheduler smoke-tests.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bamengine.scheduler import Scheduler
from tests.helpers.invariants import assert_basic_invariants


# --------------------------------------------------------------------------- #
#   single-step                                                               #
# --------------------------------------------------------------------------- #
def test_scheduler_step(tiny_sched: Scheduler) -> None:
    """
    Smoke integration test: one `Scheduler.step()`.
    """
    tiny_sched.step()
    assert_basic_invariants(tiny_sched)


# --------------------------------------------------------------------------- #
#   multi-period                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_periods", [100])
def test_scheduler_state_stable_over_time(n_periods: int) -> None:
    """
    Multi‑period smoke test: run consecutive Scheduler steps on a medium‑sized
    economy and assert that key invariants hold after **each** period.

    This catches state‑advancement bugs that single‑step tests can miss.
    """
    # Medium‑sized economy
    sch = Scheduler.init(
        n_firms=50,
        n_households=250,
        n_banks=5,
        seed=9,
    )

    for _ in range(n_periods):
        sch.step()
        assert_basic_invariants(sch)


# --------------------------------------------------------------------------- #
#   property-based size & seed fuzzing                                        #
# --------------------------------------------------------------------------- #
@given(
    n_firms=st.integers(20, 50),
    n_households=st.integers(100, 250),
    n_banks=st.integers(2, 5),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=50)
def test_scheduler_step_properties(
    n_firms: int, n_households: int, n_banks: int, seed: int
) -> None:
    """
    Property‑based smoke test for a single `Scheduler.step()`.

    Hypothesis varies the economy's size and a random seed.
    We assert only the most fundamental invariants that should
    hold *regardless of parameter values*.
    """
    sch = Scheduler.init(
        n_firms=n_firms, n_households=n_households, n_banks=n_banks, seed=seed
    )
    sch.step()
    assert_basic_invariants(sch)
