# tests/integrations/test_scheduler.py
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bamengine.scheduler import HOOK_NAMES, Scheduler
from tests.helpers.invariants import assert_basic_invariants


def test_scheduler_step(tiny_sched: Scheduler) -> None:
    """
    Smoke integration test: one `Scheduler.step()`.

    The goal is **not** to retest algebra already covered by unit tests, but to
    confirm that the driver glues the systems together correctly and keeps basic
    state invariants intact.

    Its job is to answer “Did anything explode when I call step()?”
    """
    tiny_sched.step()
    assert_basic_invariants(tiny_sched)


@pytest.mark.parametrize("steps", [10])
def test_scheduler_state_stable_over_time(steps: int) -> None:
    """
    Multi‑period smoke test: run consecutive Scheduler steps on a medium‑sized
    economy and assert that key invariants hold after **each** period.

    This catches state‑advancement bugs that single‑step tests can miss.
    """
    # Medium‑sized economy
    sch = Scheduler.init(
        n_firms=50,
        n_households=200,
        n_banks=5,
        seed=9,
    )

    for _ in range(steps):
        sch.step()
        assert_basic_invariants(sch)


@given(
    n_firms=st.integers(20, 50),
    n_households=st.integers(50, 200),
    n_banks=st.integers(2, 5),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=50)  # keeps CI fast; can raise for more fuzzing
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


def test_scheduler_hooks_called() -> None:
    """
    Attach a simple recorder callable to **every** defined hook.  After one
    ``Scheduler.step`` the recorder's call order must match
    ``Scheduler.HOOK_NAMES`` exactly.
    """
    call_order: list[str] = []

    hooks = {
        name: (lambda s, _name=name: call_order.append(_name)) for name in HOOK_NAMES
    }

    sch = Scheduler.init(n_firms=3, n_households=6, n_banks=3, seed=0)
    sch.step(**hooks)

    assert call_order == list(HOOK_NAMES)


def test_hook_after_stub_forces_no_inventory() -> None:

    def _zero_out_inventory(sched: Scheduler) -> None:
        """Callback that sets all firm inventories to 0."""
        sched.prod.inventory[:] = 0.0

    sch = Scheduler.init(n_firms=10, n_households=40, n_banks=5, seed=123)

    # --- period t --------------------------------------------------------
    sch.step(after_stub=_zero_out_inventory)  # <-- new hook name

    # Inventories stay zero because the stub already ran
    assert np.allclose(sch.prod.inventory, 0.0)

    # --- period t+1 ------------------------------------------------------
    # Plan again; with zero stock every firm takes the “up” branch
    p_avg = float(sch.prod.price.mean())
    from bamengine.systems.planning import firms_decide_desired_production

    firms_decide_desired_production(sch.prod, p_avg=p_avg, h_rho=sch.h_rho, rng=sch.rng)

    assert (sch.prod.desired_production >= sch.prod.production).all()


@pytest.mark.xfail(reason="Cover snapshot later")
def test_prepare_applications_loyalty_swap() -> None:  # pragma: no cover
    raise NotImplementedError


def test_scheduler_means() -> None:
    """
    Convenience-property sanity:

        • mean_Yd   averages desired production
        • mean_Ld   averages desired labour
        • mean_L    averages current labour
        • mean_B    averages credit demand
        • total_loans sums outstanding principal
    """
    sch = Scheduler.init(n_firms=3, n_households=3, n_banks=3)

    # fill deterministic toy values
    sch.prod.desired_production = np.array([1.0, 2.0, 3.0])
    sch.emp.desired_labor = np.array([10, 20, 30])
    sch.emp.current_labor = np.array([0, 1, 99])
    sch.bor.credit_demand = np.array([5.0, 0.0, 10.0])

    # create two dummy loans so total_loans ≠ 0
    sch.lb.principal = np.array([2.0, 3.0])
    sch.lb.rate = np.zeros_like(sch.lb.principal)
    sch.lb.interest = np.zeros_like(sch.lb.principal)
    sch.lb.debt = sch.lb.principal.copy()
    sch.lb.borrower = np.array([0, 2])
    sch.lb.lender = np.array([1, 0])
    sch.lb.size = 2

    # assertions
    assert np.isclose(sch.mean_Yd, 2.0)  # (1+2+3)/3
    assert sch.mean_Ld == 20  # (10+20+30)/3
    assert sch.mean_L == pytest.approx(100 / 3)  # (0+1+99)/3
    assert sch.mean_B == pytest.approx(5.0)  # (5+0+10)/3
    assert sch.total_loans == pytest.approx(5.0)
