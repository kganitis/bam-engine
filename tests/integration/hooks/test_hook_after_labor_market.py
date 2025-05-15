import numpy as np

from bamengine.scheduler import Scheduler


def zero_out_inventory(sched: Scheduler) -> None:
    """Callback that sets all firm inventories to 0."""
    sched.prod.inventory[:] = 0.0


def test_hook_after_stub_forces_no_inventory() -> None:
    sch = Scheduler.init(n_firms=10, n_households=40, n_banks=5, seed=123)

    # --- period t --------------------------------------------------------
    sch.step(after_stub=zero_out_inventory)  # <-- new hook name

    # Inventories stay zero because the stub already ran
    assert np.allclose(sch.prod.inventory, 0.0)

    # --- period t+1 ------------------------------------------------------
    # Plan again; with zero stock every firm takes the “up” branch
    p_avg = float(sch.prod.price.mean())
    from bamengine.systems.planning import firms_decide_desired_production

    firms_decide_desired_production(sch.prod, p_avg=p_avg, h_rho=sch.h_rho, rng=sch.rng)

    assert (sch.prod.desired_production >= sch.prod.production).all()
