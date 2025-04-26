from bamengine.scheduler import Scheduler


def test_one_period_integration() -> None:
    sched = Scheduler.init_random(n_firms=5, h_rho=0.1, seed=123)

    # Run exactly one period
    sched.step()

    # --- sanity checks ---------------------------------------------------
    assert sched.prod.desired_production.shape == (5,)
    assert sched.lab.desired_labor.shape == (5,)

    # At least one firm should adjust labour demand
    assert sched.lab.desired_labor.min() >= 1
    assert sched.lab.desired_labor.max() >= 1

    # Expected demand and desired production must be equal
    assert (sched.prod.expected_demand == sched.prod.desired_production).all()

    # Optional: simple invariant
    assert sched.mean_Ld > 0
