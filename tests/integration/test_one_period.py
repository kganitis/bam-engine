from bamengine.scheduler import Scheduler


def test_one_period_integration() -> None:
    sched = Scheduler.init(n_firms=5, h_rho=0.1, seed=123)

    # Run exactly one period
    sched.step()

    # --- sanity checks ---------------------------------------------------
    assert sched.prod.desired_production.shape == (5,)
    assert sched.lab.desired_labor.shape == (5,)
    assert sched.vac.n_vacancies.shape == (5,)

    assert (sched.prod.expected_demand > 0).all()
    assert (sched.prod.expected_demand == sched.prod.desired_production).all()

    assert (sched.lab.desired_labor > 0).all()

    assert (sched.vac.n_vacancies >= 0).all()
    assert (sched.vac.n_vacancies > 0).any()
    assert (sched.vac.n_vacancies <= sched.lab.desired_labor).all()
