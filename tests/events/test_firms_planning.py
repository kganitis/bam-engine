import numpy as np
from numpy.random import default_rng

from bamengine.components.firm_plan import (
    FirmLaborPlan,
    FirmProductionPlan,
    FirmVacancies,
)
from bamengine.events.firms_planning import firms_planning


def test_event_firms_planning() -> None:
    rng = default_rng(seed=99)

    price = np.array([1.0, 1.5, 2.0])
    inv = np.array([0.0, 5.0, 10.0])
    prod = np.array([8.0, 10.0, 12.0])
    expected_demand = np.array([12.0, 10.0, 8.0])
    desired_prod = np.zeros(3)
    labor_prod = np.ones(3)
    desired_lab = np.array([0, 0, 0])
    current_lab = np.array([8, 10, 12])
    n_vacancies = np.array([0, 0, 0])

    prod = FirmProductionPlan(
        price=price,
        inventory=inv,
        prev_production=prod,
        expected_demand=expected_demand,
        desired_production=desired_prod,
    )
    lab = FirmLaborPlan(
        desired_production=desired_prod,
        labor_productivity=labor_prod,
        desired_labor=desired_lab,
    )
    vac = FirmVacancies(
        desired_labor=desired_lab,
        current_labor=current_lab,
        n_vacancies=n_vacancies,
    )

    firms_planning(prod, lab, vac, p_avg=1.5, h_rho=0.1, rng=rng)

    # Sanity: expected_demand set, desired_labor positive
    assert (prod.expected_demand > 0).all()
    assert (lab.desired_labor >= 1).all()
    assert (vac.n_vacancies >= 0).all()
    assert (vac.n_vacancies > 0).any()
    assert (vac.n_vacancies <= vac.desired_labor).all()
    # Invariant: exp_demand == desired_production
    np.testing.assert_array_equal(prod.expected_demand, prod.desired_production)
