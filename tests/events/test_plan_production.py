import numpy as np
from numpy.random import default_rng

from bamengine.components.firm_labor import FirmLabor
from bamengine.components.firm_production import FirmProduction
from bamengine.events.plan_production import event_plan_production


def test_event_production_planning() -> None:
    rng = default_rng(seed=99)

    price = np.array([2.0, 1.5, 1.0])
    inventory = np.array([0.0, 0.0, 5.0])
    prod_prev = np.full(3, 10.0)

    desired_prod = np.zeros(3)
    prod = FirmProduction(
        price=price,
        inventory=inventory,
        prev_production=prod_prev,
        expected_demand=np.zeros(3),
        desired_production=desired_prod,
    )
    lab = FirmLabor(
        desired_production=desired_prod,
        labor_productivity=np.ones(3),
        desired_labor=np.zeros(3, dtype=np.int64),
    )

    event_plan_production(prod, lab, p_avg=1.5, h_rho=0.1, rng=rng)

    # Sanity: expected_demand set, desired_labor positive
    assert (prod.expected_demand > 0).all()
    assert (lab.desired_labor >= 1).all()
    # Invariant: exp_demand == desired_production
    np.testing.assert_array_equal(prod.expected_demand, prod.desired_production)
