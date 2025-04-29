import numpy as np
from numpy.random import default_rng

from bamengine.components.firm_plan import FirmLaborPlan, FirmProductionPlan
from bamengine.systems.planning import decide_desired_labor, decide_desired_production


def test_decide_desired_production() -> None:
    rng = default_rng(seed=1)

    prod = FirmProductionPlan(
        price=np.array([2.0, 1.5, 1.0, 1.0, 2.0]),
        inventory=np.array([0.0, 0.0, 5.0, 0.0, 5.0]),
        prev_production=np.full(5, 10.0),
        expected_demand=np.zeros(5),
        desired_production=np.zeros(5),
    )

    decide_desired_production(prod, p_avg=1.5, h_rho=0.1, rng=rng)

    # ---- Shape guards (fail early if something went wrong)
    assert prod.expected_demand.shape == (5,), "Expected demand shape mismatch"
    assert prod.desired_production.shape == (5,), "Desired production shape mismatch"

    # ---- Compute expected manually with identical RNG
    rng2 = default_rng(seed=1)
    shocks = rng2.uniform(0.0, 0.1, 5)

    expected = np.array(
        [
            10.0 * (1 + shocks[0]),  # firm 0
            10.0 * (1 + shocks[1]),  # firm 1
            10.0 * (1 - shocks[2]),  # firm 2
            10.0,  # firm 3: no update
            10.0,  # firm 4: no update
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(prod.expected_demand, expected, rtol=1e-6)
    np.testing.assert_allclose(prod.desired_production, expected, rtol=1e-6)


def test_decide_desired_labor() -> None:
    lab = FirmLaborPlan(
        desired_production=np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
        labor_productivity=np.array([1.0, 0.8, 1.2, 0.5, 2.0]),
        desired_labor=np.zeros(5, dtype=np.int64),
    )

    decide_desired_labor(lab)

    # expected_labor = ceil(desired_production / labor_productivity)
    expected_labor = np.array(
        [
            10,  # 10 / 1.0
            13,  # 10 / 0.8 = 12.5 → ceil to 13
            9,  # 10 / 1.2 ≈ 8.33 → ceil to 9
            20,  # 10 / 0.5 = 20
            5,  # 10 / 2.0 = 5
        ]
    )

    np.testing.assert_array_equal(lab.desired_labor, expected_labor)
