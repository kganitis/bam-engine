import numpy as np
from numpy.random import default_rng
from bamengine.components.firm_production import FirmProduction
from bamengine.systems.production import decide_desired_production


def test_decide_desired_production():
    rng = default_rng(seed=1)

    prod = FirmProduction(
        price=np.array([2.0, 1.5, 1.0, 1.0, 2.0]),
        inventory=np.array([0.0, 0.0, 5.0, 0.0, 5.0]),
        production_prev=np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
        expected_demand=np.zeros(5),
        desired_production=np.zeros(5),
    )

    decide_desired_production(prod, p_avg=1.5, h_rho=0.1, rng=rng)

    # Known shocks from seeded RNG
    expected_shocks = np.array(
        [
            0.05118216,  # firm 0: inv=0, price > avg → increase
            0.09504637,  # firm 1: inv=0, price == avg → increase (since >=)
            0.01441596,  # firm 2: inv>0, price < avg → decrease
            0.09912418,  # firm 3: inv=0, price < avg → no update
            0.02064253,  # firm 4: inv>0, price > avg → no update
        ]
    )

    expected = np.array(
        [
            10.0 * (1 + expected_shocks[0]),  # firm 0
            10.0 * (1 + expected_shocks[1]),  # firm 1
            10.0 * (1 - expected_shocks[2]),  # firm 2
            10.0,  # firm 3 (no condition matched)
            10.0,  # firm 4 (no condition matched)
        ]
    )

    np.testing.assert_allclose(prod.expected_demand, expected, rtol=1e-6)
    np.testing.assert_allclose(prod.desired_production, expected, rtol=1e-6)
