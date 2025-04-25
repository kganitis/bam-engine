import numpy as np
from bamengine.components.firm_labor import FirmLabor
from bamengine.systems.labor import decide_desired_labor


def test_decide_desired_labor():
    lab = FirmLabor(
        desired_production=np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
        labor_productivity=np.array([1.0, 0.8, 1.2, 0.5, 2.0]),
        desired_labor=np.zeros(5, dtype=int),
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
