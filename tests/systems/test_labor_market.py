import numpy as np

from bamengine.components.economy import Economy
from bamengine.systems.labor_market import adjust_minimum_wage


def test_adjust_minimum_wage() -> None:
    # build an economy with known price path
    ec = Economy(
        min_wage=1.0,
        avg_mrkt_price_history=np.array([1.0, 1.05, 1.10, 1.12, 1.15]),  # t = 4
        min_wage_rev_period=4,
    )
    adjust_minimum_wage(ec)
    # inflation between P_0 and P_3 (t-1) : (1.12-1.0)/1.0 = 0.12
    assert np.isclose(ec.min_wage, 1.0 * 1.12)
