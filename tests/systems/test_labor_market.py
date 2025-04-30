import numpy as np
from numpy.random import default_rng

from bamengine.components.economy import Economy
from bamengine.components.firm_labor import FirmWageOffer
from bamengine.systems.labor_market import adjust_minimum_wage, decide_wage_offer


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


def test_decide_wage_offer() -> None:
    rng = default_rng(seed=42)

    fw = FirmWageOffer(
        wage_prev=np.array([1.0, 1.2, 1.1, 1.3]),
        n_vacancies=np.array([3, 0, 2, 0]),  # 2 firms hiring
        wage_offer=np.zeros(4),
    )

    decide_wage_offer(fw, w_min=1.05, h_xi=0.1, rng=rng)

    # Firms with no vacancies should offer max(w_min , wage_prev) == w_min
    assert np.isclose(fw.wage_offer[1], 1.2)  # higher than min → unchanged
    assert np.isclose(fw.wage_offer[3], 1.3)  # higher than min → unchanged

    # Firms hiring should have wage >= w_min and >= wage_prev
    assert (fw.wage_offer[fw.n_vacancies > 0] >= fw.wage_prev[fw.n_vacancies > 0]).all()
    assert (fw.wage_offer >= 1.05).all()
