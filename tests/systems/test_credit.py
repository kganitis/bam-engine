import numpy as np
from bamengine.components.firm_fin import FirmFin
from bamengine.systems.credit import decide_credit_demand


def test_decide_credit_demand():
    fin = FirmFin(
        net_worth=np.array([10.0, 8.0, 5.0]),
        wage_bill=np.array([9.0, 12.0, 2.0]),
        credit_demand=np.zeros(3),
    )
    decide_credit_demand(fin)
    assert (fin.credit_demand == np.array([0.0, 4.0, 0.0])).all()
