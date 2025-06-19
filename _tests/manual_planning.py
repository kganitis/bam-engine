
from __future__ import annotations

import logging

import numpy as np
from numpy.random import default_rng

from bamengine.systems import firms_decide_desired_production, \
    firms_decide_desired_labor, firms_decide_vacancies
from bamengine.systems.planning import firms_calc_breakeven_price, firms_adjust_price
from helpers.factories import mock_producer, mock_employer, mock_loanbook

logging.getLogger("bamengine").setLevel(logging.DEBUG)


rng = default_rng(4)
prod = mock_producer(
    n=7,
    production=np.array([10.0, 10.0, 15.0, 10.0, 10.0, 10.0, 10.0]),
    inventory=np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 10.0]),
    price=np.array([1.5, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0]),
)
emp = mock_employer(
    n=7,
    wage_bill=np.array([10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 18.0]),
)
lb = mock_loanbook()
lb.append_loans_for_lender(
    np.intp(0),
    np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.intp),
    amount=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
    rate=np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]),
)


firms_decide_desired_production(prod, p_avg=1.5, h_rho=0.10, rng=rng)
firms_calc_breakeven_price(prod, emp, lb)
firms_adjust_price(prod, p_avg=1.5, h_eta=0.10, rng=rng)
firms_decide_desired_labor(prod, emp)
firms_decide_vacancies(emp)
