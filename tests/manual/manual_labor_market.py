from logging import getLogger

from numpy.random import default_rng

import numpy as np
import logging


from bamengine.systems import workers_decide_firms_to_apply, workers_send_one_round, \
    firms_hire_workers, firms_decide_wage_offer, firms_calc_wage_bill
from bamengine.systems.labor_market import calc_annual_inflation_rate, \
    adjust_minimum_wage
from helpers.factories import mock_employer, mock_worker, mock_economy

logging.getLogger("bamengine").setLevel(logging.DEBUG)


rng = default_rng(2)
max_M = 3
h_xi = 0.05

ec = mock_economy(
    avg_mkt_price_history=np.array([1.0, 1.05, 1.10, 1.15, 1.20]),
)

emp = mock_employer(
    n=6,
    queue_m=max_M,
    wage_offer=np.array([1.0, 1.2, 1.5, 2.0, 2.0, 2.5]),
    n_vacancies=np.array([2, 2, 0, 1, 0, 2], dtype=np.int64),
    current_labor=np.array([0, 0, 1, 0, 0, 1], dtype=np.int64),
)

wrk = mock_worker(
    n=10,
    queue_m=max_M,
    employed=np.array([False, False, False, False, False,
                       False, False, False, True, True]),
    employer=np.array([-1, -1, -1, -1, -1, -1, -1, -1, 2, 5], dtype=np.intp),
    employer_prev=np.array([2, 4, 0, 1, 3, 6, 2, 3, -1, -1], dtype=np.intp),
    contract_expired=np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 0], dtype=np.bool_),
    fired=np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.bool_),
    wage=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 2.5])
)

calc_annual_inflation_rate(ec)
adjust_minimum_wage(ec)
firms_decide_wage_offer(emp, w_min=ec.min_wage, h_xi=h_xi, rng=rng)
workers_decide_firms_to_apply(wrk, emp, max_M=max_M, rng=rng)
for _ in range(max_M):
    workers_send_one_round(wrk, emp, rng)
    firms_hire_workers(wrk, emp, theta=8, rng=rng)
firms_calc_wage_bill(emp, wrk)
getLogger("bamengine").debug(f"Employer: {wrk.employer}")
getLogger("bamengine").debug(f"Current Labor: {emp.current_labor}")
