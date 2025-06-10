
from __future__ import annotations

import logging

import numpy as np
from numpy.random import default_rng

from bamengine.systems.production import calc_unemployment_rate, firms_pay_wages, \
    workers_receive_wage, firms_run_production, workers_update_contracts
from helpers.factories import mock_producer, mock_employer, mock_worker, \
    mock_economy, mock_consumer

logging.getLogger("bamengine").setLevel(logging.DEBUG)



rng = default_rng(42)

ec = mock_economy(
    avg_mkt_price=2.0,
    avg_mkt_price_history=np.array([1.0, 1.25, 1.50, 2.00])
)

prod = mock_producer(
    n=4,
    inventory=np.array([0.0, 0.0, 3.0, 3.0]),
    price=np.array([1.0, 1.0, 3.0, 3.0]),
    alloc_scratch=False,
)

emp = mock_employer(
    n=4,
    current_labor=np.full(4, 2, dtype=np.int64),
    wage_bill=np.full(4, 2.0),
    total_funds=np.full(4, 2.0),
)

wrk = mock_worker(
    n=10,
    employed=np.array([1]*8 + [0]*2, dtype=np.bool_),
    employer=np.array([0]*2 + [1]*2 + [2]*2 + [3]*2 + [-1]*2, dtype=np.intp),
    periods_left=np.array([2, 1] * 4 + [0, 0]),
    wage=np.array([0.5, 1.5] * 2 + [1.3, 0.7] * 2 + [0.0] * 2),
)

calc_unemployment_rate(ec, wrk)
firms_pay_wages(emp)
workers_receive_wage(mock_consumer(10), wrk)
firms_run_production(prod, emp)
workers_update_contracts(wrk, emp)
calc_unemployment_rate(ec, wrk)
