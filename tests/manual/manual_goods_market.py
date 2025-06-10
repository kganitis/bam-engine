
from __future__ import annotations

import logging

import numpy as np
from numpy.random import default_rng

from bamengine.systems import consumers_calc_propensity, \
    consumers_decide_income_to_spend, consumers_decide_firms_to_visit, \
    consumers_shop_one_round, consumers_finalize_purchases
from helpers.factories import mock_producer, mock_consumer

logging.getLogger("bamengine").setLevel(logging.INFO)

rng = default_rng(0)
Z = 3
beta = 0.87

prod = mock_producer(
    n=4,
    production=np.array([5.0, 10.0, 20.0, 100.0]),
    inventory=np.array([5.0, 10.0, 20.0, 100.0]),
    price=np.array([1.0, 2.0, 5.0, 50.0]),
    alloc_scratch=False,
)
con = mock_consumer(
    n = 10,
    queue_z=Z,
    income=np.array([0.5, 1.0, 1.5, 2.0, 2.5] * 2, dtype=np.float64),
    savings=np.array([1.0, 5.0] * 5, dtype=np.float64),
    largest_prod_prev=np.array([0, 1, 1, 2, 2, 3, 3, 3, 3, 3], dtype=np.intp),
)

avg_sav = con.savings.mean()

consumers_calc_propensity(con, avg_sav=avg_sav, beta=beta)
consumers_decide_income_to_spend(con)
consumers_decide_firms_to_visit(con, prod, max_Z=Z, rng=rng)
consumers_shop_one_round(con, prod, rng)
consumers_shop_one_round(con, prod, rng)
consumers_finalize_purchases(con)
