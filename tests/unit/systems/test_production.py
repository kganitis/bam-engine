# tests/unit/systems/test_production.py
"""
Unit tests for production systems.
"""
from __future__ import annotations

import numpy as np

from bamengine.systems.production import (
    consumers_receive_wage,
    firms_pay_wages,
    firms_run_production,
)
from tests.helpers.factories import (
    mock_consumer,
    mock_employer,
    mock_producer,
    mock_worker,
)


# ------------------------------------------------------------------ #
#  firms_pay_wages                                                   #
# ------------------------------------------------------------------ #
def test_firms_pay_wages_debits_cash() -> None:
    emp = mock_employer(
        n=2,
        current_labor=np.array([4, 0]),
        wage_offer=np.array([1.0, 2.0]),
        wage_bill=np.array([4.0, 0.0]),
        total_funds=np.array([20.0, 7.0]),
    )
    before = emp.total_funds.copy()

    firms_pay_wages(emp)

    np.testing.assert_allclose(emp.total_funds, before - emp.wage_bill, rtol=1e-12)


# ------------------------------------------------------------------ #
#  consumers_receive_wage                                            #
# ------------------------------------------------------------------ #
def test_consumers_receive_wage_credits_income() -> None:
    wrk = mock_worker(
        n=2,
        employed=np.array([1, 0], dtype=np.bool_),
        wage=np.array([4.0, 3.0]),
    )
    con = mock_consumer(n=2, income=np.array([1.0, 5.0]))

    consumers_receive_wage(con, wrk)

    # only consumer-0 is paid (worker-0 employed)
    np.testing.assert_allclose(con.income, np.array([5.0, 5.0]))


# ------------------------------------------------------------------ #
#  firms_run_production                                              #
# ------------------------------------------------------------------ #
def test_firms_run_production_updates_output_and_stock() -> None:
    emp = mock_employer(
        n=2,
        current_labor=np.array([4, 0]),
    )
    prod = mock_producer(
        n=2,
        labor_productivity=np.array([2.0, 3.0]),
        production=np.zeros(2),
        inventory=np.zeros(2),
    )

    firms_run_production(prod, emp)

    expected = prod.labor_productivity * emp.current_labor
    np.testing.assert_allclose(prod.production, expected)
    np.testing.assert_allclose(prod.inventory, expected)
