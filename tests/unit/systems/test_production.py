# tests/unit/systems/test_production.py
"""
Unit tests for production systems.
"""
from __future__ import annotations

import numpy as np

from bamengine.roles import Employer, Worker
from bamengine.systems.production import (
    # firms_decide_price,
    firms_pay_wages,
    firms_run_production,
    update_avg_mkt_price,
    workers_receive_wage,
    workers_update_contracts,
)
from tests.helpers.factories import (
    mock_consumer,
    mock_economy,
    mock_employer,
    mock_producer,
    mock_worker,
)

# def test_firms_decide_price_obeys_break_even_and_shocks() -> None:
#     rng = make_rng(0)
#
#     prod = mock_producer(
#         n=4,
#         production=np.array([5.0, 5.0, 8.0, 8.0]),
#         inventory=np.array([0.0, 0.0, 3.0, 3.0]),
#         price=np.array([1.0, 1.0, 3.0, 3.0]),
#         alloc_scratch=False,
#     )
#     emp = mock_employer(
#         n=4,
#         current_labor=np.full(4, 2, dtype=np.int64),
#         wage_offer=np.full(4, 1.0),
#         wage_bill=np.full(4, 2.0),
#     )
#     # dummy LoanBook that always returns a constant vector 0.5
#     lb = mock_loanbook()
#
#     def _const_interest(_self: "LoanBook", n: int = 128) -> NDArray[np.float64]:
#         return np.array([0.1, 10.0, 0.1, 0.5])
#
#     p_avg = 2.0
#     h_eta = 0.10
#
#     interest = np.array([0.1, 10.0, 0.1, 0.5])
#     projected_output = prod.labor_productivity * emp.current_labor
#     breakeven = (emp.wage_bill + interest) / np.maximum(projected_output, 1.0e-12)
#     breakeven_capped = np.minimum(breakeven, prod.price * 2)
#
#     with patch.object(type(lb), "interest_per_borrower", _const_interest):
#         firms_decide_price(prod, emp, lb, p_avg=p_avg, h_eta=h_eta, rng=rng)
#
#     # firm-0 price ↑ at most 10 %
#     assert prod.price[0] >= 1.0
#     assert prod.price[0] <= 1.0 * (1 + h_eta) + 1.0e-12
#
#     # firm-1 price -> breakeven
#     assert prod.price[1] >= 1.0
#     assert prod.price[1] >= breakeven_capped[1] - 1.0e-12
#
#     # firm-2 price ↓ at most 10 %
#     assert prod.price[2] <= 3.0
#     assert prod.price[2] >= 3.0 * (1 - h_eta) - 1.0e-12
#
#     # firm-3 price ↓ to breakeven
#     assert prod.price[3] <= 3.0
#     assert prod.price[3] >= breakeven_capped[3] - 1.0e-12


def test_update_avg_mkt_price_appends_series() -> None:
    ec = mock_economy()
    prod = mock_producer(n=3, price=np.array([1.0, 2.0, 3.0]))

    hist_len_before = ec.avg_mkt_price_history.size
    update_avg_mkt_price(ec, prod)

    expected_avg = prod.price.mean()
    assert ec.avg_mkt_price == expected_avg
    assert ec.avg_mkt_price_history.size == hist_len_before + 1
    assert ec.avg_mkt_price_history[-1] == expected_avg


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


def test_consumers_receive_wage_credits_income() -> None:
    wrk = mock_worker(
        n=2,
        employed=np.array([1, 0], dtype=np.bool_),
        wage=np.array([4.0, 3.0]),
    )
    con = mock_consumer(n=2, income=np.array([1.0, 5.0]))

    workers_receive_wage(con, wrk)

    # only consumer-0 is paid (worker-0 employed)
    np.testing.assert_allclose(con.income, np.array([5.0, 5.0]))


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


def _mini_state() -> tuple[Employer, Worker]:
    emp = mock_employer(
        n=2,
        current_labor=np.array([2, 1]),
    )
    wrk = mock_worker(
        n=3,
        employed=np.array([1, 1, 1], dtype=np.bool_),
        employer=np.array([0, 0, 1]),
        periods_left=np.array([2, 1, 1]),
        wage=np.array([1.0, 1.0, 1.5]),
    )
    return emp, wrk


def test_contracts_expire_and_update_everything() -> None:
    emp, wrk = _mini_state()

    workers_update_contracts(wrk, emp)

    # worker-side
    assert wrk.employed.tolist() == [1, 0, 0]  # workers 1 & 2 expired
    assert wrk.contract_expired.tolist() == [0, 1, 1]
    assert wrk.employer_prev.tolist() == [-1, 0, 1]
    assert wrk.employer.tolist() == [0, -1, -1]
    assert wrk.wage.tolist() == [1.0, 0.0, 0.0]
    assert wrk.periods_left.tolist() == [1, 0, 0]

    # firm-side
    assert emp.current_labor.tolist() == [1, 0]  # one head left each firm


def test_contracts_no_expiration_no_change() -> None:
    emp, wrk = _mini_state()
    wrk.periods_left[:] = [5, 4, 3]  # far from expiry

    before_emp = np.copy(emp.current_labor)
    before_wrk = {
        name: np.copy(getattr(wrk, name))
        for name in Worker.__dataclass_fields__
        if isinstance(getattr(wrk, name), np.ndarray)
    }

    workers_update_contracts(wrk, emp)

    # firm labour unchanged
    np.testing.assert_array_equal(emp.current_labor, before_emp)

    # all worker fields unchanged *except* periods_left ↓1 for employed
    for name, arr in before_wrk.items():
        if name == "periods_left":
            np.testing.assert_array_equal(getattr(wrk, name), arr - 1)
        else:
            np.testing.assert_array_equal(getattr(wrk, name), arr)


def test_contracts_already_zero_are_handled() -> None:
    emp = mock_employer(
        n=1,
        current_labor=np.array([1]),
    )
    wrk = mock_worker(
        n=1,
        employed=np.array([1]),
        employer=np.array([0]),
        periods_left=np.array([0]),
        wage=np.array([2.0]),
    )

    workers_update_contracts(wrk, emp)

    assert wrk.employed[0] == 0
    assert emp.current_labor[0] == 0


def test_contracts_no_employed_is_noop() -> None:
    """
    When every worker is unemployed the function should return immediately
    without mutating either the worker table or the firm labour vector.
    """
    emp = mock_employer(
        n=2,
        current_labor=np.array([0, 0]),  # bookkeeping already zero
    )
    wrk = mock_worker(
        n=3,
        employed=np.zeros(3, dtype=np.bool_),  # nobody employed
        periods_left=np.array([5, 5, 5]),
    )

    before_emp = emp.current_labor.copy()
    before_wrk = {
        name: np.copy(getattr(wrk, name))
        for name in Worker.__dataclass_fields__
        if isinstance(getattr(wrk, name), np.ndarray)
    }

    workers_update_contracts(wrk, emp)

    np.testing.assert_array_equal(emp.current_labor, before_emp)
    for name, arr in before_wrk.items():
        np.testing.assert_array_equal(getattr(wrk, name), arr)
