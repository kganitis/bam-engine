"""
Unit tests for production events internal implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from bamengine import make_rng
from bamengine.events._internal.production import (
    calc_unemployment_rate,
    firms_adjust_price,
    firms_calc_breakeven_price,
    firms_pay_wages,
    firms_run_production,
    update_avg_mkt_price,
    workers_receive_wage,
    workers_update_contracts,
)
from bamengine.roles import Employer, Worker
from bamengine.utils import EPS
from tests.helpers.factories import (
    mock_consumer,
    mock_economy,
    mock_employer,
    mock_producer,
    mock_worker,
)


def test_calc_unemployment_rate_all_employed() -> None:
    """Test unemployment rate when all workers are employed."""
    ec = mock_economy()
    wrk = mock_worker(
        n=5,
        employer=np.array([0, 0, 1, 1, 2], dtype=np.intp),  # all employed
    )

    hist_len_before = ec.unemp_rate_history.size
    calc_unemployment_rate(ec, wrk)

    # 0 unemployed out of 5 workers = 0% unemployment
    assert ec.unemp_rate_history.size == hist_len_before + 1
    assert ec.unemp_rate_history[-1] == 0.0


def test_calc_unemployment_rate_all_unemployed() -> None:
    """Test unemployment rate when all workers are unemployed."""
    ec = mock_economy()
    wrk = mock_worker(
        n=5,
        employer=np.full(5, -1, dtype=np.intp),  # all unemployed
    )

    hist_len_before = ec.unemp_rate_history.size
    calc_unemployment_rate(ec, wrk)

    # 5 unemployed out of 5 workers = 100% unemployment
    assert ec.unemp_rate_history.size == hist_len_before + 1
    assert ec.unemp_rate_history[-1] == 1.0


def test_calc_unemployment_rate_partial_unemployment() -> None:
    """Test unemployment rate with partial unemployment."""
    ec = mock_economy()
    wrk = mock_worker(
        n=10,
        employer=np.array([0, 1, 2, -1, -1, 3, 4, -1, 5, 6], dtype=np.intp),
    )

    hist_len_before = ec.unemp_rate_history.size
    calc_unemployment_rate(ec, wrk)

    # 3 unemployed out of 10 workers = 30% unemployment
    expected_rate = 3.0 / 10.0
    assert ec.unemp_rate_history.size == hist_len_before + 1
    np.testing.assert_allclose(ec.unemp_rate_history[-1], expected_rate, rtol=1e-12)


def test_calc_unemployment_rate_single_worker() -> None:
    """Test unemployment rate with single worker edge case."""
    ec = mock_economy()

    # Test employed single worker
    wrk_employed = mock_worker(n=1, employer=np.array([0], dtype=np.intp))
    calc_unemployment_rate(ec, wrk_employed)
    assert ec.unemp_rate_history[-1] == 0.0

    # Test unemployed single worker
    wrk_unemployed = mock_worker(n=1, employer=np.array([-1], dtype=np.intp))
    calc_unemployment_rate(ec, wrk_unemployed)
    assert ec.unemp_rate_history[-1] == 1.0


def test_calc_unemployment_rate_appends_to_history() -> None:
    """Test that unemployment rate is correctly appended to history."""
    ec = mock_economy()
    wrk = mock_worker(n=4, employer=np.array([0, -1, 1, -1], dtype=np.intp))

    # Call multiple times to verify appending behavior
    initial_len = ec.unemp_rate_history.size

    calc_unemployment_rate(ec, wrk)
    assert ec.unemp_rate_history.size == initial_len + 1

    calc_unemployment_rate(ec, wrk)
    assert ec.unemp_rate_history.size == initial_len + 2

    # Both entries should be the same (2/4 = 0.5)
    np.testing.assert_allclose(ec.unemp_rate_history[-2:], [0.5, 0.5], rtol=1e-12)


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
        # first employed, second unemployed
        employer=np.array([0, -1], dtype=np.intp),
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
        production_prev=np.zeros(2),
        inventory=np.zeros(2),
    )

    firms_run_production(prod, emp)

    expected = prod.labor_productivity * emp.current_labor
    np.testing.assert_allclose(prod.production, expected)
    np.testing.assert_allclose(prod.production_prev, expected)  # Also updated
    np.testing.assert_allclose(prod.inventory, expected)


def _mini_state() -> tuple[Employer, Worker]:
    emp = mock_employer(
        n=2,
        current_labor=np.array([2, 1]),
    )
    wrk = mock_worker(
        n=3,
        employer=np.array([0, 0, 1], dtype=np.intp),  # all employed
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
        employer=np.array([0], dtype=np.intp),  # employed
        periods_left=np.array([0]),
        wage=np.array([2.0]),
    )

    workers_update_contracts(wrk, emp)

    assert wrk.employed[0] == 0  # should be unemployed after contract expiration
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
        employer=np.full(3, -1, dtype=np.intp),  # nobody employed
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


def test_contracts_do_not_modify_wage_bill() -> None:
    """
    workers_update_contracts should NOT recalculate wage_bill.
    wage_bill is recalculated by firms_calc_wage_bill in Phase 2.
    """
    emp, wrk = _mini_state()
    wage_bill_before = emp.wage_bill.copy()

    workers_update_contracts(wrk, emp)

    # wage_bill should be unchanged (no recalculation)
    np.testing.assert_array_equal(emp.wage_bill, wage_bill_before)


# ============================================================================
# Breakeven Price Tests (production-phase)
# ============================================================================


def test_breakeven_no_cap_equals_raw() -> None:
    """
    If no cap_factor is provided (or cap_factor <= 1), the breakeven floor
    equals raw (wage_bill + interest) / max(projected_production, _EPS).
    projected_production = labor_productivity * current_labor.
    """
    n = 3
    prod = mock_producer(n=n)
    prod.price[:] = 1.0  # irrelevant when no cap
    # labor_productivity defaults to 1.0, so projected_production = current_labor

    emp = mock_employer(n=n)
    emp.wage_bill[:] = np.array([10.0, 5.0, 8.0])
    emp.current_labor[:] = np.array([10, 5, 2], dtype=np.int64)  # projected_production

    class DummyLoanBook:
        @staticmethod
        def interest_per_borrower(nfirms: int) -> np.ndarray:
            assert nfirms == n
            return np.array([1.0, 0.0, 1.0])

    lb = DummyLoanBook()

    # noinspection PyTypeChecker
    firms_calc_breakeven_price(prod, emp, lb, cap_factor=None)

    # projected_production = labor_productivity * current_labor = 1.0 * [10, 5, 2]
    projected_production = prod.labor_productivity * emp.current_labor
    expected = (emp.wage_bill + np.array([1.0, 0.0, 1.0])) / np.maximum(
        projected_production, EPS
    )
    np.testing.assert_allclose(prod.breakeven_price, expected, rtol=1e-12, atol=0.0)


def test_breakeven_capped_by_price_times_factor() -> None:
    """
    With a cap_factor > 1, breakeven is min(raw_breakeven, price * cap_factor).
    projected_production = labor_productivity * current_labor.
    """
    n = 3
    prod = mock_producer(n=n)
    prod.price[:] = np.array([2.0, 1.0, 0.5])  # caps: [4.0, 2.0, 1.0]
    # labor_productivity defaults to 1.0, so projected_production = current_labor

    emp = mock_employer(n=n)
    emp.wage_bill[:] = np.array([100.0, 1.0, 1.0])  # raw: [100, 1, 1] when proj_prod=1
    emp.current_labor[:] = np.array(
        [1, 1, 1], dtype=np.int64
    )  # projected_production = 1

    class DummyLoanBook:
        @staticmethod
        def interest_per_borrower(nfirms: int) -> np.ndarray:
            return np.zeros(nfirms)

    lb = DummyLoanBook()

    # noinspection PyTypeChecker
    firms_calc_breakeven_price(prod, emp, lb, cap_factor=2)

    # projected_production = labor_productivity * current_labor = 1.0 * [1, 1, 1]
    projected_production = prod.labor_productivity * emp.current_labor
    raw = (emp.wage_bill + 0.0) / np.maximum(projected_production, EPS)
    cap = prod.price * 2
    expected = np.minimum(raw, cap)

    np.testing.assert_allclose(prod.breakeven_price, expected, rtol=1e-12, atol=0.0)


# ============================================================================
# Price Adjustment Tests (production-phase)
# ============================================================================


def test_price_adjust_raise_branch() -> None:
    """
    Inventory == 0 and price < p_avg → raise by (1 + shock) and respect breakeven floor.
    """
    rng = make_rng(0)
    prod = mock_producer(n=1)
    prod.inventory[:] = 0.0
    prod.price[:] = 1.0
    prod.breakeven_price[:] = 0.1  # low floor so it doesn't bind

    old = prod.price.copy()
    firms_adjust_price(prod, p_avg=2.0, h_eta=0.1, rng=rng)

    # replicate the single shock deterministically
    shock = make_rng(0).uniform(0.0, 0.1, 1)[0]
    expected = max(float(old[0] * (1.0 + shock)), float(prod.breakeven_price[0]))

    assert prod.price[0] == pytest.approx(expected)
    assert prod.price[0] >= old[0]


def test_price_adjust_cut_branch_with_floor_increase() -> None:
    """
    Inventory > 0 and price >= p_avg → cut by (1 - shock) but *not below* breakeven.
    If breakeven > old price, the price should increase to the floor (warning case).
    """
    rng = make_rng(1)
    prod = mock_producer(n=2)

    # both firms in "cut" set (inventory>0 & price >= p_avg)
    prod.inventory[:] = np.array([5.0, 5.0])
    prod.price[:] = np.array([2.0, 1.0])
    prod.breakeven_price[:] = np.array([1.5, 1.2])  # second > old price

    old = prod.price.copy()
    firms_adjust_price(prod, p_avg=1.0, h_eta=0.1, rng=rng)

    # first: normal cut (floor 1.5 below plausible cut target)
    assert prod.price[0] <= old[0]

    # second: floor-induced increase (breakeven above old)
    assert prod.price[1] == pytest.approx(1.2)
    assert prod.price[1] > old[1]


def test_price_adjust_noop_when_masks_empty() -> None:
    """
    If a firm is neither in mask_up nor mask_dn, its price must remain unchanged.
    Cases:
      - inventory == 0 and price >= p_avg  (not mask_up since price !< p_avg)
      - inventory > 0 and price < p_avg   (not mask_dn since price !>= p_avg)
    """
    rng = make_rng(3)
    prod = mock_producer(n=2)
    prod.inventory[:] = np.array([0.0, 10.0])
    prod.price[:] = np.array([2.0, 0.5])
    prod.breakeven_price[:] = np.array([0.1, 0.4])

    old = prod.price.copy()
    firms_adjust_price(prod, p_avg=2.0, h_eta=0.1, rng=rng)

    np.testing.assert_array_equal(prod.price, old)


def test_price_adjust_cut_with_no_increase_allowed() -> None:
    """
    When price_cut_allow_increase=False, prices should not increase
    even when the breakeven floor is above the old price.
    """
    rng = make_rng(1)
    prod = mock_producer(n=2)

    # Both firms in "cut" set (inventory>0 & price >= p_avg)
    prod.inventory[:] = np.array([5.0, 5.0])
    prod.price[:] = np.array([2.0, 1.0])
    # Second firm has breakeven_price > old price
    prod.breakeven_price[:] = np.array([1.5, 1.2])

    old = prod.price.copy()
    firms_adjust_price(
        prod, p_avg=1.0, h_eta=0.1, rng=rng, price_cut_allow_increase=False
    )

    # First firm: normal cut (floor 1.5 below price cut target)
    assert prod.price[0] <= old[0]

    # Second firm: with price_cut_allow_increase=False, price should NOT increase
    # even though breakeven_price (1.2) > old price (1.0)
    # Instead, floor is min(old_price, breakeven_price) = min(1.0, 1.2) = 1.0
    assert prod.price[1] <= old[1]
