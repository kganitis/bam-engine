# tests/unit/systems/test_revenue.py
"""
Revenue-systems unit-tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from bamengine.roles import Borrower, Lender, LoanBook
from bamengine.systems.revenue import (
    firms_collect_revenue,
    firms_pay_dividends,
    firms_validate_debt_commitments,
)
from tests.helpers.factories import (
    mock_borrower,
    mock_lender,
    mock_loanbook,
    mock_producer,
)


def test_collect_revenue_basic() -> None:
    """
    Revenues = p·sold and gross_profit = revenue − wage_bill
    Cash is increased by revenue.
    """
    prod = mock_producer(
        n=1,
        production=np.array([10.0]),
        inventory=np.array([2.0]),  # sold = 8
        price=np.array([1.25]),
    )
    bor = mock_borrower(
        n=1,
        wage_bill=np.array([5.0]),
        total_funds=np.array([20.0]),
    )

    firms_collect_revenue(prod, bor)

    expected_rev = 1.25 * 8.0
    assert bor.total_funds[0] == pytest.approx(20.0 + expected_rev)
    assert bor.gross_profit[0] == pytest.approx(expected_rev - 5.0)


def test_collect_revenue_zero_sales() -> None:
    """
    When nothing is sold (inventory == production) revenue is zero.
    """
    prod = mock_producer(
        n=1,
        production=np.array([5.0]),
        inventory=np.array([5.0]),
        price=np.array([3.0]),
    )
    bor = mock_borrower(n=1)

    firms_collect_revenue(prod, bor)
    assert bor.gross_profit[0] == 0.0
    assert bor.total_funds[0] == 10.0  # unchanged


def _single_loan_setup(
    gross_profit: float,
    debt: float,
    *,
    lender_equity: float = 1_000.0,
) -> tuple[Borrower, Lender, LoanBook]:
    """Build a 1-firm / 1-bank world with exactly one loan row."""
    bor = mock_borrower(
        n=1,
        gross_profit=np.array([gross_profit]),
        net_worth=np.array([50.0]),
        total_funds=np.array([100.0]),
    )
    lend = mock_lender(
        n=1,
        equity_base=np.array([lender_equity]),
    )
    # principal = interest => debt = 2×principal
    lb = mock_loanbook(n=1, size=1)
    lb.borrower[0] = 0
    lb.lender[0] = 0
    lb.principal[0] = debt / 2.0
    lb.rate[0] = 1.0
    lb.interest[0] = lb.principal[0]
    lb.debt[0] = debt

    return bor, lend, lb


def test_validate_debt_full_repayment() -> None:
    """
    gross_profit ≥ debt  →  full repayment, ledger row removed,
    lender equity increases, borrower cash decreases.
    """
    debt = 12.0
    bor, lend, lb = _single_loan_setup(gross_profit=15.0, debt=debt)
    interest = lb.interest.sum()

    firms_validate_debt_commitments(bor, lend, lb)

    # ledger emptied
    assert lb.size == 0
    # symmetric money-flow
    assert lend.equity_base[0] == pytest.approx(1_000.0 + interest)
    assert bor.total_funds[0] == pytest.approx(100.0 - debt)
    # net_profit = gross_profit − interest
    assert bor.net_profit[0] == pytest.approx(9.0)


def test_validate_debt_partial_writeoff() -> None:
    """
    gross_profit < debt  → proportional bad-debt write-off
    up to the borrower's net-worth.
    """
    # borrower 0 owes two banks
    bor = mock_borrower(
        n=1,
        gross_profit=np.array([-2.0]),
        net_worth=np.array([10.0]),
        total_funds=np.array([10.0]),  # cannot cover total debt (12.0)
    )
    lend = mock_lender(
        n=2,
        equity_base=np.array([10.0, 5.0]),
    )
    lb = mock_loanbook(n=2, size=2)
    lb.borrower[:2] = 0
    lb.lender[:2] = [0, 1]
    lb.principal[:2] = 5.0  # each → total 10.0
    lb.rate[:2] = 0.2
    lb.interest[:2] = 1.0  # each → total 2.0
    lb.debt[:2] = 6.0  # each → total 12.0

    firms_validate_debt_commitments(bor, lend, lb)

    # equity drop: each bank eats ½ of net_worth (=5.0)
    assert lend.equity_base.tolist() == pytest.approx([5.0, 0.0])
    # ledger rows *stay* (no repayment)
    assert lb.size == 2
    # net_profit = gross_profit − total_interest
    assert bor.net_profit[0] == pytest.approx(-4.0)


def test_validate_debt_no_loans_noop() -> None:
    """When a firm has zero outstanding debt nothing should change."""
    bor = mock_borrower(n=2, gross_profit=np.array([1.0, -3.0]))
    lend = mock_lender(n=1)
    lb = mock_loanbook(size=0)  # empty ledger

    equity_before = lend.equity_base.copy()

    firms_validate_debt_commitments(bor, lend, lb)

    np.testing.assert_array_equal(bor.net_profit, bor.gross_profit)
    np.testing.assert_array_equal(lend.equity_base, equity_before)


@pytest.mark.parametrize("delta", [0.0, 0.25, 0.8])
def test_pay_dividends_positive_profit(delta: float) -> None:
    """
    For positive net_profit the retained share is (1-delta)
    and cash drops by the dividend amount.
    """
    net = 10.0
    bor = mock_borrower(
        n=1,
        net_profit=np.array([net]),
        total_funds=np.array([50.0]),
        net_worth=np.array([40.0]),
    )

    firms_pay_dividends(bor, delta=delta)

    retained = net * (1 - delta)
    div = net - retained

    assert bor.retained_profit[0] == pytest.approx(retained)
    assert bor.total_funds[0] == pytest.approx(50.0 - div)


def test_pay_dividends_negative_profit() -> None:
    """
    For losses (≤0) dividends are zero and retained_profit == net_profit.
    """
    bor = mock_borrower(
        n=1,
        net_profit=np.array([-4.0]),
        total_funds=np.array([20.0]),
        net_worth=np.array([30.0]),
    )

    firms_pay_dividends(bor, delta=0.99)  # δ should be irrelevant

    assert bor.retained_profit[0] == -4.0
    assert bor.total_funds[0] == 20.0  # cash unchanged
