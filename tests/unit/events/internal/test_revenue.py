"""
Revenue events internal implementation unit-tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from bamengine.events._internal import (
    firms_collect_revenue,
    firms_pay_dividends,
    firms_validate_debt_commitments,
)
from bamengine.relationships import LoanBook
from bamengine.roles import Borrower, Lender
from tests.helpers.factories import (
    mock_borrower,
    mock_consumer,
    mock_lender,
    mock_loanbook,
    mock_producer,
    mock_shareholder,
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
    # ledger emptied after bad debt write-off (debt has been settled)
    assert lb.size == 0
    # net_profit = gross_profit − total_interest
    assert bor.net_profit[0] == pytest.approx(-4.0)


def test_validate_debt_bad_amt_capped_at_loan_value() -> None:
    """
    When net_worth > loan_value, bad_amt should be capped at the loan value.

    This tests the fix for the bug where banks could lose more than they lent
    when a defaulting firm had net_worth exceeding its debt.
    """
    # Scenario: Firm has high net_worth (50.0) but small loan (2.0)
    # Without the fix: bank would lose frac × net_worth = 1.0 × 50.0 = 50.0
    # With the fix: bank loses at most loan_value = 2.0
    bor = mock_borrower(
        n=1,
        gross_profit=np.array([-5.0]),
        net_worth=np.array([50.0]),  # Much higher than loan
        total_funds=np.array([1.0]),  # Cannot cover debt (2.0)
    )
    lend = mock_lender(
        n=1,
        equity_base=np.array([100.0]),
    )
    lb = mock_loanbook(n=1, size=1)
    lb.borrower[0] = 0
    lb.lender[0] = 0
    lb.principal[0] = 1.0
    lb.rate[0] = 1.0
    lb.interest[0] = 1.0
    lb.debt[0] = 2.0  # Total debt = 2.0, much less than net_worth (50.0)

    firms_validate_debt_commitments(bor, lend, lb)

    # Bank should lose at most the loan value (2.0), not frac × net_worth (50.0)
    # equity = 100.0 - 2.0 = 98.0
    assert lend.equity_base[0] == pytest.approx(98.0)
    assert lb.size == 0


def test_validate_debt_bad_amt_floored_at_zero() -> None:
    """
    When net_worth < 0, bad_amt should be floored at 0.

    This ensures that negative net_worth doesn't result in banks gaining equity
    from defaults (which would be economically absurd).
    """
    # Scenario: Firm has negative net_worth (-10.0)
    # Without the floor: bank would "gain" frac × (-10) = -10 (equity increases!)
    # With the floor: bank loses 0, not the full loan, because bad_amt >= 0
    bor = mock_borrower(
        n=1,
        gross_profit=np.array([-20.0]),
        net_worth=np.array([-10.0]),  # Negative net worth
        total_funds=np.array([0.0]),  # Cannot cover debt
    )
    lend = mock_lender(
        n=1,
        equity_base=np.array([100.0]),
    )
    lb = mock_loanbook(n=1, size=1)
    lb.borrower[0] = 0
    lb.lender[0] = 0
    lb.principal[0] = 5.0
    lb.rate[0] = 0.0
    lb.interest[0] = 0.0
    lb.debt[0] = 5.0

    firms_validate_debt_commitments(bor, lend, lb)

    # Bank loses 0 (floored), not -5 (which would increase equity)
    # Note: The bank still loses the loan asset, but bad_amt accounting is 0
    assert lend.equity_base[0] == pytest.approx(100.0)
    assert lb.size == 0


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
    cons = mock_consumer(n=5, savings=np.array([100.0] * 5))

    firms_pay_dividends(bor, cons, delta=delta)

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
    cons = mock_consumer(n=5, savings=np.array([100.0] * 5))

    firms_pay_dividends(bor, cons, delta=0.99)  # δ should be irrelevant

    assert bor.retained_profit[0] == -4.0
    assert bor.total_funds[0] == 20.0  # cash unchanged


def test_pay_dividends_credits_household_savings() -> None:
    """
    Dividends are distributed equally to all households,
    increasing their savings by dividend_total / n_households.
    """
    net = 10.0
    delta = 0.25
    n_households = 5

    bor = mock_borrower(
        n=1,
        net_profit=np.array([net]),
        total_funds=np.array([50.0]),
        net_worth=np.array([40.0]),
    )
    initial_savings = np.array([100.0] * n_households)
    cons = mock_consumer(n=n_households, savings=initial_savings.copy())

    firms_pay_dividends(bor, cons, delta=delta)

    # Calculate expected dividend per household
    total_dividends = net * delta  # 2.5
    dividend_per_household = total_dividends / n_households  # 0.5

    # Each household savings should increase by dividend_per_household
    expected_savings = initial_savings + dividend_per_household
    np.testing.assert_array_almost_equal(cons.savings, expected_savings)


def test_pay_dividends_stock_flow_consistency() -> None:
    """
    Verify stock-flow consistency: total dividends paid by firms equals
    total dividends received by households.
    """
    net_profits = np.array([10.0, -5.0, 20.0])  # Two profitable, one loss
    delta = 0.10
    n_households = 10

    bor = mock_borrower(
        n=3,
        net_profit=net_profits,
        total_funds=np.array([100.0, 50.0, 200.0]),
        net_worth=np.array([50.0, 20.0, 100.0]),
    )
    initial_savings = np.full(n_households, 50.0)
    cons = mock_consumer(n=n_households, savings=initial_savings.copy())

    initial_firm_funds = bor.total_funds.sum()
    initial_household_savings = cons.savings.sum()

    firms_pay_dividends(bor, cons, delta=delta)

    # Calculate total dividends from positive profits
    positive_mask = net_profits > 0
    total_dividends = (net_profits[positive_mask] * delta).sum()

    # Verify firm funds decreased by total dividends
    firm_funds_decrease = initial_firm_funds - bor.total_funds.sum()
    assert firm_funds_decrease == pytest.approx(total_dividends)

    # Verify household savings increased by same amount
    household_savings_increase = cons.savings.sum() - initial_household_savings
    assert household_savings_increase == pytest.approx(total_dividends)

    # Verify firm debit equals household credit (stock-flow consistency)
    assert firm_funds_decrease == pytest.approx(household_savings_increase)


def test_pay_dividends_sets_shareholder_dividends() -> None:
    """
    When sh is provided, sh.dividends is overwritten with per-household dividend.
    """
    net = 10.0
    delta = 0.25
    n_households = 5

    bor = mock_borrower(
        n=1,
        net_profit=np.array([net]),
        total_funds=np.array([50.0]),
    )
    cons = mock_consumer(n=n_households, savings=np.array([100.0] * n_households))
    sh = mock_shareholder(n=n_households)

    firms_pay_dividends(bor, cons, delta=delta, sh=sh)

    expected_div = net * delta / n_households  # 0.5
    np.testing.assert_array_almost_equal(sh.dividends, expected_div)


def test_pay_dividends_without_shareholder() -> None:
    """
    Backward compatibility: sh=None (default) works without error.
    """
    bor = mock_borrower(
        n=1,
        net_profit=np.array([10.0]),
        total_funds=np.array([50.0]),
    )
    cons = mock_consumer(n=5, savings=np.array([100.0] * 5))

    firms_pay_dividends(bor, cons, delta=0.25)
    # No error, and savings still updated
    assert cons.savings[0] == pytest.approx(100.5)


def test_pay_dividends_stock_flow_with_shareholder() -> None:
    """
    Firm funds decrease == savings increase == sh.dividends sum.
    """
    net_profits = np.array([10.0, -5.0, 20.0])
    delta = 0.10
    n_households = 10

    bor = mock_borrower(
        n=3,
        net_profit=net_profits,
        total_funds=np.array([100.0, 50.0, 200.0]),
    )
    initial_savings = np.full(n_households, 50.0)
    cons = mock_consumer(n=n_households, savings=initial_savings.copy())
    sh = mock_shareholder(n=n_households)

    initial_firm_funds = bor.total_funds.sum()

    firms_pay_dividends(bor, cons, delta=delta, sh=sh)

    firm_funds_decrease = initial_firm_funds - bor.total_funds.sum()
    household_savings_increase = cons.savings.sum() - initial_savings.sum()

    assert firm_funds_decrease == pytest.approx(sh.dividends.sum())
    assert household_savings_increase == pytest.approx(sh.dividends.sum())


def test_pay_dividends_shareholder_overwrites_each_call() -> None:
    """
    Two calls: second overwrites sh.dividends, not accumulates.
    """
    n_households = 5
    bor = mock_borrower(
        n=1,
        net_profit=np.array([10.0]),
        total_funds=np.array([500.0]),
    )
    cons = mock_consumer(n=n_households, savings=np.array([100.0] * n_households))
    sh = mock_shareholder(n=n_households)

    # First call
    firms_pay_dividends(bor, cons, delta=0.5, sh=sh)
    first_div = sh.dividends[0]
    assert first_div > 0

    # Second call with different profit
    bor.net_profit[:] = 20.0
    firms_pay_dividends(bor, cons, delta=0.5, sh=sh)
    second_div = sh.dividends[0]

    # Second dividend should be from the 20.0 profit, not accumulated
    expected_second = (20.0 * 0.5) / n_households
    assert second_div == pytest.approx(expected_second)
    assert second_div != pytest.approx(first_div + expected_second)
