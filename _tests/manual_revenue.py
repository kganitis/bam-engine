# tests/_tests/manual_revenue.py
from __future__ import annotations

import logging

import numpy as np

from bamengine.systems import firms_collect_revenue, firms_validate_debt_commitments, \
    firms_pay_dividends
from helpers.factories import mock_producer, mock_borrower, mock_lender, mock_loanbook


logging.getLogger("bamengine").setLevel(logging.DEBUG)


prod = mock_producer(
    n=3,
    production=np.array([20.0, 20.0, 20.0]),
    inventory=np.array([0, 10, 20], dtype=np.int64),
    price=np.array([2.0, 2.0, 2.0]),
)
bor = mock_borrower(
    n=3,
    net_worth=np.array([10.0, 10.0, 10.0]),
    total_funds=np.array([0.0, 0.0, 0.0]),
    wage_bill=np.array([20.0, 20.0, 20.0]),
    gross_profit=np.array([0.0, 0.0, 0.0]),
    net_profit=np.array([0.0, 0.0, 0.0]),
    retained_profit=np.array([0.0, 0.0, 0.0]),
)
lend = mock_lender(
    n=2,
    equity_base=np.array([2.0, 1.0]),
    credit_supply=np.array([0.0, 0.0]),
    interest_rate=np.array([0.055, 0.055]),
)
lb = mock_loanbook()
lb.append_loans_for_lender(
    lender_idx=np.intp(0),
    borrower_indices=np.array([0, 1], dtype=np.intp),
    amount=np.array([10.0, 10.0]),
    rate=np.array([0.11, 0.11]),
)
lb.append_loans_for_lender(
    lender_idx=np.intp(1),
    borrower_indices=np.array([2], dtype=np.intp),
    amount=np.array([10.0]),
    rate=np.array([0.11]),
)

firms_collect_revenue(prod, bor)
firms_validate_debt_commitments(bor, lend, lb)
firms_pay_dividends(bor, delta=0.40)
