# tests/_tests/manual_bankruptcy.py
from __future__ import annotations

import logging

import numpy as np
from numpy.random import default_rng

from bamengine.systems.bankruptcy import firms_update_net_worth, mark_bankrupt_firms, \
    mark_bankrupt_banks, spawn_replacement_firms, spawn_replacement_banks
from helpers.factories import mock_economy, mock_borrower, mock_lender, mock_loanbook, \
    mock_producer, mock_employer, mock_worker

logging.getLogger("bamengine").setLevel(logging.DEBUG)

ec = mock_economy()
prod = mock_producer(
    n=3,
    production=np.array([20.0, 20.0, 20.0]),
    inventory=np.array([0, 10, 20], dtype=np.int64),
    price=np.array([2.0, 2.0, 2.0]),
)
bor = mock_borrower(
    n=3,
    net_worth=np.array([10.0, 10.0, 10.0]),
    total_funds=np.array([21.34, 8.90, 0.0]),
    wage_bill=np.array([20.0, 20.0, 20.0]),
    gross_profit=np.array([20.0, 0.0, -20.0]),
    net_profit=np.array([18.90, -1.10, -21.10]),
    retained_profit=np.array([11.34, -1.10, -21.10]),
)
lend = mock_lender(
    n=2,
    equity_base=np.array([4.20, -9.00]),
    credit_supply=np.array([0.0, 0.0]),
    interest_rate=np.array([0.055, 0.055]),
)
lb = mock_loanbook()
lb.append_loans_for_lender(
    lender_idx=np.intp(1),
    borrower_indices=np.array([2], dtype=np.intp),
    amount=np.array([10.0]),
    rate=np.array([0.11]),
)
emp = mock_employer(
    n=3,
    current_labor=np.array([20, 20, 20], dtype=np.int64),
    wage_bill=bor.wage_bill,
)
wrk = mock_worker(
    n=60,
    employed=np.ones(60, dtype=np.bool_),
    employer=np.array([0]*20 + [1]*20 + [2]*20, dtype=np.intp),
    employer_prev=np.array([0]*20 + [1]*20 + [2]*20, dtype=np.intp),
    wage=np.ones(60),
    periods_left=np.full(60, 2, dtype=np.int64),
    contract_expired=np.full(60, 0, dtype=np.bool_),
    fired=np.full(60, 0, dtype=np.bool_),
)
rng = default_rng(0)

firms_update_net_worth(bor)
mark_bankrupt_firms(ec, emp, bor, prod, wrk, lb)
mark_bankrupt_banks(ec, lend, lb)
spawn_replacement_firms(ec, prod, emp, bor, rng=rng)
spawn_replacement_banks(ec, lend, rng=rng)
