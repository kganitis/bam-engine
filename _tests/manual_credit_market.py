from __future__ import annotations

import logging

import numpy as np
from numpy.random import default_rng

from bamengine.systems import banks_decide_credit_supply, banks_decide_interest_rate, \
    firms_decide_credit_demand, firms_calc_credit_metrics, \
    firms_prepare_loan_applications, firms_send_one_loan_app, banks_provide_loans, \
    firms_fire_workers
from helpers.factories import mock_borrower, mock_lender, mock_loanbook, mock_employer, \
    mock_worker

logging.getLogger("bamengine").setLevel(logging.DEBUG)

n_borrowers = 6
n_lenders = 3
H = 2
rng = default_rng(1)

current_labor = np.array([80, 20, 15, 10, 5, 1], dtype=np.int64)
wage_bills = np.array([80.0, 20.0, 15.0, 10.0, 5.0, 1.0])

# Generate individual wages that sum to each firm's wage bill
individual_wages = np.zeros(131)
worker_idx = 0

for firm_id, (n_workers, total_wage_bill) in enumerate(zip(current_labor, wage_bills)):
    if n_workers == 1:
        # If only one worker, they get the entire wage bill
        individual_wages[worker_idx] = total_wage_bill
    else:
        # Generate random wages using Dirichlet distribution for proper summing
        # This ensures all wages are positive and sum exactly to the target
        proportions = rng.dirichlet(np.ones(n_workers))
        firm_wages = proportions * total_wage_bill
        individual_wages[worker_idx:worker_idx + n_workers] = firm_wages

    worker_idx += n_workers

wrk = mock_worker(
    n=131,
    employed=np.ones(131, dtype=np.int64),
    employer=np.array(
        [0] * 80 + [1] * 20 + [2] * 15 + [3] * 10 + [4] * 5 + [5] * 1),
    wage=individual_wages,
)

emp = mock_employer(
    n=n_borrowers,
    current_labor=current_labor,
    wage_offer=np.ones(n_borrowers),
    wage_bill=wage_bills,
    total_funds=np.full(n_borrowers, 4.0),
)

bor = mock_borrower(
    n=n_borrowers,
    queue_h=H,
    wage_bill=emp.wage_bill,
    net_worth=emp.total_funds.copy(),
    total_funds=emp.total_funds,
)
lend = mock_lender(
    n=n_lenders,
    queue_h=H,
    equity_base=np.array([0.1, 1.0, 10.0])
)
lb = mock_loanbook()

banks_decide_credit_supply(lend, v=0.1)
banks_decide_interest_rate(lend, r_bar=0.02, h_phi=0.10, rng=rng)
firms_decide_credit_demand(bor)
firms_calc_credit_metrics(bor)
firms_prepare_loan_applications(bor, lend, max_H=H, rng=rng)
for _ in range(H):
    firms_send_one_loan_app(bor, lend, rng)
    banks_provide_loans(bor, lb, lend, r_bar=0.02, h_phi=0.10, rng=rng)
firms_fire_workers(emp, wrk, method='expensive', rng=rng)
