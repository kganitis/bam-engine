# src/bamengine/systems/bankruptcy.py
"""
Event-7  ─  Bankruptcy (mark exits, fire workers, purge loans)
Event-8  ─  Entry (spawn replacements for those exits)
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.components import (
    Borrower,
    Economy,
    Employer,
    Lender,
    LoanBook,
    Producer,
    Worker,
)
from bamengine.helpers import sample_beta_with_mean, trim_mean
from helpers.factories import mock_economy, mock_borrower, mock_lender, mock_loanbook, \
    mock_producer, mock_employer, mock_worker

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

_EPS = 1.0e-9

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
    lender_idx=1,
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


# ───────────────────────── Event-7  ─  Bankruptcy ───────────
def firms_update_net_worth(bor: Borrower) -> None:
    """Retained profit is added to net worth; syncs the cash column too."""

    log.info(
        f"  Total retained profits being added to net worth: "
        f"{bor.retained_profit.sum():,.2f}"
    )

    np.add(bor.net_worth, bor.retained_profit, out=bor.net_worth)
    bor.total_funds[:] = bor.net_worth
    np.maximum(bor.total_funds, 0)

    log.debug(
        f"  Net Worths after update:\n" f"{np.array2string(bor.net_worth, precision=2)}"
    )


def mark_bankrupt_firms(
    ec: Economy,
    prod: Producer,
    emp: Employer,
    bor: Borrower,
    wrk: Worker,
    lb: LoanBook,
) -> None:
    """
    Detect and process firm exits.

    A firm exits when **either**
    • net-worth A_i < 0   **or**
    • current production Y_i == 0

    For every exiting firm:
      • all workers are released
      • employer books are wiped
      • its loan rows are removed from the ledger
      • index is recorded in ``ec.exiting_firms``
    """
    bankrupt = np.where((bor.net_worth < -_EPS) | (prod.production <= 0))[0]

    ec.exiting_firms = bankrupt.astype(np.int64)

    if bankrupt.size == 0:
        log.info("  No new firm bankruptcies this period.")
        return

    log.info(f"--> {bankrupt.size} FIRM(S) HAVE GONE BANKRUPT: {bankrupt.tolist()}")

    # ── fire all employees of those firms ───────────────────────────────
    mask_bad_emp = np.isin(wrk.employer, bankrupt)
    if mask_bad_emp.any():
        wrk.employed[mask_bad_emp] = 0
        wrk.employer_prev[mask_bad_emp] = -1
        wrk.employer[mask_bad_emp] = -1
        wrk.wage[mask_bad_emp] = 0.0
        wrk.periods_left[mask_bad_emp] = 0
        wrk.contract_expired[mask_bad_emp] = 0
        wrk.fired[mask_bad_emp] = 0

    # ── wipe firm-side labour books ─────────────────────────────────────
    emp.current_labor[bankrupt] = 0
    emp.wage_bill[bankrupt] = 0.0

    # ── purge their loans from the ledger ───────────────────────────────
    lb.purge_borrowers(bankrupt)


def mark_bankrupt_banks(
    ec: Economy,
    lend: Lender,
    lb: LoanBook,
) -> None:
    """
    • Detect E_k < 0.
    • Record indices in `ec.exiting_banks`.
    • Purge every loan row that references the bankrupt banks.
    """
    bankrupt = np.where(lend.equity_base < -_EPS)[0]
    ec.exiting_banks = bankrupt.astype(np.int64)

    if bankrupt.size == 0:
        log.info("  No new bank bankruptcies this period.")
        return

    log.info(f"!!! {bankrupt.size} BANK(S) HAVE GONE BANKRUPT: {bankrupt} !!!")

    # Purge every loan row that references the bankrupt banks
    lb.purge_lenders(bankrupt)


# ───────────────────────── Event-8  ─  Entry  ─────────────────
def spawn_replacement_firms(
    ec: Economy,
    prod: Producer,
    emp: Employer,
    bor: Borrower,
    *,
    rng: Generator,
) -> None:
    """Create one brand-new firm *per* index stored in `ec.exiting_firms`."""
    exiting = ec.exiting_firms
    if exiting.size == 0:
        return

    if exiting.size == bor.net_worth.size:
        log.critical("!!! ALL FIRMS BANKRUPT !!!")
        # terminate simulation
        ec.destroyed = True
        return
    else:
        survivors = np.setdiff1d(
            np.arange(bor.net_worth.size), exiting, assume_unique=True
        )
        # trimmed means of survivors
        mean_net = trim_mean(bor.net_worth[survivors])
        mean_prod = trim_mean(prod.production[survivors])
        mean_wage = trim_mean(emp.wage_offer[survivors])
        log.info(
            f"  New firms initialized based on survivor averages: "
            f"mean_net={mean_net:.2f}, mean_prod={mean_prod:.2f}, "
            f"mean_wage={mean_wage:.2f}"
        )

    for i in exiting:
        # size smaller than trimmed mean
        s = sample_beta_with_mean(0.5, low=0.1, high=1.0, concentration=6, rng=rng)
        # s = 0.9
        bor.net_worth[i] = mean_net * s

        bor.total_funds[i] = bor.net_worth[i]
        bor.gross_profit[i] = bor.net_profit[i] = 0.0
        bor.retained_profit[i] = 0.0
        bor.credit_demand[i] = 0.0
        bor.projected_fragility[i] = 0.0

        prod.production[i] = mean_prod * s
        prod.inventory[i] = 0.0
        prod.expected_demand[i] = 0.0
        prod.desired_production[i] = 0.0
        prod.labor_productivity[i] = 1.0
        prod.price[i] = ec.avg_mkt_price

        emp.current_labor[i] = 0
        emp.desired_labor[i] = 0
        emp.wage_offer[i] = mean_wage
        emp.n_vacancies[i] = 0
        emp.total_funds[i] = bor.total_funds[i]
        emp.wage_bill[i] = 0.0

    ec.exiting_firms = np.empty(0, np.intp)  # clear list


def spawn_replacement_banks(
    ec: Economy,
    lend: Lender,
    *,
    rng: Generator,
) -> None:
    """Clone parameters from a random healthy peer for each exiting bank."""
    exiting = ec.exiting_banks
    if exiting.size == 0:
        return

    alive = np.setdiff1d(np.arange(lend.equity_base.size), exiting, assume_unique=True)

    for k in exiting:
        if alive.size:  # clone from peer
            src = int(rng.choice(alive))
            log.debug(f"  Cloning healthy bank {src} to replace bankrupt bank {k}.")
            lend.equity_base[k] = lend.equity_base[src]
        else:  # terminate simulation
            log.critical("!!! ALL BANKS BANKRUPT !!!")
            ec.destroyed = True
            return

        # Reset state for the new bank
        lend.credit_supply[k] = 0.0
        lend.interest_rate[k] = 0.0
        lend.recv_loan_apps_head[k] = -1
        lend.recv_loan_apps[k, :] = -1

    ec.exiting_banks = np.empty(0, np.intp)  # clear list


firms_update_net_worth(bor)
mark_bankrupt_firms(ec, prod, emp, bor, wrk, lb)
mark_bankrupt_banks(ec, lend, lb)
spawn_replacement_firms(ec, prod, emp, bor, rng=rng)
spawn_replacement_banks(ec, lend, rng=rng)
