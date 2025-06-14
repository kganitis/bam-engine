# src/bamengine/systems/bankruptcy.py
"""
Event-7  ─  Bankruptcy (mark exits, fire workers, purge loans)
Event-8  ─  Entry (spawn replacements for those exits)
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator, default_rng

from bamengine import _logging_ext
from bamengine.components import (
    Borrower,
    Economy,
    Employer,
    Lender,
    LoanBook,
    Producer,
    Worker,
)
from bamengine.helpers import trim_mean

log = _logging_ext.getLogger(__name__)

_EPS = 1.0e-9


def firms_update_net_worth(bor: Borrower) -> None:
    """
    Update firm net worth with retained earnings from the current period.

    Rule
    ----
        A_t = A_{t-1} + retained_t
        funds_t = max(0, A_t)

    t: Current Period, A: Net Worth,
    """
    log.info("--- Firms Updating Net Worth ---")

    # update net worth with retained profits 
    total_retained_profits = bor.retained_profit.sum()
    log.info(
        f"  Total retained profits being added to net worth: "
        f"{total_retained_profits:,.2f}")

    np.add(bor.net_worth, bor.retained_profit, out=bor.net_worth)

    # sync cash and clamp at zero 
    bor.total_funds[:] = bor.net_worth
    np.maximum(bor.total_funds, 0.0, out=bor.total_funds)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Net worths after update (first 10 firms): "
            f"{np.array2string(bor.net_worth, precision=2)}")
        log.debug(
            f"  Total funds (cash) after sync (first 10 firms): "
            f"{np.array2string(bor.total_funds, precision=2)}")

    log.info("--- Firms Updating Net Worth complete ---")


def mark_bankrupt_firms(
    ec: Economy,
    emp: Employer,
    bor: Borrower,
    prod: Producer,
    wrk: Worker,
    lb: LoanBook,
) -> None:
    """
    Detect and process firm bankruptcies based on net worth or production.

    Rule
    ----
    A firm is marked as bankrupt if either:
        • net-worth (A) < 0
        • current production (Y) <= 0

    For bankrupt firms, all workers are fired and loans are purged.
    """
    log.info("--- Marking Bankrupt Firms ---")

    # detect bankruptcies 
    bankrupt_mask = (bor.net_worth < _EPS) | (prod.production <= _EPS)
    bankrupt_indices = np.where(bankrupt_mask)[0]

    ec.exiting_firms = bankrupt_indices.astype(np.int64)

    if bankrupt_indices.size == 0:
        log.info("  No new firm bankruptcies this period.")
        log.info("--- Firm Bankruptcy Marking complete ---")
        return

    log.warning(
        f"  {bankrupt_indices.size} firm(s) have gone bankrupt: {bankrupt_indices.tolist()}")
    if log.isEnabledFor(logging.DEBUG):
        nw_bankrupt = np.where(bor.net_worth < _EPS)[0]
        prod_bankrupt = np.where(prod.production <= 0)[0]
        log.debug(
            f"    Bankrupt due to Net Worth < 0: {np.intersect1d(bankrupt_indices, nw_bankrupt).tolist()}")
        log.debug(
            f"    Bankrupt due to Production <= 0: {np.intersect1d(bankrupt_indices, prod_bankrupt).tolist()}")

    # fire all employees of bankrupt firms
    workers_to_fire_mask = np.isin(wrk.employer, bankrupt_indices)
    num_fired = np.sum(workers_to_fire_mask)
    if num_fired > 0:
        log.info(f"  Firing {num_fired} worker(s) from bankrupt firms.")
        wrk.employed[workers_to_fire_mask] = 0
        wrk.employer_prev[workers_to_fire_mask] = -1
        wrk.employer[workers_to_fire_mask] = -1
        wrk.wage[workers_to_fire_mask] = 0.0
        wrk.periods_left[workers_to_fire_mask] = 0
        wrk.contract_expired[workers_to_fire_mask] = 0
        wrk.fired[workers_to_fire_mask] = 0

    # wipe firm-side labor books
    log.debug(
        f"  Wiping labor and wage bill data for {bankrupt_indices.size} bankrupt firms.")
    emp.current_labor[bankrupt_indices] = 0
    emp.wage_bill[bankrupt_indices] = 0.0

    # purge their loans from the ledger
    num_purged = lb.purge_borrowers(bankrupt_indices)
    log.info(
        f"  Purged {num_purged} loans from the ledger belonging to bankrupt firms.")

    log.info("--- Firm Bankruptcy Marking complete ---")


def mark_bankrupt_banks(ec: Economy, lend: Lender, lb: LoanBook) -> None:
    """
    Detect and process bank bankruptcies based on negative equity.

    Rule
    ----
        A bank is marked as bankrupt if equity (E) < 0.

    For bankrupt banks, all associated loans are purged from the loan book.
    """
    log.info("--- Marking Bankrupt Banks ---")

    # detect bankruptcies
    bankrupt_indices = np.where(lend.equity_base < _EPS)[0]
    ec.exiting_banks = bankrupt_indices.astype(np.int64)

    if bankrupt_indices.size == 0:
        log.info("  No new bank bankruptcies this period.")
        log.info("--- Bank Bankruptcy Marking complete ---")
        return

    log.warning(
        f"  !!! {bankrupt_indices.size} BANK(S) HAVE GONE BANKRUPT: "
        f"{bankrupt_indices.tolist()} !!!")

    # purge their loans from the ledger
    num_purged = lb.purge_lenders(bankrupt_indices)
    log.info(
        f"  Purged {num_purged} loans from the ledger issued by bankrupt banks.")

    log.info("--- Bank Bankruptcy Marking complete ---")


def spawn_replacement_firms(
    ec: Economy,
    prod: Producer,
    emp: Employer,
    bor: Borrower,
    *,
    rng: Generator = default_rng(),
) -> None:
    """
    Create one brand-new firm to replace each firm that went bankrupt.

    Rule
    ----
    New firms inherit attributes based on the trimmed mean of surviving firms,
    scaled by a factor `s`.
    """
    log.info("--- Spawning Replacement Firms ---")
    exiting_indices = ec.exiting_firms
    num_exiting = exiting_indices.size
    if num_exiting == 0:
        log.info("  No firms to replace. Skipping.")
        log.info("--- Firm Spawning complete ---")
        return

    log.info(f"  Spawning {num_exiting} new firm(s) to replace bankrupt ones.")

    # handle full market collapse
    if num_exiting == bor.net_worth.size:
        log.critical("!!! ALL FIRMS ARE BANKRUPT !!! SIMULATION ENDING.")
        ec.destroyed = True
        return

    # calculate survivor metrics
    survivors = np.setdiff1d(np.arange(bor.net_worth.size), exiting_indices,
                             assume_unique=True)
    mean_net = trim_mean(bor.net_worth[survivors])
    mean_prod = trim_mean(prod.production[survivors])
    mean_wage = trim_mean(emp.wage_offer[survivors])
    log.info(
        f"  New firms will be initialized based on survivor averages: "
        f"mean_net={mean_net:.2f}, mean_prod={mean_prod:.2f}, mean_wage={mean_wage:.2f}"
    )

    # initialize new firms
    s = 0.8  # New firms start smaller than the mean of survivors
    for i in exiting_indices:
        # Reset Borrower component
        bor.net_worth[i] = mean_net * s
        bor.total_funds[i] = bor.net_worth[i]
        bor.gross_profit[i] = 0.0
        bor.net_profit[i] = 0.0
        bor.retained_profit[i] = 0.0
        bor.credit_demand[i] = 0.0
        bor.projected_fragility[i] = 0.0

        # Reset Producer component
        prod.production[i] = mean_prod
        prod.inventory[i] = 0.0
        prod.expected_demand[i] = 0.0
        prod.desired_production[i] = 0.0
        prod.labor_productivity[i] = 1.0
        prod.price[i] = ec.avg_mkt_price * 1.26

        # Reset Employer component
        emp.current_labor[i] = 0
        emp.desired_labor[i] = 0
        emp.wage_offer[i] = mean_wage * s
        emp.n_vacancies[i] = 0
        emp.total_funds[i] = bor.total_funds[i]
        emp.wage_bill[i] = 0.0

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Initialized new firm at index {i} "
                f"with net worth {bor.net_worth[i]:.2f}.")

    # clear exit list
    ec.exiting_firms = np.empty(0, np.intp)
    log.info("--- Firm Spawning complete ---")


def spawn_replacement_banks(
    ec: Economy,
    lend: Lender,
    *,
    rng: Generator = default_rng(),
) -> None:
    """
    Create one brand-new bank to replace each bank that went bankrupt.

    Rule
    ----
    New banks clone the equity of a random surviving peer. If no peers exist,
    the simulation is terminated.
    """
    log.info("--- Spawning Replacement Banks ---")
    exiting_indices = ec.exiting_banks
    num_exiting = exiting_indices.size
    if num_exiting == 0:
        log.info("  No banks to replace. Skipping.")
        log.info("--- Bank Spawning complete ---")
        return

    log.info(f"  Spawning {num_exiting} new bank(s) to replace bankrupt ones.")

    # handle full market collapse
    alive = np.setdiff1d(np.arange(lend.equity_base.size), exiting_indices,
                         assume_unique=True)
    if not alive.size:
        log.critical("!!! ALL BANKS ARE BANKRUPT !!! SIMULATION ENDING.")
        ec.destroyed = True
        return

    # initialize new banks
    for k in exiting_indices:
        src = int(rng.choice(alive))
        log.debug(f"  Cloning healthy bank {src} to replace bankrupt bank {k}.")
        lend.equity_base[k] = lend.equity_base[src]

        # Reset state for the new bank
        lend.credit_supply[k] = 0.0
        lend.interest_rate[k] = 0.0
        lend.recv_loan_apps_head[k] = -1
        lend.recv_loan_apps[k, :] = -1

    # clear exit list
    ec.exiting_banks = np.empty(0, np.intp)
    log.info("--- Bank Spawning complete ---")
