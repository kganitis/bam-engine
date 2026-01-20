"""
System functions for bankruptcy phase events.

This module contains the internal implementation functions for bankruptcy events.
Event classes wrap these functions and provide the primary documentation.

See Also
--------
bamengine.events.bankruptcy : Event classes (primary documentation source)
"""

from __future__ import annotations

import numpy as np

from bamengine import Rng, logging, make_rng
from bamengine.economy import Economy
from bamengine.relationships import LoanBook
from bamengine.roles import Borrower, Employer, Lender, Producer, Worker
from bamengine.utils import EPS, trim_mean

log = logging.getLogger(__name__)


def firms_update_net_worth(bor: Borrower) -> None:
    """
    Update firm net worth with retained profits/losses.

    See Also
    --------
    bamengine.events.bankruptcy.FirmsUpdateNetWorth : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Updating Net Worth ---")

    # update net worth with retained profits
    if info_enabled:
        total_retained_profits = bor.retained_profit.sum()
        log.info(
            f"  Total retained profits being added to net worth: "
            f"{total_retained_profits:,.2f}"
        )

    np.add(bor.net_worth, bor.retained_profit, out=bor.net_worth)

    # sync cash and clamp at zero
    bor.total_funds[:] = bor.net_worth
    np.maximum(bor.total_funds, 0.0, out=bor.total_funds)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Net worths after update (first 10 firms): "
            f"{np.array2string(bor.net_worth, precision=2)}"
        )
        log.debug(
            f"  Total funds (cash) after sync (first 10 firms): "
            f"{np.array2string(bor.total_funds, precision=2)}"
        )

    if info_enabled:
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
    Detect insolvent firms and remove them from the economy.

    See Also
    --------
    bamengine.events.bankruptcy.MarkBankruptFirms : Full documentation
    """
    # A firm is marked as bankrupt if either:
    #     • net-worth (A) < 0
    #     • previous production (Y_prev) <= 0 (ghost firm rule)
    #
    # Note: We check production_prev (not production) because production is zeroed
    # at the start of each period's planning phase. production_prev holds the
    # previous period's actual production, which is the relevant signal for
    # detecting inactive "ghost" firms.
    #
    # For bankrupt firms, all workers are fired and loans are purged.
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Marking Bankrupt Firms ---")

    # detect bankruptcies
    bankrupt_mask = (bor.net_worth < EPS) | (prod.production_prev <= EPS)
    bankrupt_indices = np.where(bankrupt_mask)[0]

    ec.exiting_firms = bankrupt_indices.astype(np.int64)

    if bankrupt_indices.size == 0:
        if info_enabled:
            log.info("  No new firm bankruptcies this period.")
            log.info("--- Firm Bankruptcy Marking complete ---")
        return

    log.warning(
        f"  {bankrupt_indices.size} firm(s) have gone bankrupt: "
        f"{bankrupt_indices.tolist()}"
    )
    if log.isEnabledFor(logging.DEBUG):  # pragma: no cover
        nw_bankrupt = np.where(bor.net_worth < EPS)[0]
        prod_bankrupt = np.where(prod.production_prev <= 0)[0]
        log.debug(
            f"    Bankrupt due to Net Worth < 0: "
            f"{np.intersect1d(bankrupt_indices, nw_bankrupt).tolist()}"
        )
        log.debug(
            f"    Bankrupt due to production_prev <= 0: "
            f"{np.intersect1d(bankrupt_indices, prod_bankrupt).tolist()}"
        )

    # fire all employees of bankrupt firms
    workers_to_fire_mask = np.isin(wrk.employer, bankrupt_indices)
    num_fired = np.sum(workers_to_fire_mask)
    if num_fired > 0:
        if info_enabled:
            log.info(f"  Firing {num_fired} worker(s) from bankrupt firms.")
        wrk.employer_prev[workers_to_fire_mask] = -1
        wrk.employer[workers_to_fire_mask] = -1
        wrk.wage[workers_to_fire_mask] = 0.0
        wrk.periods_left[workers_to_fire_mask] = 0
        wrk.contract_expired[workers_to_fire_mask] = 0
        wrk.fired[workers_to_fire_mask] = 0

    # wipe firm-side labor books
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Wiping labor and wage bill data for {bankrupt_indices.size} bankrupt firms."
        )
    emp.current_labor[bankrupt_indices] = 0
    emp.wage_bill[bankrupt_indices] = 0.0

    # purge their loans from the ledger
    num_purged = lb.purge_borrowers(bankrupt_indices)
    if info_enabled:
        log.info(
            f"  Purged {num_purged} loans from the ledger belonging to bankrupt firms."
        )

        log.info("--- Firm Bankruptcy Marking complete ---")


def mark_bankrupt_banks(ec: Economy, lend: Lender, lb: LoanBook) -> None:
    """
    Detect insolvent banks and remove them from the economy.

    See Also
    --------
    bamengine.events.bankruptcy.MarkBankruptBanks : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Marking Bankrupt Banks ---")

    # detect bankruptcies
    bankrupt_indices = np.where(lend.equity_base < EPS)[0]
    ec.exiting_banks = bankrupt_indices.astype(np.int64)

    if bankrupt_indices.size == 0:
        if info_enabled:
            log.info("  No new bank bankruptcies this period.")
            log.info("--- Bank Bankruptcy Marking complete ---")
        return

    log.warning(
        f"  !!! {bankrupt_indices.size} BANK(S) HAVE GONE BANKRUPT: "
        f"{bankrupt_indices.tolist()} !!!"
    )

    # purge their loans from the ledger
    num_purged = lb.purge_lenders(bankrupt_indices)
    if info_enabled:
        log.info(
            f"  Purged {num_purged} loans from the ledger issued by bankrupt banks."
        )

        log.info("--- Bank Bankruptcy Marking complete ---")


def spawn_replacement_firms(
    ec: Economy,
    prod: Producer,
    emp: Employer,
    bor: Borrower,
    wrk: Worker,
    *,
    new_firm_size_factor: float = 0.9,
    new_firm_production_factor: float = 0.9,
    new_firm_wage_factor: float = 0.9,
    new_firm_price_markup: float = 1.0,
    rng: Rng = make_rng(),
) -> None:
    """
    Create new firms to replace bankrupt ones.

    See Also
    --------
    bamengine.events.bankruptcy.SpawnReplacementFirms : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Spawning Replacement Firms ---")
    exiting_indices = ec.exiting_firms
    num_exiting = exiting_indices.size
    if num_exiting == 0:
        if info_enabled:
            log.info("  No firms to replace. Skipping.")
            log.info("--- Firm Spawning complete ---")
        return

    if info_enabled:
        log.info(f"  Spawning {num_exiting} new firm(s) to replace bankrupt ones.")

    # handle full market collapse
    if num_exiting == bor.net_worth.size:  # pragma: no cover - catastrophic edge case
        log.warning("!!! ALL FIRMS ARE BANKRUPT !!! SIMULATION ENDING.")
        ec.destroyed = True
        return

    # calculate survivor metrics
    survivors = np.setdiff1d(
        np.arange(bor.net_worth.size), exiting_indices, assume_unique=True
    )
    mean_net = trim_mean(bor.net_worth[survivors])
    mean_prod = trim_mean(prod.production[survivors])
    mean_wage = trim_mean(wrk.wage[wrk.employed])

    if info_enabled:
        log.info(
            f"  New firms will be initialized based on survivor averages: "
            f"mean_net={mean_net:.2f}, mean_prod={mean_prod:.2f}, mean_wage={mean_wage:.2f}"
        )

    # initialize new firms
    for i in exiting_indices:
        # Reset Borrower component
        bor.net_worth[i] = mean_net * new_firm_size_factor
        bor.total_funds[i] = bor.net_worth[i]
        bor.gross_profit[i] = 0.0
        bor.net_profit[i] = 0.0
        bor.retained_profit[i] = 0.0
        bor.credit_demand[i] = 0.0
        bor.projected_fragility[i] = 0.0

        # Reset Producer component
        prod.production[i] = 0.0
        prod.production_prev[i] = mean_prod * new_firm_production_factor
        prod.inventory[i] = 0.0
        prod.expected_demand[i] = 0.0
        prod.desired_production[i] = 0.0
        prod.price[i] = ec.avg_mkt_price * new_firm_price_markup

        # Reset Employer component
        emp.current_labor[i] = 0
        emp.desired_labor[i] = 0
        emp.wage_offer[i] = max(mean_wage * new_firm_wage_factor, ec.min_wage)
        emp.n_vacancies[i] = 0
        emp.total_funds[i] = bor.total_funds[i]
        emp.wage_bill[i] = 0.0

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Initialized new firm at index {i} "
                f"with net worth {bor.net_worth[i]:.2f}."
            )

    # clear exit list
    ec.exiting_firms = np.empty(0, np.intp)
    if info_enabled:
        log.info("--- Firm Spawning complete ---")


def spawn_replacement_banks(
    ec: Economy,
    lend: Lender,
    *,
    rng: Rng = make_rng(),
) -> None:
    """
    Create new banks to replace bankrupt ones.

    See Also
    --------
    bamengine.events.bankruptcy.SpawnReplacementBanks : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Spawning Replacement Banks ---")
    exiting_indices = ec.exiting_banks
    num_exiting = exiting_indices.size
    if num_exiting == 0:
        if info_enabled:
            log.info("  No banks to replace. Skipping.")
            log.info("--- Bank Spawning complete ---")
        return

    if info_enabled:
        log.info(f"  Spawning {num_exiting} new bank(s) to replace bankrupt ones.")

    # handle full market collapse
    alive = np.setdiff1d(
        np.arange(lend.equity_base.size), exiting_indices, assume_unique=True
    )
    if not alive.size:
        log.warning("!!! ALL BANKS ARE BANKRUPT !!! SIMULATION ENDING.")
        ec.destroyed = True
        return

    # initialize new banks
    debug_enabled = log.isEnabledFor(logging.DEBUG)
    for k in exiting_indices:
        src = int(rng.choice(alive))
        if debug_enabled:
            log.debug(f"  Cloning healthy bank {src} to replace bankrupt bank {k}.")
        lend.equity_base[k] = lend.equity_base[src]

        # Reset state for the new bank
        lend.credit_supply[k] = 0.0
        lend.interest_rate[k] = 0.0
        lend.recv_loan_apps_head[k] = -1
        lend.recv_loan_apps[k, :] = -1

    # clear exit list
    ec.exiting_banks = np.empty(0, np.intp)
    if info_enabled:
        log.info("--- Bank Spawning complete ---")
