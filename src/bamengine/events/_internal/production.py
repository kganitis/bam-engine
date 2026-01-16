"""
System functions for production phase events.

This module contains the internal implementation functions for production events.
Event classes wrap these functions and provide the primary documentation.

See Also
--------
bamengine.events.production : Event classes (primary documentation source)
"""

from __future__ import annotations

import numpy as np

from bamengine import logging
from bamengine.economy import Economy
from bamengine.roles import Consumer, Employer, Producer, Worker
from bamengine.utils import trimmed_weighted_mean

log = logging.getLogger(__name__)


def calc_unemployment_rate(
    ec: Economy,
    wrk: Worker,
) -> None:
    """
    Calculate unemployment rate from worker employment status and store in history.

    Parameters
    ----------
    ec : Economy
        Economy object (stores unemployment rate history).
    wrk : Worker
        Worker role (contains employment status for all workers).

    See Also
    --------
    bamengine.events.economy_stats.CalcUnemploymentRate : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Calculating Unemployment Rate ---")

    n_workers = wrk.employed.size
    unemployed_count = n_workers - wrk.employed.sum()
    rate = unemployed_count / n_workers

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Unemployment calculation: {unemployed_count} unemployed "
            f"out of {n_workers} total workers"
        )

    if info_enabled:
        log.info(f"  Unemployment rate: {rate * 100:.2f}%")

    # Store raw rate in history
    ec.unemp_rate_history = np.append(ec.unemp_rate_history, rate)

    if info_enabled:
        log.info("--- Unemployment Rate Calculation complete ---")


def update_avg_mkt_price(
    ec: Economy,
    prod: Producer,
    trim_pct: float = 0.0,
) -> None:
    """
    Update exponentially smoothed average market price.

    See Also
    --------
    bamengine.events.economy_stats.UpdateAvgMktPrice : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Updating Average Market Price ---")

    # calculate average market price by weighting firm prices by production output
    p_avg_trimmed = trimmed_weighted_mean(
        prod.price, trim_pct=trim_pct, weights=prod.production
    )
    previous_price = ec.avg_mkt_price

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Price calculation: trimmed_mean={p_avg_trimmed:.4f}, "
            f"previous_avg={previous_price:.4f}"
        )

    # If calculated price is 0 (all production is 0), preserve previous price
    if p_avg_trimmed <= 0 and previous_price > 0:
        log.warning(
            f"  Calculated avg price is {p_avg_trimmed:.4f} (no production). "
            f"Preserving previous price {previous_price:.4f}."
        )
        p_avg_trimmed = previous_price

    # update economy state
    ec.avg_mkt_price = p_avg_trimmed
    ec.avg_mkt_price_history = np.append(ec.avg_mkt_price_history, ec.avg_mkt_price)

    if info_enabled:
        log.info(f"  Average market price updated: {ec.avg_mkt_price:.4f}")
        log.info("--- Average Market Price Update complete ---")


def firms_pay_wages(emp: Employer) -> None:
    """
    Deduct wage bill from firm funds.

    See Also
    --------
    bamengine.events.production.FirmsPayWages : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Paying Wages ---")

    paying_firms = np.where(emp.wage_bill > 0.0)[0]
    total_wages_paid = (
        emp.wage_bill[paying_firms].sum() if paying_firms.size > 0 else 0.0
    )

    if info_enabled:
        log.info(
            f"  {paying_firms.size} firms paying total wages of {total_wages_paid:,.2f}"
        )

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Pre-payment firm funds: {emp.total_funds}")
        log.debug(f"  Wage bills: {emp.wage_bill}")

    # debit firm accounts
    np.subtract(emp.total_funds, emp.wage_bill, out=emp.total_funds)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Post-payment firm funds: {emp.total_funds}")

    if info_enabled:
        log.info("--- Firms Paying Wages complete ---")


def workers_receive_wage(con: Consumer, wrk: Worker) -> None:
    """
    Add wages to worker income.

    See Also
    --------
    bamengine.events.production.WorkersReceiveWage : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Workers Receiving Wages ---")

    employed_workers = np.where(wrk.employed == 1)[0]
    total_wages_received = (wrk.wage * wrk.employed).sum()

    if info_enabled:
        log.info(
            f"  {employed_workers.size} employed workers receiving "
            f"total wages of {total_wages_received:,.2f}"
        )

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Pre-wage consumer income: {con.income}")
        log.debug(f"  Worker wages (employed only): {wrk.wage[employed_workers]}")

    # credit household income
    np.add(con.income, wrk.wage * wrk.employed, out=con.income)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Post-wage consumer income: {con.income}")

    if info_enabled:
        log.info("--- Workers Receiving Wages complete ---")


def firms_run_production(prod: Producer, emp: Employer) -> None:
    """
    Generate production output from labor and productivity.

    See Also
    --------
    bamengine.events.production.FirmsRunProduction : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Running Production ---")

    producing_firms = np.where(emp.current_labor > 0)[0]

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  {producing_firms.size} firms with labor are producing")
        log.debug(f"  Labor productivity: {prod.labor_productivity}")
        log.debug(f"  Current labor: {emp.current_labor}")

    # calculate production output
    np.multiply(prod.labor_productivity, emp.current_labor, out=prod.production)

    # Update production_prev unconditionally.
    # Firms with production=0 will be detected as "ghost firms" by the bankruptcy
    # check next period (production_prev <= 0) and replaced with new entrants.
    prod.production_prev[:] = prod.production

    total_production = prod.production.sum()

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Production output: {prod.production}")
        log.debug(f"  Production_prev updated: {prod.production_prev}")

    if info_enabled:
        log.info(f"  Total production output: {total_production:,.2f}")

    # update inventory
    prod.inventory[:] = prod.production  # overwrite, do **not** add

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Inventory updated (replaced): {prod.inventory}")

    if info_enabled:
        log.info("--- Firms Running Production complete ---")


def workers_update_contracts(wrk: Worker, emp: Employer) -> None:
    """
    Decrement contract duration and handle contract expiration.

    See Also
    --------
    bamengine.events.production.WorkersUpdateContracts : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Updating Worker Contracts ---")

    employed_workers = np.where(wrk.employed == 1)[0]
    total_employed = employed_workers.size

    if info_enabled:
        log.info(f"  Processing contracts for {total_employed} employed workers")

    # validate contract consistency
    already_expired_mask = (wrk.employed == 1) & (wrk.periods_left == 0)
    if np.any(already_expired_mask):
        num_already_expired = np.sum(already_expired_mask)
        affected_worker_ids = np.where(already_expired_mask)[0]
        log.warning(
            f"  Found {num_already_expired} employed worker(s) "
            f"with periods_left already at 0. "
            f"Temporarily setting periods_left to 1 for normal processing."
        )

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Worker IDs with already-0 contracts: "
                f"{affected_worker_ids.tolist()}"
            )

        wrk.periods_left[already_expired_mask] = 1

    # decrement contract periods
    mask_emp = wrk.employed == 1
    if not np.any(mask_emp):
        if info_enabled:
            log.info("  No employed workers found. Skipping contract updates.")
            log.info("--- Worker Contract Update complete ---")
        return

    num_employed_ticking = np.sum(mask_emp)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Decrementing periods_left for {num_employed_ticking} workers")

    wrk.periods_left[mask_emp] -= 1

    # identify contract expirations
    expired_mask = mask_emp & (wrk.periods_left == 0)

    if not np.any(expired_mask):
        if info_enabled:
            log.info("  No worker contracts expired this step.")
            log.info("--- Worker Contract Update complete ---")
        return

    num_newly_expired = np.sum(expired_mask)
    newly_expired_worker_ids = np.where(expired_mask)[0]

    if info_enabled:
        log.info(f"  {num_newly_expired} worker contract(s) have expired")

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"    Expired worker IDs: {newly_expired_worker_ids.tolist()}")

    # gather firm data before updates
    firms_losing_workers = wrk.employer[expired_mask].copy()
    unique_firms_affected = np.unique(firms_losing_workers)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"    Firms losing workers: {unique_firms_affected.tolist()}")

    # worker‑side updates
    log.debug(
        f"      Updating state for {num_newly_expired} workers with expired contracts."
    )
    wrk.employer[expired_mask] = -1
    wrk.employer_prev[expired_mask] = firms_losing_workers
    wrk.wage[expired_mask] = 0.0
    wrk.contract_expired[expired_mask] = 1
    wrk.fired[expired_mask] = 0

    # firm‑side updates
    delta_labor = np.bincount(firms_losing_workers, minlength=emp.current_labor.size)
    affected_firms_indices = np.where(delta_labor > 0)[0]

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"      Labor count changes for firms: "
            f"indices={affected_firms_indices.tolist()}, "
            f"decreases={delta_labor[affected_firms_indices].tolist()}"
        )

    if emp.current_labor.size < delta_labor.size:
        log.warning(
            f"  delta_labor size ({delta_labor.size}) exceeds "
            f"emp.current_labor size ({emp.current_labor.size}). "
            f"Check firm ID range."
        )

    # Update firm labor counts
    max_idx_to_update = min(delta_labor.size, emp.current_labor.size)
    emp.current_labor[:max_idx_to_update] -= delta_labor[:max_idx_to_update]

    assert (emp.current_labor >= 0).all(), "negative labor after expirations"

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"      Firm labor counts updated: {emp.current_labor}")

    # Recalculate wage bills
    emp.wage_bill[:] = np.bincount(
        wrk.employer[wrk.employed == 1],
        weights=wrk.wage[wrk.employed == 1],
        minlength=emp.wage_bill.size,
    )

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"      Firm wage bills recalculated: {emp.wage_bill}")

    if info_enabled:
        log.info("--- Worker Contract Update complete ---")
