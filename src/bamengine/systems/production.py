# src/bamengine/systems/production.py
"""
Event-4 – Production systems
"""
from __future__ import annotations

import logging

import numpy as np

from bamengine import _logging_ext
from bamengine.components import Consumer, Economy, Employer, Producer, Worker
from bamengine.helpers import trimmed_weighted_mean

log = _logging_ext.getLogger(__name__)


def calc_unemployment_rate(
    ec: Economy,
    wrk: Worker,
) -> None:
    """Calculate unemployment rate and update economy history."""
    log.info("--- Calculating Unemployment Rate ---")

    n_workers = wrk.employed.size
    unemployed_count = n_workers - wrk.employed.sum()
    rate = unemployed_count / n_workers

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Unemployment calculation: {unemployed_count} unemployed "
            f"out of {n_workers} total workers")

    log.info(f"  Current unemployment rate: {rate * 100:.2f}%")

    # ---- update economy state -------------------------------------------
    ec.unemp_rate_history = np.append(ec.unemp_rate_history, rate)

    log.info("--- Unemployment Rate Calculation complete ---")


def update_avg_mkt_price(
    ec: Economy,
    prod: Producer,
    alpha: float = 1.0,
    trim_pct: float = 0.0,
) -> None:
    """
    Update exponentially smoothed average market price and update economy state.
    """
    log.info("--- Updating Average Market Price ---")

    # Sanitize alpha
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Price update parameters: alpha={alpha:.3f}, "
            f"trim_pct={trim_pct:.3f}")

    # ---- calculate trimmed weighted mean --------------------------------
    p_avg_trimmed = trimmed_weighted_mean(prod.price, trim_pct=trim_pct)
    previous_price = ec.avg_mkt_price

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Price calculation: trimmed_mean={p_avg_trimmed:.4f}, "
            f"previous_avg={previous_price:.4f}")

    # ---- update economy state -------------------------------------------
    ec.avg_mkt_price = alpha * p_avg_trimmed + (1.0 - alpha) * ec.avg_mkt_price
    ec.avg_mkt_price_history = np.append(ec.avg_mkt_price_history, ec.avg_mkt_price)

    log.info(f"  Average market price updated: {ec.avg_mkt_price:.4f}")
    log.info("--- Average Market Price Update complete ---")


def firms_pay_wages(emp: Employer) -> None:
    """Debit firm cash accounts by wage bills and update employer state."""
    log.info("--- Firms Paying Wages ---")

    paying_firms = np.where(emp.wage_bill > 0.0)[0]
    total_wages_paid = emp.wage_bill[
        paying_firms].sum() if paying_firms.size > 0 else 0.0

    log.info(
        f"  {paying_firms.size} firms paying total wages of {total_wages_paid:,.2f}")

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Pre-payment firm funds: {emp.total_funds}")
        log.debug(f"  Wage bills: {emp.wage_bill}")

    # ---- debit firm accounts --------------------------------------------
    np.subtract(emp.total_funds, emp.wage_bill, out=emp.total_funds)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Post-payment firm funds: {emp.total_funds}")

    log.info("--- Firms Paying Wages complete ---")


def workers_receive_wage(con: Consumer, wrk: Worker) -> None:
    """
    Credit household income with wages for employed workers and update consumer state.

    Rule
    ----
        income += wage · employed
    """
    log.info("--- Workers Receiving Wages ---")

    employed_workers = np.where(wrk.employed == 1)[0]
    total_wages_received = (wrk.wage * wrk.employed).sum()

    log.info(
        f"  {employed_workers.size} employed workers receiving "
        f"total wages of {total_wages_received:,.2f}")

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Pre-wage consumer income: {con.income}")
        log.debug(f"  Worker wages (employed only): {wrk.wage[employed_workers]}")

    # ---- credit household income ----------------------------------------
    np.add(con.income, wrk.wage * wrk.employed, out=con.income)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Post-wage consumer income: {con.income}")

    log.info("--- Workers Receiving Wages complete ---")


def firms_run_production(prod: Producer, emp: Employer) -> None:
    """
    Compute production output and update inventory state.

    Rule
    ----
        Y  =  a · L
        S  ←  Y

    Y: Actual Production Output, a: Labour Productivity, L: Actual Labour, S: Inventory
    """
    log.info("--- Firms Running Production ---")

    producing_firms = np.where(emp.current_labor > 0)[0]

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  {producing_firms.size} firms with labor are producing")
        log.debug(f"  Labor productivity: {prod.labor_productivity}")
        log.debug(f"  Current labor: {emp.current_labor}")

    # ---- calculate production output ------------------------------------
    np.multiply(prod.labor_productivity, emp.current_labor, out=prod.production)
    total_production = prod.production.sum()

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Production output: {prod.production}")

    log.info(f"  Total production output: {total_production:,.2f}")

    # ---- update inventory -----------------------------------------------
    prod.inventory[:] = prod.production  # overwrite, do **not** add

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Inventory updated (replaced): {prod.inventory}")

    log.info("--- Firms Running Production complete ---")


def workers_update_contracts(wrk: Worker, emp: Employer) -> None:
    """
    Decrease `periods_left` for every employed worker and let contracts that
    reach 0 expire. All worker-side flags are updated **and** the corresponding
    firm’s labour and wage-bill are brought back in sync.

    Rule
    ----
        L_i = Σ {worker employed & employer == i}
        W   = L · w

    L: Actual Labour, W: Wage Bill, w: Individual Wage
    """
    log.info("--- Updating Worker Contracts ---")

    employed_workers = np.where(wrk.employed == 1)[0]
    total_employed = employed_workers.size

    log.info(f"  Processing contracts for {total_employed} employed workers")

    # ---- validate contract consistency ----------------------------------
    already_expired_mask = (wrk.employed == 1) & (wrk.periods_left == 0)
    if np.any(already_expired_mask):
        num_already_expired = np.sum(already_expired_mask)
        affected_worker_ids = np.where(already_expired_mask)[0]
        log.warning(
            f"  Found {num_already_expired} employed worker(s) "
            f"with periods_left already at 0. "
            f"Temporarily setting periods_left to 1 for normal processing.")

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Worker IDs with already-0 contracts: "
                f"{affected_worker_ids.tolist()}")

        wrk.periods_left[already_expired_mask] = 1

    # ---- decrement contract periods -------------------------------------
    mask_emp = wrk.employed == 1
    if not np.any(mask_emp):
        log.info("  No employed workers found. Skipping contract updates.")
        log.info("--- Worker Contract Update complete ---")
        return

    num_employed_ticking = np.sum(mask_emp)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Decrementing periods_left for {num_employed_ticking} workers")

    wrk.periods_left[mask_emp] -= 1

    # ---- identify contract expirations ----------------------------------
    expired_mask = mask_emp & (wrk.periods_left == 0)

    if not np.any(expired_mask):
        log.info("  No worker contracts expired this step.")
        log.info("--- Worker Contract Update complete ---")
        return

    num_newly_expired = np.sum(expired_mask)
    newly_expired_worker_ids = np.where(expired_mask)[0]

    log.info(f"  {num_newly_expired} worker contract(s) have expired")

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"    Expired worker IDs: {newly_expired_worker_ids.tolist()}")

    # ---- gather firm data before updates -------------------------------
    firms_losing_workers = wrk.employer[expired_mask].copy()
    unique_firms_affected = np.unique(firms_losing_workers)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"    Firms losing workers: {unique_firms_affected.tolist()}")

    # ---- worker‑side updates --------------------------------------------
    log.debug(
        f"      Updating state for {num_newly_expired} workers with expired contracts.")
    wrk.employed[expired_mask] = 0
    wrk.employer[expired_mask] = -1
    wrk.employer_prev[expired_mask] = firms_losing_workers
    wrk.wage[expired_mask] = 0.0
    wrk.contract_expired[expired_mask] = 1
    wrk.fired[expired_mask] = 0

    # ---- firm‑side updates ----------------------------------------------
    delta_labor = np.bincount(firms_losing_workers, minlength=emp.current_labor.size)
    affected_firms_indices = np.where(delta_labor > 0)[0]

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"      Labor count changes for firms: "
            f"indices={affected_firms_indices.tolist()}, "
            f"decreases={delta_labor[affected_firms_indices].tolist()}")

    if emp.current_labor.size < delta_labor.size:
        log.warning(
            f"  delta_labor size ({delta_labor.size}) exceeds "
            f"emp.current_labor size ({emp.current_labor.size}). "
            f"Check firm ID range.")

    # Update firm labor counts
    max_idx_to_update = min(delta_labor.size, emp.current_labor.size)
    emp.current_labor[:max_idx_to_update] -= delta_labor[:max_idx_to_update]

    assert (emp.current_labor >= 0).all(), "negative labour after expirations"

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"      Firm labor counts updated: {emp.current_labor}")

    # Recalculate wage bills
    emp.wage_bill[:] = np.bincount(
        wrk.employer[wrk.employed == 1],
        weights=wrk.wage[wrk.employed == 1],
        minlength=emp.wage_bill.size
    )

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"      Firm wage bills recalculated: {emp.wage_bill}")

    log.info("--- Worker Contract Update complete ---")
