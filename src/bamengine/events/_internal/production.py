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

from bamengine import Rng, logging, make_rng
from bamengine.economy import Economy
from bamengine.relationships import LoanBook
from bamengine.roles import Consumer, Employer, Producer, Worker
from bamengine.utils import EPS, trimmed_weighted_mean

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

    if info_enabled:
        log.info("--- Worker Contract Update complete ---")


def firms_calc_breakeven_price(
    prod: Producer,
    emp: Employer,
    lb: LoanBook,
    *,
    cap_factor: float | None = None,
) -> None:
    """
    Calculate breakeven price from wage costs and interest payments.

    See Also
    --------
    bamengine.events.production.FirmsCalcBreakevenPrice : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Calculating Breakeven Price ---")
        log.info(
            f"  Inputs: Breakeven Cap Factor={cap_factor if cap_factor else 'None'}"
        )
        log.info(
            "  Calculation uses projected production (labor_productivity × current_labor) "
            "as the denominator."
        )

    # Breakeven calculation
    interest = lb.interest_per_borrower(prod.production.size)
    projected_production = prod.labor_productivity * emp.current_labor
    breakeven = (emp.wage_bill + interest) / np.maximum(projected_production, EPS)
    if info_enabled:
        log.info(
            f"  Total Wage Bill for calc: {emp.wage_bill.sum():,.2f}. "
            f"Total Interest for calc: {interest.sum():,.2f}"
        )
    if log.isEnabledFor(logging.DEBUG):
        valid_breakeven = breakeven[np.isfinite(breakeven)]
        log.debug(
            f"  Raw breakeven prices (before cap): "
            f"min={valid_breakeven.min() if valid_breakeven.size > 0 else 'N/A':.2f}, "
            f"max={valid_breakeven.max() if valid_breakeven.size > 0 else 'N/A':.2f}, "
            f"avg={valid_breakeven.mean() if valid_breakeven.size > 0 else 'N/A':.2f}"
        )

    # Cap breakeven
    if cap_factor and cap_factor > 1:
        # Cannot be more than current price x cap_factor. This prevents extreme jumps.
        breakeven_max_value = prod.price * cap_factor
    else:
        # If no cap_factor, the max value is effectively infinite
        if info_enabled:
            log.info(
                "  No cap_factor provided for breakeven price. "
                "Prices may jump uncontrollably."
            )
        breakeven_max_value = breakeven

    np.minimum(breakeven, breakeven_max_value, out=prod.breakeven_price)

    num_capped = np.sum(breakeven > breakeven_max_value)
    if num_capped > 0 and info_enabled:
        log.info(f"  Breakeven prices capped for {num_capped} firms.")
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Capped firm indices: "
                f"{np.where(breakeven > breakeven_max_value)[0].tolist()}"
            )

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Final (Capped) Breakeven Prices:\n"
            f"{np.array2string(prod.breakeven_price, precision=2)}"
        )
    if info_enabled:
        log.info("--- Breakeven Price Calculation complete ---")


def firms_adjust_price(
    prod: Producer,
    *,
    p_avg: float,
    h_eta: float,
    price_cut_allow_increase: bool = True,
    rng: Rng = make_rng(),
) -> None:
    """
    Adjust prices based on inventory and market position.

    See Also
    --------
    bamengine.events.production.FirmsAdjustPrice : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Adjusting Prices ---")
        log.info(
            f"  Inputs: Avg Market Price (p_avg)={p_avg:.3f}  |  "
            f"Max Price Shock (h_η)={h_eta:.3f}"
        )

    shape = prod.price.shape
    old_prices_for_log = prod.price.copy()

    # scratch buffer for shocks
    shock = prod.price_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64)  # Corrected dtype
        prod.price_shock = shock

    shock[:] = rng.uniform(0.0, h_eta, size=shape)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Generated price shocks:\n{np.array2string(shock, precision=4)}")

    # masks
    mask_up = (prod.inventory == 0.0) & (prod.price < p_avg)
    mask_dn = (prod.inventory > 0.0) & (prod.price >= p_avg)
    n_up, n_dn = np.sum(mask_up), np.sum(mask_dn)
    n_keep = shape[0] - n_up - n_dn
    if info_enabled:
        log.info(
            f"  Price adjustments: {n_up} firms ↑, {n_dn} firms ↓, {n_keep} firms ↔."
        )

    if log.isEnabledFor(logging.DEBUG):
        if n_up > 0:
            log.debug(f"    Firms increasing price: {np.where(mask_up)[0].tolist()}")
        if n_dn > 0:
            log.debug(f"    Firms decreasing price: {np.where(mask_dn)[0].tolist()}")

    # DEBUG pre-update snapshot
    if log.isEnabledFor(logging.DEBUG):
        log.debug("  --- PRICE ADJUSTMENT (EXECUTION) ---")
        log.debug(f"  P̄ (avg market price) : {p_avg:.4f}")
        log.debug(f"  mask_up: {n_up} firms → raise  |  mask_dn: {n_dn} firms → cut")
        log.debug(
            f"  Breakeven prices being used:\n"
            f"{np.array2string(prod.breakeven_price, precision=2)}"
        )

    # raise prices
    if n_up > 0:
        np.multiply(prod.price, 1.0 + shock, out=prod.price, where=mask_up)
        np.maximum(prod.price, prod.breakeven_price, out=prod.price, where=mask_up)

        if info_enabled:
            price_changes = prod.price[mask_up] - old_prices_for_log[mask_up]
            num_floored = np.sum(
                np.isclose(prod.price[mask_up], prod.breakeven_price[mask_up])
            )
            log.info(
                f"  Raised prices for {n_up} firms. "
                f"Avg change: {np.mean(price_changes):+.3f}. "
                f"{num_floored} prices set by breakeven floor."
            )
        if log.isEnabledFor(logging.DEBUG):
            for firm_idx in np.where(mask_up)[0][:5]:
                log.debug(
                    f"    Raise Firm {firm_idx}: "
                    f"OldP={old_prices_for_log[firm_idx]:.2f} -> "
                    f"NewP={prod.price[firm_idx]:.2f} "
                    f"(Breakeven={prod.breakeven_price[firm_idx]:.2f})"
                )

    # cut prices
    if n_dn > 0:
        np.multiply(prod.price, 1.0 - shock, out=prod.price, where=mask_dn)

        if price_cut_allow_increase:
            # Allow price to increase due to breakeven floor
            np.maximum(prod.price, prod.breakeven_price, out=prod.price, where=mask_dn)
        else:
            # Don't allow price increase when trying to cut - cap at old price
            # Apply breakeven floor but not above old price
            floor_price = np.minimum(old_prices_for_log, prod.breakeven_price)
            np.maximum(prod.price, floor_price, out=prod.price, where=mask_dn)

        if info_enabled:
            price_changes = prod.price[mask_dn] - old_prices_for_log[mask_dn]
            num_floored = np.sum(
                np.isclose(prod.price[mask_dn], prod.breakeven_price[mask_dn])
            )
            num_increased_due_to_floor = np.sum(
                prod.price[mask_dn] > old_prices_for_log[mask_dn]
            )
            log.info(
                f"  Cut prices for {n_dn} firms. "
                f"Avg change: {np.mean(price_changes):+.3f}. "
                f"{num_floored} prices set by breakeven floor."
            )
            if num_increased_due_to_floor > 0:
                log.info(
                    f"  !!! {num_increased_due_to_floor} firms in the 'cut price' "
                    f"group ended up INCREASING their price because their "
                    f"breakeven floor was higher than their old price."
                )
        if log.isEnabledFor(logging.DEBUG):
            for firm_idx in np.where(mask_dn)[0][:5]:
                log.debug(
                    f"    Cut Firm {firm_idx}: "
                    f"OldP={old_prices_for_log[firm_idx]:.2f} -> "
                    f"NewP={prod.price[firm_idx]:.2f} "
                    f"(Breakeven={prod.breakeven_price[firm_idx]:.2f})"
                )

    if info_enabled:
        log.info("--- Price Adjustment complete ---")
