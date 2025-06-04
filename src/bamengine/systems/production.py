# src/bamengine/systems/production.py
"""
Event-4 – Production systems
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.components import Consumer, Economy, Employer, LoanBook, Producer, Worker
from bamengine.helpers import trimmed_weighted_mean

log = logging.getLogger("bamengine")


def calc_unemployment_rate(
    ec: Economy,
    wrk: Worker,
) -> None:
    # Number of unemployed agents / total household population
    n_workers = wrk.employed.size
    unemployed_count = wrk.employed.size - wrk.employed.sum()
    rate = unemployed_count / n_workers
    ec.unemp_rate_history = np.append(ec.unemp_rate_history, rate)

    log.debug("  ----- Unemployment Rate Calculation -----")
    log.debug(f"  n_workers={n_workers}")
    log.debug(f"  unemployed_count={unemployed_count}")
    log.debug(f"  unemployment rate: {rate}")


def firms_decide_price(
    prod: Producer,
    emp: Employer,
    lb: LoanBook,
    *,
    p_avg: float,
    h_eta: float,
    rng: Generator = default_rng(),
) -> None:
    """
    Nominal price-adjustment rule (vectorised):

        shock_i ~ U(0, h_eta)

        if S_i == 0 and p_i < p̄:     p_i ← max(breakeven_i , p_i·(1+shock))
        if S_i  > 0 and p_i ≥ p̄:     p_i ← max(breakeven_i , p_i·(1-shock))
    """
    shape = prod.price.shape

    # ── scratch buffer for shocks ─────────────────────────────────────────
    shock = prod.price_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64)
        prod.price_shock = shock

    shock[:] = rng.uniform(0.0, h_eta, size=shape)

    # ── masks ─────────────────────────────────────────────────────────────
    mask_up = (prod.inventory == 0.0) & (prod.price < p_avg)
    mask_dn = (prod.inventory > 0.0) & (prod.price >= p_avg)

    # ── breakeven price: (wage bill + interest) / output ─────────────────
    interest = lb.interest_per_borrower(prod.price.size)
    projected_output = prod.labor_productivity * emp.current_labor
    breakeven = (emp.wage_bill + interest) / np.maximum(projected_output, 1.0e-12)

    # ── DEBUG pre-update snapshot ────────────────────────────────────────
    if log.isEnabledFor(logging.DEBUG):
        log.debug("----- PRICE UPDATE -----")
        log.debug("p̄ (avg market price) : %.4f", p_avg)
        log.debug(
            "mask_up: %d firms → raise  |  mask_dn: %d firms → cut",
            mask_up.sum(),
            mask_dn.sum(),
        )

    # old_prices = prod.price.copy()

    # ── raise prices ─────────────────────────────────────────────────────
    if mask_up.any():
        np.multiply(prod.price, 1.0 + shock, out=prod.price, where=mask_up)
        np.maximum(prod.price, breakeven, out=prod.price, where=mask_up)

    # ── cut prices ────────────────────────────────────────────────────────
    if mask_dn.any():
        np.multiply(prod.price, 1.0 - shock, out=prod.price, where=mask_dn)
        np.maximum(prod.price, breakeven, out=prod.price, where=mask_dn)

    # bad_mask = prod.price > old_prices * 10.0
    # if np.any(bad_mask):
    #     bad = np.where(bad_mask)[0]
    #     log.debug(
    #         f"!!! runaway price at firm(s) {bad}: {prod.price[bad]}, "
    #         f"old: {old_prices[bad]}, "
    #         f"breakeven: {breakeven[bad]}, "
    #         f"production: {prod.production[bad]}"
    #     )
    #     capped_price = old_prices[bad_mask] * (1.0 + h_eta)
    #     prod.price[bad_mask] = capped_price

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"[PRICE SANITY] max price={prod.price.max():.3g}, "
            f"min shock={shock.min():.12g} "
            f"max shock={shock.max():.12g} "
        )
        log.debug(f"  Detailed Prices:\n" f"{np.array2string(prod.price, precision=2)}")


def update_avg_mkt_price(
    ec: Economy,
    prod: Producer,
    alpha: float = 1.0,
    trim_pct: float = 0.0,
) -> None:
    """
    Update the exponentially smoothed average market price in the Economy.

    Parameters
    ----------
    ec : Economy
        The economy state, which will be updated in-place.
    prod : Producer
        Producer state, containing price and production arrays.
    alpha : float, optional
        Exponential smoothing factor, in [0, 1]. alpha=1 means no smoothing
        (just use the current computed average). alpha < 1 blends the new
        average with the previous average.
    trim_pct : float, optional
        Fraction (0 to 0.5) to trim from both ends of the price distribution
        (by price value) before averaging, for robustness to outliers.

    Returns
    -------
    None. Updates ec.avg_mkt_price and ec.avg_mkt_price_history in-place.
    """
    # Sanitize alpha
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    p_avg_trimmed = trimmed_weighted_mean(prod.price, trim_pct=trim_pct)
    ec.avg_mkt_price = alpha * p_avg_trimmed + (1.0 - alpha) * ec.avg_mkt_price
    ec.avg_mkt_price_history = np.append(ec.avg_mkt_price_history, ec.avg_mkt_price)
    log.debug(f"  Average market price: {ec.avg_mkt_price:.4f}")


def calc_annual_inflation_rate(ec: Economy) -> None:
    """
    Calculate and store the annual inflation rate for the current period.

    π_t = (P_{t} - P_{t-4}) / P_{t-4}
    Stores result in ec.inflation_history (appended each call).
    """
    hist = ec.avg_mkt_price_history
    if hist.size <= 4:
        # not enough periods
        ec.inflation_history = np.append(ec.inflation_history, 0.0)
        return

    p_now = hist[-1]
    p_prev = hist[-5]
    inflation = (p_now - p_prev) / p_prev

    # ── debug -------------------------------------------------------------
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"Inflation calc – t={hist.size - 1}: "
            f"p_now={p_now:.4f}, p_prev={p_prev:.4f}, π={inflation:+.3%}"
        )

    ec.inflation_history = np.append(ec.inflation_history, inflation)


def firms_pay_wages(emp: Employer) -> None:
    """
    Debit each firm’s cash account by its wage-bill (vectorised).
    """
    np.subtract(emp.total_funds, emp.wage_bill, out=emp.total_funds)


def workers_receive_wage(con: Consumer, wrk: Worker) -> None:
    """
    Credit household income by the wage that *employed* workers earned:

        income_h += wage_h · employed_h
    """
    np.add(con.income, wrk.wage * wrk.employed, out=con.income)


def firms_run_production(prod: Producer, emp: Employer) -> None:
    """
    Compute current-period output and replace inventories:

        Y_i  =  a_i · L_i
        S_i  ←  Y_i
    """
    np.multiply(prod.labor_productivity, emp.current_labor, out=prod.production)
    prod.inventory[:] = prod.production  # overwrite, do **not** add

    log.debug(
        f"  Detailed Production:\n" f"{np.array2string(prod.production, precision=2)}"
    )


def workers_update_contracts(wrk: Worker, emp: Employer) -> None:
    """
    Decrease `periods_left` for every employed worker and let contracts that
    reach 0 expire.  All worker-side flags are updated **and** the corresponding
    firm’s labour and wage-bill are brought back in sync.

        • L_i := Σ 1{worker employed & employer == i}
        • W_i := L_i · w_i
    """
    log.info("--- Updating Worker Contracts ---")

    # ---- step 0: guard against impossible ‘already-0’ contracts ----------
    already_expired_mask = (wrk.employed == 1) & (wrk.periods_left == 0)
    if np.any(already_expired_mask):
        num_already_expired = np.sum(already_expired_mask)
        affected_worker_ids = np.where(already_expired_mask)[0]
        log.warning(
            f"  Found {num_already_expired} employed worker(s) "
            f"with periods_left already at 0. "
            f"Temporarily setting periods_left to 1 for these workers "
            f"to allow normal expiration processing this period."
        )
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Worker IDs with already-0 contracts (will be set to 1): "
                f"{affected_worker_ids.tolist()}"
            )
        wrk.periods_left[already_expired_mask] = 1

    # ---- step 1: tick down only for currently employed -------------------
    mask_emp = wrk.employed == 1
    if not np.any(mask_emp):
        log.info("  No employed workers found. Skipping contract period updates.")
        return

    num_employed_ticking = np.sum(mask_emp)
    log.info(
        f"  Decrementing 'periods_left' for {num_employed_ticking} employed worker(s)."
    )
    if log.isEnabledFor(logging.DEBUG):
        # Log a sample before decrementing for comparison, if needed (can be verbose)
        # For example, log periods_left for the first few employed workers
        # sample_indices = np.where(mask_emp)[0][:5]
        # log.debug(f"    Sample 'periods_left' before decrement for workers
        # {sample_indices.tolist()}: {wrk.periods_left[sample_indices].tolist()}")
        pass  # Keeping it less verbose for now

    wrk.periods_left[mask_emp] -= 1
    if log.isEnabledFor(logging.DEBUG):
        # log.debug(f"    Sample 'periods_left' after decrement for workers
        # {sample_indices.tolist()}: {wrk.periods_left[sample_indices].tolist()}")
        pass

    # ---- step 2: detect expirations --------------------------------------
    # `mask_emp` is from before decrement, `wrk.periods_left` is after.
    # So, we need to re-evaluate based on current `periods_left`.
    # The original `expired` mask correctly uses `mask_emp`
    # which refers to `wrk.employed == 1`.
    expired_mask = mask_emp & (wrk.periods_left == 0)

    if not np.any(expired_mask):
        log.info("  No worker contracts expired this period after decrementing.")
        return

    num_newly_expired = np.sum(expired_mask)
    newly_expired_worker_ids = np.where(expired_mask)[0]
    log.info(f"  {num_newly_expired} worker contract(s) have newly expired.")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"    Newly expired worker IDs: {newly_expired_worker_ids.tolist()}")

    # Snapshot of firms losing workers, before employer IDs are overwritten
    firms_losing_workers = wrk.employer[
        expired_mask
    ].copy()  # Make a copy before modification
    if log.isEnabledFor(logging.DEBUG):
        unique_firms_affected = np.unique(firms_losing_workers)
        log.debug(
            f"    Firms losing workers due to these expirations: "
            f"{unique_firms_affected.tolist()}"
        )

    # -------- worker-side updates -----------------------------------------
    log.debug("    Updating worker-side attributes for expired contracts...")
    wrk.employed[expired_mask] = 0
    wrk.employer[expired_mask] = -1  # Mark as unemployed by this firm
    wrk.employer_prev[expired_mask] = (
        firms_losing_workers  # Store the firm they just left
    )
    wrk.wage[expired_mask] = 0.0
    wrk.contract_expired[expired_mask] = 1  # Set flag indicating contract expired
    wrk.fired[expired_mask] = 0  # They were not fired, contract ended

    # -------- firm-side labour bookkeeping --------------------------------
    log.info("    Updating firm-side labor counts and wage bills due to expirations.")
    # `firms_losing_workers` contains the employer ID for each expired contract
    delta_labor = np.bincount(firms_losing_workers, minlength=emp.current_labor.size)

    affected_firms_indices = np.where(delta_labor > 0)[0]
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"    Labor count changes (delta_labor) for firms: "
            f"(Indices: {affected_firms_indices.tolist()}, "
            f"Changes: {-delta_labor[affected_firms_indices].tolist()})"
        )

    if emp.current_labor.size < delta_labor.size:
        # This case should ideally not happen
        log.warning(
            f"  delta_labor (size {delta_labor.size}) "
            f"is larger than emp.current_labor (size {emp.current_labor.size}). "
            f"Resizing might be needed or check firm ID range."
        )

    # Ensure we only subtract where delta_labor
    # has relevant indices for emp.current_labor
    max_idx_to_update = min(delta_labor.size, emp.current_labor.size)
    emp.current_labor[:max_idx_to_update] -= delta_labor[:max_idx_to_update]

    assert (emp.current_labor >= 0).all(), "negative labour after expirations"
    log.debug("    Firm labor counts updated.")

    # keep wage-bill consistent with the new labour vector
    if log.isEnabledFor(logging.DEBUG):
        old_total_wage_bill = emp.wage_bill.sum()  # For summary
        # Store wage bills of affected firms for detailed logging
        # wage_bills_before_update = emp.wage_bill[affected_firms_indices].copy()

    np.multiply(emp.current_labor, emp.wage_offer, out=emp.wage_bill)
    log.debug("    Firm wage bills recalculated based on new labor counts.")

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"    Total wage bill changed from {old_total_wage_bill:,.2f} "
            f"to {emp.wage_bill.sum():,.2f}."
        )
        # for idx, firm_idx_affected in enumerate(affected_firms_indices):
        #     log.debug(f"      Firm {firm_idx_affected}: "
        #               f"wage bill from {wage_bills_before_update[idx]:.2f} "
        #               f"to {emp.wage_bill[firm_idx_affected]:.2f}")

    log.info("--- Worker Contract Update complete ---")
