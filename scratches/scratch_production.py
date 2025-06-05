
from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator, default_rng

from bamengine.components import Consumer, Economy, Employer, LoanBook, Producer, Worker
from bamengine.helpers import trimmed_weighted_mean
from helpers.factories import mock_producer, mock_employer, mock_loanbook, mock_worker, \
    mock_economy, mock_consumer

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def _mini_state() -> tuple[Economy, Producer, Employer, Worker, LoanBook, Generator]:
    rng = default_rng(42)

    ec = mock_economy(
        avg_mkt_price=2.0,
        avg_mkt_price_history=np.array([1.0, 1.25, 1.50, 2.00])
    )

    prod = mock_producer(
        n=4,
        inventory=np.array([0.0, 0.0, 3.0, 3.0]),
        price=np.array([1.0, 1.0, 3.0, 3.0]),
        alloc_scratch=False,
    )
    emp = mock_employer(
        n=4,
        current_labor=np.full(4, 2, dtype=np.int64),
        wage_offer=np.full(4, 1.0),
        wage_bill=np.full(4, 2.0),
    )

    lb = mock_loanbook()

    def _const_interest(_self: "LoanBook", n: int = 128) -> NDArray[np.float64]:
        return np.array([0.1, 10.0, 0.1, 10.0])

    patch.object(type(lb), "interest_per_borrower", _const_interest)

    wrk = mock_worker(
        n=10,
        employed=np.array([1]*8 + [0]*2, dtype=np.bool_),
        employer=np.array([0]*2 + [1]*2 + [2]*2 + [3]*2 + [-1]*2, dtype=np.intp),
        periods_left=np.array([2, 1] * 4 + [0, 0]),
        wage=np.array([1.0] * 8 + [0.0] * 2),
    )
    return ec, prod, emp, wrk, lb, rng


ec, prod, emp, wrk, lb, rng = _mini_state()


def calc_unemployment_rate(
    ec: Economy,
    wrk: Worker,
) -> None:
    # Number of unemployed workers / total household population
    n_workers = wrk.employed.size
    unemployed_count = n_workers - wrk.employed.sum()
    rate = unemployed_count / n_workers
    ec.unemp_rate_history = np.append(ec.unemp_rate_history, rate)

    log.debug("  ----- Unemployment Rate Calculation -----")
    log.debug(f"  n_workers={n_workers}")
    log.debug(f"  unemployed_count={unemployed_count}")
    log.debug(f"  unemployment rate: {rate * 100}%")



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
    log.info("  ----- Firms Deciding Prices -----")
    log.info(
        f"  Inputs: Avg Market Price (p_avg)={p_avg:.4f}, Max Price Shock (h_eta)={h_eta:.3f}")

    shape = prod.price.shape
    old_prices_for_log = prod.price.copy()  # For logging changes accurately

    # ── scratch buffer for shocks ─────────────────────────────────────────
    shock = prod.price_shock
    if shock is None or shock.shape != shape:
        log.debug("  Initializing price shock buffer.")
        shock = np.empty(shape, dtype=np.float64)
        prod.price_shock = shock

    shock[:] = rng.uniform(0.0, h_eta, size=shape)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Generated price shocks: {shock})")

    # ── masks ─────────────────────────────────────────────────────────────
    mask_up = (prod.inventory == 0.0) & (prod.price < p_avg)
    mask_dn = (prod.inventory > 0.0) & (prod.price >= p_avg)
    num_up = np.sum(mask_up)
    num_dn = np.sum(mask_dn)
    num_no_change_rule = shape[0] - num_up - num_dn
    log.info(
        f"  Price adjustment candidates: Raise Price={num_up}, Cut Price={num_dn}, No Rule={num_no_change_rule}")

    if log.isEnabledFor(logging.DEBUG):
        if num_up > 0:
            log.debug(
                f"    Firms with S=0 & P<p_avg (mask_up): {np.where(mask_up)[0].tolist()}")
        if num_dn > 0:
            log.debug(
                f"    Firms with S>0 & P>=p_avg (mask_dn): {np.where(mask_dn)[0].tolist()}")

    # ── breakeven price: (wage bill + interest) / output ─────────────────
    interest = lb.interest_per_borrower(prod.price.size)
    interest = np.array([0.1, 10.0, 0.1, 10.0])
    projected_output = prod.labor_productivity * emp.current_labor  # Output based on current labor

    # Raw breakeven calculation
    raw_breakeven = (emp.wage_bill + interest) / np.maximum(projected_output, 1.0e-12)

    # Cap breakeven: cannot be more than 2x current price. This prevents extreme jumps.
    # prod.price here is the price from *before* this period's adjustment.
    breakeven_cap_factor = 2.0
    breakeven_capped_at_value = old_prices_for_log * breakeven_cap_factor
    breakeven_capped = np.minimum(raw_breakeven, breakeven_capped_at_value)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Interest: {interest}")
        log.debug(
            f"  Projected outputs: {projected_output}")
        valid_raw_breakeven = raw_breakeven[np.isfinite(raw_breakeven)]
        log.debug(f"  Initial prices: {prod.price}")
        log.debug(f"  Raw breakeven prices (finite values): {valid_raw_breakeven}")
        log.debug(
            f"  Breakeven prices capped for firms "
            f"{np.where(raw_breakeven > breakeven_capped_at_value)[0]}"
            f"(max increase limited to {breakeven_cap_factor:.1f}x old price).")
        log.debug(
            f"  Capped breakeven prices (finite values): {breakeven_capped}")

    # ── DEBUG pre-update snapshot ────────────────────────────────────────
    if log.isEnabledFor(logging.DEBUG):
        log.debug("----- PRICE UPDATE (EXECUTION) -----")  # Renamed for clarity
        log.debug(f"  p̄ (avg market price) : {p_avg}")
        log.debug(f"  mask_up: {num_up} firms → raise  |"
                  f"  mask_dn: {num_dn} firms → cut")

    # ── raise prices ─────────────────────────────────────────────────────
    if mask_up.any():
        # --- Original logic ---
        np.multiply(prod.price, 1.0 + shock, out=prod.price, where=mask_up)
        np.maximum(prod.price, breakeven_capped, out=prod.price, where=mask_up)
        # --- Logging for this block ---
        price_changes_raise = prod.price[mask_up] - old_prices_for_log[mask_up]
        avg_change_raise = np.mean(price_changes_raise) if num_up > 0 else 0
        # Check how many prices were actually set to the breakeven_capped value
        num_floored_by_breakeven_raise = np.sum(
            np.isclose(prod.price[mask_up],
                       breakeven_capped[mask_up]) &  # Price is now breakeven
            ((old_prices_for_log[mask_up] * (1.0 + shock[mask_up])) < breakeven_capped[
                mask_up])  # And it was lower before max()
        )
        log.info(
            f"  Raising prices for {num_up} firms. Avg price change: {avg_change_raise:+.3f}. {num_floored_by_breakeven_raise} prices set by (capped) breakeven.")
        if log.isEnabledFor(logging.DEBUG):
            up_indices = np.where(mask_up)[0]
            for i_loop_idx, firm_actual_idx in enumerate(
                    up_indices[:min(3, num_up)]):  # Log first 3 (or fewer)
                price_after_shock_mult = old_prices_for_log[firm_actual_idx] * (
                            1.0 + shock[firm_actual_idx])
                log.debug(
                    f"    Raise Firm {firm_actual_idx}: OldP={old_prices_for_log[firm_actual_idx]:.2f}, "
                    f"Shock={shock[firm_actual_idx]:.3f} (max_eta={h_eta:.3f}), "
                    f"P_after_shock_mult={price_after_shock_mult:.2f}, "
                    f"BreakevenCap={breakeven_capped[firm_actual_idx]:.2f}, FinalNewP={prod.price[firm_actual_idx]:.2f}")

    # ── cut prices ────────────────────────────────────────────────────────
    if mask_dn.any():
        # --- Original logic ---
        np.multiply(prod.price, 1.0 - shock, out=prod.price, where=mask_dn)
        np.maximum(prod.price, breakeven_capped, out=prod.price, where=mask_dn)
        # --- Logging for this block ---
        price_changes_cut = prod.price[mask_dn] - old_prices_for_log[
            mask_dn]  # Should be negative
        avg_change_cut = np.mean(price_changes_cut) if num_dn > 0 else 0
        num_floored_by_breakeven_cut = np.sum(
            np.isclose(prod.price[mask_dn], breakeven_capped[mask_dn]) &
            ((old_prices_for_log[mask_dn] * (1.0 - shock[mask_dn])) < breakeven_capped[
                mask_dn])
        )
        log.info(
            f"  Cutting prices for {num_dn} firms. Avg price change: {avg_change_cut:+.3f}. {num_floored_by_breakeven_cut} prices set by (capped) breakeven.")
        if log.isEnabledFor(logging.DEBUG):
            dn_indices = np.where(mask_dn)[0]
            for i_loop_idx, firm_actual_idx in enumerate(
                    dn_indices[:min(3, num_dn)]):  # Log first 3 (or fewer)
                price_after_shock_mult = old_prices_for_log[firm_actual_idx] * (
                            1.0 - shock[firm_actual_idx])
                log.debug(
                    f"    Cut Firm {firm_actual_idx}: OldP={old_prices_for_log[firm_actual_idx]:.2f}, "
                    f"Shock={shock[firm_actual_idx]:.3f} (max_eta={h_eta:.3f}), "
                    f"P_after_shock_mult={price_after_shock_mult:.2f}, "
                    f"BreakevenCap={breakeven_capped[firm_actual_idx]:.2f}, FinalNewP={prod.price[firm_actual_idx]:.2f}")

    # User's existing final log block is good.
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"[PRICE SANITY] max price={prod.price.max() if prod.price.size > 0 else 'N/A':.3g}, "
            f"min shock={shock.min() if shock.size > 0 else 'N/A':.3f} "  # Corrected format for shock
            f"max shock={shock.max() if shock.size > 0 else 'N/A':.3f} "
        )
        log.debug(
            f"  Detailed New Prices (all firms):\n" f"{np.array2string(prod.price, precision=2)}")
    log.info("--- Price Decision complete ---")


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
        f"  Production:\n{np.array2string(prod.production, precision=2)}"
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
            f"Changes: {(-1 * delta_labor[affected_firms_indices]).tolist()})"
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
    log.debug(f"    Firm labor counts updated: {emp.current_labor}")

    np.multiply(emp.current_labor, emp.wage_offer, out=emp.wage_bill)
    log.debug(f"    Firm wage bills recalculated based on new labor counts: "
              f"{emp.wage_bill}")

    log.info("--- Worker Contract Update complete ---")


calc_unemployment_rate(ec, wrk)
firms_pay_wages(emp)
workers_receive_wage(mock_consumer(10), wrk)
firms_decide_price(prod, emp, lb, p_avg=ec.avg_mkt_price, h_eta=0.10, rng=rng)
update_avg_mkt_price(ec, prod)
calc_annual_inflation_rate(ec)
firms_run_production(prod, emp)
workers_update_contracts(wrk, emp)
