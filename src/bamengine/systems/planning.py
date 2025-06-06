# src/bamengine/systems/planning.py
from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.components import Employer, Producer, LoanBook

log = logging.getLogger(__name__)

_EPS = 1.0e-9


def firms_decide_desired_production(
        prod: Producer,
        *,
        p_avg: float,
        h_rho: float,
        rng: Generator = default_rng(),
) -> None:
    """
    Update `prod.expected_demand` and `prod.desired_production` **in‑place**.

    Rule
    ----
      if S_i == 0 and P_i ≥ p̄   → raise   by (1 + shock)
      if S_i  > 0 and P_i < p̄   → cut     by (1 − shock)
      otherwise                 → keep previous level
    """
    log.info("  --- Firms Deciding Desired Production ---")
    log.info(
        f"  Inputs: Avg Market Price (p_avg)={p_avg:.2f}, "
        f"Max Production Shock (h_ρ)={h_rho:.2f}")
    shape = prod.price.shape

    # --- permanent scratches ---------------
    shock = prod.prod_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64)
        prod.prod_shock = shock

    up_mask = prod.prod_mask_up
    if up_mask is None or up_mask.shape != shape:
        up_mask = np.empty(shape, dtype=np.bool_)
        prod.prod_mask_up = up_mask

    dn_mask = prod.prod_mask_dn
    if dn_mask is None or dn_mask.shape != shape:
        dn_mask = np.empty(shape, dtype=np.bool_)
        prod.prod_mask_dn = dn_mask

    # --- fill buffers in‑place ---------------
    shock[:] = rng.uniform(0.0, h_rho, size=shape)
    np.logical_and(prod.inventory == 0.0, prod.price >= p_avg, out=up_mask)
    np.logical_and(prod.inventory > 0.0, prod.price < p_avg, out=dn_mask)

    n_up = np.sum(up_mask)
    n_dn = np.sum(dn_mask)
    n_keep = prod.price.size - n_up - n_dn
    log.info(f"  Production changes: {n_up} firms ↑, {n_dn} firms ↓, {n_keep} firms ↔.")

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Avg market price (p̄): {p_avg:.2f}")
        log.debug(
            f"  Generated production shocks (h_rho={h_rho:.2f}):\n"
            f"{np.array2string(shock, precision=4)}")
        log.debug(
            f"  Inventories (S_i):\n{np.array2string(prod.inventory, precision=2)}"
        )
        log.debug(
            f"  Previous Production (Y_{{t-1}}):\n"
            f"{np.array2string(prod.production, precision=2)}"
        )
        if n_up > 0: log.debug(
            f"  Firms increasing production: {np.where(up_mask)[0].tolist()}")
        if n_dn > 0: log.debug(
            f"  Firms decreasing production: {np.where(dn_mask)[0].tolist()}")

    # --- core rule ----------------------------------
    prod.expected_demand[:] = prod.production
    prod.expected_demand[up_mask] *= 1.0 + shock[up_mask]
    prod.expected_demand[dn_mask] *= 1.0 - shock[dn_mask]
    log.debug(
        f"  Expected Demand set based on production changes:\n"
        f"{np.array2string(prod.expected_demand, precision=2)}")
    prod.desired_production[:] = prod.expected_demand

    log.debug(
        f"  Desired Production (Yd_i):\n"
        f"{np.array2string(prod.desired_production, precision=2)}"
    )
    log.info(f"  Total Desired Production: {prod.desired_production.sum():,.2f}")
    log.info("  --- Desired Production Decision complete ---")


def firms_calc_breakeven_price(
        prod: Producer,
        emp: Employer,
        lb: LoanBook,
        *,
        cap_factor: int | None = None,
):
    log.info("  --- Firms Calculating Breakeven Price ---")
    log.info(f"  Inputs: Breakeven Cap Factor={cap_factor}")

    # --- Breakeven calculation -----------------------------------------------
    interest = lb.interest_per_borrower(prod.production.size)
    # TODO Ensure wage bill is updated
    breakeven = ((emp.wage_bill + interest) /
                 np.maximum(prod.production, _EPS))
    log.info(
        f"  Total Wage Bill for calc: {emp.wage_bill.sum():,.2f}. "
        f"Total Interest for calc: {interest.sum():,.2f}")
    if log.isEnabledFor(logging.DEBUG):
        valid_breakeven = breakeven[np.isfinite(breakeven)]
        log.debug(
            f"  Raw breakeven prices (before cap): "
            f"min={valid_breakeven.min() if valid_breakeven.size > 0 else 'N/A':.2f}, "
            f"max={valid_breakeven.max() if valid_breakeven.size > 0 else 'N/A':.2f}, "
            f"avg={valid_breakeven.mean() if valid_breakeven.size > 0 else 'N/A':.2f}")

    # --- Cap breakeven -------------------------------------------------------
    if cap_factor and cap_factor > 1:
        # Cannot be more than current price x cap_factor. This prevents extreme jumps.
        breakeven_max_value = prod.price * cap_factor
    else:
        # If no cap_factor, the max value is effectively infinite
        breakeven_max_value = breakeven

    np.minimum(breakeven, breakeven_max_value, out=prod.breakeven_price)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Wage bill array:\n{emp.wage_bill}")
        log.debug(f"  Interest array:\n{interest}")
        log.debug(f"  Production array:\n{prod.production,}")
        log.debug(f"  Initial prices array:\n{prod.price}")
        num_capped = np.sum(breakeven > breakeven_max_value)
        log.debug(
            f"  Breakeven prices capped for {num_capped} firms "
            f"(indices: {np.where(breakeven > breakeven_max_value)[0].tolist()}) "
            f"with max increase limited to {cap_factor}x old price."
        )
        log.debug(f"  Final (Capped) Breakeven Prices:\n{prod.breakeven_price}")
    log.info("  --- Breakeven Price Calculation complete ---")


def firms_adjust_price(
        prod: Producer,
        *,
        p_avg: float,
        h_eta: float,
        rng: Generator = default_rng(),
) -> None:
    """
    Nominal price-adjustment rule (vectorised):

        shock ~ U(0, h_eta)

        if S == 0 and p < p̄:     p ← max(Pl , p·(1+shock))
        if S  > 0 and p ≥ p̄:     p ← max(Pl , p·(1-shock))
    """
    log.info("  --- Firms Adjusting Prices ---")
    log.info(
        f"  Inputs: Avg Market Price (p_avg)={p_avg:.2f}, "
        f"Max Price Shock (h_eta)={h_eta:.4f}"
    )

    shape = prod.price.shape
    old_prices_for_log = prod.price.copy()

    # --- scratch buffer for shocks -------------------------------------------
    shock = prod.price_shock
    if shock is None or shock.shape != shape:
        log.debug("  Initializing price shock buffer.")
        shock = np.empty(shape)
        prod.price_shock = shock

    shock[:] = rng.uniform(0.0, h_eta, size=shape)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Generated price shocks:\n{np.array2string(shock, precision=4)}")

    # --- masks ----------------------------------------------------------------
    mask_up = (prod.inventory == 0.0) & (prod.price < p_avg)
    mask_dn = (prod.inventory > 0.0) & (prod.price >= p_avg)
    num_up = np.sum(mask_up)
    num_dn = np.sum(mask_dn)
    num_no_change_rule = shape[0] - num_up - num_dn
    log.info(
        f"  Price adjustment candidates: "
        f"Raise Price={num_up}, Cut Price={num_dn}, No Rule={num_no_change_rule}"
    )

    if log.isEnabledFor(logging.DEBUG):
        if num_up > 0:
            log.debug(
                f"    Firms with S=0 & P<p̄ (mask_up): "
                f"{np.where(mask_up)[0].tolist()}"
            )
        if num_dn > 0:
            log.debug(
                f"    Firms with S>0 & P>=p̄ (mask_dn): "
                f"{np.where(mask_dn)[0].tolist()}"
            )

    # --- DEBUG pre-update snapshot -------------------------------------------
    if log.isEnabledFor(logging.DEBUG):
        log.debug("----- PRICE UPDATE (EXECUTION) -----")
        log.debug(f"  p̄ (avg market price) : {p_avg:.4f}")
        log.debug(
            f"  mask_up: {num_up} firms → raise  |  mask_dn: {num_dn} firms → cut"
        )
        log.debug(
            f"  Breakeven prices being used:\n"
            f"{np.array2string(prod.breakeven_price, precision=2)}")

    # --- raise prices --------------------------------------------------------
    if mask_up.any():
        if log.isEnabledFor(logging.DEBUG):
            up_indices = np.where(mask_up)[0]
            for firm_idx in up_indices[:min(3, num_up)]:
                log.debug(
                    f"    Raise Firm {firm_idx} PRE-UPDATE: "
                    f"OldP={old_prices_for_log[firm_idx]:.2f}, "
                    f"Shock={shock[firm_idx]:.4f}, "
                    f"Breakeven={prod.breakeven_price[firm_idx]:.2f}")

        np.multiply(prod.price, 1.0 + shock, out=prod.price, where=mask_up)
        np.maximum(prod.price, prod.breakeven_price, out=prod.price, where=mask_up)

        price_changes_raise = prod.price[mask_up] - old_prices_for_log[mask_up]
        avg_change_raise = np.mean(price_changes_raise) if num_up > 0 else 0
        num_floored_by_breakeven_raise = np.sum(
            np.isclose(
                prod.price[mask_up], prod.breakeven_price[mask_up]
            )  # Price is now breakeven
            & (
                    (old_prices_for_log[mask_up] * (1.0 + shock[mask_up]))
                    < prod.breakeven_price[mask_up]
            )  # And it was lower before max()
        )
        log.info(
            f"  Raising prices for {num_up} firms. "
            f"Avg price change: {avg_change_raise:+.2f}. "
            f"{num_floored_by_breakeven_raise} prices set by breakeven."
        )

    # --- cut prices ----------------------------------------------------------
    if mask_dn.any():
        if log.isEnabledFor(logging.DEBUG):
            dn_indices = np.where(mask_dn)[0]
            for firm_idx in dn_indices[:min(3, num_dn)]:
                log.debug(
                    f"    Cut Firm {firm_idx} PRE-UPDATE: "
                    f"OldP={old_prices_for_log[firm_idx]:.2f}, "
                    f"Shock={shock[firm_idx]:.4f}, "
                    f"Breakeven={prod.breakeven_price[firm_idx]:.2f}")

        np.multiply(prod.price, 1.0 - shock, out=prod.price, where=mask_dn)
        np.maximum(prod.price, prod.breakeven_price, out=prod.price, where=mask_dn)

        price_changes_cut = prod.price[mask_dn] - old_prices_for_log[mask_dn]
        avg_change_cut = np.mean(price_changes_cut) if num_dn > 0 else 0
        num_floored_by_breakeven_cut = np.sum(
            np.isclose(prod.price[mask_dn], prod.breakeven_price[mask_dn])
            & (
                    (old_prices_for_log[mask_dn] * (1.0 - shock[mask_dn]))
                    < prod.breakeven_price[mask_dn]
            )
        )
        log.info(
            f"  Cutting prices for {num_dn} firms. "
            f"Avg price change: {avg_change_cut:+.2f}. "
            f"{num_floored_by_breakeven_cut} prices set by breakeven."
        )

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Final Updated prices:\n{np.array2string(prod.price, precision=2)}")
    log.info("  --- Price Decision complete ---")


def firms_decide_desired_labor(prod: Producer, emp: Employer) -> None:
    """
    Desired labor demand (vectorised):

        Ld = ceil(Yd / a)
    """
    log.info("--- Firms Deciding Desired Labor ---")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Inputs: "
            f"Desired Production:\n{prod.desired_production.sum():,.2f}, "
            f"Labor Productivity:\n{np.array2string(
                prod.labor_productivity, precision=3)}")

    # --- validation -----------------------------------------------------------
    invalid = (~np.isfinite(prod.labor_productivity)) | (
            prod.labor_productivity <= _EPS
    )
    if invalid.any():
        n_invalid = np.sum(invalid)
        log.warning(
            f"  Found and clamped {n_invalid} firms "
            f"with invalid (zero, negative, or non-finite) labor productivity."
        )
        prod.labor_productivity[invalid] = _EPS

    # --- core rule -----------------------------------------------------------
    desired_labor_frac = prod.desired_production / prod.labor_productivity
    np.ceil(desired_labor_frac, out=desired_labor_frac)
    emp.desired_labor[:] = desired_labor_frac.astype(np.int64)

    # --- logging -----------------------------------------------------------
    log.info(f"  Total desired labor across all firms: {emp.desired_labor.sum()}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Desired Labor (Ld = ceil(Yd / a)):\n{emp.desired_labor}")
    log.info("  --- Desired Labor Decision complete ---")


def firms_decide_vacancies(emp: Employer) -> None:
    """
    Vector rule: V = max( Ld – L , 0 )
    """
    log.info("  --- Firms Deciding Vacancies ---")

    # --- core rule -----------------------------------------------------------
    np.subtract(
        emp.desired_labor,
        emp.current_labor,
        out=emp.n_vacancies,
        dtype=np.int64,
    )
    np.maximum(emp.n_vacancies, 0, out=emp.n_vacancies)

    # --- logging -----------------------------------------------------------
    log.info(f"  Total open vacancies in the economy: {emp.n_vacancies.sum()}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Current Labor (L):\n{emp.current_labor}")
        log.debug(f"  Final Vacancies (V = max(0, Ld - L)):\n{emp.n_vacancies}")
    log.info("--- Vacancy Decision complete ---")
