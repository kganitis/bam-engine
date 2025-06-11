# src/bamengine/systems/planning.py
from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator, default_rng

from bamengine import _logging_ext
from bamengine.components import Employer, Producer, LoanBook

log = _logging_ext.getLogger(__name__)

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
      shock ~ U(0, h_ρ)
      if S == 0 and P ≥ P̄   → raise by (1 + shock)
      if S  > 0 and P < P̄   → cut   by (1 − shock)
      otherwise             → keep previous level

    S: Inventory, P: Indivudual Price, P̄: Avg Market Price, h_ρ: Max Price Growth
    """
    log.info("--- Firms Deciding Desired Production ---")
    log.info(
        f"  Inputs: Avg Market Price (p_avg)={p_avg:.3f}  |  "
        f"Max Production Shock (h_ρ)={h_rho:.3f}")
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
        log.debug(f"  Avg market price (P̄): {p_avg:.2f}")
        log.debug(
            f"  Generated production shocks (h_ρ)={h_rho:.2f}):\n"
            f"{np.array2string(shock, precision=4)}")
        log.debug(
            f"  Inventories (S):\n{np.array2string(prod.inventory, precision=2)}"
        )
        log.debug(
            f"  Previous Production (Y_{{t-1}}):\n"
            f"{np.array2string(prod.production, precision=2)}"
        )
        if n_up > 0:
            log.debug(f"  Firms increasing production: {np.where(up_mask)[0].tolist()}")
        if n_dn > 0:
            log.debug(f"  Firms decreasing production: {np.where(dn_mask)[0].tolist()}")

    # --- core rule ----------------------------------
    prod.expected_demand[:] = prod.production
    prod.expected_demand[up_mask] *= 1.0 + shock[up_mask]
    prod.expected_demand[dn_mask] *= 1.0 - shock[dn_mask]
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Expected Demand set based on production changes:\n"
                  f"{np.array2string(prod.expected_demand, precision=2)}")

    # TODO Separate desired production system
    prod.desired_production[:] = prod.expected_demand
    log.info(f"  Total Desired Production for economy: "
             f"{prod.desired_production.sum():,.2f}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Desired Production (Yd):\n"
                  f"{np.array2string(prod.desired_production, precision=2)}")
    log.info("--- Desired Production Decision complete ---")


def firms_calc_breakeven_price(
        prod: Producer,
        emp: Employer,
        lb: LoanBook,
        *,
        cap_factor: int | None = None,
):
    # TODO Decide when to calc breakeven price
    #    - Case 1: During planning (use values from last period)
    #              - Use wage bill after firing or after contracts expiring?
    #              - Use interest before or after debt repayment?
    #    - Case 2: After credit market (where labor and interest are finalized)
    #              - Use projected production based on updated labor
    log.info("--- Firms Calculating Breakeven Price ---")
    log.info(f"  Inputs: Breakeven Cap Factor={cap_factor if cap_factor else 'None'}")
    log.info("  Calculation uses `prod.production` (from t-1) as the denominator. "
             "Ensure this is the intended production base.")

    # --- Breakeven calculation -----------------------------------------------
    interest = lb.interest_per_borrower(prod.production.size)
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
        log.info("  No cap_factor provided for breakeven price. "
                 "Prices may jump uncontrollably.")
        breakeven_max_value = breakeven

    np.minimum(breakeven, breakeven_max_value, out=prod.breakeven_price)

    num_capped = np.sum(breakeven > breakeven_max_value)
    if num_capped > 0:
        log.info(f"  Breakeven prices capped for {num_capped} firms.")
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"    Capped firm indices: "
                      f"{np.where(breakeven > breakeven_max_value)[0].tolist()}")

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Final (Capped) Breakeven Prices:\n"
                  f"{np.array2string(prod.breakeven_price, precision=2)}")
    log.info("--- Breakeven Price Calculation complete ---")


def firms_adjust_price(
        prod: Producer,
        *,
        p_avg: float,
        h_eta: float,
        rng: Generator = default_rng(),
) -> None:
    """
    Nominal price-adjustment rule.
    """
    # TODO Decide on whether to cap P to Pl or not
    #      As implemented, the new price is `max(shocked_price, breakeven_price)`.
    #      This can lead to prices jumping to the breakeven floor, which may be
    #      higher than the old price, even in a price-cutting scenario.
    #      Warn for extreme jumps.
    log.info("--- Firms Adjusting Prices ---")
    log.info(
        f"  Inputs: Avg Market Price (p_avg)={p_avg:.3f}  |  "
        f"Max Price Shock (h_η)={h_eta:.3f}"
    )

    shape = prod.price.shape
    old_prices_for_log = prod.price.copy()

    # --- scratch buffer for shocks -------------------------------------------
    shock = prod.price_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64) # Corrected dtype
        prod.price_shock = shock

    shock[:] = rng.uniform(0.0, h_eta, size=shape)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Generated price shocks:\n{np.array2string(shock, precision=4)}")

    # --- masks ----------------------------------------------------------------
    mask_up = (prod.inventory == 0.0) & (prod.price < p_avg)
    mask_dn = (prod.inventory > 0.0) & (prod.price >= p_avg)
    n_up, n_dn = np.sum(mask_up), np.sum(mask_dn)
    n_keep = shape[0] - n_up - n_dn
    log.info(f"  Price adjustments: {n_up} firms ↑, {n_dn} firms ↓, {n_keep} firms ↔.")

    if log.isEnabledFor(logging.DEBUG):
        if n_up > 0:
            log.debug(f"    Firms increasing price: {np.where(mask_up)[0].tolist()}")
        if n_dn > 0:
            log.debug(f"    Firms decreasing price: {np.where(mask_dn)[0].tolist()}")

    # --- DEBUG pre-update snapshot -------------------------------------------
    if log.isEnabledFor(logging.DEBUG):
        log.debug("  --- PRICE ADJUSTMENT (EXECUTION) ---")
        log.debug(f"  P̄ (avg market price) : {p_avg:.4f}")
        log.debug(
            f"  mask_up: {n_up} firms → raise  |  mask_dn: {n_dn} firms → cut"
        )
        log.debug(
            f"  Breakeven prices being used:\n"
            f"{np.array2string(prod.breakeven_price, precision=2)}")

    # --- raise prices --------------------------------------------------------
    if n_up > 0:
        np.multiply(prod.price, 1.0 + shock, out=prod.price, where=mask_up)
        np.maximum(prod.price, prod.breakeven_price, out=prod.price, where=mask_up)

        price_changes = prod.price[mask_up] - old_prices_for_log[mask_up]
        num_floored = np.sum(np.isclose(
            prod.price[mask_up], prod.breakeven_price[mask_up]))
        log.info(f"  Raised prices for {n_up} firms. "
                 f"Avg change: {np.mean(price_changes):+.3f}. "
                 f"{num_floored} prices set by breakeven floor.")
        if log.isEnabledFor(logging.DEBUG):
            for firm_idx in np.where(mask_up)[0][:5]:
                log.debug(f"    Raise Firm {firm_idx}: "
                          f"OldP={old_prices_for_log[firm_idx]:.2f} -> "
                          f"NewP={prod.price[firm_idx]:.2f} "
                          f"(Breakeven={prod.breakeven_price[firm_idx]:.2f})")

    # --- cut prices ----------------------------------------------------------
    if n_dn > 0:
        np.multiply(prod.price, 1.0 - shock, out=prod.price, where=mask_dn)
        np.maximum(prod.price, prod.breakeven_price, out=prod.price, where=mask_dn)

        price_changes = prod.price[mask_dn] - old_prices_for_log[mask_dn]
        num_floored = np.sum(np.isclose(prod.price[mask_dn],
                                        prod.breakeven_price[mask_dn]))
        num_increased_due_to_floor = np.sum(
            prod.price[mask_dn] > old_prices_for_log[mask_dn])
        log.info(f"  Cut prices for {n_dn} firms. "
                 f"Avg change: {np.mean(price_changes):+.3f}. "
                 f"{num_floored} prices set by breakeven floor.")
        if num_increased_due_to_floor > 0:
            log.warning(f"  !!! {num_increased_due_to_floor} firms in the 'cut price' "
                        f"group ended up INCREASING their price because their "
                        f"breakeven floor was higher than their old price.")
        if log.isEnabledFor(logging.DEBUG):
            for firm_idx in np.where(mask_dn)[0][:5]:
                log.debug(f"    Cut Firm {firm_idx}: "
                          f"OldP={old_prices_for_log[firm_idx]:.2f} -> "
                          f"NewP={prod.price[firm_idx]:.2f} "
                          f"(Breakeven={prod.breakeven_price[firm_idx]:.2f})")

    log.info("--- Price Adjustment complete ---")


def firms_decide_desired_labor(prod: Producer, emp: Employer) -> None:
    """
    Rule
    ----
        Ld = ceil(Yd / a)

    Ld: Desired Labour, Yd: Desired Production, a: Labour Productivity
    """
    log.info("--- Firms Deciding Desired Labor ---")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Inputs: Total Desired Production={prod.desired_production.sum():,.2f}"
            f"  |  Avg Labor Productivity={prod.labor_productivity.mean():.3f}")

    # --- validation -----------------------------------------------------------
    invalid_mask = np.logical_or(~np.isfinite(prod.labor_productivity),
                                 prod.labor_productivity <= _EPS)
    if invalid_mask.any():
        n_invalid = np.sum(invalid_mask)
        log.warning(f"  Found and clamped {n_invalid} firms "
                    f"with invalid labor productivity.")
        prod.labor_productivity[invalid_mask] = _EPS

    # --- core rule -----------------------------------------------------------
    desired_labor_frac = prod.desired_production / prod.labor_productivity
    np.ceil(desired_labor_frac, out=desired_labor_frac)
    emp.desired_labor[:] = desired_labor_frac.astype(np.int64)

    # --- logging -----------------------------------------------------------
    log.info(f"  Total desired labor across all firms: {emp.desired_labor.sum():,}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Desired Labor (Ld):\n{emp.desired_labor}")
    log.info("--- Desired Labor Decision complete ---")


def firms_decide_vacancies(emp: Employer) -> None:
    """
    Rule
    ----
        V = max( Ld – L , 0 )

    V: Number of Open Vacancies, Ld: Desired Labour, L: Actual Labour
    """
    log.info("--- Firms Deciding Vacancies ---")
    log.info(f"  Inputs: Total Desired Labor={emp.desired_labor.sum():,}  |"
             f"  Total Current Labor={emp.current_labor.sum():,}")

    # --- core rule -----------------------------------------------------------
    np.subtract(emp.desired_labor, emp.current_labor,
                out=emp.n_vacancies, dtype=np.int64)
    np.maximum(emp.n_vacancies, 0, out=emp.n_vacancies)

    # --- logging -----------------------------------------------------------
    log.info(f"  Total open vacancies in the economy: {emp.n_vacancies.sum():,}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Final Vacancies (V):\n{emp.n_vacancies}")
    log.info("--- Vacancy Decision complete ---")
