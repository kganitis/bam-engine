# src/bamengine/systems/planning.py
from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.components import Employer, Producer, LoanBook

log = logging.getLogger(__name__)

CAP_LAB_PROD = 1.0e-6  # labor productivity cap if below from or equal to zero


def firms_decide_desired_production(  # noqa: C901
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
    shape = prod.price.shape

    # ── 1. permanent scratches ---------------
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

    # ── 2. fill buffers in‑place ---------------
    shock[:] = rng.uniform(0.0, h_rho, size=shape)
    np.logical_and(prod.inventory == 0.0, prod.price >= p_avg, out=up_mask)
    np.logical_and(prod.inventory > 0.0, prod.price < p_avg, out=dn_mask)

    n_up = np.sum(up_mask)
    n_dn = np.sum(dn_mask)
    n_keep = len(prod.price) - n_up - n_dn
    log.info(f"  Production changes: {n_up} firms ↑, {n_dn} firms ↓, {n_keep} firms ↔.")

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Avg market price (p_avg): {p_avg:.4f}")
        log.debug(
            f"  Inventories (S_i):\n{np.array2string(prod.inventory, precision=2)}"
        )
        log.debug(
            f"  Previous Production (Y_{{t-1}}):\n"
            f"{np.array2string(prod.production, precision=2)}"
        )

    # ── 3. core rule ----------------------------------
    prod.expected_demand[:] = prod.production
    prod.expected_demand[up_mask] *= 1.0 + shock[up_mask]
    prod.expected_demand[dn_mask] *= 1.0 - shock[dn_mask]
    prod.desired_production[:] = prod.expected_demand

    log.debug(
        f"  Desired Production (Yd_i):\n"
        f"{np.array2string(prod.desired_production, precision=2)}"
    )


def firms_decide_desired_labor(prod: Producer, emp: Employer) -> None:
    """
    Desired labor demand (vectorised):

        Ld_i = ceil(Yd_i / a_i)
    """
    # --- validation -----------------------------------------------------------
    invalid = (~np.isfinite(prod.labor_productivity)) | (
        prod.labor_productivity <= CAP_LAB_PROD
    )
    if invalid.any():
        n_invalid = np.sum(invalid)
        log.warning(
            f"  {n_invalid} firms have too low/non-finite labor productivity; clamping."
        )
        prod.labor_productivity[invalid] = CAP_LAB_PROD

    # --- core rule -----------------------------------------------------------
    desired_labor = prod.desired_production / prod.labor_productivity
    np.ceil(desired_labor, out=desired_labor)
    emp.desired_labor[:] = desired_labor.astype(np.int64)

    # --- logging -----------------------------------------------------------
    log.info(f"  Total desired labor across all firms: {emp.desired_labor.sum()}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Desired Labor (Ld_i):\n{emp.desired_labor}")


def firms_decide_vacancies(emp: Employer) -> None:
    """
    Vector rule: V_i = max( Ld_i – L_i , 0 )
    """
    # --- core rule -----------------------------------------------------------
    np.subtract(
        emp.desired_labor,
        emp.current_labor,
        out=emp.n_vacancies,
        dtype=np.int64,
        casting="unsafe",  # makes MyPy/NumPy on Windows happy
    )
    np.maximum(emp.n_vacancies, 0, out=emp.n_vacancies)

    # --- logging -----------------------------------------------------------
    total_vacancies = emp.n_vacancies.sum()
    log.info(f"  Total open vacancies in the economy: {total_vacancies}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Current Labor (L_i):\n{emp.current_labor}")
        log.debug(f"  Vacancies (V_i):\n{emp.n_vacancies}")


def firms_decide_price_new(
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
        f"  Inputs: Avg Market Price (p_avg)={p_avg:.4f}, "
        f"Max Price Shock (h_eta)={h_eta:.3f}"
    )

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
        log.debug(f"  Generated price shocks: {shock})")

    # ── masks ─────────────────────────────────────────────────────────────
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
                f"    Firms with S=0 & P<p_avg (mask_up): "
                f"{np.where(mask_up)[0].tolist()}"
            )
        if num_dn > 0:
            log.debug(
                f"    Firms with S>0 & P>=p_avg (mask_dn): "
                f"{np.where(mask_dn)[0].tolist()}"
            )

    # ── breakeven price: (wage bill + interest) / output ─────────────────
    interest = lb.interest_per_borrower(prod.price.size)

    # Raw breakeven calculation
    raw_breakeven = (emp.wage_bill + interest) / np.maximum(prod.production, 1.0e-12)

    # Cap breakeven: cannot be more than 2x current price. This prevents extreme jumps.
    # prod.price here is the price from *before* this period's adjustment.
    breakeven_cap_factor = 100.0
    breakeven_capped_at_value = old_prices_for_log * breakeven_cap_factor
    breakeven_capped = np.minimum(raw_breakeven, breakeven_capped_at_value)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Wage bill: {emp.wage_bill}")
        log.debug(f"  Interest: {interest}")
        log.debug(f"  Production: {prod.production,}")
        valid_raw_breakeven = raw_breakeven[np.isfinite(raw_breakeven)]
        log.debug(f"  Initial prices: {prod.price}")
        log.debug(f"  Raw breakeven prices (finite values): {valid_raw_breakeven}")
        log.debug(
            f"  Breakeven prices capped for firms "
            f"{np.where(raw_breakeven > breakeven_capped_at_value)[0]}"
            f"(max increase limited to {breakeven_cap_factor:.1f}x old price)."
        )
        log.debug(f"  Capped breakeven prices (finite values): {breakeven_capped}")

    # ── DEBUG pre-update snapshot ────────────────────────────────────────
    if log.isEnabledFor(logging.DEBUG):
        log.debug("----- PRICE UPDATE (EXECUTION) -----")
        log.debug(f"  p̄ (avg market price) : {p_avg}")
        log.debug(
            f"  mask_up: {num_up} firms → raise  |" f"  mask_dn: {num_dn} firms → cut"
        )

    # ── raise prices ─────────────────────────────────────────────────────
    if mask_up.any():
        np.multiply(prod.price, 1.0 + shock, out=prod.price, where=mask_up)
        np.maximum(prod.price, breakeven_capped, out=prod.price, where=mask_up)
        price_changes_raise = prod.price[mask_up] - old_prices_for_log[mask_up]
        avg_change_raise = np.mean(price_changes_raise) if num_up > 0 else 0
        # Check how many prices were actually set to the breakeven_capped value
        num_floored_by_breakeven_raise = np.sum(
            np.isclose(
                prod.price[mask_up], breakeven_capped[mask_up]
            )  # Price is now breakeven
            & (
                (old_prices_for_log[mask_up] * (1.0 + shock[mask_up]))
                < breakeven_capped[mask_up]
            )  # And it was lower before max()
        )
        log.info(
            f"  Raising prices for {num_up} firms. "
            f"Avg price change: {avg_change_raise:+.3f}. "
            f"{num_floored_by_breakeven_raise} prices set by (capped) breakeven."
        )

    # ── cut prices ────────────────────────────────────────────────────────
    if mask_dn.any():
        np.multiply(prod.price, 1.0 - shock, out=prod.price, where=mask_dn)
        np.maximum(prod.price, breakeven_capped, out=prod.price, where=mask_dn)
        price_changes_cut = prod.price[mask_dn] - old_prices_for_log[mask_dn]
        avg_change_cut = np.mean(price_changes_cut) if num_dn > 0 else 0
        num_floored_by_breakeven_cut = np.sum(
            np.isclose(prod.price[mask_dn], breakeven_capped[mask_dn])
            & (
                (old_prices_for_log[mask_dn] * (1.0 - shock[mask_dn]))
                < breakeven_capped[mask_dn]
            )
        )
        log.info(
            f"  Cutting prices for {num_dn} firms. "
            f"Avg price change: {avg_change_cut:+.3f}. "
            f"{num_floored_by_breakeven_cut} prices set by (capped) breakeven."
        )

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Updated prices:\n" f"{np.array2string(prod.price, precision=2)}")
    log.info("--- Price Decision complete ---")
