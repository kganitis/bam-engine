"""
System functions for planning phase events.

This module contains the internal implementation functions for planning events.
Event classes wrap these functions and provide the primary documentation.

See Also
--------
bamengine.events.planning : Event classes (primary documentation source)
"""

from __future__ import annotations

import numpy as np

from bamengine import Rng, logging, make_rng
from bamengine.relationships import LoanBook
from bamengine.roles import Employer, Producer, Worker
from bamengine.utils import EPS

log = logging.getLogger(__name__)


def firms_decide_desired_production(
    prod: Producer, *, p_avg: float, h_rho: float, rng: Rng = make_rng()
) -> None:
    """
    Set production targets based on inventory levels and market position.

    See Also
    --------
    bamengine.events.planning.FirmsDecideDesiredProduction : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Deciding Desired Production ---")

    # Zero out current production at start of planning phase.
    # production_prev retains previous period's value for use as planning signal.
    prod.production[:] = 0.0
    if info_enabled:
        log.info(
            f"  Inputs: Avg Market Price (p_avg)={p_avg:.3f}  |  "
            f"Max Production Shock (h_ρ)={h_rho:.3f}"
        )
    shape = prod.price.shape

    # permanent scratches
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

    # fill buffers in‑place
    shock[:] = rng.uniform(0.0, h_rho, size=shape)
    np.logical_and(prod.inventory == 0.0, prod.price >= p_avg, out=up_mask)
    np.logical_and(prod.inventory > 0.0, prod.price < p_avg, out=dn_mask)

    n_up = np.sum(up_mask)
    n_dn = np.sum(dn_mask)
    n_keep = prod.price.size - n_up - n_dn
    if info_enabled:
        log.info(
            f"  Production changes: {n_up} firms ↑, {n_dn} firms ↓, {n_keep} firms ↔."
        )

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Avg market price (P̄): {p_avg:.2f}")
        log.debug(
            f"  Generated production shocks (h_ρ)={h_rho:.2f}):\n"
            f"{np.array2string(shock, precision=4)}"
        )
        log.debug(f"  Inventories (S):\n{np.array2string(prod.inventory, precision=2)}")
        log.debug(
            f"  Previous Production (Y_{{t-1}}):\n"
            f"{np.array2string(prod.production_prev, precision=2)}"
        )
        if n_up > 0:
            log.debug(f"  Firms increasing production: {np.where(up_mask)[0].tolist()}")
        if n_dn > 0:
            log.debug(f"  Firms decreasing production: {np.where(dn_mask)[0].tolist()}")

    # core rule - use production_prev as baseline for expected demand
    prod.expected_demand[:] = prod.production_prev
    prod.expected_demand[up_mask] *= 1.0 + shock[up_mask]
    prod.expected_demand[dn_mask] *= 1.0 - shock[dn_mask]
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Expected Demand set based on production changes:\n"
            f"{np.array2string(prod.expected_demand, precision=2)}"
        )

    prod.desired_production[:] = prod.expected_demand
    if info_enabled:
        log.info(
            f"  Total Desired Production for economy: {prod.desired_production.sum():,.2f}"
        )
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Desired Production (Yd):\n"
            f"{np.array2string(prod.desired_production, precision=2)}"
        )
    if info_enabled:
        log.info("--- Desired Production Decision complete ---")


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
    bamengine.events.planning.FirmsCalcBreakevenPrice : Full documentation
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
    bamengine.events.planning.FirmsAdjustPrice : Full documentation
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


def firms_decide_desired_labor(prod: Producer, emp: Employer) -> None:
    """
    Calculate desired labor from production targets and productivity.

    See Also
    --------
    bamengine.events.planning.FirmsDecideDesiredLabor : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Deciding Desired Labor ---")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Inputs: Total Desired Production={prod.desired_production.sum():,.2f}"
            f"  |  Avg Labor Productivity={prod.labor_productivity.mean():.3f}"
        )

    # validation
    invalid_mask = np.logical_or(
        ~np.isfinite(prod.labor_productivity), prod.labor_productivity <= EPS
    )
    if invalid_mask.any():
        n_invalid = np.sum(invalid_mask)
        log.warning(
            f"  Found and clamped {n_invalid} firms with invalid labor productivity."
        )
        prod.labor_productivity[invalid_mask] = EPS

    # core rule
    desired_labor_frac = prod.desired_production / prod.labor_productivity
    np.ceil(desired_labor_frac, out=desired_labor_frac)
    emp.desired_labor[:] = desired_labor_frac.astype(np.int64)

    # logging
    if info_enabled:
        log.info(f"  Total desired labor across all firms: {emp.desired_labor.sum():,}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Desired Labor (Ld):\n{emp.desired_labor}")
    if info_enabled:
        log.info("--- Desired Labor Decision complete ---")


def firms_decide_vacancies(emp: Employer) -> None:
    """
    Calculate number of job vacancies from labor gap.

    See Also
    --------
    bamengine.events.planning.FirmsDecideVacancies : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Deciding Vacancies ---")
        log.info(
            f"  Inputs: Total Desired Labor={emp.desired_labor.sum():,}  |"
            f"  Total Current Labor={emp.current_labor.sum():,}"
        )

    # core rule
    np.subtract(
        emp.desired_labor, emp.current_labor, out=emp.n_vacancies, dtype=np.int64
    )
    np.maximum(emp.n_vacancies, 0, out=emp.n_vacancies)

    # logging
    if info_enabled:
        log.info(f"  Total open vacancies in the economy: {emp.n_vacancies.sum():,}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Final Vacancies:\n{emp.n_vacancies}")
    if info_enabled:
        log.info("--- Vacancy Decision complete ---")


def firms_fire_excess_workers(
    emp: Employer,
    wrk: Worker,
    *,
    method: str = "random",
    rng: Rng = make_rng(),
) -> None:
    """
    Firms lay off workers when current labor exceeds desired labor.

    See Also
    --------
    bamengine.events.planning.FirmsFireExcessWorkers : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Firing Excess Workers ---")

    excess = emp.current_labor - emp.desired_labor
    firing_ids = np.where(excess > 0)[0]
    total_excess = excess[excess > 0].sum() if excess.size > 0 else 0

    if info_enabled:
        log.info(
            f"  {firing_ids.size} firms have excess labor totaling {total_excess:,} workers "
            f"using '{method}' method."
        )

    total_workers_fired_this_step = 0

    for i in firing_ids:
        n_to_fire = excess[i]
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"  Processing firm {i} (current_labor: {emp.current_labor[i]}, "
                f"desired_labor: {emp.desired_labor[i]}, excess: {n_to_fire})"
            )

        # Validate workforce consistency
        workforce = np.where((wrk.employed == 1) & (wrk.employer == i))[0]
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Firm {i} workforce validation: "
                f"real={workforce.size}, recorded={emp.current_labor[i]}"
            )

        if workforce.size != emp.current_labor[i]:
            log.critical(
                f"    Firm {i}: Real workforce ({workforce.size}) INCONSISTENT "
                f"with bookkeeping ({emp.current_labor[i]})."
            )

        if workforce.size == 0:  # pragma: no cover
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"    Firm {i}: No workers to fire. Skipping.")
            continue

        # Determine workers to fire
        worker_wages = wrk.wage[workforce]
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Firm {i} worker wages: {worker_wages} "
                f"(total: {worker_wages.sum():.2f})"
            )

        if method not in ("random", "expensive"):  # pragma: no cover
            log.error(
                f"    Unknown firing method '{method}' specified. "
                f"Defaulting to 'random'."
            )
            method = "random"

        if method == "expensive":
            # Fire most expensive workers first
            sorted_indices = np.argsort(worker_wages)[::-1]  # Descending order
            victims_indices = sorted_indices[:n_to_fire]
            victims = workforce[victims_indices]
        else:  # method == "random"
            # Random firing
            shuffled_indices = rng.permutation(workforce.size)
            victims_indices = shuffled_indices[:n_to_fire]
            victims = workforce[victims_indices]

        fired_wages = wrk.wage[victims]
        total_fired_wage = fired_wages.sum()

        # Extra validation, should never trigger
        if victims.size == 0:  # pragma: no cover
            continue

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Firm {i} is firing {victims.size} worker(s): "
                f"{victims.tolist()} (total wage savings: {total_fired_wage:.2f})"
            )
        total_workers_fired_this_step += victims.size

        # Worker-side updates
        log.debug(f"      Updating state for {victims.size} fired workers.")
        wrk.employer[victims] = -1
        wrk.employer_prev[victims] = i
        wrk.wage[victims] = 0.0
        wrk.periods_left[victims] = 0
        wrk.contract_expired[victims] = 0
        wrk.fired[victims] = 1

        # Firm-side updates
        emp.current_labor[i] -= victims.size

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"      Firm {i} state updated: "
                f"current_labor={emp.current_labor[i]}, "
                f"desired_labor={emp.desired_labor[i]}"
            )

    if info_enabled:
        log.info(
            f"  Total workers fired this step across all firms: "
            f"{total_workers_fired_this_step}"
        )
        log.info("--- Firms Firing Excess Workers complete ---")
