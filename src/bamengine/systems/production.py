# src/bamengine/systems/production.py
"""
Event-4 – Production systems
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator

from bamengine.components import Consumer, Economy, Employer, LoanBook, Producer, Worker

log = logging.getLogger(__name__)
log.setLevel(logging.CRITICAL)


# ------------------------------------------------------------------ #
# 0.  Calc unemployment rate                                         #
# ------------------------------------------------------------------ #
def calc_unemployment_rate(
    ec: Economy,
    wrk: Worker,
) -> None:
    # Number of unemployed agents / total household population
    n_workers = wrk.employed.size
    unemployed_count = wrk.employed.size - wrk.employed.sum()
    ec.unemp_rate_history = np.append(
        ec.unemp_rate_history, unemployed_count / n_workers
    )


# ------------------------------------------------------------------ #
# 1.  Firms: decide selling price                                    #
# ------------------------------------------------------------------ #
def firms_decide_price(
    prod: Producer,
    emp: Employer,
    lb: LoanBook,
    *,
    p_avg: float,
    h_eta: float,
    rng: Generator,
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
    breakeven = (emp.wage_bill + interest) / np.maximum(prod.production, 1.0e-12)

    # ── DEBUG pre-update snapshot ────────────────────────────────────────
    if log.isEnabledFor(logging.DEBUG):
        log.debug("----- PRICE UPDATE -----")
        log.debug("p̄ (avg market price) : %.4f", p_avg)
        log.debug(
            "mask_up: %d firms → raise  |  mask_dn: %d firms → cut",
            mask_up.sum(),
            mask_dn.sum(),
        )

    old_prices = prod.price.copy()

    # ── raise prices ─────────────────────────────────────────────────────
    if mask_up.any():
        np.multiply(prod.price, 1.0 + shock, out=prod.price, where=mask_up)
        np.maximum(prod.price, breakeven, out=prod.price, where=mask_up)

    # ── cut prices ────────────────────────────────────────────────────────
    if mask_dn.any():
        np.multiply(prod.price, 1.0 - shock, out=prod.price, where=mask_dn)
        np.maximum(prod.price, breakeven, out=prod.price, where=mask_dn)

    bad_mask = prod.price > old_prices * 10.0
    if np.any(bad_mask):
        bad = np.where(bad_mask)[0]
        log.debug(
            f"!!! runaway price at firm(s) {bad}: {prod.price[bad]}, "
            f"old: {old_prices[bad]}, "
            f"breakeven: {breakeven[bad]}, "
            f"production: {prod.production[bad]}"
        )
        capped_price = old_prices[bad_mask] * (1.0 + h_eta)
        prod.price[bad_mask] = capped_price

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"[PRICE SANITY] max price={prod.price.max():.3g}, "
            f"min shock={shock.min():.12g} "
            f"max shock={shock.max():.12g} "
        )
        log.debug(f"  Detailed Prices:\n" f"{np.array2string(prod.price, precision=2)}")


# --------------------------------------------------------------------- #
# 2.  Update average market price                                       #
# --------------------------------------------------------------------- #
def update_avg_mkt_price(ec: Economy, prod: Producer) -> None:
    """ """
    ec.avg_mkt_price = float(prod.price.mean())
    ec.avg_mkt_price_history = np.append(ec.avg_mkt_price_history, ec.avg_mkt_price)


# --------------------------------------------------------------------- #
# 3.  Calculate inflation rate                                          #
# --------------------------------------------------------------------- #
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


# --------------------------------------------------------------------- #
# 3.  Wage payment – firm-side only                                     #
# --------------------------------------------------------------------- #
def firms_pay_wages(emp: Employer) -> None:
    """
    Debit each firm’s cash account by its wage-bill (vectorised).
    """
    np.subtract(emp.total_funds, emp.wage_bill, out=emp.total_funds)


# --------------------------------------------------------------------- #
# 4.  Wage receipt – household side                                     #
# --------------------------------------------------------------------- #
def workers_receive_wage(con: Consumer, wrk: Worker) -> None:
    """
    Credit household income by the wage that *employed* workers earned:

        income_h += wage_h · employed_h
    """
    inc = wrk.wage * wrk.employed
    np.add(con.income, inc, out=con.income)


# --------------------------------------------------------------------- #
# 5.  Physical production                                               #
# --------------------------------------------------------------------- #
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


# --------------------------------------------------------------------- #
# 6.  Contract-expiration mechanic                                      #
# --------------------------------------------------------------------- #
def workers_update_contracts(wrk: Worker, emp: Employer) -> None:
    """
    Decrease `periods_left` for every employed worker and let contracts that
    reach 0 expire.  All worker-side flags are updated **and** the corresponding
    firm’s labour and wage-bill are brought back in sync.

        • L_i := Σ 1{worker employed & employer == i}
        • W_i := L_i · w_i
    """

    # ---- step 0: guard against impossible ‘already-0’ contracts ----------
    already_expired = (wrk.employed == 1) & (wrk.periods_left == 0)
    if already_expired.any():  # treat them as “1 → 0”
        wrk.periods_left[already_expired] = 1

    # ---- step 1: tick down only for currently employed -------------------
    mask_emp = wrk.employed == 1
    if not mask_emp.any():  # nothing to do
        return

    wrk.periods_left[mask_emp] -= 1

    # ---- step 2: detect expirations --------------------------------------
    expired = mask_emp & (wrk.periods_left == 0)
    if not expired.any():  # no contract hit zero
        return

    firms = wrk.employer[expired]  # snapshot before overwrite

    # -------- worker-side updates -----------------------------------------
    wrk.employed[expired] = 0
    wrk.employer[expired] = -1
    wrk.employer_prev[expired] = firms
    wrk.wage[expired] = 0.0
    wrk.contract_expired[expired] = 1
    wrk.fired[expired] = 0

    # -------- firm-side labour bookkeeping --------------------------------
    delta = np.bincount(firms, minlength=emp.current_labor.size)
    emp.current_labor[: delta.size] -= delta
    assert (emp.current_labor >= 0).all(), "negative labour after expirations"

    # keep wage-bill consistent with the new labour vector
    np.multiply(emp.current_labor, emp.wage_offer, out=emp.wage_bill)
