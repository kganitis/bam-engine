# src/bamengine/systems/production.py
"""
Event-4 – Production systems (vectorised, zero allocations)
"""
from __future__ import annotations

import numpy as np

from bamengine.components import Consumer, Employer, Producer, Worker


# --------------------------------------------------------------------- #
# 1.  Wage payment – firm-side only                                     #
# --------------------------------------------------------------------- #
def firms_pay_wages(emp: Employer) -> None:
    """
    Debit each firm’s cash account by its wage-bill (vectorised).
    """
    np.subtract(emp.total_funds, emp.wage_bill, out=emp.total_funds)


# --------------------------------------------------------------------- #
# 2.  Wage receipt – household side                                     #
# --------------------------------------------------------------------- #
def workers_receive_wage(con: Consumer, wrk: Worker) -> None:
    """
    Credit household income by the wage that *employed* workers earned:

        income_h += wage_h · employed_h
    """
    inc = wrk.wage * wrk.employed
    np.add(con.income, inc, out=con.income)


# --------------------------------------------------------------------- #
# 3.  Physical production                                               #
# --------------------------------------------------------------------- #
def firms_run_production(prod: Producer, emp: Employer) -> None:
    """
    Compute current-period output and replace inventories:

        Y_i  =  a_i · L_i
        S_i  ←  Y_i
    """
    np.multiply(prod.labor_productivity, emp.current_labor, out=prod.production)
    prod.inventory[:] = prod.production  # overwrite, do **not** add


# --------------------------------------------------------------------- #
# 4.  Contract-expiration mechanic                                      #
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
