# src/bamengine/systems/production.py
"""
Event-4 – Production systems (vectorised, zero allocations)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bamengine.components import Consumer, Employer, Producer, Worker
from bamengine.typing import Idx1D


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
    Decrease `periods_left` for every *employed* worker.
    When the counter reaches 0 the contract expires:

        • worker becomes unemployed (`employed = 0`)
        • `contract_expired = 1`
        • `employer_prev` stores the firm index
        • firm labour count `current_labor` is decremented
        • queue-related scratch is left untouched (will be filled later)

    All updates are fully vectorised except the firm-side labour
    adjustment, which uses a single `np.bincount`.
    """

    # --- step 1: tick down only for currently employed -----------------
    mask_emp: NDArray[np.bool_] = wrk.employed == 1
    if not mask_emp.any():
        return  # nothing to do

    wrk.periods_left[mask_emp] -= 1

    # --- step 2: detect expirations -----------------------------------
    expired: NDArray[np.bool_] = mask_emp & (wrk.periods_left == 0)
    if not expired.any():
        return

    # snapshot firm indices before we overwrite them
    firms: Idx1D = wrk.employer[expired]

    # worker-side state
    wrk.employed[expired] = 0
    wrk.employer[expired] = -1
    wrk.employer_prev[expired] = firms
    wrk.wage[expired] = 0.0
    wrk.contract_expired[expired] = 1
    wrk.fired[expired] = 0  # explicit: this was an expiration

    # firm-side labour book-keeping
    delta = np.bincount(firms, minlength=emp.current_labor.size)
    emp.current_labor[: delta.size] -= delta
    # guard against numerical slip
    assert (emp.current_labor >= 0).all(), "negative labour after expirations"
