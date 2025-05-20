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
def consumers_receive_wage(con: Consumer, wrk: Worker) -> None:
    """
    Credit household income by the wage that *employed* workers earned:

        income_h += wage_h · employed_h
    """
    add = wrk.wage * wrk.employed.astype(np.float64)
    np.add(con.income, add, out=con.income)


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
