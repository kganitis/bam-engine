"""
Event‑3  –  Credit‑market systems  (vectorised, no new allocations at runtime)
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator

from bamengine.components.bank_credit import BankCreditSupply, BankInterestRate
from bamengine.components.firm_credit import FirmCreditPlan
from bamengine.typing import FloatA, IdxA

log = logging.getLogger(__name__)


# ─────────────────────── banks: vacancies & rate ───────────────────────

def banks_decide_credit_supply(cs: BankCreditSupply, *, v: float) -> None:
    """
    C_k = E_k · v
    """
    np.multiply(cs.equity_base, v, out=cs.credit_supply)


def banks_decide_interest_rate(
    br: BankInterestRate,
    *,
    r_bar: float,
    h_phi: float,
    rng: Generator,
) -> None:
    """
    Nominal interest rate rule:

    r_k = r̄ · (1 + U(0, h_φ))
    """
    shape = br.interest_rate.shape

    # permanent scratch
    shock = br.credit_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64)

    # fill buffer in-place
    shock[:] = rng.uniform(0.0, h_phi, size=shape)

    # core rule
    br.interest_rate[:] = r_bar * (1.0 + shock)


def firms_decide_credit_demand(fp: FirmCreditPlan) -> None:
    """
    B_i = max( W_i − A_i , 0 )
    """
    np.subtract(fp.wage_bill, fp.net_worth, out=fp.credit_demand)
    np.maximum(fp.credit_demand, 0.0, out=fp.credit_demand)