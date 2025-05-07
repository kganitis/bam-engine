# src/bamengine/systems/credit_market.py
"""
Event‑3  –  Credit‑market systems  (vectorised, no new allocations at runtime)
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator

from bamengine.components.bank_credit import BankCreditSupply, BankInterestRate, \
    BankReceiveLoanApplication, BankProvideLoan
from bamengine.components.firm_credit import FirmCreditDemand, FirmLoanApplication, \
    FirmLoan
from bamengine.typing import FloatA, IdxA

log = logging.getLogger(__name__)


# ─────────────────────── banks: vacancies & rate ───────────────────────

def banks_decide_credit_supply(banks: BankCreditSupply, *, v: float) -> None:
    """
    C_k = E_k · v
    """
    np.multiply(banks.equity_base, v, out=banks.credit_supply)


def banks_decide_interest_rate(
    banks: BankInterestRate,
    *,
    r_bar: float,
    h_phi: float,
    rng: Generator,
) -> None:
    """
    Nominal interest rate rule:

    r_k = r̄ · (1 + U(0, h_φ))
    """
    shape = banks.interest_rate.shape

    # permanent scratch
    shock = banks.credit_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64)

    # fill buffer in-place
    shock[:] = rng.uniform(0.0, h_phi, size=shape)

    # core rule
    banks.interest_rate[:] = r_bar * (1.0 + shock)


def firms_decide_credit_demand(firms: FirmCreditDemand) -> None:
    """
    B_i = max( W_i − A_i , 0 )
    """
    np.subtract(firms.wage_bill, firms.net_worth, out=firms.credit_demand)
    np.maximum(firms.credit_demand, 0.0, out=firms.credit_demand)


def _topk_lowest_rate(values: FloatA, k: int) -> IdxA:
    """
    argpartition on **+rate** (cheapest first)
    """
    if k >= values.shape[-1]:
        return np.argpartition(values, kth=0, axis=-1)
    part = np.argpartition(values, kth=k - 1, axis=-1)
    return part[..., :k]


def firms_prepare_loan_applications(
    firms: FirmLoanApplication,
    banks: BankInterestRate,
    *,
    max_H: int,
    rng: Generator,
) -> None:
    """
    * draws H random banks per borrower
    * keeps the H *cheapest* (lowest r_k) via partial sort
    * writes indices into ``loan_apps_targets`` and resets ``loan_apps_head``
    """
    n_banks = banks.interest_rate.size
    borrowers = np.where(firms.credit_demand > 0.0)[0]
    if borrowers.size == 0:
        firms.loan_apps_head.fill(-1)
        return

    sample = rng.integers(0, n_banks, size=(borrowers.size, max_H), dtype=np.int64)
    topk = _topk_lowest_rate(banks.interest_rate[sample], k=max_H)
    sorted_sample = np.take_along_axis(sample, topk, axis=1)

    stride = max_H
    firms.loan_apps_targets.fill(-1)
    firms.loan_apps_head.fill(-1)

    for k, f in enumerate(borrowers):
        firms.loan_apps_targets[f, :stride] = sorted_sample[k]
        firms.loan_apps_head[f] = f * stride  # start of that row


def firms_send_one_loan_app(
    firms: FirmLoanApplication,
    banks: BankReceiveLoanApplication
) -> None:
    """
    """
    stride = firms.loan_apps_targets.shape[1]

    for f in np.where(firms.credit_demand > 0.0)[0]:
        h = firms.loan_apps_head[f]
        if h < 0:
            continue
        row, col = divmod(h, stride)
        bank_idx = firms.loan_apps_targets[row, col]
        if bank_idx < 0:
            firms.loan_apps_head[f] = -1
            continue

        # bounded queue – reuse recv_apps_head logic
        ptr = banks.recv_apps_head[bank_idx] + 1
        if ptr >= banks.recv_apps.shape[1]:
            continue
        banks.recv_apps_head[bank_idx] = ptr
        banks.recv_apps[bank_idx, ptr] = f

        firms.loan_apps_head[f] = h + 1
        firms.loan_apps_targets[row, col] = -1


def banks_provide_loans(
    firms: FirmLoan,
    banks: BankProvideLoan,
    *,
    r_bar: float,
) -> None:
    """
    Process queued applications **in‑place**:

        • grant up to available credit_supply
        • contractual rate  = r̄ · (1 + frag_i)   (frag placeholder = 0)
        • reduce firm's credit_demand and bank's credit_supply
    """
    total_loans = 0.0
    for k in np.where(banks.credit_supply > 0.0)[0]:
        n_recv = banks.recv_apps_head[k] + 1
        if n_recv <= 0:
            continue

        firms_ = banks.recv_apps[k, :n_recv]
        firms_ = firms_[firms_ >= 0]

        for f in firms_:
            if banks.credit_supply[k] <= 0.0:
                break

            # to ChatGPT: shouldn't contractual rate be calculated here based on firm's fincancial fragility?
            amount = min(firms.credit_demand[f], banks.credit_supply[k])
            if amount <= 0.0:
                continue

            # update balances
            # to ChatGPT: shouldn't the loan data (along with the contractual rate) be kept somewhere?
            firms.credit_demand[f] -= amount
            banks.credit_supply[k] -= amount
            total_loans += amount

        # flush queue
        banks.recv_apps_head[k] = -1
        banks.recv_apps[k, :n_recv] = -1

    log.debug("banks_provide_loans: total_L=%.2f", total_loans)