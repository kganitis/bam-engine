# src/bamengine/systems/credit_market.py
"""
Event‑3  –  Credit‑market systems  (vectorised, no new allocations at runtime)
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator

from bamengine.components.bank_credit import (
    BankCreditSupply,
    BankInterestRate,
    BankProvideLoan,
    BankReceiveLoanApplication,
)
from bamengine.components.credit import LoanBook
from bamengine.components.firm_credit import (
    FirmCreditDemand,
    FirmLoan,
    FirmLoanApplication, FirmFinancialFragility,
)
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
        banks.credit_shock = shock

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


def firms_calc_financial_fragility(firms: FirmFinancialFragility) -> None:
    """Populate `projected_leverage` and `projected_fragility` in-place."""
    shape = firms.net_worth.shape

    # ensure / reuse leverage buffer
    plv = firms.projected_leverage
    if plv is None or plv.shape != shape:
        plv = np.empty(shape, dtype=np.float64)
        firms.projected_leverage = plv

    # avoid division by zero via `where=`
    np.divide(
        firms.credit_demand, firms.net_worth,
        out=plv,
        where=firms.net_worth > 0.0,
    )

    # ensure / reuse fragility buffer
    frag = firms.projected_fragility
    if frag is None or frag.shape != shape:
        frag = np.empty(shape, dtype=np.float64)
        firms.projected_fragility = frag

    np.multiply(plv, firms.rnd_intensity_mu, out=frag)


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
    firms: FirmLoanApplication, banks: BankReceiveLoanApplication
) -> None:
    """ """
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


def _ensure_capacity(book: LoanBook, extra: int) -> None:
    if book.size + extra <= book.capacity:
        return

    new_cap = max(book.capacity * 2, book.size + extra, 32)

    for name in ("firm", "bank", "principal", "rate", "interest", "debt"):
        arr = getattr(book, name)
        new_arr = np.resize(arr, new_cap)  # returns *new* ndarray; O(1) amortized
        setattr(book, name, new_arr)

    book.capacity = new_cap



def _append_loan(book: LoanBook,
                 i: int, k: int,
                 amount: float, r: float) -> None:
    _ensure_capacity(book, 1)
    j = book.size
    book.firm[j]      = i
    book.bank[j]      = k
    book.principal[j] = amount
    book.rate[j]      = r
    book.interest[j]  = amount * r
    book.debt[j]      = amount * (1.0 + r)
    book.size += 1


def banks_provide_loans(
    firms: FirmLoan,
    ledger: LoanBook,
    banks: BankProvideLoan,
    *,
    r_bar: float,
) -> None:
    """
    Process queued applications **in‑place**:

        • contractual r_ik = r_bar * (1 + frag_i)
        • satisfy queues until each bank’s credit_supply is exhausted
        • update both firm credit-demand **and** edge-list ledger
    """
    for k in np.where(banks.credit_supply > 0.0)[0]:
        n_recv = banks.recv_apps_head[k] + 1
        if n_recv <= 0:
            continue
        queue = banks.recv_apps[k, :n_recv]
        queue = queue[queue >= 0]

        for f in queue:
            if banks.credit_supply[k] <= 0.0:
                break
            amount = min(firms.credit_demand[f], banks.credit_supply[k])
            if amount <= 0.0:
                continue

            rate = r_bar * (1.0 + firms.projected_fragility[f])

            # ----- update ledger ------------------------------------
            _append_loan(ledger, f, k, amount, rate)

            # ----- balances --------------------------------------------
            firms.credit_demand[f] -= amount
            banks.credit_supply[k] -= amount

        banks.recv_apps_head[k] = -1
        banks.recv_apps[k, :n_recv] = -1
