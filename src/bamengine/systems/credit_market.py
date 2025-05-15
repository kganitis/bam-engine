# src/bamengine/systems/credit_market.py
"""
Event‑3  –  Credit‑market systems  (vectorised, no new allocations at runtime)
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator

from bamengine.components import Borrower, Employer, Lender, LoanBook, Worker
from bamengine.typing import Float1D, Idx1D

log = logging.getLogger(__name__)

CAP_FRAG = 1.0e6  # fragility cap when net worth is zero


def banks_decide_credit_supply(lend: Lender, *, v: float) -> None:
    """
    C_k = E_k · v

    v : capital requirement coefficient
    """
    np.multiply(lend.equity_base, v, out=lend.credit_supply)


def banks_decide_interest_rate(
    lend: Lender,
    *,
    r_bar: float,
    h_phi: float,
    rng: Generator,
) -> None:
    """
    Nominal interest rate rule:

    r_k = r̄ · (1 + U(0, h_φ))
    """
    shape = lend.interest_rate.shape

    # permanent scratch
    shock = lend.opex_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64)
        lend.opex_shock = shock

    # fill buffer in-place
    shock[:] = rng.uniform(0.0, h_phi, size=shape)

    # core rule
    lend.interest_rate[:] = r_bar * (1.0 + shock)


def firms_decide_credit_demand(bor: Borrower) -> None:
    """
    B_i = max( W_i − A_i , 0 )
    """
    np.subtract(bor.wage_bill, bor.net_worth, out=bor.credit_demand)
    np.maximum(bor.credit_demand, 0.0, out=bor.credit_demand)


def firms_calc_credit_metrics(bor: Borrower) -> None:
    """
    projected_fragility[i] = μ_i · B_i / A_i
    """
    shape = bor.net_worth.shape

    frag = bor.projected_fragility
    if frag is None or frag.shape != shape:
        frag = np.empty(shape, dtype=np.float64)
        bor.projected_fragility = frag

    # frag ←  B_i / A_i  (safe divide)
    np.divide(
        bor.credit_demand,
        bor.net_worth,
        out=frag,
        where=bor.net_worth > 0.0,
    )
    frag[bor.net_worth == 0.0] = CAP_FRAG

    # frag *= μ_i
    np.multiply(frag, bor.rnd_intensity, out=frag)


def _topk_lowest_rate(values: Float1D, k: int) -> Idx1D:
    """Return indices of the *k* cheapest elements (unsorted)."""
    if k >= values.shape[-1]:
        return np.argpartition(values, kth=0, axis=-1)
    part = np.argpartition(values, kth=k - 1, axis=-1)
    return part[..., :k]


def firms_prepare_loan_applications(
    bor: Borrower,
    lend: Lender,
    *,
    max_H: int,
    rng: Generator,
) -> None:
    """
    * draws H random banks per borrower
    * keeps the H *cheapest* (lowest r_k) via partial sort
    * writes indices into ``loan_apps_targets`` and resets ``loan_apps_head``
    """
    n_banks = lend.interest_rate.size
    borrowers = np.where(bor.credit_demand > 0.0)[0]
    if borrowers.size == 0:
        bor.loan_apps_head.fill(-1)
        return

    sample = rng.integers(0, n_banks, size=(borrowers.size, max_H), dtype=np.int64)
    topk = _topk_lowest_rate(lend.interest_rate[sample], k=max_H)
    sorted_sample = np.take_along_axis(sample, topk, axis=1)

    # flush vectors
    bor.loan_apps_targets.fill(-1)
    bor.loan_apps_head.fill(-1)

    stride = max_H
    for k, f in enumerate(borrowers):
        bor.loan_apps_targets[f, :stride] = sorted_sample[k]
        bor.loan_apps_head[f] = f * stride  # start of that row


def firms_send_one_loan_app(bor: Borrower, lend: Lender) -> None:
    """ """
    stride = bor.loan_apps_targets.shape[1]

    for f in np.where(bor.credit_demand > 0.0)[0]:
        h = bor.loan_apps_head[f]
        if h < 0:
            continue
        row, col = divmod(h, stride)
        bank_idx = bor.loan_apps_targets[row, col]
        if bank_idx < 0:
            bor.loan_apps_head[f] = -1
            continue

        # bounded queue
        ptr = lend.recv_apps_head[bank_idx] + 1
        if ptr >= lend.recv_apps.shape[1]:
            continue
        lend.recv_apps_head[bank_idx] = ptr
        lend.recv_apps[bank_idx, ptr] = f

        bor.loan_apps_head[f] = h + 1
        bor.loan_apps_targets[row, col] = -1


def _ensure_capacity(book: LoanBook, extra: int) -> None:

    needed = book.size + extra
    if needed <= book.capacity:
        return

    new_cap = max(book.capacity * 2, needed, 128)

    for name in ("borrower", "lender", "principal", "rate", "interest", "debt"):
        arr = getattr(book, name)
        new_arr = np.resize(arr, new_cap)  # returns *new* ndarray; O(1) amortized;
        # large simulations should pre-allocate capacity = n_borrowers * H
        setattr(book, name, new_arr)

    book.capacity = new_cap
    assert all(
        getattr(book, n).size == new_cap
        for n in ("borrower", "lender", "principal", "rate", "interest", "debt")
    )


def _append_loans(
    ledger: LoanBook,
    bor: Idx1D,
    lender_idx: int,
    amount: Float1D,
    rate: Float1D,
) -> None:
    _ensure_capacity(ledger, amount.size)
    start, stop = ledger.size, ledger.size + amount.size

    ledger.borrower[start:stop] = bor
    ledger.lender[start:stop] = lender_idx  # ← scalar broadcast
    ledger.principal[start:stop] = amount
    ledger.rate[start:stop] = rate
    ledger.interest[start:stop] = amount * rate
    ledger.debt[start:stop] = amount * (1.0 + rate)
    ledger.size = stop


def banks_provide_loans(
    bor: Borrower,
    ledger: LoanBook,
    lend: Lender,
    *,
    r_bar: float,
) -> None:
    """
    Process queued applications **in‑place**:

        • contractual r_ik = r_bar * (1 + frag_i)
        • satisfy queues until each bank’s credit_supply is exhausted
        • update both firm credit-demand **and** edge-list ledger
    """
    for k in np.where(lend.credit_supply > 0.0)[0]:
        n_recv = lend.recv_apps_head[k] + 1
        if n_recv <= 0:
            continue

        queue = lend.recv_apps[k, :n_recv]
        queue = queue[queue >= 0]  # compact valid requests

        if queue.size == 0:
            # flush empty / invalid queues immediately
            lend.recv_apps_head[k] = -1
            lend.recv_apps[k, :n_recv] = -1

        # --- gather data ------------------------------------------------
        cd = bor.credit_demand[queue]
        frag = bor.projected_fragility[queue]
        max_grant = np.minimum(cd, lend.credit_supply[k])

        # amount actually granted (zero once the pot is empty)
        cum = np.cumsum(max_grant, dtype=np.float64)
        cut = cum > lend.credit_supply[k]
        if cut.any():
            first_exceed = cut.argmax()
            max_grant[first_exceed] -= cum[first_exceed] - lend.credit_supply[k]
            max_grant[first_exceed + 1 :] = 0.0

        mask = max_grant > 0.0
        if not mask.any():
            continue

        amount = max_grant[mask]
        borrowers = queue[mask]
        rate = r_bar * (1.0 + frag[mask])

        # --- update ledger (vectorised append) ---------------------------------
        _append_loans(ledger, borrowers, k, amount, rate)

        # --- balances ---------------------------------------------------
        bor.total_funds[borrowers] += amount
        bor.credit_demand[borrowers] -= amount
        lend.credit_supply[k] -= amount.sum()
        assert (bor.credit_demand >= -1e-12).all(), "negative credit_demand"

        # reset queue
        lend.recv_apps_head[k] = -1
        lend.recv_apps[k, :n_recv] = -1


def firms_fire_workers(
    emp: Employer,
    wrk: Worker,
    *,
    rng: Generator,
) -> None:
    """
    If `wage_bill[i]` exceeds `total_funds[i]` the firm lays off just enough
    workers to close the gap:

        n_fire = ceil( gap / wage[i] )   but never more than current labour.

    Workers are picked **uniformly at random** from the current workforce.
    """
    n_firms = emp.current_labor.size

    for i in range(n_firms):
        gap = emp.wage_bill[i] - emp.total_funds[i]
        if gap <= 0.0 or emp.current_labor[i] == 0:
            continue

        # how many heads must roll?
        n_fire = int(
            min(
                emp.current_labor[i],
                np.ceil(gap / emp.wage_offer[i]),
            )
        )
        if n_fire == 0:  # float quirks
            continue

        # choose victims uniformly
        workforce = np.where((wrk.employed == 1) & (wrk.employer == i))[0]
        if workforce.size == 0:
            continue
        victims = rng.choice(workforce, size=n_fire, replace=False)

        # --- worker-side -------------------------------------------
        wrk.employed[victims] = 0
        wrk.fired[victims] = 1
        wrk.contract_expired[victims] = 0  # explicit: this was a firing

        # --- firm-side ---------------------------------------------
        emp.current_labor[i] -= n_fire
        emp.wage_bill[i] -= n_fire * emp.wage_offer[i]
