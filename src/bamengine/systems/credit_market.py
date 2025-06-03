# src/bamengine/systems/credit_market.py
"""
Event‑3  –  Credit‑market systems  (vectorised, no new allocations at runtime)
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.components import Borrower, Employer, Lender, LoanBook, Worker
from bamengine.helpers import select_top_k_indices_sorted

log = logging.getLogger("bamengine")

CAP_FRAG = 1.0e6  # fragility cap when net worth is zero


def banks_decide_credit_supply(lend: Lender, *, v: float) -> None:
    """
    C_k = E_k / v

    v : capital requirement coefficient
    """
    np.divide(lend.equity_base, v, out=lend.credit_supply)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"banks_decide_credit_supply:\n"
            f"equity_base={lend.equity_base}\n"
            f"max_leverage (1/v)={1/v}\n"
            f"credit_supply={lend.credit_supply}\n"
            f"Total credit supply update = {lend.credit_supply.sum():.2f} "
        )


def banks_decide_interest_rate(
    lend: Lender,
    *,
    r_bar: float,
    h_phi: float,
    rng: Generator = default_rng(),
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
    lend.interest_rate[:] = r_bar * (1.0 + shock)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"banks_decide_interest_rate: r_bar={r_bar:.4f}, h_phi={h_phi:.4f}\n"
            f"shocks={shock}\n"
            f"interest rates={lend.interest_rate}"
        )


def firms_decide_credit_demand(bor: Borrower) -> None:
    """
    B_i = max( W_i − A_i , 0 )
    """
    np.subtract(bor.wage_bill, bor.net_worth, out=bor.credit_demand)
    np.maximum(bor.credit_demand, 0.0, out=bor.credit_demand)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"firms_decide_credit_demand:\n"
                  f"credit_demand={bor.credit_demand}")


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
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"firms_calc_credit_metrics:\n"
            f"net_worth={bor.net_worth}\n"
            f"projected_fragility={frag}"
        )


def firms_prepare_loan_applications(
    bor: Borrower,
    lend: Lender,
    *,
    max_H: int,
    rng: Generator = default_rng(),
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
    topk = select_top_k_indices_sorted(lend.interest_rate[sample], k=max_H)
    sorted_sample = np.take_along_axis(sample, topk, axis=1)

    # flush vectors
    bor.loan_apps_targets.fill(-1)
    bor.loan_apps_head.fill(-1)

    stride = max_H
    for k, f in enumerate(borrowers):
        bor.loan_apps_targets[f, :stride] = sorted_sample[k]
        bor.loan_apps_head[f] = f * stride  # start of that row


def firms_send_one_loan_app(
    bor: Borrower, lend: Lender, rng: Generator = default_rng()
) -> None:
    """ """
    stride = bor.loan_apps_targets.shape[1]
    borrowers_indices = np.where(bor.credit_demand > 0.0)[0]
    rng.shuffle(borrowers_indices)

    for f in borrowers_indices:
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


def banks_provide_loans(
    bor: Borrower,
    lb: LoanBook,
    lend: Lender,
    *,
    r_bar: float,
    rng: Generator = default_rng(0),
) -> None:
    """
    Process queued applications **in‑place**:

        • contractual r_ik = r_bar * (1 + frag_i)
        • satisfy queues until each bank’s credit_supply is exhausted
        • update both firm credit-demand **and** edge-list ledger
    """
    lenders_indices = np.where(lend.credit_supply > 0.0)[0]
    rng.shuffle(lenders_indices)

    for k in lenders_indices:
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
        rate = lend.interest_rate[k] * (1.0 + frag[mask])

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"Bank {k}: granting loans to borrowers {borrowers}\n"
                f"amounts={amount}\n"
                f"rates={rate}"
            )

        # --- update ledger (vectorised append) ---------------------------------
        lb.append_loans(borrowers, k, amount, rate)

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
    rng: Generator = default_rng(),
) -> None:
    """
    If a firm’s wage-bill exceeds its cash, lay off just enough workers
    to close the funding gap – never sampling more workers than actually
    exist on the roster.

        n_fire = ceil(gap / w_i)   capped by true roster size
    """
    for i in range(emp.current_labor.size):
        gap = emp.wage_bill[i] - emp.total_funds[i]
        if gap <= 0.0:
            continue

        # real roster: might differ from bookkeeping
        workforce = np.where((wrk.employed == 1) & (wrk.employer == i))[0]
        if workforce.size == 0:  # no one to fire
            continue

        needed = int(np.ceil(gap / float(emp.wage_offer[i])))
        n_fire = min(needed, workforce.size)
        if n_fire == 0:  # numerical quirk
            continue

        victims = rng.choice(workforce, size=n_fire, replace=False)
        # log.debug(f"Firm {i}: calculated {n_fire} workers to fire.")

        # -------- worker-side updates --------------------------------
        wrk.employed[victims] = 0
        wrk.employer[victims] = -1
        wrk.employer_prev[victims] = i
        wrk.wage[victims] = 0.0
        wrk.periods_left[victims] = 0
        wrk.contract_expired[victims] = 0
        wrk.fired[victims] = 1

        emp.current_labor[i] -= n_fire
        np.multiply(emp.current_labor, emp.wage_offer, out=emp.wage_bill)

    log.debug("Firing complete")
    log.debug(f"  Current Labor after firing (L_i):\n{emp.current_labor}")
    log.debug(
        f"  Wage bills after firing:\n{np.array2string(emp.wage_bill, precision=2)}"
    )
