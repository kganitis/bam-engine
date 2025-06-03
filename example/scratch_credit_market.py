from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.components import Borrower, Employer, Lender, LoanBook, Worker
from bamengine.helpers import select_top_k_indices_sorted
from helpers.factories import mock_borrower, mock_lender, mock_loanbook, mock_employer

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

CAP_FRAG = 1.0e6  # fragility cap when net worth is zero


def _mini_state(
    *,
    n_borrowers: int = 6,
    n_lenders: int = 3,
    H: int = 2,
    seed: int = 7,
) -> tuple[Employer, Borrower, Lender, LoanBook, Generator, int]:
    """
    Build minimal Borrower, Lender, and empty LoanBook components plus an RNG.

    * Borrowers: wage_bill > net_worth for the first 3 borrowers so they demand credit.
    * Lenders:   non-zero credit_supply, distinct interest rates.
    """
    rng = default_rng(seed)

    emp = mock_employer(
        n=n_borrowers,
        current_labor=np.array([100, 20, 15, 10, 5, 1], dtype=np.int64),
        wage_offer=np.ones(n_borrowers),
        wage_bill=np.array([100.0, 20.0, 15.0, 10.0, 5.0, 1.0]),
        total_funds=np.full(n_borrowers, 10.0),
    )
    bor = mock_borrower(
        n=n_borrowers,
        queue_h=H,
        wage_bill=emp.wage_bill,
        net_worth=emp.total_funds.copy(),
        total_funds=emp.total_funds,
    )
    lend = mock_lender(
        n=n_lenders,
        queue_h=H,
        equity_base=np.array([0.1, 1.0, 10.0])
    )
    ledger = mock_loanbook()

    return emp, bor, lend, ledger, rng, H


emp, bor, lend, ledger, rng, H = _mini_state()


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
            f"banks_decide_interest_rate (r_bar={r_bar:.4f}, h_phi={h_phi:.4f})\n"
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
    lenders = np.where(lend.credit_supply > 0)[0]  # bank ids
    borrowers = np.where(bor.credit_demand > 0.0)[0]  # firm ids
    if borrowers.size == 0:
        bor.loan_apps_head.fill(-1)
        return

    H_eff = min(max_H, lenders.size)
    log.debug(f"  Effective applications per firm (H_eff): {H_eff}")
    sample = np.empty((borrowers.size, H_eff), dtype=np.int64)
    for row, w in enumerate(borrowers):
        sample[row] = rng.choice(lenders, size=H_eff, replace=False)
    if log.isEnabledFor(logging.DEBUG) and unemp.size > 0:
        log.debug(
            f"  Initial random banks sample (first 5 firms, if any):\n" f"{sample[:5]}"
        )

    topk = select_top_k_indices_sorted(lend.interest_rate[sample], k=H_eff)
    sorted_sample = np.take_along_axis(sample, topk, axis=1)
    if log.isEnabledFor(logging.DEBUG) and borrowers.size > 0:
        log.debug(
            f"  Sorted bank sample by interest (first 5 firms, if any):\n"
            f"{sorted_sample[:5]}"
        )

    # flush vectors
    bor.loan_apps_targets.fill(-1)
    bor.loan_apps_head.fill(-1)

    stride = max_H
    for k, f in enumerate(borrowers):
        bor.loan_apps_targets[f, :stride] = sorted_sample[k]
        bor.loan_apps_head[f] = f * stride  # start of that row


banks_decide_credit_supply(lend, v=0.1)
banks_decide_interest_rate(lend, r_bar=0.02, h_phi=0.10, rng=rng)
firms_decide_credit_demand(bor)
firms_calc_credit_metrics(bor)
