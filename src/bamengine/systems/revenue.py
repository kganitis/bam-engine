# src/bamengine/systems/revenue.py
"""
Event-6 – Revenue, debt-service & dividends
Vectorised, allocation-free hot path.
"""
from __future__ import annotations

import numpy as np

from bamengine.components import Borrower, Lender, LoanBook, Producer

_EPS = 1.0e-12


# ------------------------------------------------------------------ #
# 1.  Firms collect revenues & compute gross profit                  #
# ------------------------------------------------------------------ #
def firms_collect_revenue(prod: Producer, bor: Borrower) -> None:
    """
    revenue_i      = p_i · (Y_i − S_i)
    gross_profit_i = revenue_i − W_i
    cash           += revenue
    """
    sold = prod.production - prod.inventory
    revenue = prod.price * sold

    np.add(bor.total_funds, revenue, out=bor.total_funds)

    bor.gross_profit[:] = revenue - bor.wage_bill


# ------------------------------------------------------------------ #
# 2.  Validate debt commitments                                      #
# ------------------------------------------------------------------ #
def firms_validate_debt_commitments(
    bor: Borrower,
    lend: Lender,
    lb: LoanBook,
) -> None:
    """
    If gross_profit ≥ total_debt  →  full repayment (principal+interest).
    Else: proportional write-off up to net-worth.
    """
    n_firms = bor.total_funds.size
    debt_tot = lb.debt_per_borrower(n_firms)  # Σ_j debt_ij

    repay_mask = bor.gross_profit >= debt_tot - _EPS
    unable_mask = ~repay_mask & (debt_tot > _EPS)

    # ================================================================ #
    # 2.1  Full repayments                                             #
    # ================================================================ #
    repay_firms = np.where(repay_mask & (debt_tot > 0.0))[0]
    if repay_firms.size:
        # cash ↓ and net profit computed
        bor.total_funds[repay_firms] -= debt_tot[repay_firms]

        # aggregate per-lender payments
        row_sel = np.isin(lb.borrower[: lb.size], repay_firms)
        np.add.at(
            lend.equity_base,
            lb.lender[: lb.size][row_sel],
            lb.debt[: lb.size][row_sel],
        )

        # remove fully repaid rows from ledger (cheap in-place compaction)
        keep = ~row_sel
        keep_size = int(keep.sum())
        for name in ("borrower", "lender", "principal", "rate", "interest", "debt"):
            arr = getattr(lb, name)
            arr[:keep_size] = arr[: lb.size][keep]
        lb.size = keep_size

    # ================================================================ #
    # 2.2  Bad-debt write-offs                                         #
    # ================================================================ #
    if unable_mask.any() and lb.size:
        bor_ids = lb.borrower[: lb.size]
        bad_rows = np.isin(bor_ids, np.where(unable_mask)[0])

        # per-row bad-debt  =  (debt_row / debt_tot_borrower) · net_worth_borrower
        d_tot_map = debt_tot[bor_ids[bad_rows]]
        frac = lb.debt[: lb.size][bad_rows] / np.maximum(d_tot_map, _EPS)
        bad_amt = frac * bor.net_worth[bor_ids[bad_rows]]

        np.subtract.at(lend.equity_base, lb.lender[: lb.size][bad_rows], bad_amt)

    # ---------------- net profit ------------------------------------- #
    bor.net_profit[:] = bor.gross_profit - debt_tot


# ------------------------------------------------------------------ #
# 3.  Payout dividends & retain earnings                             #
# ------------------------------------------------------------------ #
def firms_pay_dividends(bor: Borrower, *, delta: float) -> None:
    """
    retained_i = net_profit_i         ( ≤ 0 case)
               = net_profit_i·(1-δ)   ( > 0 case)

    • Cash ↓ by dividends
    • Net-worth ↑ by retained earnings
    """
    pos = bor.net_profit > 0.0
    bor.retained_profit[:] = bor.net_profit  # default (≤0)
    bor.retained_profit[pos] *= 1.0 - delta

    dividends = bor.net_profit - bor.retained_profit  # non-neg

    np.subtract(bor.total_funds, dividends, out=bor.total_funds)
    np.add(bor.net_worth, bor.retained_profit, out=bor.net_worth)
