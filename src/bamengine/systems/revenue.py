# src/bamengine/systems/revenue.py
"""
Event-6 – Revenue, debt-service & dividends
Vectorised, allocation-free hot path.
"""
from __future__ import annotations

import logging

import numpy as np

from bamengine.components import Borrower, Lender, LoanBook, Producer

_EPS = 1.0e-9

log = logging.getLogger("bamengine")


# ------------------------------------------------------------------ #
# 1.  Firms collect revenues & compute gross profit                  #
# ------------------------------------------------------------------ #
def firms_collect_revenue(prod: Producer, bor: Borrower) -> None:
    """
    revenue_i      = p_i · (Y_i − S_i)
    gross_profit_i = revenue_i − W_i
    cash           += revenue
    """
    quantity_sold = prod.production - prod.inventory
    revenue = prod.price * quantity_sold

    log.info(
        f"Total Revenue: {revenue.sum():,.2f}, "
        f"Total Wage Bill: {bor.wage_bill.sum():,.2f}"
    )

    np.add(bor.total_funds, revenue, out=bor.total_funds)

    bor.gross_profit[:] = revenue - bor.wage_bill
    log.info(f"Total Gross Profit for the economy: {bor.gross_profit.sum():,.2f}")
    # log.debug(
    #     f"  Gross Profits per firm:\n{np.array2string(bor.gross_profit, precision=2)}"
    # )


# ------------------------------------------------------------------ #
# 2.  Validate debt commitments                                      #
# ------------------------------------------------------------------ #
def firms_validate_debt_commitments(
    bor: Borrower,
    lend: Lender,
    lb: LoanBook,
) -> None:
    """
    If total_funds ≥ total_debt  →  full repayment (principal+interest).
    Else: proportional write-off up to net-worth.
    """
    log.info("Firms validating debt commitments...")
    n_firms = bor.total_funds.size
    debt_tot = lb.debt_per_borrower(n_firms)
    interest_tot = lb.interest_per_borrower(n_firms)
    log.info(f"Total outstanding debt to be serviced: {debt_tot.sum():,.2f}")

    repay_mask = bor.total_funds - debt_tot >= -_EPS
    unable_mask = ~repay_mask & (debt_tot > _EPS)
    log.info(
        f"{repay_mask.sum()} firms can fully service their debt; "
        f"{unable_mask.sum()} firms are at risk of default."
    )

    # ================================================================ #
    # 2.1  Full repayments                                             #
    # ================================================================ #
    repay_firms = np.where(repay_mask & (debt_tot > _EPS))[0]
    if repay_firms.size > 0:
        log.info(
            f"Processing full repayments for {repay_firms.size} firms, "
            f"totaling {debt_tot[repay_firms].sum():,.2f}."
        )
        # cash ↓ and net profit computed
        bor.total_funds[repay_firms] -= debt_tot[repay_firms]

        # TODO Break lender-side repayment into a separate system
        # aggregate per-lender payments
        row_sel = np.isin(lb.borrower[: lb.size], repay_firms)
        log.debug(f"  Aggregating {row_sel.sum()} repayments to lender equity.")
        np.add.at(
            lend.equity_base,
            lb.lender[: lb.size][row_sel],
            lb.debt[: lb.size][row_sel],
        )

        # remove fully repaid rows from ledger (cheap in-place compaction)
        keep = ~row_sel
        keep_size = int(keep.sum())
        log.debug(
            f"  Compacting loan book: removing {row_sel.sum()} repaid loans. "
            f"New size: {keep_size}."
        )
        #  TODO Make a LoanBook method for this
        for name in ("borrower", "lender", "principal", "rate", "interest", "debt"):
            arr = getattr(lb, name)
            arr[:keep_size] = arr[: lb.size][keep]
        lb.size = keep_size

    # ================================================================ #
    # 2.2  Bad-debt write-offs                                         #
    # ================================================================ #
    bad_firms = np.where(unable_mask & (debt_tot > _EPS))[0]
    if bad_firms.size > 0:
        log.info(
            f"Processing bad-debt write-offs for {unable_mask.sum()} struggling firms."
        )
        # zero out bad firms cash
        bor.total_funds[bad_firms] = 0

        # TODO Break bad debt handling into a separate system
        bor_ids = lb.borrower[: lb.size]
        bad_rows = np.isin(bor_ids, np.where(unable_mask)[0])
        if np.any(bad_rows):
            # per-row bad-debt  =  (debt_row / debt_tot_borrower) · net_worth_borrower
            d_tot_map = debt_tot[bor_ids[bad_rows]]
            frac = lb.debt[: lb.size][bad_rows] / np.maximum(d_tot_map, _EPS)
            bad_amt = frac * bor.net_worth[bor_ids[bad_rows]]
            log.info(
                f"  Total bad debt write-off impacting lender equity: "
                f"{bad_amt.sum():,.2f}."
            )
            np.subtract.at(lend.equity_base, lb.lender[: lb.size][bad_rows], bad_amt)
        else:
            log.info(
                "  No outstanding loans for the firms at risk; no write-offs needed."
            )

    # ---------------- net profit ------------------------------------- #
    bor.net_profit[:] = bor.gross_profit - interest_tot
    log.info(f"Final net profit for the economy: {bor.net_profit.sum():,.2f}")
    # log.debug(
    #     f"  Net Profits per firm:\n{np.array2string(bor.net_profit, precision=2)}"
    # )


# ------------------------------------------------------------------ #
# 3.  Payout dividends & retain earnings                             #
# ------------------------------------------------------------------ #
def firms_pay_dividends(bor: Borrower, *, delta: float) -> None:
    # TODO Separate dividends from retained profit calculation
    """
    retained_i = net_profit_i         ( ≤ 0 case)
               = net_profit_i·(1-δ)   ( > 0 case)

    • Cash ↓ by dividends
    • Net-worth **not** updated here
    """
    log.info(
        f"Firms paying out dividends (DPR δ={delta:.2f}) and retaining earnings..."
    )

    bor.retained_profit[:] = bor.net_profit  # default case
    bor.retained_profit[bor.net_profit > 0.0] *= 1.0 - delta  # ( > 0 case)

    dividends = bor.net_profit - bor.retained_profit  # non-neg
    np.subtract(bor.total_funds, dividends, out=bor.total_funds)

    log.info(
        f"Total dividends paid out: {dividends.sum():,.2f}. "
        f"Total earnings retained: {bor.retained_profit.sum():,.2f}."
    )
