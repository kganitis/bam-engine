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

log = logging.getLogger(__name__)


def firms_collect_revenue(prod: Producer, bor: Borrower) -> None:
    """
    revenue_i      = p_i · (Y_i − S_i)
    gross_profit_i = revenue_i − W_i
    cash           += revenue
    """
    log.info("--- Firms Collecting Revenue & Calculating Gross Profit ---")
    quantity_sold = prod.production - prod.inventory
    log.info(
        f"  Total quantity produced: {prod.production.sum():,.2f}, "
        f"Total inventory: {prod.inventory.sum():,.2f} -> "
        f"Total quantity sold: {quantity_sold.sum():,.2f}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Quantity Sold per firm:\n{np.array2string(quantity_sold, precision=2)}")

    revenue = prod.price * quantity_sold

    log.info(
        f"  Total Revenue: {revenue.sum():,.2f}, "
        f"Total Wage Bill: {bor.wage_bill.sum():,.2f}"
    )
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Revenue per firm:\n{np.array2string(revenue, precision=2)}")
        log.debug(
            f"  Total Funds (cash) BEFORE revenue collection:\n"
            f"{np.array2string(bor.total_funds, precision=2)}")

    np.add(bor.total_funds, revenue, out=bor.total_funds)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Total Funds (cash) AFTER revenue collection:\n"
            f"{np.array2string(bor.total_funds, precision=2)}")

    bor.gross_profit[:] = revenue - bor.wage_bill
    log.info(f"  Total Gross Profit for the economy: {bor.gross_profit.sum():,.2f}")
    log.debug(
        f"  Gross Profits per firm:\n{np.array2string(bor.gross_profit, precision=2)}"
    )


def firms_validate_debt_commitments(
        bor: Borrower,
        lend: Lender,
        lb: LoanBook,
) -> None:
    """
    If total_funds ≥ total_debt  →  full repayment (principal+interest).
    Else: proportional write-off up to net-worth
    """
    log.info("--- Firms Validating Debt Commitments ---")
    n_firms = bor.total_funds.size
    total_debt = lb.debt_per_borrower(n_firms)
    total_interest = lb.interest_per_borrower(n_firms)

    log.info(f"  Total outstanding debt (principal + interest) to be serviced: "
             f"{total_debt.sum():,.2f}")
    log.info(f"  Total interest component of debt: {total_interest.sum():,.2f}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Total Debt per firm:\n"
                  f"{np.array2string(total_debt, precision=2)}")
        log.debug(
            f"  Interest Total per firm:\n"
            f"{np.array2string(total_interest, precision=2)}")
        log.debug(
            f"  Borrower Total Funds (cash) before debt validation:\n"
            f"{np.array2string(bor.total_funds, precision=2)}")

    # Repayment condition now based on total_funds (cash on hand) vs total debt
    repay_mask = bor.total_funds - total_debt >= -_EPS
    unable_mask = ~repay_mask & (
                total_debt > _EPS)  # Firms that can't fully repay and have debt

    log.info(
        f"  {repay_mask.sum()} firms can repay their debt (total_funds >= total_debt);"
        f"  {unable_mask.sum()} firms are unable to repay (total_funds < total_debt)."
    )

    # ================================================================ #
    # 2.1  Full repayments                                             #
    # ================================================================ #
    repay_firms = np.where(repay_mask & (total_debt > _EPS))[
        0]
    if repay_firms.size > 0:
        log.info(
            f"  Processing full repayments for {repay_firms.size} firms, "
            f"totaling {total_debt[repay_firms].sum():,.2f}."
        )
        if log.isEnabledFor(logging.DEBUG):
            sample_repay_firms = repay_firms[:min(5, repay_firms.size)]
            log.debug(
                f"    Sample of repaying firms (IDs): {sample_repay_firms.tolist()}")
            for firm_idx in sample_repay_firms:
                log.debug(
                    f"      Firm {firm_idx}: "
                    f"total_funds before debt pay: {bor.total_funds[firm_idx]:.2f}, "
                    f"paying debt: {total_debt[firm_idx]:.2f}")

        # cash ↓
        bor.total_funds[repay_firms] -= total_debt[repay_firms]
        if log.isEnabledFor(logging.DEBUG):
            for firm_idx in sample_repay_firms:
                log.debug(
                    f"      Firm {firm_idx}: "
                    f"total_funds after debt pay: {bor.total_funds[firm_idx]:.2f}")

        # TODO Break lender-side repayment into a separate system
        # aggregate per-lender payments
        row_sel = np.isin(lb.borrower[: lb.size], repay_firms)
        num_loans_repaid_to_lenders = row_sel.sum()
        log.debug(
            f"  Aggregating {num_loans_repaid_to_lenders} "
            f"loan repayments to lender equity.")
        if num_loans_repaid_to_lenders > 0 and log.isEnabledFor(logging.DEBUG):
            affected_lenders_repayment = np.unique(lb.lender[: lb.size][row_sel])
            old_lender_equity_repayment = lend.equity_base[
                affected_lenders_repayment].copy()

        np.add.at(
            lend.equity_base,
            lb.lender[: lb.size][row_sel],
            lb.debt[: lb.size][row_sel],
        )
        if num_loans_repaid_to_lenders > 0 and log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Lender equity updated for "
                f"{affected_lenders_repayment.size} lenders due to repayments.")
            for i_lender, lender_idx in enumerate(
                    affected_lenders_repayment[
                    :min(5, affected_lenders_repayment.size)]):
                log.debug(
                    f"      Lender {lender_idx}: "
                    f"equity from {old_lender_equity_repayment[i_lender]:.2f} "
                    f"to {lend.equity_base[lender_idx]:.2f}")

        # remove fully repaid rows from ledger (cheap in-place compaction)
        keep = ~row_sel
        old_lb_size = lb.size
        keep_size = int(keep.sum())
        log.debug(
            f"  Compacting loan book: removing {row_sel.sum()} repaid loans. "
            f"Old size: {old_lb_size}, New size: {keep_size}."
        )
        #  TODO Make a LoanBook method for this
        for name in ("borrower", "lender", "principal", "rate", "interest", "debt"):
            arr = getattr(lb, name)
            arr[:keep_size] = arr[: lb.size][keep]
        lb.size = keep_size

    # ================================================================ #
    # 2.2  Bad-debt write-offs                                         #
    # ================================================================ #
    bad_firms = np.where(unable_mask & (total_debt > _EPS))[0]
    if bad_firms.size > 0:
        log.info(
            f"  Processing bad-debt write-offs for "
            f"{bad_firms.size} defaulting firms (unable to meet full debt obligations)."
            # Clarified from unable_mask.sum()
        )

        # Zero out cash for defaulting firms
        log.info(
            f"  Zeroing out total_funds (cash) for {bad_firms.size} defaulting firms.")
        if log.isEnabledFor(logging.DEBUG):
            sample_default_firms = bad_firms[:min(3, bad_firms.size)]
            for firm_idx in sample_default_firms:
                log.debug(
                    f"    Firm {firm_idx}: "
                    f"total_funds changing from {bor.total_funds[firm_idx]:.2f} to 0.")
        bor.total_funds[bad_firms] = 0.0

        # TODO Break bad debt handling into a separate system
        borrowers_from_lb = lb.borrower[: lb.size]
        bad_rows_in_lb_mask = np.isin(borrowers_from_lb, bad_firms)

        if np.any(bad_rows_in_lb_mask):
            log.debug(
                f"  {bad_rows_in_lb_mask.sum()} "
                f"loans in loan book belong to these defaulting firms.")

            d_tot_map = total_debt[borrowers_from_lb[bad_rows_in_lb_mask]]
            frac = (lb.debt[: lb.size][bad_rows_in_lb_mask] /
                    np.maximum(d_tot_map, _EPS))

            bad_amt_per_loan = frac * bor.net_worth[
                borrowers_from_lb[bad_rows_in_lb_mask]]

            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Calculating bad_amt per loan for write-off "
                    f"(frac * firm_net_worth):")
                sample_bad_loan_indices = np.where(bad_rows_in_lb_mask)[0][
                                          :min(5, np.sum(bad_rows_in_lb_mask))]
                for i_loan in sample_bad_loan_indices:
                    b_id = borrowers_from_lb[i_loan]
                    log.debug(
                        f"      Loan {i_loan} (Borrower {b_id}): "
                        f"loan_val={lb.debt[i_loan]:.2f}, "
                        f"borrower_total_debt_for_map="
                        f"{d_tot_map[np.where(borrowers_from_lb[bad_rows_in_lb_mask] 
                                              == b_id)[0][0]]:.2f}, "
                        f"frac={frac[np.where(borrowers_from_lb[bad_rows_in_lb_mask] 
                                              == b_id)[0][0]]:.3f}, "
                        f"borrower_net_worth={bor.net_worth[b_id]:.2f} -> "
                        f"bad_amt_for_this_loan="
                        f"{bad_amt_per_loan[
                            np.where(borrowers_from_lb[bad_rows_in_lb_mask] 
                                     == b_id)[0][0]]:.2f}")

            log.info(
                f"  Total bad debt write-off value (sum of bad_amt_per_loan) impacting lender equity: "
                f"{bad_amt_per_loan.sum():,.2f}."
            )
            affected_lenders_default = np.unique(
                lb.lender[: lb.size][bad_rows_in_lb_mask])
            old_lender_equity_default = lend.equity_base[
                affected_lenders_default].copy()

            np.subtract.at(lend.equity_base,
                           lb.lender[: lb.size][bad_rows_in_lb_mask],
                           bad_amt_per_loan)

            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Lender equity updated for "
                    f"{affected_lenders_default.size} lenders due to defaults.")
                for i_lender, lender_idx in enumerate(
                        affected_lenders_default[
                        :min(5, affected_lenders_default.size)]):
                    log.debug(
                        f"      Lender {lender_idx}: "
                        f"equity from {old_lender_equity_default[i_lender]:.2f} "
                        f"to {lend.equity_base[lender_idx]:.2f}")
            # Note: Loans from defaulting firms are NOT removed from the loan book here.
            # This means they might carry over.
            # TODO They should be removed
        else:
            log.info(
                "  No outstanding loans in the loan book for the firms identified "
                "as 'at risk of default'; no specific loan write-offs needed."
            )

    # ---------------- net profit ------------------------------------- #
    # net_profit = gross_profit - total_interest (NOT total_debt)
    log.info(
        f"Calculating Net Profit as Gross Profit - Total Interest "
        f"(Interest component only, not full debt repayment).")
    bor.net_profit[:] = bor.gross_profit - total_interest

    log.info(f"  Final net profit for the economy: {bor.net_profit.sum():,.2f}")
    log.debug(
        f"  Net Profits per firm:\n{np.array2string(bor.net_profit, precision=2)}"
    )


def firms_pay_dividends(bor: Borrower, *, delta: float) -> None:
    # TODO Separate dividends from retained profit calculation (User's TODO)
    """
    retained_i = net_profit_i         ( ≤ 0 case)
               = net_profit_i·(1-δ)   ( > 0 case)

    • Cash ↓ by dividends
    • Net-worth **not** updated here (User's note - this is a correction from previous versions)
    """
    log.info(
        f"--- Firms Paying Dividends (Payout Ratio δ for profits = {delta:.2f}) ---")

    positive_profit_mask = bor.net_profit > 0.0
    num_paying_dividends = np.sum(positive_profit_mask)
    log.info(
        f"  {num_paying_dividends} firms have positive net profit and will consider paying dividends.")

    bor.retained_profit[
    :] = bor.net_profit  # default case (all net profit is retained if not positive)
    bor.retained_profit[positive_profit_mask] *= (
                1.0 - delta)  # ( > 0 case, retain (1-delta) portion)

    dividends = bor.net_profit - bor.retained_profit  # non-neg; for profitable firms, this is net_profit * delta

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Dividends per firm:\n{np.array2string(dividends, precision=2)}")
        log.debug(
            f"  Retained Profit per firm:\n{np.array2string(bor.retained_profit, precision=2)}")
        log.debug(
            f"  Total Funds (cash) BEFORE paying dividends:\n{np.array2string(bor.total_funds, precision=2)}")

    np.subtract(bor.total_funds, dividends,
                out=bor.total_funds)  # Cash decreases by dividend amount

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Total Funds (cash) AFTER paying dividends:\n{np.array2string(bor.total_funds, precision=2)}")

    # User's existing log, good.
    log.info(
        f"  Total dividends paid out: {dividends.sum():,.2f}. "
        f"Total earnings retained: {bor.retained_profit.sum():,.2f}."
    )
    # User has correctly noted that Net-worth is NOT updated here.
    # It will be updated in bankruptcy.py with `firms_update_net_worth` which adds retained_profit.
    log.debug(
        "  Net worth is NOT directly updated in this function by retained earnings; that happens in firms_update_net_worth.")
    log.info("--- Dividend Payout & Earnings Retainment complete ---")