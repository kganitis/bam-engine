"""
System functions for revenue phase events.

This module contains the internal implementation functions for revenue events.
Event classes wrap these functions and provide the primary documentation.

See Also
--------
bamengine.events.revenue : Event classes (primary documentation source)
"""

from __future__ import annotations

import numpy as np

from bamengine import logging
from bamengine.relationships import LoanBook
from bamengine.roles import Borrower, Consumer, Lender, Producer
from bamengine.utils import EPS

log = logging.getLogger(__name__)


def firms_collect_revenue(prod: Producer, bor: Borrower) -> None:
    """
    Collect revenue from sales and calculate gross profit.

    See Also
    --------
    bamengine.events.revenue.FirmsCollectRevenue : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Collecting Revenue & Calculating Gross Profit ---")

    # calculate quantities sold
    quantity_sold = prod.production - prod.inventory
    total_quantity_sold = quantity_sold.sum()

    if info_enabled:
        log.info(
            f"  Total quantity produced: {prod.production.sum():,.2f}, "
            f"Total inventory: {prod.inventory.sum():,.2f} -> "
            f"Total quantity sold: {total_quantity_sold:,.2f}"
        )

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Quantity sold per firm: {quantity_sold}")

    # calculate revenue
    revenue = prod.price * quantity_sold
    total_revenue = revenue.sum()
    total_wage_bill = bor.wage_bill.sum()

    if info_enabled:
        log.info(
            f"  Total revenue: {total_revenue:,.2f}, "
            f"Total wage bill: {total_wage_bill:,.2f}"
        )

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Total funds (cash) before revenue collection: {bor.total_funds}")
        log.debug(f"  Revenue per firm: {revenue}")

    # update firm cash accounts
    np.add(bor.total_funds, revenue, out=bor.total_funds)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Total funds (cash) after revenue collection: {bor.total_funds}")

    # calculate gross profit
    bor.gross_profit[:] = revenue - bor.wage_bill
    total_gross_profit = bor.gross_profit.sum()

    if info_enabled:
        log.info(f"  Total gross profit for the economy: {total_gross_profit:,.2f}")

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Gross profit per firm: {bor.gross_profit}")

    if info_enabled:
        log.info("--- Firms Collecting Revenue & Calculating Gross Profit complete ---")


def firms_validate_debt_commitments(
    bor: Borrower,
    lend: Lender,
    lb: LoanBook,
) -> None:
    """
    Repay debts or write off if insufficient funds.

    See Also
    --------
    bamengine.events.revenue.FirmsValidateDebtCommitments : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Validating Debt Commitments ---")

    # calculate debt obligations
    n_firms = bor.total_funds.size
    total_debt = lb.debt_per_borrower(n_firms)
    total_interest = lb.interest_per_borrower(n_firms)

    total_outstanding_debt = total_debt.sum()
    total_interest_component = total_interest.sum()

    if info_enabled:
        log.info(
            f"  Total outstanding debt (principal + interest) to be serviced: "
            f"{total_outstanding_debt:,.2f}"
        )
        log.info(f"  Total interest component of debt: {total_interest_component:,.2f}")

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Total debt per firm: {total_debt}")
        log.debug(f"  Interest total per firm: {total_interest}")
        log.debug(
            f"  Borrower total funds (cash) before debt validation: {bor.total_funds}"
        )

    # classify firms by repayment ability
    repay_mask = bor.total_funds - total_debt >= -EPS
    unable_mask = ~repay_mask & (total_debt > EPS)

    num_can_repay = repay_mask.sum()
    num_unable_repay = unable_mask.sum()

    if info_enabled:
        log.info(
            f"  {num_can_repay} firms can repay their debt (total_funds >= total_debt); "
            f"{num_unable_repay} firms are unable to repay (total_funds < total_debt)."
        )

    # process full repayments
    repay_firms = np.where(repay_mask & (total_debt > EPS))[0]
    if repay_firms.size > 0:
        if info_enabled:
            log.info(
                f"  Processing full repayments for {repay_firms.size} firms, "
                f"totaling {total_debt[repay_firms].sum():,.2f}."
            )

        if log.isEnabledFor(logging.DEBUG):
            sample_repay_firms = repay_firms[: min(5, repay_firms.size)]
            log.debug(
                f"    Sample of repaying firms (IDs): {sample_repay_firms.tolist()}"
            )

            for firm_idx in sample_repay_firms:
                log.debug(
                    f"      Firm {firm_idx}: "
                    f"total_funds before debt pay: {bor.total_funds[firm_idx]:.2f}, "
                    f"paying debt: {total_debt[firm_idx]:.2f}"
                )

        # debit firm cash accounts
        bor.total_funds[repay_firms] -= total_debt[repay_firms]

        if log.isEnabledFor(logging.DEBUG):
            # noinspection PyUnboundLocalVariable
            for firm_idx in sample_repay_firms:
                log.debug(
                    f"      Firm {firm_idx}: "
                    f"total_funds after debt pay: {bor.total_funds[firm_idx]:.2f}"
                )

        # aggregate per-lender payments
        row_sel = np.isin(lb.borrower[: lb.size], repay_firms)
        num_loans_repaid = row_sel.sum()

        log.debug(f"  Aggregating {num_loans_repaid} loan repayments to lender equity.")

        if num_loans_repaid > 0 and log.isEnabledFor(logging.DEBUG):
            affected_lenders_repayment = np.unique(lb.lender[: lb.size][row_sel])
            old_lender_equity_repayment = lend.equity_base[
                affected_lenders_repayment
            ].copy()

        # Credit lender equity with interest payments
        np.add.at(
            lend.equity_base,
            lb.lender[: lb.size][row_sel],
            lb.interest[: lb.size][row_sel],
        )

        if num_loans_repaid > 0 and log.isEnabledFor(logging.DEBUG):
            # noinspection PyUnboundLocalVariable
            log.debug(
                f"    Lender equity updated for "
                f"{affected_lenders_repayment.size} lenders due to repayments."
            )

            for i_lender, lender_idx in enumerate(
                affected_lenders_repayment[: min(5, affected_lenders_repayment.size)]
            ):
                # noinspection PyUnboundLocalVariable
                log.debug(
                    f"      Lender {lender_idx}: "
                    f"equity from {old_lender_equity_repayment[i_lender]:.2f} "
                    f"to {lend.equity_base[lender_idx]:.2f}"
                )

        # remove repaid loans from loan book
        removed = lb.drop_rows(row_sel)
        log.debug(
            f"  Compacting loan book: removed {removed} repaid loans. "
            f"New size={lb.size}"
        )

    # process bad-debt write-offs
    bad_firms = np.where(unable_mask & (total_debt > EPS))[0]
    if bad_firms.size > 0:
        if info_enabled:
            log.info(
                f"  Processing bad-debt write-offs for {bad_firms.size} defaulting firms."
            )

            # zero out cash for defaulting firms
            log.info(
                f"  Zeroing out total_funds (cash) for {bad_firms.size} defaulting firms."
            )

        if log.isEnabledFor(logging.DEBUG):
            sample_default_firms = bad_firms[: min(5, bad_firms.size)]
            for firm_idx in sample_default_firms:
                log.debug(
                    f"    Firm {firm_idx}: "
                    f"total_funds changing from {bor.total_funds[firm_idx]:.1f} to 0.0"
                )

        bor.total_funds[bad_firms] = 0.0

        # process loan book write-offs
        borrowers_from_lb = lb.borrower[: lb.size]
        bad_rows_in_lb_mask = np.isin(borrowers_from_lb, bad_firms)

        if np.any(bad_rows_in_lb_mask):
            num_bad_loans = bad_rows_in_lb_mask.sum()
            log.debug(
                f"  {num_bad_loans} loans in loanbook belong to these defaulting firms."
            )

            # calculate proportional write-offs
            # per-row bad-debt = (debt_row / debt_tot_borrower) · net_worth_borrower
            #
            # When a firm defaults on its debt, the bank must write down its assets.
            # The value of this write-down (the bad debt) is calculated as a share
            # of the defaulting firm's remaining equity (net worth).

            # Calculate the lender's share (`frac`) of the defaulting firm's total debt.
            # For a given loan, this is: (this loan's value) / (firm's total debt).
            # This determines the proportion of the equity-based loss
            # this bank absorbs for this loan.
            d_tot_map = total_debt[borrowers_from_lb[bad_rows_in_lb_mask]]
            frac = lb.debt[: lb.size][bad_rows_in_lb_mask] / np.maximum(d_tot_map, EPS)

            # Calculate the bad debt amount for this loan.
            # This is the bank's `frac` multiplied by the firm's net worth.
            # Bad debt is bounded: 0 <= bad_amt <= loan_value
            # - Floor at 0: negative net_worth means bank loses the full loan
            # - Cap at loan: bank can't lose more than it lent
            loan_values = lb.debt[: lb.size][bad_rows_in_lb_mask]
            bad_amt_per_loan = np.clip(
                frac * bor.net_worth[borrowers_from_lb[bad_rows_in_lb_mask]],
                0.0,
                loan_values,
            )

            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "    Calculating bad_amt per loan for write-off "
                    "(capped at loan value):"
                )
                sample_bad_loan_indices = np.where(bad_rows_in_lb_mask)[0][
                    : min(5, int(np.sum(bad_rows_in_lb_mask)))
                ]

                for i_loan in sample_bad_loan_indices:
                    b_id = borrowers_from_lb[i_loan]
                    matching_indices = np.where(
                        borrowers_from_lb[bad_rows_in_lb_mask] == b_id
                    )[0]
                    if len(matching_indices) > 0:
                        idx = matching_indices[0]
                        log.debug(
                            f"      Loan {i_loan} (Borrower {b_id}): "
                            f"loan_val={lb.debt[i_loan]:.2f}, "
                            f"borrower_total_debt_for_map={d_tot_map[idx]:.2f}, "
                            f"frac={frac[idx]:.3f}, "
                            f"borrower_net_worth={bor.net_worth[b_id]:.2f} -> "
                            f"bad_amt_for_this_loan={bad_amt_per_loan[idx]:.2f}"
                        )

            total_bad_debt_writeoff = bad_amt_per_loan.sum()
            if info_enabled:
                log.info(
                    f"  Total bad debt write-off value (sum of bad_amt_per_loan) "
                    f"impacting lender equity: {total_bad_debt_writeoff:,.2f}."
                )

            # update lender equity
            affected_lenders_default = np.unique(
                lb.lender[: lb.size][bad_rows_in_lb_mask]
            )

            if log.isEnabledFor(logging.DEBUG):
                old_lender_equity_default = lend.equity_base[
                    affected_lenders_default
                ].copy()

            # Debit lender equity by bad debt amounts
            np.subtract.at(
                lend.equity_base,
                lb.lender[: lb.size][bad_rows_in_lb_mask],
                bad_amt_per_loan,
            )

            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Lender equity updated for "
                    f"{affected_lenders_default.size} lenders due to defaults."
                )

                for i_lender, lender_idx in enumerate(
                    affected_lenders_default[: min(5, affected_lenders_default.size)]
                ):
                    # noinspection PyUnboundLocalVariable
                    log.debug(
                        f"      Lender {lender_idx}: "
                        f"equity from {old_lender_equity_default[i_lender]:.2f} "
                        f"to {lend.equity_base[lender_idx]:.2f}"
                    )

            # Remove defaulted loans from loan book (debt has been written off)
            removed = lb.drop_rows(bad_rows_in_lb_mask)
            log.debug(
                f"  Compacting loan book: removed {removed} defaulted loans. "
                f"New size={lb.size}"
            )
        else:
            if info_enabled:
                log.info(
                    "  No outstanding loans in the loan book for the firms identified "
                    "as 'at risk of default'; no specific loan write-offs needed."
                )

    # calculate net profit
    if info_enabled:
        log.info("  Calculating net profit (gross profit - total interest)")
    bor.net_profit[:] = bor.gross_profit - total_interest
    total_net_profit = bor.net_profit.sum()

    if info_enabled:
        log.info(f"  Final net profit for the economy: {total_net_profit:,.2f}")

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Net profits per firm: {bor.net_profit}")

    if info_enabled:
        log.info("--- Firms Validating Debt Commitments complete ---")


def firms_pay_dividends(bor: Borrower, cons: Consumer, *, delta: float) -> None:
    """
    Distribute dividends from positive profits and credit households.

    See Also
    --------
    bamengine.events.revenue.FirmsPayDividends : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info(
            f"--- Firms Paying Dividends (Payout Ratio δ for profits = {delta:.2f}) ---"
        )

    # identify firms with positive profits
    positive_profit_mask = bor.net_profit > 0.0
    num_paying_dividends = np.sum(positive_profit_mask)

    if info_enabled:
        log.info(
            f"  {num_paying_dividends} firms have net profit and will pay dividends."
        )

    # calculate retained profits and dividends
    # Default case: all net profit is retained if not positive
    bor.retained_profit[:] = bor.net_profit

    # For positive profits: retain (1-delta) portion
    bor.retained_profit[positive_profit_mask] *= 1.0 - delta
    dividends = bor.net_profit - bor.retained_profit  # net_profit * delta

    total_dividends = dividends.sum()
    total_retained = bor.retained_profit.sum()

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Dividends per firm: {dividends}")
        log.debug(f"  Retained profit per firm: {bor.retained_profit}")
        log.debug(f"  Total funds (cash) before paying dividends: {bor.total_funds}")

    # debit firm cash accounts by dividend amount
    np.subtract(bor.total_funds, dividends, out=bor.total_funds)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Total funds (cash) after paying dividends: {bor.total_funds}")

    if info_enabled:
        log.info(
            f"  Total dividends paid out: {total_dividends:,.2f}. "
            f"Total earnings retained: {total_retained:,.2f}."
        )

    # distribute dividends to households (equal distribution)
    # This is a modeling simplification: rather than introducing a separate
    # "capitalist" role, dividends flow to all households equally. This preserves
    # stock-flow consistency while maintaining model parsimony.
    # TODO: Implement a Capitalist role with firm ownership relationships to allow
    #  dividends to flow to specific households (shareholders) rather than equally.
    n_households = cons.savings.size
    dividend_per_household = total_dividends / n_households

    if info_enabled:
        log.info(
            f"  Distributing dividends to {n_households} households "
            f"({dividend_per_household:,.4f} per household)"
        )

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Household savings before dividends: {cons.savings}")

    # credit household savings with dividends
    np.add(cons.savings, dividend_per_household, out=cons.savings)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Household savings after dividends: {cons.savings}")

    if info_enabled:
        log.info("--- Firms Paying Dividends complete ---")
