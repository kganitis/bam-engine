"""
System functions for credit market phase events.

This module contains the internal implementation functions for credit market events.
Event classes wrap these functions and provide the primary documentation.

See Also
--------
bamengine.events.credit_market : Event classes (primary documentation source)
"""

from __future__ import annotations

import numpy as np

from bamengine import Rng, logging, make_rng
from bamengine.relationships import LoanBook
from bamengine.roles import Borrower, Employer, Lender, Worker
from bamengine.typing import Idx1D
from bamengine.utils import EPS, select_top_k_indices_sorted

log = logging.getLogger(__name__)


def banks_decide_credit_supply(lend: Lender, *, v: float) -> None:
    """
    Calculate bank credit supply from equity base.

    See Also
    --------
    bamengine.events.credit_market.BanksDecideCreditSupply : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Banks Deciding Credit Supply ---")
        log.info(
            f"  Inputs: Capital Requirement (v)={v:.3f} (Max Leverage={1 / v:.2f}x)"
        )

    # Core Rule
    np.divide(lend.equity_base, v, out=lend.credit_supply)
    np.maximum(lend.credit_supply, 0.0, out=lend.credit_supply)

    # Logging
    if info_enabled:
        total_supply = lend.credit_supply.sum()
        log.info(f"  Total credit supply in the economy: {total_supply:,.2f}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Equity Base (E) (first 10 borrowers): "
            f"{np.array2string(lend.equity_base[:10], precision=2)}"
        )
        log.debug(
            f"  Credit Supply (C = E / v) (first 10 borrowers): "
            f"{np.array2string(lend.credit_supply[:10], precision=2)}"
        )
    if info_enabled:
        log.info("--- Credit Supply Decision complete ---")


def banks_decide_interest_rate(
    lend: Lender,
    *,
    r_bar: float,
    h_phi: float,
    rng: Rng = make_rng(),
) -> None:
    """
    Set bank interest rates with random markup over base rate.

    See Also
    --------
    bamengine.events.credit_market.BanksDecideInterestRate : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Banks Deciding Interest Rate ---")
        log.info(
            f"  Inputs: Base Rate (r_bar)={r_bar:.4f}  |"
            f"  Max Markup Shock (h_phi)={h_phi:.4f}"
        )
    shape = lend.interest_rate.shape

    # Permanent scratch buffer
    shock = lend.opex_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64)
        lend.opex_shock = shock

    # Core Rule
    shock[:] = rng.uniform(0.0, h_phi, size=shape)
    lend.interest_rate[:] = r_bar * (1.0 + shock)

    # Logging
    if info_enabled:
        avg_rate = lend.interest_rate.mean() * 100
        log.info(f"  Interest rates set. Average rate: {avg_rate:.3f}%")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Generated shocks (first 10 borrowers): "
            f"{np.array2string(shock[:10], precision=4)}"
        )
        log.debug(
            f"  Interest Rates (%) (first 10 borrowers): "
            f"{np.array2string(lend.interest_rate[:10] * 100, precision=4)}"
        )
    if info_enabled:
        log.info("--- Interest Rate Decision complete ---")


def firms_decide_credit_demand(bor: Borrower) -> None:
    """
    Calculate firm credit demand from wage bill and total funds gap.

    See Also
    --------
    bamengine.events.credit_market.FirmsDecideCreditDemand : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Borrowers Deciding Credit Demand ---")
        log.info(
            f"  Inputs: Total Wage Bill={bor.wage_bill.sum():,.2f}  |"
            f"  Total Net Worth={bor.total_funds.sum():,.2f}"
        )

    # Core Rule
    np.subtract(bor.wage_bill, bor.total_funds, out=bor.credit_demand)
    np.maximum(bor.credit_demand, 0.0, out=bor.credit_demand)

    # Logging
    if info_enabled:
        total_demand = bor.credit_demand.sum()
        num_borrowers = np.sum(bor.credit_demand > 0)
        log.info(
            f"  {num_borrowers} borrowers demand credit, for a total of {total_demand:,.2f}"
        )
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Credit Demand per borrower (B = max(0, W-A)) (first 10 borrowers): "
            f"{np.array2string(bor.credit_demand[:10], precision=2)}"
        )
    if info_enabled:
        log.info("--- Credit Demand Decision complete ---")


def firms_calc_financial_fragility(
    bor: Borrower,
    *,
    fragility_cap_method: str = "credit_demand",
) -> None:
    """
    Calculate firm projected financial fragility metric for loan applications.

    See Also
    --------
    bamengine.events.credit_market.FirmsCalcFinancialFragility : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Borrowers Calculating Credit Metrics ---")
    shape = bor.net_worth.shape

    # Permanent scratch buffer (may be None on first call)
    frag = bor.projected_fragility
    if frag is None or frag.shape != shape:  # type: ignore[redundant-expr]
        frag = np.empty(shape, dtype=np.float64)
        bor.projected_fragility = frag

    # Core Rule
    np.divide(bor.credit_demand, bor.net_worth, out=frag, where=bor.net_worth > 0.0)

    # Cap fragility for borrowers with zero or negative net worth at amount B
    zero_nw_mask = bor.net_worth <= EPS
    if np.any(zero_nw_mask):
        num_zero_nw = np.sum(zero_nw_mask)
        log.warning(
            f"  {num_zero_nw} borrower(s) have zero/negative net worth. "
            f"Capping their financial fragility to their credit demand amount."
        )
        frag[zero_nw_mask] = bor.credit_demand[zero_nw_mask]

    # Apply a global fragility cap for all borrowers (if enabled).
    # This handles borrowers whose fragility may have exploded due to
    # very small (but positive) net worth, capping them at amount B.
    if fragility_cap_method == "credit_demand":
        general_cap_mask = frag > bor.credit_demand
        if np.any(general_cap_mask) and info_enabled:
            num_generally_capped = np.sum(general_cap_mask)
            log.info(
                f"  Capping fragility for {num_generally_capped} borrower(s) "
                f"whose calculated fragility exceeded the allowed limit."
            )
        np.minimum(frag, bor.credit_demand, out=frag)
    else:
        # fragility_cap_method == "none" - skip global cap
        if info_enabled:
            log.info("  Fragility cap disabled (fragility_cap_method='none').")

    # Logging
    if info_enabled:
        valid_frag = frag[np.isfinite(frag)]
        avg_fragility = valid_frag.mean() if valid_frag.size > 0 else 0.0
        log.info(
            f"  Average projected fragility across all borrowers: {avg_fragility:.4f}"
        )
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Projected Fragility per borrower (first 10 borrowers): "
            f"{np.array2string(frag[:10], precision=3)}"
        )
    if info_enabled:
        log.info("--- Credit Metrics Calculation complete ---")


def firms_prepare_loan_applications(
    bor: Borrower,
    lend: Lender,
    *,
    max_H: int,
    rng: Rng = make_rng(),
) -> None:
    """
    Firms build loan application queue sorted by interest rate.

    See Also
    --------
    bamengine.events.credit_market.FirmsPrepareLoanApplications : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Borrowers Preparing Loan Applications ---")
    lenders = np.where(lend.credit_supply > 0)[0]
    borrowers = np.where(bor.credit_demand > 0.0)[0]

    if info_enabled:
        log.info(
            f"  {borrowers.size} borrowers are seeking loans "
            f"from {lenders.size} available lenders (max apps per borrower, H={max_H})."
        )

    if borrowers.size == 0 or lenders.size == 0:
        if info_enabled:
            log.info(
                "  No borrowers or no available lenders. "
                "Skipping loan application preparation."
            )
        bor.loan_apps_head.fill(-1)
        if info_enabled:
            log.info("--- Loan Application Preparation complete ---")
        return

    # Sample H random lending banks per borrower
    H_eff = min(max_H, lenders.size)
    if info_enabled:
        log.info(f"  Effective applications per borrower (H_eff): {H_eff}")
    sample = np.array(
        [rng.choice(lenders, size=H_eff, replace=False) for _ in range(borrowers.size)]
    )
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Initial random bank sample (first 10 borrowers):\n{sample[:10]}")

    # Sort applications by ascending interest rate
    topk = select_top_k_indices_sorted(
        lend.interest_rate[sample], k=H_eff, descending=False
    )
    sorted_sample = np.take_along_axis(sample, topk, axis=1)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Sorted bank sample by interest rate (first 5 borrowers):\n"
            f"{sorted_sample[:5]}"
        )

    # Write buffers
    log.debug("  Writing application targets and head pointers for all borrowers...")
    bor.loan_apps_targets.fill(-1)
    bor.loan_apps_head.fill(-1)
    stride = max_H

    for i, f_id in enumerate(borrowers):
        bor.loan_apps_targets[f_id, :H_eff] = sorted_sample[i]
        if H_eff < max_H:
            bor.loan_apps_targets[f_id, H_eff:max_H] = -1
        bor.loan_apps_head[f_id] = f_id * stride

        if log.isEnabledFor(logging.TRACE) and i < 10:
            log.trace(
                f"    Borrower {f_id}: targets={bor.loan_apps_targets[f_id]}, "
                f"head_ptr={bor.loan_apps_head[f_id]}"
            )

    if info_enabled:
        log.info("--- Loan Application Preparation complete ---")


def firms_send_one_loan_app(bor: Borrower, lend: Lender, rng: Rng = make_rng()) -> None:
    """
    Process one round of loan applications from firms to banks.

    See Also
    --------
    bamengine.events.credit_market.FirmsSendOneLoanApp : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Borrowers Sending One Round of Applications ---")
    stride = bor.loan_apps_targets.shape[1]  # max_H
    borrowers = np.where(bor.credit_demand > 0.0)[0]  # borrowers ids
    active_applicants_mask = bor.loan_apps_head[borrowers] >= 0
    borrowers_applying = borrowers[active_applicants_mask]

    if borrowers_applying.size == 0:
        if info_enabled:
            log.info("  No borrowers with pending applications found. Skipping round.")
            log.info("--- Application Sending Round complete ---")
        return

    if info_enabled:
        log.info(
            f"  Processing {borrowers_applying.size} borrowers with pending applications "
            f"(Stride={stride})."
        )

    rng.shuffle(borrowers_applying)  # order randomly chosen at each time step

    # Counters for logging
    apps_sent_successfully = 0
    apps_dropped_queue_full = 0
    apps_dropped_no_credit = 0

    for i in borrowers_applying:
        head = bor.loan_apps_head[i]
        if head < 0:
            log.warning(
                f"  Borrower {i} was in applying list but head is {head}. Skipping."
            )
            continue

        row_from_head, col = divmod(head, stride)
        if row_from_head != i:
            log.error(
                f"  CRITICAL MISMATCH for borrower {i}: "
                f"head={head} decoded to row {row_from_head}."
            )

        if head >= (i + 1) * stride:
            # Normal exit condition for an applicant who finished their list.
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Borrower {i} has exhausted all {stride} application slots. "
                    f"Setting head to -1."
                )
            bor.loan_apps_head[i] = -1
            continue

        lend_id = bor.loan_apps_targets[row_from_head, col]
        if lend_id < 0:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Borrower {i} encountered sentinel (-1) at col {col}. "
                    f"End of list. Setting head to -1."
                )
            bor.loan_apps_head[i] = -1
            continue

        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"    Borrower {i} applying to bank {lend_id} (app #{col + 1}).")

        # Check for remaining credit before checking queue space
        if lend.credit_supply[lend_id] <= 0:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"  Bank {lend_id} has no more credit to supply. "
                    f"Borrower {i} application dropped."
                )
            apps_dropped_no_credit += 1
            bor.loan_apps_head[i] = head + 1
            bor.loan_apps_targets[row_from_head, col] = -1
            continue

        # Check bank's application queue available space
        ptr = lend.recv_loan_apps_head[lend_id] + 1
        if ptr >= lend.recv_loan_apps.shape[1]:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Bank {lend_id} application queue full. "
                    f"Borrower {i} application dropped."
                )
            apps_dropped_queue_full += 1
            bor.loan_apps_head[i] = head + 1
            bor.loan_apps_targets[row_from_head, col] = -1
            continue

        # Application is successful
        lend.recv_loan_apps_head[lend_id] = ptr
        lend.recv_loan_apps[lend_id, ptr] = i
        apps_sent_successfully += 1
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"  Borrower {i} application queued at bank {lend_id} slot {ptr}."
            )

        bor.loan_apps_head[i] = head + 1
        bor.loan_apps_targets[row_from_head, col] = -1

    # Summary log
    total_dropped = apps_dropped_queue_full + apps_dropped_no_credit
    if info_enabled:
        log.info(
            f"  Round Summary: "
            f"{apps_sent_successfully} applications successfully queued, "
            f"{total_dropped} dropped."
        )
    if total_dropped > 0 and log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"    Dropped breakdown -> Queue Full: {apps_dropped_queue_full},"
            f" No Credit: {apps_dropped_no_credit}"
        )
    if info_enabled:
        log.info("--- Application Sending Round complete ---")


def _clean_queue(
    slice_: Idx1D,
    bor: Borrower,
    bank_idx_for_log: int,
    *,
    priority_method: str = "by_net_worth",
) -> Idx1D:  # pragma: no cover
    """
    Return a queue (Idx1D) of *unique* borrower ids that still demand credit
    from the raw queue slice (may contain -1 sentinels and duplicates).

    Parameters
    ----------
    slice_ : Idx1D
        Raw queue slice (may contain -1 sentinels and duplicates)
    bor : Borrower
        Borrower role with credit_demand and net_worth
    bank_idx_for_log : int
        Bank index for logging purposes
    priority_method : str, default "by_net_worth"
        Sorting method for the queue:
        - "by_net_worth": Sort by net worth (descending) - safest borrowers first
        - "by_leverage": Sort by leverage (ascending) - least leveraged first
          (uses projected_fragility as proxy for leverage)
        - "by_appearance": Preserve application order (first-come-first-served)

    Returns
    -------
    Idx1D
        Cleaned queue of unique borrower ids with positive credit demand
    """
    if log.isEnabledFor(logging.TRACE):
        log.trace(
            f"    Bank {bank_idx_for_log}: Cleaning queue. Initial raw slice: {slice_}"
        )

    # Drop -1 sentinels
    cleaned_slice = slice_[slice_ >= 0]
    if cleaned_slice.size == 0:
        if log.isEnabledFor(logging.TRACE):
            log.trace(
                f"    Bank {bank_idx_for_log}: Queue empty after dropping sentinels."
            )
        return cleaned_slice.astype(np.intp)

    if log.isEnabledFor(logging.TRACE):
        log.trace(
            f"    Bank {bank_idx_for_log}: "
            f"Queue after dropping sentinels: {cleaned_slice}"
        )

    # Unique *without* sorting
    first_idx = np.unique(cleaned_slice, return_index=True)[1]
    unique_slice = cleaned_slice[np.sort(first_idx)]
    if log.isEnabledFor(logging.TRACE):
        log.trace(
            f"    Bank {bank_idx_for_log}: "
            f"Queue after unique (order kept): {unique_slice}"
        )

    # Keep only positive-demand firms
    cd_mask = bor.credit_demand[unique_slice] > 0
    filtered_queue = unique_slice[cd_mask]
    if filtered_queue.size == 0:
        if log.isEnabledFor(logging.TRACE):
            log.trace(
                f"    Bank {bank_idx_for_log}: "
                f"No borrowers left after credit-demand filter."
            )
        return filtered_queue

    # Sort queue based on priority method
    # TODO: Introduce a separate `leverage` variable for Borrower role.
    # Currently we use projected_fragility as a proxy for leverage. In future
    # variations, projected_fragility will be calculated as rnd_intensity * leverage,
    # but we will still need only leverage for loan applicant sorting.
    if priority_method == "by_net_worth":
        # Sort by net worth (descending) - safest borrowers first
        sort_idx = np.argsort(-bor.net_worth[filtered_queue])
        ordered_queue = filtered_queue[sort_idx]
        if log.isEnabledFor(logging.TRACE):
            log.trace(
                f"    Bank {bank_idx_for_log}: "
                f"Final cleaned queue (net_worth-desc): {ordered_queue}"
            )
        return ordered_queue
    # TODO Keep only by_leverage method
    elif priority_method == "by_leverage":
        # Sort by leverage (ascending) - least leveraged first
        # Uses projected_fragility as proxy for leverage
        sort_idx = np.argsort(bor.projected_fragility[filtered_queue])
        ordered_queue = filtered_queue[sort_idx]
        if log.isEnabledFor(logging.TRACE):
            log.trace(
                f"    Bank {bank_idx_for_log}: "
                f"Final cleaned queue (leverage-asc): {ordered_queue}"
            )
        return ordered_queue
    else:
        # by_appearance: Preserve application order (first-come-first-served)
        if log.isEnabledFor(logging.TRACE):
            log.trace(
                f"    Bank {bank_idx_for_log}: "
                f"Final cleaned queue (FCFS order): {filtered_queue}"
            )
        return filtered_queue


def banks_provide_loans(
    bor: Borrower,
    lb: LoanBook,
    lend: Lender,
    *,
    r_bar: float,
    h_phi: float,
    loan_priority_method: str = "by_net_worth",
    max_loan_to_net_worth: float = 0.0,
) -> None:
    """
    Banks process applications and provide loans ranked by net worth.

    See Also
    --------
    bamengine.events.credit_market.BanksProvideLoans : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Banks Providing Loans ---")
    bank_ids = np.where(lend.credit_supply > 0.0)[0]
    total_credit_supply = lend.credit_supply.sum()
    if info_enabled:
        log.info(
            f"  {bank_ids.size} banks have {total_credit_supply:,} "
            f"total credit supply and are attempting to provide loans."
        )

    total_loans_this_round = 0

    for k in bank_ids:
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"  Processing bank {k} (credit supply: {lend.credit_supply[k]})")

        n_recv = lend.recv_loan_apps_head[k] + 1
        if n_recv <= 0:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"    Bank {k} has no applications. Skipping.")
            continue

        raw_queue = lend.recv_loan_apps[k, :n_recv].copy()
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Bank {k} raw application queue "
                f"({n_recv} applications): {raw_queue}"
            )

        queue = _clean_queue(
            raw_queue, bor, bank_idx_for_log=k, priority_method=loan_priority_method
        )

        if queue.size == 0:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Bank {k}: no valid (unique, positive credit demand) "
                    f"applicants in queue. Flushing."
                )
            lend.recv_loan_apps_head[k] = -1
            lend.recv_loan_apps[k, :n_recv] = -1
            continue

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Bank {k} has {queue.size} valid potential borrowers: {queue}"
            )

        # gather loan data
        cd = bor.credit_demand[queue]
        frag = bor.projected_fragility[queue]
        max_grant = np.minimum(cd, lend.credit_supply[k])

        # Apply loan-to-net-worth cap if configured
        if max_loan_to_net_worth > 0.0:
            nw_cap = bor.net_worth[queue] * max_loan_to_net_worth
            max_grant = np.minimum(max_grant, nw_cap)
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Bank {k}: Applied max_loan_to_net_worth={max_loan_to_net_worth} "
                    f"cap. nw_cap={nw_cap}, max_grant after cap={max_grant}"
                )

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Bank {k} loan data: credit_demand={cd}, "
                f"fragility={frag}, max_grant={max_grant}"
            )

        # determine actual loan amounts
        cumsum = np.cumsum(max_grant, dtype=np.float64)
        cut = cumsum > lend.credit_supply[k]
        if cut.any():
            first_exceed = cut.argmax()
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Bank {k} credit supply exceeded at position {first_exceed}. "
                    f"Adjusting loan amounts."
                )
            max_grant[first_exceed] -= cumsum[first_exceed] - lend.credit_supply[k]
            max_grant[first_exceed + 1 :] = 0.0

        mask = max_grant > 0.0
        # check if any loans were granted, should never trigger
        if not mask.any():  # pragma: no cover
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"    Bank {k}: No loans granted this round. Flushing queue.")
            lend.recv_loan_apps_head[k] = -1
            lend.recv_loan_apps[k, :n_recv] = -1
            continue

        final_borrowers = queue[mask]
        final_amounts = max_grant[mask]
        final_rates = r_bar * (1.0 + h_phi * frag[mask])

        # extra validation, should never trigger
        if final_borrowers.size == 0:  # pragma: no cover
            lend.recv_loan_apps_head[k] = -1
            lend.recv_loan_apps[k, :n_recv] = -1
            continue

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Bank {k} is granting loans to "
                f"{final_borrowers.size} borrower(s): {final_borrowers.tolist()}"
            )
        total_loans_this_round += final_amounts.sum()

        # ledger updates
        log.debug(f"      Updating ledger for {final_borrowers.size} new loans.")
        lb.purge_borrowers(final_borrowers)
        lb.append_loans_for_lender(k, final_borrowers, final_amounts, final_rates)

        # borrower‑side updates
        bor.total_funds[final_borrowers] += final_amounts
        bor.credit_demand[final_borrowers] -= final_amounts
        assert (bor.credit_demand >= -EPS).all(), "negative credit_demand"
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "      Borrower state updated: "
                "total_funds increased, credit_demand decreased"
            )

        # lender‑side updates
        lend.credit_supply[k] -= final_amounts.sum()
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"      Bank {k} state updated: credit_supply={lend.credit_supply[k]}"
            )

        # flush inbound queue for this bank
        lend.recv_loan_apps_head[k] = -1
        lend.recv_loan_apps[k, :n_recv] = -1
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"    Bank {k} application queue flushed.")

    if info_enabled:
        log.info(
            f"  Total loan amount provided this round across all banks: "
            f"{total_loans_this_round}"
        )
        log.info("--- Banks Providing Loans complete ---")


def firms_fire_workers(
    emp: Employer,
    wrk: Worker,
    *,
    method: str = "random",
    rng: Rng = make_rng(),
) -> None:
    """
    Firms lay off workers when credit is insufficient for wage bill.

    See Also
    --------
    bamengine.events.credit_market.FirmsFireWorkers : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Firing Workers ---")

    # Find firms with financing gaps
    gaps = emp.wage_bill - emp.total_funds
    firing_ids = np.where(gaps > 0.0)[0]
    total_gap = gaps[gaps > EPS].sum() if gaps.size > 0 else 0.0

    if info_enabled:
        log.info(
            f"  {firing_ids.size} firms have financing gaps totaling {total_gap:,.2f} "
            f"and need to fire workers using '{method}' method."
        )

    total_workers_fired_this_step = 0

    for i in firing_ids:
        gap = gaps[i]
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"  Processing firm {i} (wage bill: {emp.wage_bill[i]:.2f}, "
                f"total funds: {emp.total_funds[i]:.2f}, gap: {gap:.2f})"
            )

        # validate workforce consistency
        workforce = np.where((wrk.employed == 1) & (wrk.employer == i))[0]
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Firm {i} workforce validation: "
                f"real={workforce.size}, recorded={emp.current_labor[i]}"
            )

        if workforce.size != emp.current_labor[i]:
            log.critical(
                f"    Firm {i}: Real workforce ({workforce.size}) INCONSISTENT "
                f"with bookkeeping ({emp.current_labor[i]})."
            )

        if workforce.size == 0:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"    Firm {i}: No workers to fire. Skipping.")
            continue

        # determine workers to fire
        worker_wages = wrk.wage[workforce]
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Firm {i} worker wages: {worker_wages} "
                f"(total: {worker_wages.sum():.2f})"
            )

        if method not in ("random", "expensive"):  # pragma: no cover
            log.error(
                f"    Unknown firing method '{method}' specified. "
                f"Defaulting to 'random'."
            )
            method = "random"

        if method == "expensive":
            # Fire most expensive workers first
            sorted_indices = np.argsort(worker_wages)[::-1]  # Descending order
            sorted_wages = worker_wages[sorted_indices]
            cumsum_wages = np.cumsum(sorted_wages)

            # Find first position where cumulative wages >= gap
            sufficient_mask = cumsum_wages >= gap
            if sufficient_mask.any():
                n_fire = sufficient_mask.argmax() + 1
            else:
                n_fire = workforce.size  # Fire everyone if still not enough

            victims_indices = sorted_indices[:n_fire]
            victims = workforce[victims_indices]

        else:  # method == "random"
            # Sequential random firing until gap is covered
            shuffled_indices = rng.permutation(workforce.size)
            shuffled_wages = worker_wages[shuffled_indices]
            cumsum_wages = np.cumsum(shuffled_wages)

            # Find first position where cumulative wages >= gap
            sufficient_mask = cumsum_wages >= gap
            if sufficient_mask.any():
                n_fire = sufficient_mask.argmax() + 1
            else:
                n_fire = workforce.size  # Fire everyone if still not enough

            victims_indices = shuffled_indices[:n_fire]
            victims = workforce[victims_indices]

        fired_wages = wrk.wage[victims]
        total_fired_wage = fired_wages.sum()

        # extra validation, should never trigger
        if victims.size == 0:  # pragma: no cover
            continue

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Firm {i} is firing {victims.size} worker(s): "
                f"{victims.tolist()} (total wage savings: {total_fired_wage:.2f})"
            )
        total_workers_fired_this_step += victims.size

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Firm {i} firing details: gap={gap:.2f}, "
                f"wage_savings={total_fired_wage:.2f}, "
                f"coverage={min(100.0, (total_fired_wage / gap) * 100):.1f}%"
            )

        # worker‑side updates
        log.debug(f"      Updating state for {victims.size} fired workers.")
        wrk.employer[victims] = -1
        wrk.employer_prev[victims] = i
        wrk.wage[victims] = 0.0
        wrk.periods_left[victims] = 0
        wrk.contract_expired[victims] = 0
        wrk.fired[victims] = 1

        # firm‑side updates
        emp.current_labor[i] -= victims.size
        # Recalculate wage bill based on remaining workers
        remaining_workforce = np.where((wrk.employed == 1) & (wrk.employer == i))[0]
        emp.wage_bill[i] = (
            wrk.wage[remaining_workforce].sum() if remaining_workforce.size > 0 else 0.0
        )

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"      Firm {i} state updated: "
                f"current_labor={emp.current_labor[i]}, "
                f"wage_bill={emp.wage_bill[i]:.2f}"
            )

    if info_enabled:
        log.info(
            f"  Total workers fired this step across all firms: "
            f"{total_workers_fired_this_step}"
        )
    if log.isEnabledFor(logging.DEBUG):
        remaining_gaps = emp.wage_bill - emp.total_funds
        firms_with_gaps = np.flatnonzero(remaining_gaps > EPS)
        if firms_with_gaps.size > 0:
            log.warning(
                f"[REMAINING GAPS] {firms_with_gaps.size} firms still have "
                f"financing gaps after firing."
            )
            for i_gap in firms_with_gaps:
                log.warning(
                    f"  Firm {i_gap}: remaining gap={remaining_gaps[i_gap]:.2f}"
                )
        else:
            log.debug("[GAPS RESOLVED] All financing gaps resolved after firing.")
    if info_enabled:
        log.info("--- Firms Firing Workers complete ---")
