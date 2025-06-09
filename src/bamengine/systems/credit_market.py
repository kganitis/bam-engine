# src/bamengine/systems/credit_market.py
"""
Event‑3  –  Credit‑market systems
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from numpy.random import Generator, default_rng

from bamengine import _logging_ext
from bamengine.components import Borrower, Employer, Lender, LoanBook, Worker
from bamengine.helpers import select_top_k_indices_sorted
from bamengine.typing import Idx1D

log = _logging_ext.getLogger(__name__)

CAP_FRAG = 1.0e6  # Fragility cap when net worth is zero or negative


def banks_decide_credit_supply(lend: Lender, *, v: float) -> None:
    """
    Banks determine their total credit supply based on their equity.

    Rule
    ----
        C = E / v

    `C: Credit Supply, E: Equity Base, v: Capital Requirement Coefficient`
    """
    log.info("--- Banks Deciding Credit Supply ---")
    log.info(f"  Inputs: Capital Requirement (v)={v:.3f} (Max Leverage={1 / v:.2f}x)")

    # --- Core Rule ---
    np.divide(lend.equity_base, v, out=lend.credit_supply)
    np.maximum(lend.credit_supply, 0.0, out=lend.credit_supply)

    # --- Logging ---
    total_supply = lend.credit_supply.sum()
    log.info(f"  Total credit supply in the economy: {total_supply:,.2f}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Equity Base (E) (first 10 firms): "
            f"{np.array2string(lend.equity_base[:10], precision=2)}")
        log.debug(
            f"  Credit Supply (C = E / v) (first 10 firms): "
            f"{np.array2string(lend.credit_supply[:10], precision=2)}")
    log.info("--- Credit Supply Decision complete ---")


def banks_decide_interest_rate(
    lend: Lender,
    *,
    r_bar: float,
    h_phi: float,
    rng: Generator = default_rng(),
) -> None:
    """
    Banks set their nominal interest rate as a markup over a base rate.

    Rule
    ----
        r = r̄ · (1 + shock)
        shock ~ U(0, h_φ)

    `r : Nominal Rate, r̄ : Baseline (Policy) Rate, h_φ : Banks Max Opex Growth`
    """
    log.info("--- Banks Deciding Interest Rate ---")
    log.info(
        f"  Inputs: Base Rate (r_bar)={r_bar:.4f}  |"
        f"  Max Markup Shock (h_phi)={h_phi:.4f}")
    shape = lend.interest_rate.shape

    # --- Permanent scratch buffer ---
    shock = lend.opex_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64)
        lend.opex_shock = shock

    # --- Core Rule ---
    shock[:] = rng.uniform(0.0, h_phi, size=shape)
    lend.interest_rate[:] = r_bar * (1.0 + shock)

    # --- Logging ---
    avg_rate = lend.interest_rate.mean() * 100
    log.info(f"  Interest rates set. Average rate: {avg_rate:.3f}%")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Generated shocks (first 10 firms): "
                  f"{np.array2string(shock[:10], precision=4)}")
        log.debug(
            f"  Interest Rates (%) (first 10 firms): "
            f"{np.array2string(lend.interest_rate[:10] * 100, precision=4)}")
    log.info("--- Interest Rate Decision complete ---")


def firms_decide_credit_demand(bor: Borrower) -> None:
    """
    Firms determine their credit demand
    as the gap between their wage bill and net worth.

    Rule
    ----
        B = max(W − A, 0)

    `B: Credit Demand, W: Wage Bill, A: Net Worth`
    """
    log.info("--- Firms Deciding Credit Demand ---")
    log.info(
        f"  Inputs: Total Wage Bill={bor.wage_bill.sum():,.2f}  |"
        f"  Total Net Worth={bor.net_worth.sum():,.2f}")

    # --- Core Rule ---
    np.subtract(bor.wage_bill, bor.net_worth, out=bor.credit_demand)
    np.maximum(bor.credit_demand, 0.0, out=bor.credit_demand)

    # --- Logging ---
    total_demand = bor.credit_demand.sum()
    num_borrowers = np.sum(bor.credit_demand > 0)
    log.info(
        f"  {num_borrowers} firms demand credit, for a total of {total_demand:,.2f}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Credit Demand per firm (B = max(0, W-A)) (first 10 firms): "
            f"{np.array2string(bor.credit_demand[:10], precision=2)}")
    log.info("--- Credit Demand Decision complete ---")


def firms_calc_credit_metrics(bor: Borrower) -> None:
    """
    Firms calculate their projected fragility based on credit demand and net worth.

    Rule
    ----
        l = B / A
        f = μ · l

    f: Fragility, μ: R&D Intensity, l: Leverage, B: Credit Demand, A: Net Worth
    """
    log.info("--- Firms Calculating Credit Metrics ---")
    shape = bor.net_worth.shape

    # --- Permanent scratch buffer ---
    frag = bor.projected_fragility
    if frag is None or frag.shape != shape:
        frag = np.empty(shape, dtype=np.float64)
        bor.projected_fragility = frag

    # --- Core Rule ---
    # Calculate raw fragility as B / A for firms with positive net worth.
    np.divide(bor.credit_demand, bor.net_worth, out=frag, where=bor.net_worth > 0.0)

    # Cap fragility for firms with zero or negative net worth at CAP_FRAG.
    zero_nw_mask = bor.net_worth <= 0.0
    if np.any(zero_nw_mask):
        num_zero_nw = np.sum(zero_nw_mask)
        log.warning(
            f"  {num_zero_nw} firm(s) have zero/negative net worth. "
            f"Setting their initial fragility to the cap of {CAP_FRAG:,.0f}."
        )
        frag[zero_nw_mask] = CAP_FRAG

    # Apply a global fragility cap for all firms.
    # This handles firms whose fragility may have exploded due to very small
    # (but positive) net worth, capping them at CAP_FRAG.
    general_cap_mask = frag > CAP_FRAG
    if np.any(general_cap_mask):
        num_generally_capped = np.sum(general_cap_mask)
        log.warning(
            f"  Capping fragility for {num_generally_capped} firm(s) "
            f"whose calculated fragility exceeded the {CAP_FRAG:,.0f} limit."
        )
    np.minimum(frag, CAP_FRAG, out=frag)

    # Final adjustment by R&D intensity (μ).
    np.multiply(frag, bor.rnd_intensity, out=frag)

    # --- Logging ---
    valid_frag = frag[np.isfinite(frag)]
    avg_fragility = valid_frag.mean() if valid_frag.size > 0 else 0.0
    log.info(f"  Average projected fragility across all firms: {avg_fragility:.4f}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Projected Fragility per firm (first 10 firms): "
            f"{np.array2string(frag[:10], precision=3)}")
    log.info("--- Credit Metrics Calculation complete ---")


def firms_prepare_loan_applications(
    bor: Borrower,
    lend: Lender,
    *,
    max_H: int,
    rng: Generator = default_rng(),
) -> None:
    """
    Firms with credit demand choose up to `max_H` banks to apply to, 
    sorted by interest rate.
    """
    log.info("--- Firms Preparing Loan Applications ---")
    lenders = np.where(lend.credit_supply > 0)[0]
    borrowers = np.where(bor.credit_demand > 0.0)[0]

    log.info(
        f"  {borrowers.size} firms are seeking loans "
        f"from {lenders.size} available lenders (max apps per firm, H={max_H}).")

    if borrowers.size == 0 or lenders.size == 0:
        log.info(
            "  No borrowers or no available lenders. "
            "Skipping loan application preparation.")
        bor.loan_apps_head.fill(-1)
        log.info("--- Loan Application Preparation complete ---")
        return

    # --- Sample H random lending banks per firm ---
    H_eff = min(max_H, lenders.size)
    log.info(f"  Effective applications per firm (H_eff): {H_eff}")
    # TODO Optimize loop
    sample = np.array(
        [rng.choice(lenders, size=H_eff, replace=False) for _ in range(borrowers.size)])
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Initial random bank sample (first 10 firms):\n{sample[:10]}")

    # --- Sort applications by ascending interest rate ---
    topk = select_top_k_indices_sorted(lend.interest_rate[sample], k=H_eff,
                                       descending=False)
    sorted_sample = np.take_along_axis(sample, topk, axis=1)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Sorted bank sample by interest rate (first 5 firms):\n"
            f"{sorted_sample[:5]}")

    # --- Write buffers ---
    log.debug(
        "  Writing application targets and head pointers for all borrowing firms...")
    bor.loan_apps_targets.fill(-1)
    bor.loan_apps_head.fill(-1)
    stride = max_H

    for k, f_idx in enumerate(borrowers):
        bor.loan_apps_targets[f_idx, :H_eff] = sorted_sample[k]
        if H_eff < max_H:
            bor.loan_apps_targets[f_idx, H_eff:max_H] = -1
        bor.loan_apps_head[f_idx] = f_idx * stride

        if log.isEnabledFor(_logging_ext.DEEP_DEBUG) and k < 10:
            log.deep(
                f"    Firm {f_idx}: targets={bor.loan_apps_targets[f_idx]}, "
                f"head_ptr={bor.loan_apps_head[f_idx]}")

    log.info("--- Loan Application Preparation complete ---")


def firms_send_one_loan_app(
    bor: Borrower, lend: Lender, rng: Generator = default_rng()
) -> None:
    """ """
    log.debug("  --- Borrowers Sending One Round of Applications ---")
    stride = bor.loan_apps_targets.shape[1]  # max_H
    log.debug(f"  stride={stride}")
    borrowers = np.where(bor.credit_demand > 0.0)[0]  # borrowers ids
    log.debug(f"  borrowers={borrowers}")
    active_applicants_mask = bor.loan_apps_head[borrowers] >= 0
    borrowers_applying = borrowers[active_applicants_mask]
    log.debug(f"  borrowers_applying={borrowers_applying}")

    log.debug(
        f"  Number of borrowers with pending applications: {borrowers_applying.size}"
    )
    if borrowers_applying.size == 0:
        return

    rng.shuffle(borrowers_applying)  # Shuffle to ensure fairness
    log.debug(f"  borrowers_applying={borrowers_applying}")

    for f_idx in borrowers_applying:
        log.deep(f"  --- Processing firm {f_idx} ---")
        head = bor.loan_apps_head[f_idx]
        log.deep(f"  head={head}")
        # If head is still -1 after filtering, something is off, but proceed by spec
        if head < 0:
            log.warning(
                f"  Firm {f_idx} was in borrowers_applying but head is {head}. "
                f"Skipping."
            )
            continue

        # Decode row and col from the 'head'
        # which is borrower_id * stride + current_app_idx
        # This implies row should be f_idx,
        # and col is the actual application index for that firm
        row, col = divmod(head, stride)
        log.deep(f"  (row, col)=({row}, {col})")

        if row != f_idx:
            log.error(
                f"  CRITICAL MISMATCH for firm {f_idx}: "
                f"head={head} decoded to row {row} (stride={stride}). "
                f"This indicates a bug in head setting or decoding."
            )

        log.deep(
            f"  Processing worker {f_idx}: "
            f"head={head} -> decoded_row={row}, app_col_idx={col}"
        )

        if col >= stride:  # Worker has sent all their M_eff applications
            log.deep(
                f"    Firm {f_idx} has exhausted all {stride} application slots. "
                f"Setting head to -1."
            )
            bor.loan_apps_head[f_idx] = -1
            continue

        bank_idx = bor.loan_apps_targets[row, col]
        log.deep(f"  Firm {f_idx} loan_apps_head: {bor.loan_apps_head[f_idx]}")
        log.deep(f"  Firm {f_idx} loan_apps_targets: {bor.loan_apps_targets[f_idx]}")
        log.deep(f"  Firm {f_idx} applying to bank {bank_idx} (app #{col + 1})")

        if bank_idx < 0:  # exhausted list for this firm for this round or padded -1
            log.deep(
                f"    Firm {f_idx} encountered sentinel bank_idx ({bank_idx}) "
                f"at col {col}. Effective end of list for this firm or padded slot."
            )
            bor.loan_apps_head[f_idx] = -1  # Mark as done
            continue

        if lend.credit_supply[bank_idx] <= 0:  # bank has no more credit
            log.deep(
                f"  Bank {bank_idx} has no more credit to supply. "
                f"Firm {f_idx} application dropped."
            )
            # Firm still tried, advance their pointer
            bor.loan_apps_head[f_idx] = head + 1
            # Clear the target so they don't re-apply if logic changes
            bor.loan_apps_targets[row, col] = -1
            continue

        # bounded queue for banks
        ptr = lend.recv_loan_apps_head[bank_idx] + 1
        log.deep(
            f"  Bank {bank_idx} current app queue head: "
            f"{lend.recv_loan_apps_head[bank_idx]}, "
            f"new ptr: {ptr}"
        )
        lend.recv_loan_apps_head[bank_idx] = ptr
        lend.recv_loan_apps[bank_idx, ptr] = f_idx
        log.deep(
            f"  Firm {f_idx} application successfully queued "
            f"at bank {bank_idx} slot {ptr}."
        )

        # advance pointer & clear slot
        bor.loan_apps_head[f_idx] = head + 1
        bor.loan_apps_targets[row, col] = -1
        log.deep(
            f"  Firm {f_idx} head advanced to {bor.loan_apps_head[f_idx]}. "
            f"Cleared target slot [{row},{col}]."
        )
        log.deep(f"  Firm {f_idx} job_apps_targets: {bor.loan_apps_targets[f_idx]}")


def _clean_queue(slice_: Idx1D, bor: Borrower, bank_idx_for_log: int) -> Idx1D:
    """
    Return a queue (Idx1D) of *unique* firm ids that still demand credit,
    ordered by the borrower's `net_worth` (highest first).

    Steps
    -----
    1.  Drop `-1` sentinels.
    2.  Deduplicate while preserving the first appearance.
    3.  Keep only firms with `credit_demand > 0`.
    4.  Sort the remaining ids by `net_worth` descending.
    """
    log.debug(
        f"    Bank {bank_idx_for_log}: Cleaning queue. "
        f"Initial raw slice: {slice_}"
    )

    # ------------------------------------------------------------------ #
    # 1. drop -1 sentinels                                               #
    # ------------------------------------------------------------------ #
    cleaned_slice = slice_[slice_ >= 0]
    if cleaned_slice.size == 0:
        log.debug(
            f"    Bank {bank_idx_for_log}: "
            f"Queue empty after dropping sentinels."
        )
        return cleaned_slice.astype(np.intp)

    log.debug(
        f"    Bank {bank_idx_for_log}: "
        f"Queue after dropping sentinels: {cleaned_slice}"
    )

    # ------------------------------------------------------------------ #
    # 2. deduplicate (keep first appearance)                             #
    # ------------------------------------------------------------------ #
    first_idx = np.unique(cleaned_slice, return_index=True)[1]
    unique_slice = cleaned_slice[np.sort(first_idx)]  # keep original order
    log.debug(
        f"    Bank {bank_idx_for_log}: "
        f"Queue after unique (order kept): {unique_slice}"
    )

    # ------------------------------------------------------------------ #
    # 3. keep only positive-demand firms                                 #
    # ------------------------------------------------------------------ #
    cd_mask      = bor.credit_demand[unique_slice] > 0
    filtered     = unique_slice[cd_mask]
    if filtered.size == 0:
        log.debug(
            f"    Bank {bank_idx_for_log}: "
            f"No firms left after credit-demand filter."
        )
        return cast(Idx1D, filtered)

    # ------------------------------------------------------------------ #
    # 4. sort by net worth (descending)                                  #
    # ------------------------------------------------------------------ #
    sort_idx     = np.argsort(-bor.net_worth[filtered])   # negative sign ⇒ desc
    ordered_ids  = filtered[sort_idx]

    log.debug(
        f"    Bank {bank_idx_for_log}: "
        f"Final cleaned queue (net_worth-desc): {ordered_ids}"
    )
    return cast(Idx1D, ordered_ids)


def banks_provide_loans(
    bor: Borrower,
    lb: LoanBook,
    lend: Lender,
    *,
    rng: Generator = default_rng(0),
) -> None:
    """
    Process queued applications **in‑place**:

        • contractual r_ik = r_k * (1 + frag_i)
        • satisfy queues until each bank’s credit_supply is exhausted
        • update both firm credit-demand **and** edge-list ledger
    """
    log.info("  --- Banks Providing Loans ---")
    banks = np.where(lend.credit_supply > 0.0)[0]  # bank ids
    log.debug(f"  {banks.size} banks have available credit to supply.")
    rng.shuffle(banks)
    log.debug(f"  Banks={banks}")

    total_loans_this_round = 0

    for k in banks:
        log.deep(f"    --- Processing bank {k} (supply: {lend.credit_supply[k]}) ---")

        n_recv = lend.recv_loan_apps_head[k] + 1
        log.deep(f"    Bank {k} recv_loan_apps_head={lend.recv_loan_apps_head[k]}")
        log.deep(f"    Bank {k} received {n_recv} applications (raw count).")
        if n_recv <= 0:
            log.deep(f"    Bank {k} has no applications. Skipping.")
            continue

        # copy for logging before cleaning
        queue = lend.recv_loan_apps[k, :n_recv].copy()
        log.deep(f"    Bank {k} raw application queue: {queue}")

        # Clean the queue: remove sentinels, duplicates, and already employed workers
        queue = _clean_queue(queue, bor, bank_idx_for_log=k)

        if queue.size == 0:
            log.deep(
                f"    Bank {k}: no valid (unique, pos_credit_demand) "
                f"applicants in queue."
            )
            # nothing useful in the queue → just flush it
            lend.recv_loan_apps_head[k] = -1
            lend.recv_loan_apps[k, :n_recv] = -1
            continue

        log.deep(f"    Bank {k} has {queue.size} valid potential borrowers: {queue}")

        # --- gather data ------------------------------------------------
        cd = bor.credit_demand[queue]
        frag = bor.projected_fragility[queue]
        max_grant = np.minimum(cd, lend.credit_supply[k])
        log.deep(f"Credit demand={cd}, fragility={frag}, max_grant={max_grant}")

        # amount actually granted (zero once the pot is empty)
        cumsum = np.cumsum(max_grant, dtype=np.float64)
        log.deep(f"    Bank {k} cumsum: {cumsum}")
        cut = cumsum > lend.credit_supply[k]
        log.deep(f"    Bank {k} cut: {cut}")
        if cut.any():
            first_exceed = cut.argmax()
            log.deep(f"    Bank {k} first exceed: {first_exceed}")
            max_grant[first_exceed] -= cumsum[first_exceed] - lend.credit_supply[k]
            log.deep(f"    Bank {k} max grant: {max_grant}")
            max_grant[first_exceed + 1 :] = 0.0
            log.deep(f"    Bank {k} max grant: {max_grant}")

        mask = max_grant > 0.0
        if not mask.any():
            log.deep(f"    Bank {k}: No loans granted this round. Flushing queue.")
            lend.recv_loan_apps_head[k] = -1
            lend.recv_loan_apps[k, :n_recv] = -1
            continue

        amount = max_grant[mask]
        borrowers = queue[mask]
        rate = lend.interest_rate[k] * (1.0 + frag[mask])
        if log.isEnabledFor(logging.DEBUG):
            log.deep(
                f"    Bank {k}: granting loans to borrowers {borrowers}\n"
                f"amounts={amount}\n"
                f"rates={rate}"
            )
        total_loans_this_round += amount.sum()

        # --- update ledger (vectorised append) ---------------------------------
        lb.purge_borrowers(borrowers)
        lb.append_loans_for_lender(k, borrowers, amount, rate)

        # --- balances ---------------------------------------------------
        bor.total_funds[borrowers] += amount
        bor.credit_demand[borrowers] -= amount
        lend.credit_supply[k] -= amount.sum()
        assert (bor.credit_demand >= -1e-12).all(), "negative credit_demand"
        log.deep(f"    Firms total funds updated: {bor.total_funds}")
        log.deep(f"    Firms credit demand updated: {bor.credit_demand}")
        log.deep(f"    Bank {k} credit_supply updated: {lend.credit_supply[k]}")

        # flush inbound queue for this bank
        lend.recv_loan_apps_head[k] = -1
        lend.recv_loan_apps[k, :n_recv] = -1

    log.info(f"    Total loans amount provided this round: {total_loans_this_round}")

    log.debug(f"    recv_loan_apps_head={lend.recv_loan_apps_head}")
    log.debug(f"    recv_loan_apps=\n{lend.recv_loan_apps}")


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
    log.info("--- Firms Firing Workers ---")

    total_workers_fired = 0

    for i in range(emp.current_labor.size):
        log.deep(
            f"  --- Processing firm {i} "
            f"(wage bill: {emp.wage_bill[i]}, "
            f"total funds: {emp.total_funds[i]}) ---"
        )

        gap = emp.wage_bill[i] - emp.total_funds[i]
        log.deep(f"    Firm {i} financing gap: {gap}")
        if gap <= 0.0:
            log.deep(f"    Firm {i}: No need for firing")
            continue

        # real roster: might differ from bookkeeping
        workforce = np.where((wrk.employed == 1) & (wrk.employer == i))[0]
        log.deep(f"    Firm {i} real workforce: {workforce.size}")
        log.deep(f"    Firm {i} bookkeeping labor: {emp.current_labor[i]}")
        if workforce.size != emp.current_labor[i]:
            log.critical("    Real workforce INCONSISTENT with bookkeeping.")
        if workforce.size == 0:  # no one to fire
            continue

        needed = int(np.ceil(gap / float(emp.wage_offer[i])))
        log.deep(f"    Firm {i}: workers needed to fire: {needed}")
        n_fire = min(needed, workforce.size)
        if n_fire == 0:  # numerical quirk
            continue

        victims = rng.choice(workforce, size=n_fire, replace=False)
        log.deep(
            f"    Firm {i}: calculated {n_fire} workers to fire "
            f"(capped by available workforce)."
        )

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

        total_workers_fired += n_fire

    log.info("--- Firing Complete ---")
    log.info(f"  Total workers fired: {total_workers_fired}")
    log.debug(f"  Current Labor after firing (L_i):\n{emp.current_labor}")
    log.debug(
        f"  Wage bills after firing:\n{np.array2string(emp.wage_bill, precision=2)}"
    )
