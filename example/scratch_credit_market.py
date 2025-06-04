from __future__ import annotations

import logging
from typing import cast

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.components import Borrower, Employer, Lender, LoanBook, Worker
from bamengine.helpers import select_top_k_indices_sorted
from bamengine.typing import Idx1D
from helpers.factories import mock_borrower, mock_lender, mock_loanbook, mock_employer, \
    mock_worker

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

CAP_FRAG = 1.0e6  # fragility cap when net worth is zero


def _mini_state(
    *,
    n_borrowers: int = 6,
    n_lenders: int = 3,
    H: int = 2,
    seed: int = 1,
) -> tuple[Employer, Borrower, Lender, LoanBook, Generator, int]:
    """
    Build minimal Firm, Lender, and empty LoanBook components plus an RNG.

    * Borrowers: wage_bill > net_worth for the first 3 borrowers so they demand credit.
    * Lenders:   non-zero credit_supply, distinct interest rates.
    """
    rng = default_rng(seed)

    wrk = mock_worker(
        n=131,
        employed=np.ones(131, dtype=np.int64),
        employer=np.array([0]*80 + [1]*20 + [2]*15 + [3]*10+ [4]*5 + [5]*1),
    )
    emp = mock_employer(
        n=n_borrowers,
        current_labor=np.array([80, 20, 15, 10, 5, 1], dtype=np.int64),
        wage_offer=np.ones(n_borrowers),
        wage_bill=np.array([80.0, 20.0, 15.0, 10.0, 5.0, 1.0]),
        total_funds=np.full(n_borrowers, 4.0),
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

    return wrk, emp, bor, lend, ledger, rng, H


wrk, emp, bor, lend, lb, rng, H = _mini_state()


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
            f"interest rates (%) = "
            f"{np.array2string(lend.interest_rate * 100, precision=4)}"
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
        log.info(f"Total credit demand = {bor.credit_demand.sum()}")


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
    * draws H random banks per firm
    * keeps the H *cheapest* (lowest r_k) via partial sort
    * writes indices into ``loan_apps_targets`` and resets ``loan_apps_head``
    """
    log.info("")
    log.info("----- Firms Deciding Banks to Apply -----")
    lenders = np.where(lend.credit_supply > 0)[0]  # bank ids
    borrowers = np.where(bor.credit_demand > 0.0)[0]  # firm ids

    log.debug(f"  Borrowers: {borrowers}")
    log.debug(f"  Lenders: {lenders}")
    log.debug(f"  Max H: {max_H}")

    if borrowers.size == 0:
        log.debug(f"  No borrowers available. Reseting firm queues heads.")
        bor.loan_apps_head.fill(-1)
        return

    # ── sample H random lending banks per firm (with replacement) ─────
    H_eff = min(max_H, lenders.size)
    log.debug(f"  Effective applications per firm (H_eff): {H_eff}")
    sample = np.empty((borrowers.size, H_eff), dtype=np.int64)
    for row, w in enumerate(borrowers):
        sample[row] = rng.choice(lenders, size=H_eff, replace=False)
    if log.isEnabledFor(logging.DEBUG) and borrowers.size > 0:
        log.debug(
            f"\n\nSample=\n" f"{sample}\n"
        )

    # ── interest-ascending partial sort ------------------------------------
    topk = select_top_k_indices_sorted(
        lend.interest_rate[sample], k=H_eff, descending=False
    )
    sorted_sample = np.take_along_axis(sample, topk, axis=1)
    if log.isEnabledFor(logging.DEBUG) and borrowers.size > 0:
        log.debug(
            f"\n\nSorted sample by interest rate:\n"
            f"{sorted_sample}\n"
        )

    # flush queue before filling
    bor.loan_apps_targets.fill(-1)
    bor.loan_apps_head.fill(-1)

    log.debug("  Writing application targets and heads for borrowers...")
    stride = max_H
    for k, f_idx in enumerate(borrowers):
        bor.loan_apps_targets[f_idx, :H_eff] = sorted_sample[k]
        # pad any remaining columns with -1
        if H_eff < max_H:
            bor.loan_apps_targets[f_idx, H_eff:max_H] = -1
        bor.loan_apps_head[f_idx] = (
                f_idx * stride
        )  # Using actual firm ID for the row in conceptual 2D array
        log.debug(
            f"  Firm {f_idx}: targets={bor.loan_apps_targets[f_idx]}, "
            f"head_raw_val={bor.loan_apps_head[f_idx]}"
        )
    log.debug(f"  loan_apps_head: {bor.loan_apps_head}")
    log.debug(f"  loan_apps_targets:\n{bor.loan_apps_targets}")

    log.info(f"  {borrowers.size} borrowing firms prepared {H_eff} applications each.")


def firms_send_one_loan_app(
    bor: Borrower, lend: Lender, rng: Generator = default_rng()
) -> None:
    """ """
    log.debug("----- Borrowers Sending One Round of Applications -----")
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
        log.debug(f"  --- Processing firm {f_idx} ---")
        head = bor.loan_apps_head[f_idx]
        log.debug(f"  head={head}")
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
        log.debug(f"  (row, col)=({row}, {col})")

        if row != f_idx:
            log.error(
                f"  CRITICAL MISMATCH for firm {f_idx}: "
                f"head={head} decoded to row {row} (stride={stride}). "
                f"This indicates a bug in head setting or decoding."
                )

        log.debug(
            f"  Processing worker {f_idx}: "
            f"head={head} -> decoded_row={row}, app_col_idx={col}"
        )

        if col >= stride:  # Worker has sent all their M_eff applications
            log.debug(
                f"    Firm {f_idx} has exhausted all {stride} application slots. "
                f"Setting head to -1."
            )
            bor.loan_apps_head[f_idx] = -1
            continue

        bank_idx = bor.loan_apps_targets[row, col]
        log.debug(f"  Firm {f_idx} loan_apps_head: {bor.loan_apps_head[f_idx]}")
        log.debug(f"  Firm {f_idx} loan_apps_targets: {bor.loan_apps_targets[f_idx]}")
        log.debug(f"  Firm {f_idx} applying to bank {bank_idx} (app #{col + 1})")

        if bank_idx < 0:  # exhausted list for this firm for this round or padded -1
            log.debug(
                f"    Firm {f_idx} encountered sentinel bank_idx ({bank_idx}) "
                f"at col {col}. Effective end of list for this firm or padded slot."
            )
            bor.loan_apps_head[f_idx] = -1  # Mark as done
            continue

        if lend.credit_supply[bank_idx] <= 0:  # bank has no more credit
            log.debug(
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
        log.debug(
            f"  Bank {bank_idx} current app queue head: "
            f"{lend.recv_loan_apps_head[bank_idx]}, "
            f"new ptr: {ptr}"
        )
        lend.recv_loan_apps_head[bank_idx] = ptr
        lend.recv_loan_apps[bank_idx, ptr] = f_idx
        log.debug(
            f"  Firm {f_idx} application successfully queued "
            f"at bank {bank_idx} slot {ptr}."
        )

        # advance pointer & clear slot
        bor.loan_apps_head[f_idx] = head + 1
        bor.loan_apps_targets[row, col] = -1
        log.debug(
            f"  Firm {f_idx} head advanced to {bor.loan_apps_head[f_idx]}. "
            f"Cleared target slot [{row},{col}]."
        )
        log.debug(f"  Firm {f_idx} job_apps_targets: {bor.loan_apps_targets[f_idx]}")


def _clean_queue(slice_: Idx1D, bor: Borrower, bank_idx_for_log: int) -> Idx1D:
    """
    Return a *unique* array of firm ids that still demand credit
    from the raw queue slice (which may contain -1 sentinels and duplicates),
    preserving the original order of first appearance.
    """
    log.debug(f"    Bank {bank_idx_for_log}: "
              f"Cleaning queue. Initial raw slice: {slice_}")

    # 1. Drop -1 sentinels
    cleaned_slice = slice_[slice_ >= 0]
    if cleaned_slice.size == 0:
        log.debug(f"    Bank {bank_idx_for_log}: Queue empty after dropping sentinels.")
        return cleaned_slice.astype(np.intp)        # Ensure Idx1D type

    log.debug(f"    Bank {bank_idx_for_log}: "
              f"Queue after dropping sentinels: {cleaned_slice}")

    # 2. Unique *without* sorting
    # indices of first occurrences
    first_idx = np.unique(cleaned_slice, return_index=True)[1]
    unique_slice = cleaned_slice[np.sort(first_idx)]             # keep original order
    log.debug(f"    Bank {bank_idx_for_log}:"
              f" Queue after unique (order kept): {unique_slice}")

    # 3. Keep only firms with positive credit demand
    cd_mask = bor.credit_demand[unique_slice] > 0
    final_queue = unique_slice[cd_mask]
    log.debug(
        f"    Bank {bank_idx_for_log}: "
        f"Final cleaned queue (unique, pos_credit_demand): {final_queue}"
    )
    return cast(Idx1D, final_queue)


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
    log.info("")
    log.info("  Banks Providing Loans -----------------------------------------------")
    banks = np.where(lend.credit_supply > 0.0)[0]  # bank ids
    log.debug(f"  {banks.size} banks have available credit to supply.")
    rng.shuffle(banks)
    log.debug(f"  Banks={banks}")

    total_loans_this_round = 0

    for k in banks:
        log.debug(f"    --- Processing bank {k} (supply: {lend.credit_supply[k]}) ---")

        n_recv = lend.recv_loan_apps_head[k] + 1
        log.debug(f"    Bank {k} recv_loan_apps_head={lend.recv_loan_apps_head[k]}")
        log.debug(f"    Bank {k} received {n_recv} applications (raw count).")
        if n_recv <= 0:
            log.debug(f"    Bank {k} has no applications. Skipping.")
            continue

        # copy for logging before cleaning
        queue = lend.recv_loan_apps[k, :n_recv].copy()
        log.debug(f"    Bank {k} raw application queue: {queue}")

        # Clean the queue: remove sentinels, duplicates, and already employed workers
        queue = _clean_queue(queue, bor, bank_idx_for_log=k)

        # queue = queue[queue >= 0]  # compact valid requests

        if queue.size == 0:
            log.debug(
                f"    Bank {k}: no valid (unique, pos_credit_demand) "
                f"applicants in queue."
            )
            # nothing useful in the queue → just flush it
            lend.recv_loan_apps_head[k] = -1
            lend.recv_loan_apps[k, :n_recv] = -1
            continue

        log.debug(
            f"    Bank {k} has {queue.size} valid potential borrowers: {queue}"
        )

        # --- gather data ------------------------------------------------
        cd = bor.credit_demand[queue]
        frag = bor.projected_fragility[queue]
        max_grant = np.minimum(cd, lend.credit_supply[k])
        log.debug(f"Credit demand={cd}, fragility={frag}, max_grant={max_grant}")

        # amount actually granted (zero once the pot is empty)
        cumsum = np.cumsum(max_grant, dtype=np.float64)
        log.debug(f"    Bank {k} cumsum: {cumsum}")
        cut = cumsum > lend.credit_supply[k]
        log.debug(f"    Bank {k} cut: {cut}")
        if cut.any():
            first_exceed = cut.argmax()
            log.debug(f"    Bank {k} first exceed: {first_exceed}")
            max_grant[first_exceed] -= cumsum[first_exceed] - lend.credit_supply[k]
            log.debug(f"    Bank {k} max grant: {max_grant}")
            max_grant[first_exceed + 1 :] = 0.0
            log.debug(f"    Bank {k} max grant: {max_grant}")

        mask = max_grant > 0.0
        if not mask.any():
            log.debug(f"    Bank {k}: No loans granted this round. Flushing queue.")
            lend.recv_loan_apps_head[k] = -1
            lend.recv_loan_apps[k, :n_recv] = -1
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
        total_loans_this_round += amount.sum()

        # --- update ledger (vectorised append) ---------------------------------
        lb.append_loans(borrowers, k, amount, rate)

        # --- balances ---------------------------------------------------
        bor.total_funds[borrowers] += amount
        bor.credit_demand[borrowers] -= amount
        lend.credit_supply[k] -= amount.sum()
        assert (bor.credit_demand >= -1e-12).all(), "negative credit_demand"
        log.debug(f"    Firms total funds updated: {bor.total_funds}")
        log.debug(f"    Firms credit demand updated: {bor.credit_demand}")
        log.debug(f"    Bank {k} credit_supply updated: {lend.credit_supply[k]}")

        # flush inbound queue for this bank
        lend.recv_loan_apps_head[k] = -1
        lend.recv_loan_apps[k, :n_recv] = -1

    log.info(f"Total loans amount provided this round: {total_loans_this_round}")

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
    log.info("")
    log.info("  Firms Firing Workers -----------------------------------------------")

    for i in range(emp.current_labor.size):
        log.debug(f"    --- Processing firm {i} "
                  f"(wage bill: {emp.wage_bill[i]}, "
                  f"total funds: {emp.total_funds[i]}) ---")

        gap = emp.wage_bill[i] - emp.total_funds[i]
        log.debug(f"    Firm {i} financing gap: {gap}")
        if gap <= 0.0:
            log.debug(f"    Firm {i}: No need for firing")
            continue

        # real roster: might differ from bookkeeping
        workforce = np.where((wrk.employed == 1) & (wrk.employer == i))[0]
        log.debug(f"    Firm {i} real workforce: {workforce.size}")
        log.debug(f"    Firm {i} bookkeeping labor: {emp.current_labor[i]}")
        if workforce.size != emp.current_labor[i]:
            log.critical(f"    Real workforce INCONSISTENT with bookkeeping.")
        if workforce.size == 0:  # no one to fire
            continue

        needed = int(np.ceil(gap / float(emp.wage_offer[i])))
        log.debug(f"    Firm {i}: workers needed to fire: {needed}")
        n_fire = min(needed, workforce.size)
        if n_fire == 0:  # numerical quirk
            continue

        victims = rng.choice(workforce, size=n_fire, replace=False)
        log.debug(f"    Firm {i}: calculated {n_fire} workers to fire "
                  f"(capped by available workforce).")

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

    log.debug("  Firing complete")
    log.debug(f"  Current Labor after firing (L_i):\n{emp.current_labor}")
    log.debug(
        f"  Wage bills after firing:\n{np.array2string(emp.wage_bill, precision=2)}"
    )


banks_decide_credit_supply(lend, v=0.1)
banks_decide_interest_rate(lend, r_bar=0.02, h_phi=0.10, rng=rng)
firms_decide_credit_demand(bor)
firms_calc_credit_metrics(bor)
firms_prepare_loan_applications(bor, lend, max_H=H, rng=rng)
firms_send_one_loan_app(bor, lend, rng)
banks_provide_loans(bor, lb, lend, rng=rng)
firms_send_one_loan_app(bor, lend, rng)
banks_provide_loans(bor, lb, lend, rng=rng)
firms_fire_workers(emp, wrk, rng=rng)
