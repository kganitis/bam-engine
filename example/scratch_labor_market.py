from typing import cast

from numpy.random import Generator, default_rng

import numpy as np
import logging

from bamengine.components import Employer, Worker
from bamengine.helpers import select_top_k_indices_sorted
from bamengine.typing import Idx1D, Int1D
from helpers.factories import mock_employer, mock_worker

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def _mini_state(
    *,
    n_workers: int = 10,
    n_employers: int = 6,
    M: int = 3,
    seed: int = 2,
) -> tuple[Employer, Worker, Generator, int]:

    rng = default_rng(seed)

    emp = mock_employer(
        n=n_employers,
        queue_m=M,
        wage_offer=np.array([1.0, 1.2, 1.5, 2.0, 2.0, 2.5]),
        n_vacancies=np.array([2, 2, 0, 1, 0, 2], dtype=np.int64),
        current_labor=np.array([0, 0, 1, 0, 0, 1], dtype=np.int64),
    )

    wrk = mock_worker(
        n=n_workers,
        queue_m=M,
        employed=np.array([False, False, False, False, False,
                           False, False, False, True, True]),
        employer=np.array([-1, -1, -1, -1, -1, -1, -1, -1, 2, 5], dtype=np.intp),
        employer_prev=np.array([2, 4, 0, 1, 3, 6, 2, 3, -1, -1], dtype=np.intp),
        contract_expired=np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 0], dtype=np.bool_),
        fired=np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.bool_)
    )

    return emp, wrk, rng, M


emp, wrk, rng, max_M = _mini_state()

def workers_decide_firms_to_apply(
    wrk: Worker,
    emp: Employer,
    *,
    max_M: int,
    rng: Generator = default_rng()
):
    log.info("--- Workers Deciding Firms to Apply ---")
    hiring = np.where(emp.n_vacancies > 0)[0]  # hiring ids
    unemp = np.where(wrk.employed == 0)[0]  # unemployed ids

    log.debug(f"  Unemployed workers: {unemp}")
    log.debug(f"  Firms with vacancies (hiring): {hiring}")
    log.debug(f"  Max applications per worker (max_M): {max_M}")

    # ── sample M random hiring firms per worker (with replacement) ─────
    M_eff = min(max_M, hiring.size)
    log.debug(f"  Effective applications per worker (M_eff): {M_eff}")
    sample = np.empty((unemp.size, M_eff), dtype=np.int64)
    for row, w in enumerate(unemp):
        sample[row] = rng.choice(hiring, size=M_eff, replace=False)
        log.debug(f"  Worker {row}: {sample[row]}, "
                  f"previous: {wrk.employer_prev[w]}, "
                  f"contract_expired: {wrk.contract_expired[w]}, "
                  f"fired: {wrk.fired[w]}")
    if unemp.size > 0:
        log.debug(
            f"\n\nSample=\n" f"{sample}\n"
        )

    # ── wage-descending partial sort ------------------------------------
    topk = select_top_k_indices_sorted(
        emp.wage_offer[sample], k=M_eff, descending=True
    )  # Use M_eff for k
    log.debug(f"\nTop K:\n{topk}\n")
    sorted_sample = np.take_along_axis(sample, topk, axis=1)
    if log.isEnabledFor(logging.DEBUG) and unemp.size > 0:
        log.debug(
            f"\n\nSorted sample by wage:\n"
            f"{sorted_sample}\n"
        )

    # ── loyalty rule ----------------------------------------------------
    loyal_mask = (
        (wrk.contract_expired[unemp] == 1)  # only if the contract has just expired
        & (wrk.fired[unemp] == 0)  # only if not fired from previous employer
        & np.isin(wrk.employer_prev[unemp], hiring)  # only if previous employer is hiring
    )
    if loyal_mask.any():
        # Log the state of sorted_sample BEFORE this specific loyalty adjustment
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"  Sorted sample BEFORE post-sort loyalty adjustment (showing all rows if loyal_mask.any()):\n{sorted_sample}"
            )

        # Get the row indices in `unemp` (and thus in `sorted_sample`) that correspond to loyal workers
        loyal_row_indices = np.where(loyal_mask)[0]

        for row_idx in loyal_row_indices:
            actual_worker_id = unemp[row_idx]  # Get the actual ID of the loyal worker
            prev_employer_id = wrk.employer_prev[actual_worker_id]

            # Get a view of the current worker's application list (a row in sorted_sample)
            # Modifications to 'application_row' will modify 'sorted_sample' in-place.
            application_row = sorted_sample[row_idx]
            num_applications = application_row.shape[0]  # Should be M_eff

            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Adjusting for loyalty: Worker ID {actual_worker_id} (row_idx {row_idx} in unemp/sorted_sample), "
                    f"Previous Employer: {prev_employer_id}"
                )
                log.debug(
                    f"    Application row BEFORE adjustment: {application_row.copy()}")

            try:
                # Check if the previous employer is already in the application list
                current_pos_of_prev_emp = np.where(application_row == prev_employer_id)[0][
                    0]

                # If it is present and not already at the first position
                if current_pos_of_prev_emp != 0:
                    log.debug(
                        f"    Previous employer {prev_employer_id} found at position {current_pos_of_prev_emp}. Moving to front.")
                    # Element is present but not at the start.
                    # Store the element, then shift elements from [0...current_pos-1] to [1...current_pos]
                    # and place the stored element at [0].

                    # Pull out the previous employer ID
                    employer_to_move = application_row[current_pos_of_prev_emp]

                    # Shift elements from the start up to its original position one step to the right
                    # application_row[1 : current_pos_of_prev_emp + 1] = application_row[0 : current_pos_of_prev_emp]
                    # A more robust way to do this (handles M_eff=1 correctly if current_pos_of_prev_emp is 0):
                    for j in range(current_pos_of_prev_emp, 0, -1):
                        application_row[j] = application_row[j - 1]
                    application_row[0] = employer_to_move  # Place it at the front
                else:
                    log.debug(
                        f"    Previous employer {prev_employer_id} is already at the first position. No change needed.")

            except IndexError:  # np.where(...)[0][0] will fail if prev_employer_id is not found
                # Previous employer was NOT in the list.
                # Place it at the front and shift other elements to the right, dropping the last one.
                log.debug(
                    f"    Previous employer {prev_employer_id} not found in application list. Inserting at front.")
                if num_applications > 0:  # Ensure there's space to do anything
                    # Shift all existing elements one position to the right
                    # The last element application_row[num_applications-1] will be overwritten
                    if num_applications > 1:
                        application_row[1:num_applications] = application_row[
                                                              0:num_applications - 1]
                    # Place the previous employer at the first position
                    application_row[0] = prev_employer_id
                # If num_applications is 0 (empty row, though M_eff should prevent this), this block does nothing.
                # If num_applications is 1, application_row[0] is just set to prev_employer_id.

            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"    Application row AFTER adjustment:  {application_row}")

        # Log the state of sorted_sample AFTER all loyal workers have been processed
        if log.isEnabledFor(
                logging.DEBUG) and loyal_mask.any():  # Check loyal_mask.any() again as it was the entry condition
            log.debug(
                f"    Sorted sample AFTER post-sort loyalty adjustment (all rows if loyal_mask.any()):\n{sorted_sample}\n"
            )

    # ── write buffers ────────────────────────────────────────────────----
    log.debug("  Writing application targets and heads for workers...")
    stride = max_M  # Stride should be based on the full width of job_apps_targets
    for k, w_idx in enumerate(unemp):  # w_idx is the actual worker ID
        wrk.job_apps_targets[w_idx, :M_eff] = sorted_sample[k]
        # pad any remaining columns with -1
        if M_eff < max_M:
            wrk.job_apps_targets[w_idx, M_eff:max_M] = -1
        wrk.job_apps_head[w_idx] = (
            w_idx * stride
        )  # Using actual worker ID for the row in conceptual 2D array
        log.debug(
            f"  Worker {w_idx}: targets={wrk.job_apps_targets[w_idx]}, "
            f"head_raw_val={wrk.job_apps_head[w_idx]}"
        )
    log.debug(f"  job_apps_head: {wrk.job_apps_head}")
    log.debug(f"  job_apps_targets:\n{wrk.job_apps_targets}")

    # reset flags
    wrk.contract_expired[unemp] = 0
    wrk.fired[unemp] = 0

    log.info(f"  {unemp.size} unemployed workers prepared {M_eff} applications each.")


def workers_send_one_round(wrk: Worker, emp: Employer, rng: Generator = default_rng()):
    log.debug("----- Workers Sending One Round of Applications -----")
    stride = wrk.job_apps_targets.shape[1]  # This is max_M
    log.debug(f"  stride={stride}")
    unemp_ids = np.where(wrk.employed == 0)[0]  # Get currently unemployed
    log.debug(f"  unemp_ids={unemp_ids}")
    active_applicants_mask = wrk.job_apps_head[unemp_ids] >= 0
    unemp_ids_applying = unemp_ids[active_applicants_mask]
    log.debug(f"  unemp_ids_applying={unemp_ids_applying}")

    log.debug(
        f"  Number of workers with pending applications: {unemp_ids_applying.size}"
    )
    if unemp_ids_applying.size == 0:
        return

    rng.shuffle(unemp_ids_applying)  # Shuffle to ensure fairness
    log.debug(f"  unemp_ids_applying={unemp_ids_applying}")

    for w_idx in unemp_ids_applying:  # w_idx is actual worker ID
        log.debug(f"  --- Processing worker {w_idx} ---")
        head = wrk.job_apps_head[w_idx]
        log.debug(f"  head={head}")
        # If head is still -1 after filtering, something is off, but proceed by spec
        if head < 0:
            log.warning(
                f"  Worker {w_idx} was in unemp_ids_applying but head is {head}. "
                f"Skipping."
            )
            continue

        # Decode row and col from the 'head'
        # which is worker_id * stride + current_app_idx
        # This implies row should be w_idx,
        # and col is the actual application index for that worker
        row, col = divmod(head, stride)
        log.debug(f"  (row, col)=({row}, {col})")

        if row != w_idx:
            log.error(
                f"  CRITICAL MISMATCH for worker {w_idx}: "
                f"head={head} decoded to row {row} (stride={stride}). "
                f"This indicates a bug in head setting or decoding."
                )

        log.debug(
            f"  Processing worker {w_idx}: "
            f"head={head} -> decoded_row={row}, app_col_idx={col}"
        )

        if col >= stride:  # Worker has sent all their M_eff applications
            log.debug(
                f"    Worker {w_idx} has exhausted all {stride} application slots. "
                f"Setting head to -1."
            )
            wrk.job_apps_head[w_idx] = -1
            continue

        firm_idx = wrk.job_apps_targets[row, col]  # Use decoded row
        log.debug(f"  Worker {w_idx} job_apps_head: {wrk.job_apps_head[w_idx]}")
        log.debug(f"  Worker {w_idx} job_apps_targets: {wrk.job_apps_targets[w_idx]}")
        log.debug(f"  Worker {w_idx} applying to firm {firm_idx} (app #{col + 1})")

        if firm_idx < 0:  # exhausted list for this worker for this round or padded -1
            log.debug(
                f"    Worker {w_idx} encountered sentinel firm_idx ({firm_idx}) "
                f"at col {col}. Effective end of list for this worker or padded slot."
            )
            wrk.job_apps_head[w_idx] = -1  # Mark as done
            continue

        if emp.n_vacancies[firm_idx] == 0:  # firm has no more vacancies, drop application
            log.debug(
                f"  Firm {firm_idx} has no more open vacancies. "
                f"Worker {w_idx} application dropped."
            )
            # Worker still tried, advance their pointer
            wrk.job_apps_head[w_idx] = head + 1
            # Clear the target so they don't re-apply if logic changes
            wrk.job_apps_targets[row, col] = -1
            continue

        # bounded queue for firms
        current_firm_queue_head = emp.recv_job_apps_head[firm_idx]
        ptr = current_firm_queue_head + 1
        log.debug(
            f"  Firm {firm_idx} current app queue head: {current_firm_queue_head}, "
            f"new ptr: {ptr}"
        )

        if ptr >= emp.recv_job_apps.shape[1]:
            log.debug(
                f"  Firm {firm_idx} application queue full "
                f"({emp.recv_job_apps.shape[1]}). Worker {w_idx} application dropped."
            )
            # Worker still tried, advance their pointer
            wrk.job_apps_head[w_idx] = head + 1
            # Clear the target so they don't re-apply if logic changes
            wrk.job_apps_targets[row, col] = -1
            continue

        emp.recv_job_apps_head[firm_idx] = ptr
        emp.recv_job_apps[firm_idx, ptr] = w_idx
        log.debug(
            f"  Worker {w_idx} application successfully queued "
            f"at firm {firm_idx} slot {ptr}."
        )

        # advance pointer & clear slot
        wrk.job_apps_head[w_idx] = head + 1
        wrk.job_apps_targets[row, col] = -1
        log.debug(
            f"  Worker {w_idx} head advanced to {wrk.job_apps_head[w_idx]}. "
            f"Cleared target slot [{row},{col}]."
        )
        log.debug(f"  Worker {w_idx} job_apps_targets: {wrk.job_apps_targets[w_idx]}")


def _clean_queue(slice_: Idx1D, wrk: Worker, firm_idx_for_log: int) -> Idx1D:
    """
    Return a *unique* array of still-unemployed worker ids
    from the raw queue slice (may contain -1 sentinels and duplicates),
    preserving the original order of first appearance.
    """
    log.debug(f"    Firm {firm_idx_for_log}: Cleaning queue. Initial raw slice: {slice_}")

    # 1. Drop -1 sentinels
    cleaned_slice = slice_[slice_ >= 0]
    if cleaned_slice.size == 0:
        log.debug(f"    Firm {firm_idx_for_log}: Queue empty after dropping sentinels.")
        return cleaned_slice.astype(np.intp)        # Ensure Idx1D type

    log.debug(f"    Firm {firm_idx_for_log}: Queue after dropping sentinels: {cleaned_slice}")

    # 2. Unique *without* sorting
    first_idx = np.unique(cleaned_slice, return_index=True)[1]   # indices of first occurrences
    unique_slice = cleaned_slice[np.sort(first_idx)]             # keep original order
    log.debug(f"    Firm {firm_idx_for_log}: Queue after unique (order kept): {unique_slice}")

    # 3. Keep only unemployed workers
    unemployed_mask = wrk.employed[unique_slice] == 0
    final_queue = unique_slice[unemployed_mask]
    log.debug(
        f"    Firm {firm_idx_for_log}: "
        f"Final cleaned queue (unique, unemployed): {final_queue}"
    )
    return cast(Idx1D, final_queue)


def _safe_bincount_employed(wrk: Worker, n_firms: int) -> Int1D:
    """
    Return head-counts per firm, *ignoring* any corrupted rows where
    wrk.employed == 1 but wrk.employer < 0.
    Also logs those rows so you can trace them later.
    """
    mask_good = (wrk.employed == 1) & (wrk.employer >= 0)
    mask_bad = (wrk.employed == 1) & (wrk.employer < 0)

    if mask_bad.any():  # Removed log.isEnabledFor
        bad_idx = np.where(mask_bad)[0]
        log.error(  # Changed to error
            f"[CORRUPT WORKER DATA] {bad_idx.size} worker rows have "
            f"employed=1 but employer<0; indices={bad_idx.tolist()}"
        )

    return np.bincount(
        wrk.employer[mask_good].astype(np.int64),  # Ensure employer is int for bincount
        minlength=n_firms,
    ).astype(
        np.int64
    )  # Ensure output is Int1D


def firms_hire_workers(
    wrk: Worker,
    emp: Employer,
    *,
    theta: int,
    rng: Generator = default_rng(),
):
    log.info("")
    log.info("  Firms Hiring Workers -----------------------------------------------")
    hiring_ids = np.where(emp.n_vacancies > 0)[0]
    log.debug(f"  {hiring_ids.size} firms have vacancies and are attempting to hire.")
    rng.shuffle(hiring_ids)  # Process firms in random order
    log.debug(f"  hiring_ids={hiring_ids}")

    total_hires_this_step = 0

    for i in hiring_ids:  # i is firm_idx
        log.debug(f"    --- Processing firm {i} (vacancies: {emp.n_vacancies[i]}) ---")

        n_recv = emp.recv_job_apps_head[i] + 1  # queue length (−1 ⇒ 0)
        log.debug(f"    Firm {i} recv_job_apps_head={emp.recv_job_apps_head[i]}")
        log.debug(f"    Firm {i} received {n_recv} applications (raw count).")
        if n_recv <= 0:
            log.debug(f"    Firm {i} has no applications. Skipping.")
            continue

        queue = emp.recv_job_apps[i, :n_recv].copy()  # copy for logging before cleaning
        log.debug(f"    Firm {i} raw application queue: {queue}")

        # Clean the queue: remove sentinels, duplicates, and already employed workers
        hires_potential = _clean_queue(queue, wrk, firm_idx_for_log=i)

        if hires_potential.size == 0:
            log.debug(
                f"    Firm {i}: no valid (unique, unemployed) applicants in queue."
            )
            # nothing useful in the queue → just flush it
            emp.recv_job_apps_head[i] = -1
            emp.recv_job_apps[i, :n_recv] = -1
            continue

        log.debug(
            f"    Firm {i} has {hires_potential.size} "
            f"valid potential hires: {hires_potential}"
        )

        # cap by remaining vacancies
        num_to_actually_hire = min(hires_potential.size, emp.n_vacancies[i])
        if num_to_actually_hire < hires_potential.size:
            log.debug(
                f"    Firm {i} capping potential hires from {hires_potential.size} "
                f"to {num_to_actually_hire} due to vacancy limit."
            )

        final_hires_for_firm = hires_potential[:num_to_actually_hire]

        if (
            final_hires_for_firm.size == 0
        ):  # Should not happen if hires_potential.size > 0, but defensive check
            log.debug(f"    Firm {i} hiring 0 workers after vacancy cap.")
            emp.recv_job_apps_head[i] = -1  # Still flush
            emp.recv_job_apps[i, :n_recv] = -1
            continue

        log.debug(
            f"    Firm {i} hiring {final_hires_for_firm.size} "
            f"worker(s): {final_hires_for_firm.tolist()}"
        )
        total_hires_this_step += final_hires_for_firm.size

        # ---- worker‑side updates ----------------------------------------
        wrk.employed[final_hires_for_firm] = 1
        wrk.employer[final_hires_for_firm] = i
        wrk.wage[final_hires_for_firm] = emp.wage_offer[i]
        wrk.periods_left[final_hires_for_firm] = theta
        wrk.contract_expired[final_hires_for_firm] = 0  # Reset flags
        wrk.fired[final_hires_for_firm] = 0
        # Workers who are hired should stop applying elsewhere
        wrk.job_apps_head[final_hires_for_firm] = -1
        wrk.job_apps_targets[final_hires_for_firm, :] = -1

        # ---- firm‑side updates ------------------------------------------
        emp.current_labor[i] += final_hires_for_firm.size
        emp.n_vacancies[i] -= final_hires_for_firm.size
        log.debug(
            f"    Firm {i} updated: current_labor={emp.current_labor[i]}, "
            f"n_vacancies={emp.n_vacancies[i]}"
        )

        # flush inbound queue for this firm
        emp.recv_job_apps_head[i] = -1
        emp.recv_job_apps[i, :n_recv] = -1
        log.debug(f"    Firm {i} application queue flushed.")

    log.info(f"  Total hires made this step across all firms: {total_hires_this_step}")

    # -------- global cross-check -----------------------------------------
    if log.isEnabledFor(logging.DEBUG):  # This existing check is good
        true_labor_counts = _safe_bincount_employed(wrk, emp.current_labor.size)
        mismatched_firms = np.flatnonzero(emp.current_labor != true_labor_counts)
        if mismatched_firms.size:
            log.error(  # Changed to error
                f"[GLOBAL LABOR MISMATCH] "
                f"{mismatched_firms.size} firm(s) have labor counts inconsistent "
                f"with worker table after hiring round: "
                f"indices={mismatched_firms.tolist()}"
            )
            for i_mismatch in mismatched_firms:
                log.error(
                    f"  Firm {i_mismatch}: "
                    f"recorded_labor={emp.current_labor[i_mismatch]}, "
                    f"true_labor_from_workers={true_labor_counts[i_mismatch]}"
                )
        else:
            log.debug(
                "[GLOBAL LABOR CONSISTENCY] All firm labor counts match worker table."
            )

    log.debug(f"    recv_job_apps_head={emp.recv_job_apps_head}")
    log.debug(f"    recv_job_apps=\n{emp.recv_job_apps}")


workers_decide_firms_to_apply(wrk, emp, max_M=max_M, rng=rng)
for _ in range(max_M):
    workers_send_one_round(wrk, emp, rng)
    firms_hire_workers(wrk, emp, theta=8, rng=rng)

workers_decide_firms_to_apply(wrk, emp, max_M=max_M, rng=rng)
for _ in range(max_M):
    workers_send_one_round(wrk, emp, rng)
    firms_hire_workers(wrk, emp, theta=8, rng=rng)