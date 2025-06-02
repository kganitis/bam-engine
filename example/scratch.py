from numpy.random import Generator, default_rng

import numpy as np
import logging

from bamengine.components import Employer, Worker
from bamengine.helpers import select_top_k_indices_sorted
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
        n_vacancies=np.array([3, 1, 0, 1, 0, 1], dtype=np.int64),
        current_labor=np.array([0, 0, 1, 0, 1, 1], dtype=np.int64),
    )

    wrk = mock_worker(
        n=n_workers,
        queue_m=M,
        employed=np.array([False, False, False, False, False,
                           False, False, True, True, True]),
        employer_prev=np.array([2, 4, 0, 1, 3, 6, 2, -1, -1, -1], dtype=np.int64),
        contract_expired=np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 0], dtype=np.bool_),
        fired=np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.bool_)
    )

    return emp, wrk, rng, M


emp, wrk, rng, max_M = _mini_state()

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
                f"      Application row BEFORE adjustment: {application_row.copy()}")

        try:
            # Check if the previous employer is already in the application list
            current_pos_of_prev_emp = np.where(application_row == prev_employer_id)[0][
                0]

            # If it is present and not already at the first position
            if current_pos_of_prev_emp != 0:
                log.debug(
                    f"      Previous employer {prev_employer_id} found at position {current_pos_of_prev_emp}. Moving to front.")
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
                    f"      Previous employer {prev_employer_id} is already at the first position. No change needed.")

        except IndexError:  # np.where(...)[0][0] will fail if prev_employer_id is not found
            # Previous employer was NOT in the list.
            # Place it at the front and shift other elements to the right, dropping the last one.
            log.debug(
                f"      Previous employer {prev_employer_id} not found in application list. Inserting at front.")
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
            log.debug(f"      Application row AFTER adjustment:  {application_row}")

    # Log the state of sorted_sample AFTER all loyal workers have been processed
    if log.isEnabledFor(
            logging.DEBUG) and loyal_mask.any():  # Check loyal_mask.any() again as it was the entry condition
        log.debug(
            f"  Sorted sample AFTER post-sort loyalty adjustment (all rows if loyal_mask.any()):\n{sorted_sample}\n"
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
        f"    Worker {w_idx}: targets={wrk.job_apps_targets[w_idx]}, "
        f"head_raw_val={wrk.job_apps_head[w_idx]}"
    )
log.debug(f"job_apps_head: {wrk.job_apps_head}")
log.debug(f"job_apps_targets:\n{wrk.job_apps_targets}")

# reset flags
wrk.contract_expired[unemp] = 0
wrk.fired[unemp] = 0

log.info(f"  {unemp.size} unemployed workers prepared {M_eff} applications each.")
