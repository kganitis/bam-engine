# src/bamengine/systems/labor_market.py
"""
Event‑2  –  Labour‑market systems
"""

import logging
from typing import cast

import numpy as np
from numpy.random import Generator, default_rng

from bamengine import _logging_ext
from bamengine.components import Economy, Employer, Worker
from bamengine.helpers import select_top_k_indices_sorted
from bamengine.typing import Idx1D, Int1D

log = _logging_ext.getLogger(__name__)


def calc_annual_inflation_rate(ec: Economy) -> None:
    """
    Calculate and store the annual inflation rate for the current period.

    Rule
    ----
        π_t = (P̄_{t} - P̄_{t-4}) / P̄_{t-4}

    t: Current Period, π: Annual Inflation Rate, P̄: Average Market Price
    """
    log.info("--- Calculating Annual Inflation Rate ---")
    hist = ec.avg_mkt_price_history
    if hist.size <= 4:
        log.info(
            "  Not enough history to calculate annual inflation (<5 periods). "
            "Setting to 0.0.")
        ec.inflation_history = np.append(ec.inflation_history, 0.0)
        return

    p_now = hist[-1]
    p_prev = hist[-5]  # Price from 4 periods ago (e.g., if t=5, compare p_5 and p_1)

    if p_prev <= 0:
        log.warning(
            "  Cannot calculate inflation, previous price level was zero or negative. "
            "Setting to 0.0.")
        inflation = 0.0
    else:
        inflation = (p_now - p_prev) / p_prev

    ec.inflation_history = np.append(ec.inflation_history, inflation)
    log.info(
        f"  Annual inflation calculated for period t={hist.size - 1}: {inflation:+.3%}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"    Calculation: (p_now={p_now:.3f} / p_t-4={p_prev:.3f}) - 1")
    log.info("--- Annual Inflation Calculation complete ---")


def calc_inflation_and_adjust_minimum_wage(ec: Economy) -> None:
    """
    [DEPRECATED/ALTERNATIVE]
    Every `min_wage_rev_period` periods, calculate annual inflation π and update ŵ_t.
    """
    log.warning("--- Adjusting Minimum Wage (using Combined Calculation) ---")
    m = ec.min_wage_rev_period
    if ec.avg_mkt_price_history.size <= m:
        log.debug(
            f"  Skipping min wage adjustment: "
            f"not enough history ({ec.avg_mkt_price_history.size} <= {m}).")
        return
    if (ec.avg_mkt_price_history.size - 1) % m != 0:
        log.debug(f"  Skipping min wage adjustment: not a revision period.")
        return

    p_now = ec.avg_mkt_price_history[-1]
    p_prev = ec.avg_mkt_price_history[-m - 1]

    if p_prev <= 0:
        log.warning("  Cannot adjust min wage, past price level was zero or negative.")
        return

    inflation = (p_now - p_prev) / p_prev
    old_min_wage = ec.min_wage
    ec.min_wage = ec.min_wage * (1.0 + inflation)
    log.info(
        f"  Minimum wage revision: "
        f"Inflation over last {m} periods: {inflation:+.3%}. "
        f"Min wage: {old_min_wage:.3f} → {ec.min_wage:.3f}")
    log.warning("--- Minimum Wage Adjustment (Combined) complete ---")


def adjust_minimum_wage(ec: Economy) -> None:
    """
    Every `min_wage_rev_period` periods update ŵ_t by realised inflation π.

    Rule
    ----
        ŵ_t = ŵ_{t-1} · π_t

    t: Current Period, ŵ: Minimum Wage, π: Annual Inflation Rate,
    """
    log.info("--- Adjusting Minimum Wage (based on history) ---")
    m = ec.min_wage_rev_period
    if ec.avg_mkt_price_history.size <= m:
        log.debug(
            f"  Skipping: not enough history ({ec.avg_mkt_price_history.size} <= {m}).")
        return
    if (ec.avg_mkt_price_history.size - 1) % m != 0:
        log.debug(f"  Skipping: not a revision period.")
        return

    # TODO: Make sure inflation history is up to date
    inflation = ec.inflation_history[-1]
    old_min_wage = ec.min_wage
    ec.min_wage = ec.min_wage * (1.0 + inflation)
    log.info(
        f"  Minimum wage revision: "
        f"Using most recent annual inflation from history ({inflation:+.3%}).")
    log.info(f"  Min wage: {old_min_wage:.3f} → {ec.min_wage:.3f}")
    log.info("--- Minimum Wage Adjustment complete ---")


def firms_decide_wage_offer(
        emp: Employer,
        *,
        w_min: float,
        h_xi: float,
        rng: Generator = default_rng(),
) -> None:
    """
    Firms with vacancies post a wage offer as a markup over their previous offer.

    Rule
    ----
        shock ~ U(0, h_ξ)
        w_t = max( ŵ_t, w_{t-1} · (1 + shock) )

    t: Current Period, w: Offered Wage, ŵ: Minimum Wage, h_ξ: Max Wage Growth
    """
    # TODO Observe how many firm offers are near the minimum wage
    log.info("--- Firms Deciding Wage Offers ---")
    log.info(
        f"  Inputs: Min Wage (w_min)={w_min:.3f} | Max Wage Shock (h_ξ)={h_xi:.3f}")
    shape = emp.wage_offer.shape

    # permanent scratch
    shock = emp.wage_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64)
        emp.wage_shock = shock

    # Draw one shock per firm, then mask where V_i==0.
    shock[:] = rng.uniform(0.0, h_xi, size=shape)
    shock[emp.n_vacancies == 0] = 0.0

    # core rule
    np.multiply(emp.wage_offer, 1.0 + shock, out=emp.wage_offer)
    np.maximum(emp.wage_offer, w_min, out=emp.wage_offer)

    hiring_firms_mask = emp.n_vacancies > 0
    num_hiring_firms = np.sum(hiring_firms_mask)
    avg_offer_hiring = emp.wage_offer[
        hiring_firms_mask].mean() if num_hiring_firms > 0 else 0.0

    log.info(f"  {num_hiring_firms} firms with vacancies are setting wage offers.")
    log.info(
        f"  Min wage: {w_min:.3f}. "
        f"Average offer from hiring firms: {avg_offer_hiring:.3f}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Wage offers (first 10 firms): "
            f"{np.array2string(emp.wage_offer[:10], precision=2)}")
    log.info("--- Wage Offer Decision complete ---")


def workers_decide_firms_to_apply(
        wrk: Worker,
        emp: Employer,
        *,
        max_M: int,
        rng: Generator = default_rng(),
) -> None:
    """
    Unemployed workers choose up to `max_M` firms to apply to, sorted by wage.
    Workers remain loyal to their last employer if their contract has just expired.
    """
    log.info("--- Workers Deciding Firms to Apply ---")
    hiring = np.where(emp.n_vacancies > 0)[0]
    unemp = np.where(wrk.employed == 0)[0]

    log.info(f"  {unemp.size} unemployed workers "
             f"prepare up to {max_M} applications each "
             f"to {hiring.size} firms "
             f"with a total of {emp.n_vacancies.sum():,} open vacancies.")

    # fast exits
    if unemp.size == 0:
        log.info("  No unemployed workers; skipping application phase.")
        log.info("--- Workers Deciding Firms to Apply complete ---")
        wrk.job_apps_head.fill(-1)
        return

    if hiring.size == 0:
        log.info("  No firm is hiring this period – all application queues cleared.")
        log.info("--- Workers Deciding Firms to Apply complete ---")
        wrk.job_apps_head[unemp] = -1
        wrk.job_apps_targets[unemp, :].fill(-1)
        return

    # sample M random hiring firms per worker (with replacement)
    M_eff = min(max_M, hiring.size)
    log.info(f"  Effective applications per worker (M_eff): {M_eff}")
    sample = np.empty((unemp.size, M_eff), dtype=np.int64)
    for row, j in enumerate(unemp):
        sample[row] = rng.choice(hiring, size=M_eff, replace=False)
        if log.isEnabledFor(_logging_ext.DEEP_DEBUG):
            log.deep(f"  Worker {j}: initial sample={sample[row]}, "
                     f"previous: {wrk.employer_prev[j]}, "
                     f"contract_expired: {wrk.contract_expired[j]}, "
                     f"fired: {wrk.fired[j]}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Initial random firm sample (first 10 workers, if any):\n"
            f"{sample[:10]}"
        )

    # wage-descending partial sort
    topk = select_top_k_indices_sorted(
        emp.wage_offer[sample], k=M_eff, descending=True
    )
    sorted_sample = np.take_along_axis(sample, topk, axis=1)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Sorted firm sample by wage (first 10 workers, if any):\n"
            f"{sorted_sample[:10]}"
        )

    # loyalty rule
    # TODO Move loyalty logic to a separate system
    loyal_mask = (
            (wrk.contract_expired[unemp] == 1)
            & (wrk.fired[unemp] == 0)
            & np.isin(wrk.employer_prev[unemp], hiring)
    )
    num_loyal_workers = np.sum(loyal_mask)
    log.info(f"  Applying loyalty rule for {num_loyal_workers} worker(s).")

    if loyal_mask.any():
        loyal_row_indices = np.where(loyal_mask)[0]

        for row in loyal_row_indices:
            actual_worker_id = unemp[row]
            prev_employer_id = wrk.employer_prev[actual_worker_id]
            application_row = sorted_sample[row]
            num_applications = application_row.shape[0]

            if log.isEnabledFor(_logging_ext.DEEP_DEBUG):
                log.deep(
                    f"      Adjusting for loyalty: "
                    f"Worker ID {actual_worker_id} (row {row}), "
                    f"Prev Emp: {prev_employer_id}")
                log.deep(f"      Application row BEFORE: {application_row.copy()}")

            try:
                current_pos_of_prev_emp = (
                    np.where(application_row == prev_employer_id))[0][0]
                if current_pos_of_prev_emp != 0:
                    employer_to_move = application_row[current_pos_of_prev_emp]
                    for j in range(current_pos_of_prev_emp, 0, -1):
                        application_row[j] = application_row[j - 1]
                    application_row[0] = employer_to_move
                # No log needed for 'else' case as it's a no-op.
            except IndexError:
                if num_applications > 0:
                    if num_applications > 1:
                        application_row[1:num_applications] = application_row[
                                                              0: num_applications - 1]
                    application_row[0] = prev_employer_id

            if log.isEnabledFor(_logging_ext.DEEP_DEBUG):
                log.deep(f"      Application row AFTER:  {application_row}")

        if log.isEnabledFor(logging.DEBUG) and loyal_mask.any():
            log.debug(
                f"    Sorted sample AFTER post-sort loyalty adjustment "
                f"(first 10 rows if any loyal):\n{sorted_sample[:10]}"
            )

    # write buffers
    stride = max_M
    for k, j in enumerate(unemp):
        wrk.job_apps_targets[j, :M_eff] = sorted_sample[k]
        if M_eff < max_M:
            wrk.job_apps_targets[j, M_eff:max_M] = -1
        wrk.job_apps_head[j] = j * stride

        if k < 10 and log.isEnabledFor(logging.DEBUG):  # first 10 workers
            log.debug(
                f"    Worker {j}: targets={wrk.job_apps_targets[j]}, "
                f"head_ptr={wrk.job_apps_head[j]}"
            )

    # reset flags
    wrk.contract_expired[unemp] = 0
    wrk.fired[unemp] = 0

    log.info(f"  {unemp.size} unemployed workers prepared {M_eff} applications each.")
    log.info(f"--- Workers Deciding Firms to Apply complete ---")


def workers_send_one_round(
        wrk: Worker, emp: Employer, rng: Generator = default_rng()
) -> None:
    """A single round of job applications being sent and received."""
    log.info("--- Workers Sending One Round of Applications ---")
    stride = wrk.job_apps_targets.shape[1]
    unemp_ids = np.where(wrk.employed == 0)[0]
    active_applicants_mask = wrk.job_apps_head[unemp_ids] >= 0
    unemp_ids_applying = unemp_ids[active_applicants_mask]

    if unemp_ids_applying.size == 0:
        log.info(f"  No workers with pending applications found. Skipping round.")
        log.info("--- Application Sending Round complete ---")
        return

    log.info(
        f"  Processing {unemp_ids_applying.size} workers with pending applications "
        f"(Stride={stride})."
    )

    rng.shuffle(unemp_ids_applying)  # order randomly chosen at each time step

    # Counters for logging
    apps_sent_successfully = 0
    apps_dropped_queue_full = 0
    apps_dropped_no_vacancy = 0

    for j in unemp_ids_applying:
        head = wrk.job_apps_head[j]
        if head < 0:
            log.warning(
                f"  Worker {j} in applying list but head is {head}. Skipping.")
            continue

        row_from_head, col = divmod(head, stride)
        if row_from_head != j:
            log.error(
                f"  CRITICAL MISMATCH for worker {j}: "
                f"head={head} decoded to row {row_from_head}.")

        if col >= stride:
            # Normal exit condition for a worker who finished their list.
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Worker {j} exhausted all {stride} application slots. "
                    f"Setting head to -1.")
            wrk.job_apps_head[j] = -1
            continue

        firm_id = wrk.job_apps_targets[row_from_head, col]
        if firm_id < 0:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Worker {j} encountered sentinel (-1) at col {col}. "
                    f"End of list. Setting head to -1.")
            wrk.job_apps_head[j] = -1
            continue

        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"    Worker {j} applying to firm {firm_id} (app #{col + 1}).")

        # Check for vacancy before checking queue space
        if emp.n_vacancies[firm_id] <= 0:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"  Firm {firm_id} has no more open vacancies. "
                    f"Worker {j} application dropped.")
            apps_dropped_no_vacancy += 1
            wrk.job_apps_head[j] = head + 1
            wrk.job_apps_targets[row_from_head, col] = -1
            continue

        # Check firm's application queue available space
        ptr = emp.recv_job_apps_head[firm_id] + 1
        if ptr >= emp.recv_job_apps.shape[1]:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Firm {firm_id} application queue full. "
                    f"Worker {j} application dropped.")
            apps_dropped_queue_full += 1
            wrk.job_apps_head[j] = head + 1
            wrk.job_apps_targets[row_from_head, col] = -1
            continue

        # Application is successful
        emp.recv_job_apps_head[firm_id] = ptr
        emp.recv_job_apps[firm_id, ptr] = j
        apps_sent_successfully += 1
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Worker {j} application queued at firm {firm_id} slot {ptr}.")

        wrk.job_apps_head[j] = head + 1
        wrk.job_apps_targets[row_from_head, col] = -1

    # Summary log
    total_dropped = apps_dropped_queue_full + apps_dropped_no_vacancy
    log.info(
        f"  Round Summary: "
        f"{apps_sent_successfully} applications successfully queued, "
        f"{total_dropped} dropped.")
    if total_dropped > 0 and log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"    Dropped breakdown -> Queue Full: {apps_dropped_queue_full},"
            f" No Vacancy: {apps_dropped_no_vacancy}")
    log.info("--- Application Sending Round complete ---")


def _check_labor_consistency(tag: str, i: int, wrk: Worker, emp: Employer) -> bool:
    """
    Compare firm‐side bookkeeping (`emp.current_labor[i]`)
    with the ground truth reconstructed from the Worker table.
    """
    true_headcount = np.count_nonzero((wrk.employed == 1) & (wrk.employer == i))
    recorded = int(emp.current_labor[i])

    if true_headcount != recorded:
        log.warning(
            f"[{tag:^10s}] LABOR INCONSISTENCY: Firm {i:3d} | "
            f"Recorded Labor: {recorded:3d}, True Headcount: {true_headcount:3d}, "
            f"Δ={true_headcount - recorded:+d}"
        )
        return False
    elif log.isEnabledFor(_logging_ext.DEEP_DEBUG):
        log.deep(
            f"[{tag:^10s}] Labor consistent for firm {i:3d}: {recorded:3d} workers."
        )
    return True


def _safe_bincount_employed(wrk: Worker, n_firms: int) -> Int1D:
    """
    Return head-counts per firm, *ignoring* any corrupted rows where
    wrk.employed == 1 but wrk.employer < 0.
    Also log those rows in order to trace them later.
    """
    mask_good = (wrk.employed == 1) & (wrk.employer >= 0)
    mask_bad = (wrk.employed == 1) & (wrk.employer < 0)

    if mask_bad.any():
        bad_idx = np.where(mask_bad)[0]
        log.error(
            f"[CORRUPT WORKER DATA] {bad_idx.size} worker rows have "
            f"employed=1 but employer<0; indices={bad_idx.tolist()}"
        )

    return np.bincount(
        wrk.employer[mask_good].astype(np.int64),
        minlength=n_firms,
    ).astype(np.int64)


def _clean_queue(slice_: Idx1D, wrk: Worker, firm_idx_for_log: int) -> Idx1D:
    """
    Return a *unique* array of still-unemployed worker ids
    from the raw queue slice (may contain -1 sentinels and duplicates),
    preserving the original order of first appearance.
    """
    if log.isEnabledFor(_logging_ext.DEEP_DEBUG):
        log.deep(
            f"    Firm {firm_idx_for_log}: "
            f"Cleaning queue. Initial raw slice: {slice_}")

    # Drop -1 sentinels
    cleaned_slice = slice_[slice_ >= 0]
    if cleaned_slice.size == 0:
        if log.isEnabledFor(_logging_ext.DEEP_DEBUG):
            log.deep(
                f"    Firm {firm_idx_for_log}: "
                f"Queue empty after dropping sentinels.")
        return cleaned_slice.astype(np.intp)

    if log.isEnabledFor(_logging_ext.DEEP_DEBUG):
        log.deep(
            f"    Firm {firm_idx_for_log}: "
            f"Queue after dropping sentinels: {cleaned_slice}")

    # Unique *without* sorting
    first_idx = np.unique(cleaned_slice, return_index=True)[1]
    unique_slice = cleaned_slice[np.sort(first_idx)]
    if log.isEnabledFor(_logging_ext.DEEP_DEBUG):
        log.deep(
            f"    Firm {firm_idx_for_log}: "
            f"Queue after unique (order kept): {unique_slice}")

    # Keep only unemployed workers
    unemployed_mask = wrk.employed[unique_slice] == 0
    final_queue = unique_slice[unemployed_mask]
    if log.isEnabledFor(_logging_ext.DEEP_DEBUG):
        log.deep(
            f"    Firm {firm_idx_for_log}: "
            f"Final cleaned queue (unique, unemployed): {final_queue}")

    return cast(Idx1D, final_queue)


def firms_hire_workers(
        wrk: Worker,
        emp: Employer,
        *,
        theta: int,
        rng: Generator = default_rng(),
) -> None:
    """Match firms with queued applicants and update all related state."""
    log.info("--- Firms Hiring Workers ---")
    hiring_ids = np.where(emp.n_vacancies > 0)[0]
    total_vacancies = emp.n_vacancies.sum()
    log.info(
        f"  {hiring_ids.size} firms have {total_vacancies:,} "
        f"total vacancies and are attempting to hire.")

    total_hires_this_round = 0

    for i in hiring_ids:
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"  Processing firm {i} (vacancies: {emp.n_vacancies[i]})")

        _check_labor_consistency("PRE-hire", i, wrk, emp)

        n_recv = emp.recv_job_apps_head[i] + 1
        if n_recv <= 0:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"    Firm {i} has no applications. Skipping.")
            continue

        raw_queue = emp.recv_job_apps[i, :n_recv].copy()
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Firm {i} raw application queue "
                f"({n_recv} applications): {raw_queue}")

        queue = _clean_queue(raw_queue, wrk, firm_idx_for_log=i)

        if queue.size == 0:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Firm {i}: no valid (unique, unemployed) "
                    f"applicants in queue. Flushing.")
            emp.recv_job_apps_head[i] = -1
            emp.recv_job_apps[i, :n_recv] = -1
            continue

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Firm {i} has {queue.size} valid potential hires: {queue}")

        num_to_hire = min(queue.size, emp.n_vacancies[i])
        if num_to_hire < queue.size:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"    Firm {i} capping hires from {queue.size} "
                    f"to {num_to_hire} due to vacancy limit.")

        final_hires = queue[:num_to_hire]

        if final_hires.size == 0:
            emp.recv_job_apps_head[i] = -1
            emp.recv_job_apps[i, :n_recv] = -1
            continue

        log.info(
            f"    Firm {i} is hiring {final_hires.size} worker(s): "
            f"{final_hires.tolist()}")
        total_hires_this_round += final_hires.size

        # worker‑side updates
        log.debug(f"      Updating state for {final_hires.size} newly hired workers.")
        wrk.employed[final_hires] = 1
        wrk.employer[final_hires] = i
        wrk.wage[final_hires] = emp.wage_offer[i]
        wrk.periods_left[final_hires] = theta + rng.poisson(10)
        wrk.contract_expired[final_hires] = 0
        wrk.fired[final_hires] = 0
        wrk.job_apps_head[final_hires] = -1
        wrk.job_apps_targets[final_hires, :] = -1

        # firm‑side updates
        emp.current_labor[i] += final_hires.size
        emp.n_vacancies[i] -= final_hires.size
        log.debug(
            f"      Firm {i} state updated: "
            f"current_labor={emp.current_labor[i]}, "
            f"n_vacancies={emp.n_vacancies[i]}")

        # flush inbound queue for this firm
        emp.recv_job_apps_head[i] = -1
        emp.recv_job_apps[i, :n_recv] = -1
        if log.isEnabledFor(logging.DEBUG):
            log.debug(f"    Firm {i} application queue flushed.")

        _check_labor_consistency("POST-hire", i, wrk, emp)

    log.info(f"  Total hires made this step across all firms: {total_hires_this_round}")
    if log.isEnabledFor(logging.DEBUG):
        true_labor_counts = _safe_bincount_employed(wrk, emp.current_labor.size)
        mismatched_firms = np.flatnonzero(emp.current_labor != true_labor_counts)
        if mismatched_firms.size:
            log.error(
                f"[GLOBAL LABOR MISMATCH] {mismatched_firms.size} firms "
                f"have inconsistent labor counts.")
            for i_mismatch in mismatched_firms:
                log.error(
                    f"  Firm {i_mismatch}: recorded={emp.current_labor[i_mismatch]}, "
                    f"true={true_labor_counts[i_mismatch]}")
        else:
            log.debug(
                "[GLOBAL LABOR CONSISTENCY] "
                "All firm labor counts match worker table after hiring.")
    log.info("--- Firms Hiring Workers complete ---")


def firms_calc_wage_bill(emp: Employer, wrk: Worker) -> None:
    """
    Rule
    ----
        W_i = Σ w_j for all j employed by firm i

    W: Wage Bill, w: (Individual) Wage
    """
    log.info("--- Firms Calculating Wage Bill ---")

    employed_mask = wrk.employed == 1
    num_employed = np.sum(employed_mask)
    log.info(
        f"  Calculating wage bill based on "
        f"{num_employed:,} currently employed workers.")

    n_firms = emp.wage_offer.size
    emp.wage_bill[:] = np.bincount(
        wrk.employer[employed_mask],
        weights=wrk.wage[employed_mask],
        minlength=n_firms
    )

    total_wage_bill = emp.wage_bill.sum()
    avg_wage_of_employed = wrk.wage[employed_mask].mean()
    log.info(
        f"  Total economy-wide wage bill calculated: {total_wage_bill:,.2f} "
        f"(Avg wage for employed workers: {avg_wage_of_employed:.3f})"
    )
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Final Wage Bill per firm (first 10 firms): "
            f"{np.array2string(emp.wage_bill[:10], precision=2)}")
    log.info("--- Wage Bill Calculation complete ---")
