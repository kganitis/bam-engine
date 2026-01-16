"""
System functions for labor market phase events.

This module contains the internal implementation functions for labor market events.
Event classes wrap these functions and provide the primary documentation.

See Also
--------
bamengine.events.labor_market : Event classes (primary documentation source)
"""

from __future__ import annotations

from typing import cast

import numpy as np

from bamengine import Rng, logging, make_rng
from bamengine.economy import Economy
from bamengine.roles import Employer, Worker
from bamengine.typing import Idx1D, Int1D
from bamengine.utils import select_top_k_indices_sorted

log = logging.getLogger(__name__)


def calc_annual_inflation_rate(ec: Economy) -> None:
    """
    Calculate year-over-year inflation from price history.

    See Also
    --------
    bamengine.events.labor_market.CalcAnnualInflationRate : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Calculating Annual Inflation Rate ---")
    hist = ec.avg_mkt_price_history
    if hist.size <= 4:
        if info_enabled:
            log.info(
                "  Not enough history to calculate annual inflation (<5 periods). "
                "Setting to 0.0."
            )
        ec.inflation_history = np.append(ec.inflation_history, 0.0)
        return

    p_now = hist[-1]
    p_prev = hist[-5]  # Price from 4 periods ago (e.g., if t=5, compare p_5 and p_1)

    if p_prev <= 0:
        log.warning(
            "  Cannot calculate inflation, previous price level was zero or negative. "
            "Setting to 0.0."
        )
        inflation = 0.0
    else:
        inflation = (p_now - p_prev) / p_prev

    ec.inflation_history = np.append(ec.inflation_history, inflation)
    if info_enabled:
        log.info(
            f"  Annual inflation calculated for period t={hist.size - 1}: {inflation:+.3%}"
        )
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"    Calculation: (p_now={p_now:.3f} / p_t-4={p_prev:.3f}) - 1")
    if info_enabled:
        log.info("--- Annual Inflation Calculation complete ---")


def adjust_minimum_wage(ec: Economy, wrk: Worker) -> None:
    """
    Periodically index minimum wage to inflation.

    See Also
    --------
    bamengine.events.labor_market.AdjustMinimumWage : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Adjusting Minimum Wage (based on history) ---")
    m = ec.min_wage_rev_period
    if ec.avg_mkt_price_history.size <= m:
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"  Skipping: not enough history ({ec.avg_mkt_price_history.size} <= {m})."
            )
        return
    if (ec.avg_mkt_price_history.size - 1) % m != 0:
        if log.isEnabledFor(logging.DEBUG):
            log.debug("  Skipping: not a revision period.")
        return

    inflation = float(ec.inflation_history[-1])
    old_min_wage = ec.min_wage
    ec.min_wage = float(ec.min_wage) * (1.0 + inflation)
    if info_enabled:
        log.info(
            f"  Minimum wage revision: "
            f"Using most recent annual inflation from history ({inflation:+.3%})."
        )
        log.info(f"  Min wage: {old_min_wage:.3f} → {ec.min_wage:.3f}")

    # Update existing worker wages to meet new minimum
    employed_mask = wrk.employer >= 0
    below_min_mask = employed_mask & (wrk.wage < ec.min_wage)
    if below_min_mask.any():
        count = int(below_min_mask.sum())
        wrk.wage[below_min_mask] = ec.min_wage
        if info_enabled:
            log.info(f"  Updated {count} employed workers to new minimum wage")

    if info_enabled:
        log.info("--- Minimum Wage Adjustment complete ---")


def firms_decide_wage_offer(
    emp: Employer,
    *,
    w_min: float,
    h_xi: float,
    rng: Rng = make_rng(),
) -> None:
    """
    Firms set wage offers with random markup.

    See Also
    --------
    bamengine.events.labor_market.FirmsDecideWageOffer : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Deciding Wage Offers ---")
        log.info(
            f"  Inputs: Min Wage (w_min)={w_min:.3f} | Max Wage Shock (h_ξ)={h_xi:.3f}"
        )
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

    if info_enabled:
        hiring_firms_mask = emp.n_vacancies > 0
        num_hiring_firms = np.sum(hiring_firms_mask)
        avg_offer_hiring = (
            emp.wage_offer[hiring_firms_mask].mean() if num_hiring_firms > 0 else 0.0
        )

        log.info(f"  {num_hiring_firms} firms with vacancies are setting wage offers.")
        log.info(
            f"  Min wage: {w_min:.3f}. "
            f"Average offer from hiring firms: {avg_offer_hiring:.3f}"
        )

    # Log firms with offers near minimum wage (within x% threshold)
    if log.isEnabledFor(logging.DEBUG):
        hiring_firms_mask = emp.n_vacancies > 0
        num_hiring_firms = np.sum(hiring_firms_mask)
        if num_hiring_firms > 0:
            threshold = 0.05  # 5%
            near_min_threshold = w_min * (1.0 + threshold)
            near_min_mask = hiring_firms_mask & (emp.wage_offer <= near_min_threshold)
            num_near_min = int(np.sum(near_min_mask))
            pct_near_min = (
                (num_near_min / num_hiring_firms * 100.0)
                if num_hiring_firms > 0
                else 0.0
            )
            log.debug(
                f"  {num_near_min} ({pct_near_min:.1f}%) hiring firms "
                f"offering wages within {threshold * 100:.0f}% of minimum "
                f"({near_min_threshold:.3f})"
            )
        log.debug(
            f"  Wage offers (first 10 firms): "
            f"{np.array2string(emp.wage_offer[:10], precision=2)}"
        )
    if info_enabled:
        log.info("--- Wage Offer Decision complete ---")


def workers_decide_firms_to_apply(
    wrk: Worker,
    emp: Employer,
    *,
    max_M: int,
    job_search_method: str = "vacancies_only",
    rng: Rng = make_rng(),
) -> None:
    """
    Unemployed workers build job application queue sorted by wage.

    See Also
    --------
    bamengine.events.labor_market.WorkersDecideFirmsToApply : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Workers Deciding Firms to Apply ---")
    if job_search_method == "vacancies_only":
        hiring = np.where(emp.n_vacancies > 0)[0]
    else:  # "all_firms"
        hiring = np.arange(emp.n_vacancies.size)
    unemp = np.where(wrk.employed == 0)[0]

    if info_enabled:
        log.info(
            f"  {unemp.size} unemployed workers "
            f"prepare up to {max_M} applications each "
            f"to {hiring.size} firms "
            f"with a total of {emp.n_vacancies.sum():,} open vacancies."
        )

    # fast exits
    if unemp.size == 0:
        if info_enabled:
            log.info("  No unemployed workers; skipping application phase.")
            log.info("--- Workers Deciding Firms to Apply complete ---")
        wrk.job_apps_head.fill(-1)
        return

    if hiring.size == 0:
        if info_enabled:
            log.info(
                "  No firm is hiring this period – all application queues cleared."
            )
            log.info("--- Workers Deciding Firms to Apply complete ---")
        wrk.job_apps_head[unemp] = -1
        wrk.job_apps_targets[unemp, :].fill(-1)
        return

    # sample M random hiring firms per worker (with replacement)
    M_eff = min(max_M, hiring.size)
    if info_enabled:
        log.info(f"  Effective applications per worker (M_eff): {M_eff}")
    sample = np.empty((unemp.size, M_eff), dtype=np.int64)
    trace_enabled = log.isEnabledFor(logging.TRACE)
    for row, j in enumerate(unemp):
        sample[row] = rng.choice(hiring, size=M_eff, replace=False)
        if trace_enabled:
            log.trace(
                f"  Worker {j}: initial sample={sample[row]}, "
                f"previous: {wrk.employer_prev[j]}, "
                f"contract_expired: {wrk.contract_expired[j]}, "
                f"fired: {wrk.fired[j]}"
            )
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Initial random firm sample (first 10 workers, if any):\n{sample[:10]}"
        )

    # wage-descending partial sort
    topk = select_top_k_indices_sorted(emp.wage_offer[sample], k=M_eff, descending=True)
    sorted_sample = np.take_along_axis(sample, topk, axis=1)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Sorted firm sample by wage (first 10 workers, if any):\n"
            f"{sorted_sample[:10]}"
        )

    # loyalty rule
    loyal_mask = (
        (wrk.contract_expired[unemp] == 1)
        & (wrk.fired[unemp] == 0)
        & np.isin(wrk.employer_prev[unemp], hiring)
    )
    num_loyal_workers = np.sum(loyal_mask)
    if info_enabled:
        log.info(f"  Applying loyalty rule for {num_loyal_workers} worker(s).")

    if loyal_mask.any():
        loyal_row_indices = np.where(loyal_mask)[0]

        for row in loyal_row_indices:
            actual_worker_id = unemp[row]
            prev_employer_id = wrk.employer_prev[actual_worker_id]
            application_row = sorted_sample[row]
            num_applications = application_row.shape[0]

            if trace_enabled:
                log.trace(
                    f"      Adjusting for loyalty: "
                    f"Worker ID {actual_worker_id} (row {row}), "
                    f"Prev Emp: {prev_employer_id}"
                )
                log.trace(f"      Application row BEFORE: {application_row.copy()}")

            try:
                current_pos_of_prev_emp = (
                    np.where(application_row == prev_employer_id)
                )[0][0]
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
                            0 : num_applications - 1
                        ]
                    application_row[0] = prev_employer_id

            if trace_enabled:
                log.trace(f"      Application row AFTER:  {application_row}")

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

    if info_enabled:
        log.info(
            f"  {unemp.size} unemployed workers prepared {M_eff} applications each."
        )
        log.info("--- Workers Deciding Firms to Apply complete ---")


def workers_send_one_round(
    wrk: Worker,
    emp: Employer,
    rng: Rng = make_rng(),
    *,
    matching_method: str = "sequential",
) -> None:
    """
    Process one round of job applications from workers to firms.

    See Also
    --------
    bamengine.events.labor_market.WorkersSendOneRound : Full documentation
    """
    if matching_method == "simultaneous":
        _workers_send_one_round_simultaneous(wrk, emp, rng)
    else:
        _workers_send_one_round_sequential(wrk, emp, rng)


def _workers_send_one_round_sequential(wrk: Worker, emp: Employer, rng: Rng) -> None:
    """
    Sequential matching: workers shuffled, apply one at a time.

    This is the traditional efficient matching where workers are processed
    in random order and each application is handled immediately.
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Workers Sending One Round of Applications (Sequential) ---")
    stride = wrk.job_apps_targets.shape[1]
    unemp_ids = np.where(wrk.employed == 0)[0]
    active_applicants_mask = wrk.job_apps_head[unemp_ids] >= 0
    unemp_ids_applying = unemp_ids[active_applicants_mask]

    if unemp_ids_applying.size == 0:
        if info_enabled:
            log.info("  No workers with pending applications found. Skipping round.")
            log.info("--- Application Sending Round complete ---")
        return

    if info_enabled:
        log.info(
            f"  Processing {unemp_ids_applying.size} workers with pending applications "
            f"(Stride={stride})."
        )

    rng.shuffle(unemp_ids_applying)  # order randomly chosen at each time step

    # Counters for logging
    apps_sent_successfully = 0
    apps_dropped_queue_full = 0
    apps_dropped_no_vacancy = 0

    # Cache log level check outside the loop
    debug_enabled = log.isEnabledFor(logging.DEBUG)

    for j in unemp_ids_applying:
        head = wrk.job_apps_head[j]
        if head < 0:  # TODO branch is uncovered by unit tests
            log.warning(f"  Worker {j} in applying list but head is {head}. Skipping.")
            continue

        row_from_head, col = divmod(head, stride)
        if row_from_head != j:
            log.error(
                f"  CRITICAL MISMATCH for worker {j}: "
                f"head={head} decoded to row {row_from_head}."
            )

        if head >= (j + 1) * stride:  # TODO branch is uncovered by unit tests
            # Normal exit condition for a worker who finished their list.
            if debug_enabled:
                log.debug(
                    f"    Worker {j} exhausted all {stride} application slots. "
                    f"Setting head to -1."
                )
            wrk.job_apps_head[j] = -1
            continue

        firm_id = wrk.job_apps_targets[row_from_head, col]
        if firm_id < 0:
            if debug_enabled:
                log.debug(
                    f"    Worker {j} encountered sentinel (-1) at col {col}. "
                    f"End of list. Setting head to -1."
                )
            wrk.job_apps_head[j] = -1
            continue

        if debug_enabled:
            log.debug(f"    Worker {j} applying to firm {firm_id} (app #{col + 1}).")

        # Check for vacancy before checking queue space
        if emp.n_vacancies[firm_id] <= 0:
            if debug_enabled:
                log.debug(
                    f"  Firm {firm_id} has no more open vacancies. "
                    f"Worker {j} application dropped."
                )
            apps_dropped_no_vacancy += 1
            wrk.job_apps_head[j] = head + 1
            wrk.job_apps_targets[row_from_head, col] = -1
            continue

        # Check firm's application queue available space
        ptr = emp.recv_job_apps_head[firm_id] + 1
        if ptr >= emp.recv_job_apps.shape[1]:
            if debug_enabled:
                log.debug(
                    f"    Firm {firm_id} application queue full. "
                    f"Worker {j} application dropped."
                )
            apps_dropped_queue_full += 1
            wrk.job_apps_head[j] = head + 1
            wrk.job_apps_targets[row_from_head, col] = -1
            continue

        # Application is successful
        emp.recv_job_apps_head[firm_id] = ptr
        emp.recv_job_apps[firm_id, ptr] = j
        apps_sent_successfully += 1
        if debug_enabled:
            log.debug(
                f"    Worker {j} application queued at firm {firm_id} slot {ptr}."
            )

        wrk.job_apps_head[j] = head + 1
        wrk.job_apps_targets[row_from_head, col] = -1

    # Summary log
    total_dropped = apps_dropped_queue_full + apps_dropped_no_vacancy
    if info_enabled:
        log.info(
            f"  Round Summary: "
            f"{apps_sent_successfully} applications successfully queued, "
            f"{total_dropped} dropped."
        )
    if total_dropped > 0 and debug_enabled:
        log.debug(
            f"    Dropped breakdown -> Queue Full: {apps_dropped_queue_full},"
            f" No Vacancy: {apps_dropped_no_vacancy}"
        )
    if info_enabled:
        log.info("--- Application Sending Round complete ---")


def _workers_send_one_round_simultaneous(wrk: Worker, emp: Employer, rng: Rng) -> None:
    """
    Simultaneous matching: all workers apply at once, creating crowding.

    This matching method creates coordination failure by having all workers
    simultaneously choose their best remaining firm. Multiple workers "crowd"
    at popular (high-wage) firms, and firms randomly select from applicants.
    This creates natural unemployment even when vacancies exceed jobseekers.

    Algorithm
    ---------
    1. ALL unemployed workers simultaneously pick their current best target
    2. Workers "crowd" at popular firms (coordination failure)
    3. Each firm's queue receives all simultaneous applicants
    4. Firms will later hire a random subset up to vacancy limit
    5. Unhired workers have their target marked as visited for next round
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Workers Sending One Round of Applications (Simultaneous) ---")
    stride = wrk.job_apps_targets.shape[1]
    unemp_ids = np.where(wrk.employed == 0)[0]
    active_applicants_mask = wrk.job_apps_head[unemp_ids] >= 0
    unemp_ids_applying = unemp_ids[active_applicants_mask]

    if unemp_ids_applying.size == 0:
        if info_enabled:
            log.info("  No workers with pending applications found. Skipping round.")
            log.info("--- Application Sending Round complete ---")
        return

    if info_enabled:
        log.info(
            f"  Processing {unemp_ids_applying.size} workers simultaneously "
            f"(Stride={stride})."
        )

    # Counters for logging
    apps_sent_successfully = 0
    apps_dropped_no_vacancy = 0
    workers_exhausted_list = 0

    # Cache log level check outside the loops
    debug_enabled = log.isEnabledFor(logging.DEBUG)

    # Phase 1: ALL workers simultaneously pick their best remaining target
    # Build a mapping of firm -> list of applicants
    firm_applicants: dict[int, list[int]] = {}

    for j in unemp_ids_applying:
        head = wrk.job_apps_head[j]
        if head < 0:  # pragma: no cover
            continue

        row_from_head, col = divmod(head, stride)

        # Check if worker exhausted their list
        if head >= (j + 1) * stride:
            if debug_enabled:
                log.debug(
                    f"    Worker {j} exhausted all {stride} application slots. "
                    f"Setting head to -1."
                )
            wrk.job_apps_head[j] = -1
            workers_exhausted_list += 1
            continue

        firm_id = wrk.job_apps_targets[row_from_head, col]
        if firm_id < 0:
            if debug_enabled:
                log.debug(
                    f"    Worker {j} encountered sentinel (-1) at col {col}. "
                    f"End of list. Setting head to -1."
                )
            wrk.job_apps_head[j] = -1
            workers_exhausted_list += 1
            continue

        # Check for vacancy at target firm
        if emp.n_vacancies[firm_id] <= 0:
            if debug_enabled:
                log.debug(
                    f"  Firm {firm_id} has no vacancies. "
                    f"Worker {j} application dropped, advances to next target."
                )
            apps_dropped_no_vacancy += 1
            # Advance to next target for next round
            wrk.job_apps_head[j] = head + 1
            wrk.job_apps_targets[row_from_head, col] = -1
            continue

        # Worker applies to this firm (simultaneously with all other workers)
        if firm_id not in firm_applicants:
            firm_applicants[firm_id] = []
        firm_applicants[firm_id].append(j)

        if debug_enabled:
            log.debug(f"    Worker {j} targeting firm {firm_id} (app #{col + 1}).")

    # Phase 2: Queue all simultaneous applications at each firm
    for firm_id, applicants in firm_applicants.items():
        n_applicants = len(applicants)

        if debug_enabled:
            log.debug(
                f"  Firm {firm_id}: {n_applicants} workers crowding "
                f"(vacancies: {emp.n_vacancies[firm_id]})"
            )

        # Add all applicants to firm's queue (up to queue capacity)
        for worker_id in applicants:
            ptr = emp.recv_job_apps_head[firm_id] + 1
            if ptr >= emp.recv_job_apps.shape[1]:
                if debug_enabled:
                    log.debug(
                        f"    Firm {firm_id} queue full. "
                        f"Worker {worker_id} application dropped."
                    )
                continue

            emp.recv_job_apps_head[firm_id] = ptr
            emp.recv_job_apps[firm_id, ptr] = worker_id
            apps_sent_successfully += 1

            if debug_enabled:
                log.debug(
                    f"    Worker {worker_id} application queued "
                    f"at firm {firm_id} slot {ptr}."
                )

        # Advance head pointer for ALL applicants (they all "visited" this firm)
        for worker_id in applicants:
            head = wrk.job_apps_head[worker_id]
            row_from_head, col = divmod(head, stride)
            wrk.job_apps_head[worker_id] = head + 1
            wrk.job_apps_targets[row_from_head, col] = -1

    # Log crowding statistics
    if info_enabled and firm_applicants:
        crowding_counts = [len(apps) for apps in firm_applicants.values()]
        max_crowding = max(crowding_counts)
        avg_crowding = sum(crowding_counts) / len(crowding_counts)
        firms_with_crowding = sum(1 for c in crowding_counts if c > 1)

        log.info(
            f"  Crowding stats: {len(firm_applicants)} firms received applications, "
            f"avg {avg_crowding:.1f} workers/firm, max {max_crowding}, "
            f"{firms_with_crowding} firms with >1 applicant"
        )

    # Summary log
    if info_enabled:
        log.info(
            f"  Round Summary: "
            f"{apps_sent_successfully} applications queued simultaneously, "
            f"{apps_dropped_no_vacancy} dropped (no vacancy), "
            f"{workers_exhausted_list} workers exhausted their list."
        )
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
    elif log.isEnabledFor(logging.TRACE):
        log.trace(
            f"[{tag:^10s}] Labor consistent for firm {i:3d}: {recorded:3d} workers."
        )
    return True


def _safe_bincount_employed(wrk: Worker, n_firms: int) -> Int1D:  # pragma: no cover
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


def _clean_queue(
    slice_: Idx1D, wrk: Worker, firm_idx_for_log: int, *, trace_enabled: bool = False
) -> Idx1D:
    """
    Return a *unique* array of still-unemployed worker ids
    from the raw queue slice (may contain -1 sentinels and duplicates),
    preserving the original order of first appearance.
    """
    if trace_enabled:
        log.trace(
            f"    Firm {firm_idx_for_log}: Cleaning queue. Initial raw slice: {slice_}"
        )

    # Drop -1 sentinels
    cleaned_slice = slice_[slice_ >= 0]
    if cleaned_slice.size == 0:
        if trace_enabled:
            log.trace(
                f"    Firm {firm_idx_for_log}: Queue empty after dropping sentinels."
            )
        return cleaned_slice.astype(np.intp)

    if trace_enabled:
        log.trace(
            f"    Firm {firm_idx_for_log}: "
            f"Queue after dropping sentinels: {cleaned_slice}"
        )

    # Unique *without* sorting
    first_idx = np.unique(cleaned_slice, return_index=True)[1]
    unique_slice = cleaned_slice[np.sort(first_idx)]
    if trace_enabled:
        log.trace(
            f"    Firm {firm_idx_for_log}: "
            f"Queue after unique (order kept): {unique_slice}"
        )

    # Keep only unemployed workers
    unemployed_mask = wrk.employed[unique_slice] == 0
    final_queue = unique_slice[unemployed_mask]
    if trace_enabled:
        log.trace(
            f"    Firm {firm_idx_for_log}: "
            f"Final cleaned queue (unique, unemployed): {final_queue}"
        )

    return cast(Idx1D, final_queue)


def firms_hire_workers(
    wrk: Worker,
    emp: Employer,
    *,
    theta: int,
    contract_poisson_mean: int = 10,
    matching_method: str = "sequential",
    rng: Rng = make_rng(),
) -> None:
    """
    Firms process applications and hire workers to fill vacancies.

    See Also
    --------
    bamengine.events.labor_market.FirmsHireWorkers : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info(f"--- Firms Hiring Workers ({matching_method}) ---")
    hiring_ids = np.where(emp.n_vacancies > 0)[0]
    total_vacancies = emp.n_vacancies.sum()
    if info_enabled:
        log.info(
            f"  {hiring_ids.size} firms have {total_vacancies:,} "
            f"total vacancies and are attempting to hire."
        )

    total_hires_this_round = 0
    total_rejected_this_round = 0

    # Cache log level checks outside the loop
    debug_enabled = log.isEnabledFor(logging.DEBUG)
    trace_enabled = log.isEnabledFor(logging.TRACE)

    for i in hiring_ids:
        if debug_enabled:
            log.debug(f"  Processing firm {i} (vacancies: {emp.n_vacancies[i]})")

        if debug_enabled:
            _check_labor_consistency("PRE-hire", i, wrk, emp)

        n_recv = emp.recv_job_apps_head[i] + 1
        if n_recv <= 0:
            if debug_enabled:
                log.debug(f"    Firm {i} has no applications. Skipping.")
            continue

        raw_queue = emp.recv_job_apps[i, :n_recv].copy()
        if debug_enabled:
            log.debug(
                f"    Firm {i} raw application queue "
                f"({n_recv} applications): {raw_queue}"
            )

        queue = _clean_queue(
            raw_queue, wrk, firm_idx_for_log=i, trace_enabled=trace_enabled
        )

        if queue.size == 0:
            if debug_enabled:
                log.debug(
                    f"    Firm {i}: no valid (unique, unemployed) "
                    f"applicants in queue. Flushing."
                )
            emp.recv_job_apps_head[i] = -1
            emp.recv_job_apps[i, :n_recv] = -1
            continue

        if debug_enabled:
            log.debug(f"    Firm {i} has {queue.size} valid potential hires: {queue}")

        num_to_hire = min(queue.size, emp.n_vacancies[i])
        num_rejected = queue.size - num_to_hire

        if num_to_hire < queue.size:
            if debug_enabled:
                log.debug(
                    f"    Firm {i} capping hires from {queue.size} "
                    f"to {num_to_hire} due to vacancy limit "
                    f"({num_rejected} rejected)."
                )
            total_rejected_this_round += num_rejected

        # Selection method depends on matching_method
        if matching_method == "simultaneous":
            # Random selection: shuffles queue and takes first num_to_hire
            # This creates coordination failure - workers who "crowded" at
            # the firm may be randomly rejected even if they arrived "first"
            rng.shuffle(queue)
            final_hires = queue[:num_to_hire]
            if debug_enabled and num_rejected > 0:
                log.debug(
                    f"    Firm {i} randomly selected {num_to_hire} from "
                    f"{queue.size} applicants (simultaneous mode)"
                )
        else:
            # FIFO selection: takes first num_to_hire from queue order
            final_hires = queue[:num_to_hire]

        # extra validation, should never trigger
        if final_hires.size == 0:  # pragma: no cover
            emp.recv_job_apps_head[i] = -1
            emp.recv_job_apps[i, :n_recv] = -1
            continue

        if info_enabled:
            log.info(
                f"    Firm {i} is hiring {final_hires.size} worker(s): "
                f"{final_hires.tolist()}"
            )
        total_hires_this_round += final_hires.size

        # worker‑side updates
        if debug_enabled:
            log.debug(
                f"      Updating state for {final_hires.size} newly hired workers."
            )
        wrk.employer[final_hires] = i
        wrk.wage[final_hires] = emp.wage_offer[i]
        if contract_poisson_mean > 0:
            wrk.periods_left[final_hires] = theta + rng.poisson(contract_poisson_mean)
        else:
            wrk.periods_left[final_hires] = theta
        wrk.contract_expired[final_hires] = 0
        wrk.fired[final_hires] = 0
        wrk.job_apps_head[final_hires] = -1
        wrk.job_apps_targets[final_hires, :] = -1

        # firm‑side updates
        emp.current_labor[i] += final_hires.size
        emp.n_vacancies[i] -= final_hires.size
        if debug_enabled:
            log.debug(
                f"      Firm {i} state updated: "
                f"current_labor={emp.current_labor[i]}, "
                f"n_vacancies={emp.n_vacancies[i]}"
            )

        # flush inbound queue for this firm
        emp.recv_job_apps_head[i] = -1
        emp.recv_job_apps[i, :n_recv] = -1
        if debug_enabled:
            log.debug(f"    Firm {i} application queue flushed.")

        if debug_enabled:
            _check_labor_consistency("POST-hire", i, wrk, emp)

    if info_enabled:
        log.info(
            f"  Total hires made this step across all firms: {total_hires_this_round}"
            + (
                f", {total_rejected_this_round} rejected due to crowding"
                if total_rejected_this_round > 0
                else ""
            )
        )
    if log.isEnabledFor(logging.DEBUG):
        true_labor_counts = _safe_bincount_employed(wrk, emp.current_labor.size)
        mismatched_firms = np.flatnonzero(emp.current_labor != true_labor_counts)
        if mismatched_firms.size:
            log.error(
                f"[GLOBAL LABOR MISMATCH] {mismatched_firms.size} firms "
                f"have inconsistent labor counts."
            )
            for i_mismatch in mismatched_firms:
                log.error(
                    f"  Firm {i_mismatch}: recorded={emp.current_labor[i_mismatch]}, "
                    f"true={true_labor_counts[i_mismatch]}"
                )
        else:
            log.debug(
                "[GLOBAL LABOR CONSISTENCY] "
                "All firm labor counts match worker table after hiring."
            )
    if info_enabled:
        log.info("--- Firms Hiring Workers complete ---")


def firms_calc_wage_bill(emp: Employer, wrk: Worker) -> None:
    """
    Calculate total wage bill per firm.

    See Also
    --------
    bamengine.events.labor_market.FirmsCalcWageBill : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Firms Calculating Wage Bill ---")

    employed_mask = wrk.employed == 1
    num_employed = np.sum(employed_mask)
    if info_enabled:
        log.info(
            f"  Calculating wage bill based on {num_employed:,} currently employed workers."
        )

    n_firms = emp.wage_offer.size
    emp.wage_bill[:] = np.bincount(
        wrk.employer[employed_mask], weights=wrk.wage[employed_mask], minlength=n_firms
    )

    if info_enabled:
        total_wage_bill = emp.wage_bill.sum()
        avg_wage_of_employed = (
            wrk.wage[employed_mask].mean() if num_employed > 0 else 0.0
        )
        log.info(
            f"  Total economy-wide wage bill calculated: {total_wage_bill:,.2f} "
            f"(Avg wage for employed workers: {avg_wage_of_employed:.3f})"
        )
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Final Wage Bill per firm (first 10 firms): "
            f"{np.array2string(emp.wage_bill[:10], precision=2)}"
        )
    if info_enabled:
        log.info("--- Wage Bill Calculation complete ---")
