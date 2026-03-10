"""
System functions for labor market phase events.

This module contains the internal implementation functions for labor market events.
Event classes wrap these functions and provide the primary documentation.

See Also
--------
bamengine.events.labor_market : Event classes (primary documentation source)
"""

from __future__ import annotations

import numpy as np

from bamengine import Rng, logging, make_rng
from bamengine.economy import Economy
from bamengine.roles import Employer, Worker
from bamengine.utils import resolve_conflicts

log = logging.getLogger(__name__)


def calc_inflation_rate(ec: Economy) -> None:
    """
    Calculate inflation rate and append to history.

    Parameters
    ----------
    ec : Economy
        Economy object containing price history and inflation history.

    See Also
    --------
    bamengine.events.labor_market.CalcInflationRate : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Calculating Inflation Rate ---")
    hist = ec.avg_mkt_price_history

    # Year-over-year method: compare P_t to P_{t-4}
    # Needs at least 5 periods of history
    if len(hist) <= 4:
        if info_enabled:
            log.info(
                "  Not enough history to calculate annual inflation (<5 periods). "
                "Setting to 0.0."
            )
        ec.inflation_history.append(0.0)
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

    ec.inflation_history.append(inflation)
    if info_enabled:
        log.info(
            f"  Annual inflation calculated for period t={len(hist) - 1}: {inflation:+.3%}"
        )
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"    Calculation: (p_now={p_now:.3f} / p_t-4={p_prev:.3f}) - 1")

    if info_enabled:
        log.info("--- Inflation Calculation complete ---")


def adjust_minimum_wage(ec: Economy, wrk: Worker) -> None:
    """
    Periodically index minimum wage to inflation.

    Parameters
    ----------
    ec : Economy
        Economy state with min_wage and price history.
    wrk : Worker
        Worker state (wages updated if below new minimum).

    See Also
    --------
    bamengine.events.labor_market.AdjustMinimumWage : Full documentation
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Adjusting Minimum Wage (based on history) ---")
    m = ec.min_wage_rev_period
    if len(ec.avg_mkt_price_history) <= m:
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"  Skipping: not enough history ({len(ec.avg_mkt_price_history)} <= {m})."
            )
        return
    if (len(ec.avg_mkt_price_history) - 1) % m != 0:
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

    # --- Vectorized sampling using random priorities + argpartition ---
    n_unemp = unemp.size
    n_hiring = hiring.size
    M_eff = min(max_M, n_hiring)
    if info_enabled:
        log.info(f"  Effective applications per worker (M_eff): {M_eff}")

    # Generate random priorities for all (worker, firm) pairs
    priorities = rng.random((n_unemp, n_hiring))

    # Select top M_eff firms per worker using argpartition (O(n) per row)
    if M_eff < n_hiring:
        top_k_local = np.argpartition(-priorities, kth=M_eff - 1, axis=1)[:, :M_eff]
    else:
        top_k_local = np.broadcast_to(np.arange(n_hiring), (n_unemp, n_hiring)).copy()

    # Map local indices to firm IDs
    sample = hiring[top_k_local]

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Initial random firm sample (first 10 workers, if any):\n{sample[:10]}"
        )

    # Sort selected firms by wage (descending) using argsort
    wages = emp.wage_offer[sample]
    wage_order = np.argsort(-wages, axis=1)
    sorted_sample = np.take_along_axis(sample, wage_order, axis=1)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Sorted firm sample by wage (first 10 workers, if any):\n"
            f"{sorted_sample[:10]}"
        )

    # Loyalty rule: prev employer goes to position 0 for eligible workers
    loyal_mask = (
        (wrk.contract_expired[unemp] == 1)
        & (wrk.fired[unemp] == 0)
        & np.isin(wrk.employer_prev[unemp], hiring)
    )
    loyal_idx = np.where(loyal_mask)[0]
    num_loyal_workers = loyal_idx.size
    if info_enabled:
        log.info(f"  Applying loyalty rule for {num_loyal_workers} worker(s).")

    if loyal_idx.size > 0:
        prev_firm_ids = wrk.employer_prev[unemp[loyal_idx]]
        match = sorted_sample[loyal_idx] == prev_firm_ids[:, None]
        has_prev = match.any(axis=1)

        # Case 1: prev employer already in sample — move to front
        in_rows = loyal_idx[has_prev]
        if in_rows.size > 0:
            in_cols = np.argmax(match[has_prev], axis=1)
            for pos_val in range(1, M_eff):
                at_pos = in_cols == pos_val
                if at_pos.any():
                    rows = in_rows[at_pos]
                    saved = sorted_sample[rows, pos_val].copy()
                    sorted_sample[rows, 1 : pos_val + 1] = sorted_sample[
                        rows, 0:pos_val
                    ]
                    sorted_sample[rows, 0] = saved

        # Case 2: prev employer NOT in sample — drop last, insert at front
        out_rows = loyal_idx[~has_prev]
        if out_rows.size > 0:
            out_prev = prev_firm_ids[~has_prev]
            if M_eff > 1:
                sorted_sample[out_rows, 1:M_eff] = sorted_sample[
                    out_rows, 0 : M_eff - 1
                ]
            sorted_sample[out_rows, 0] = out_prev

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"    Sorted sample AFTER post-sort loyalty adjustment "
                f"(first 10 rows if any loyal):\n{sorted_sample[:10]}"
            )

    # Write buffers (vectorized)
    stride = max_M
    wrk.job_apps_targets[unemp, :M_eff] = sorted_sample
    if M_eff < max_M:
        wrk.job_apps_targets[unemp, M_eff:max_M] = -1
    wrk.job_apps_head[unemp] = unemp * stride

    if log.isEnabledFor(logging.DEBUG):
        for k in range(min(10, n_unemp)):
            j = unemp[k]
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


def labor_market_round(
    emp: Employer,
    wrk: Worker,
    *,
    theta: int,
    rng: Rng = make_rng(),
) -> None:
    """One vectorized round of labor market matching.

    All unemployed applicants simultaneously send their next application,
    conflicts are resolved randomly, and accepted workers are batch-hired.

    Parameters
    ----------
    emp : Employer
        Employer role (firms).
    wrk : Worker
        Worker role (households).
    theta : int
        Minimum contract duration.
    rng : Rng
        Random generator for conflict resolution.
    """
    info_enabled = log.isEnabledFor(logging.INFO)
    if info_enabled:
        log.info("--- Labor Market Round ---")

    stride = wrk.job_apps_targets.shape[1]  # max_M
    n_firms = emp.current_labor.size

    # 1. Find active unemployed applicants with pending applications
    unemp_ids = np.where(wrk.employed == 0)[0]
    if unemp_ids.size == 0:
        if info_enabled:
            log.info("  No unemployed workers. Skipping round.")
            log.info("--- Labor Market Round complete ---")
        return

    active_mask = wrk.job_apps_head[unemp_ids] >= 0
    active = unemp_ids[active_mask]

    if active.size == 0:
        if info_enabled:
            log.info("  No workers with pending applications. Skipping round.")
            log.info("--- Labor Market Round complete ---")
        return

    # 2. Decode targets from head pointers
    heads = wrk.job_apps_head[active]
    rows, cols = np.divmod(heads, stride)
    target_firms = wrk.job_apps_targets[rows, cols]

    # 3. Filter: valid targets (not -1)
    valid_target_mask = target_firms >= 0
    # Also filter for firms with vacancies
    has_vacancy_mask = valid_target_mask.copy()
    has_vacancy_mask[valid_target_mask] &= (
        emp.n_vacancies[target_firms[valid_target_mask]] > 0
    )

    # 4. Advance ALL head pointers (even for invalid/no-vacancy targets)
    new_heads = heads + 1
    exhausted = (new_heads >= (active + 1) * stride) | (~valid_target_mask)
    new_heads[exhausted] = -1
    wrk.job_apps_head[active] = new_heads
    # Clear consumed slots
    wrk.job_apps_targets[rows, cols] = -1

    # 5. Get senders with valid, vacancy-having targets
    valid_senders = active[has_vacancy_mask]
    valid_targets = target_firms[has_vacancy_mask]

    if valid_senders.size == 0:
        if info_enabled:
            log.info("  No valid applications this round.")
            log.info("--- Labor Market Round complete ---")
        return

    # 6. Conflict resolution
    accepted_mask = resolve_conflicts(
        valid_senders, valid_targets, emp.n_vacancies, n_firms, rng
    )

    hired_workers = valid_senders[accepted_mask]
    hired_firms = valid_targets[accepted_mask]

    if hired_workers.size == 0:
        if info_enabled:
            log.info("  No hires this round after conflict resolution.")
            log.info("--- Labor Market Round complete ---")
        return

    if info_enabled:
        log.info(f"  Hiring {hired_workers.size} worker(s) this round.")

    # 7. Batch hire: update worker state
    wrk.employer[hired_workers] = hired_firms
    wrk.wage[hired_workers] = emp.wage_offer[hired_firms]
    wrk.periods_left[hired_workers] = theta
    wrk.contract_expired[hired_workers] = 0
    wrk.fired[hired_workers] = 0
    wrk.job_apps_head[hired_workers] = -1
    wrk.job_apps_targets[hired_workers, :] = -1

    # 8. Batch hire: update firm state
    hire_counts = np.bincount(hired_firms, minlength=n_firms).astype(
        emp.current_labor.dtype
    )
    emp.current_labor += hire_counts
    emp.n_vacancies -= hire_counts

    if info_enabled:
        log.info("--- Labor Market Round complete ---")
