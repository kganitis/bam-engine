# src/bamengine/systems/labor_market.py
import logging
from typing import cast

import numpy as np
from numpy.random import Generator, default_rng

from bamengine.components.economy import Economy
from bamengine.components.employer import Employer
from bamengine.components.worker import Worker
from bamengine.helpers import select_top_k_indices
from bamengine.typing import Idx1D, Int1D

log = logging.getLogger("bamengine")


def adjust_minimum_wage(ec: Economy) -> None:
    """
    Every `min_wage_rev_period` periods update ŵ_t by realised inflation:

        π = (P_{t-1} - P_{t-m}) / P_{t-m}
        ŵ_t = ŵ_{t-1} * (1 + π)
    """
    m = ec.min_wage_rev_period
    if ec.avg_mkt_price_history.size <= m:
        return  # not enough data yet
    if (ec.avg_mkt_price_history.size - 1) % m != 0:
        return  # not a revision step

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Minimum-wage revision step reached – computing inflation …")

    p_now = ec.avg_mkt_price_history[-1]  # price of period t-1
    p_prev = ec.avg_mkt_price_history[-m - 1]  # price of period t-m
    inflation = (p_now - p_prev) / p_prev

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"Min-wage revision: p_now={p_now:.3f}, "
            f"p_prev={p_prev:.3f}, π={inflation:+.3%}"
        )

    new_min_wage = ec.min_wage * (1.0 + inflation)
    log.info(
        f"Revision period. Inflation over last {m} periods: {inflation:+.3%}. "
        f"Min wage: {ec.min_wage:.3f} → {new_min_wage:.3f}"
    )
    ec.min_wage = new_min_wage


def firms_decide_wage_offer(
    emp: Employer,
    *,
    w_min: float,
    h_xi: float,
    rng: Generator = default_rng(),
) -> None:
    """
    Vector rule:

        shock_i ~ U(0, h_xi)  if V_i>0 else 0
        w_i^b   = max( w_min , w_{i,t-1} * (1 + shock_i) )

    Works fully in-place, no temporary allocations.
    """
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

    log.info(
        f"Min wage: {w_min:.3f}. "
        f"Average offer from firms with vacancies: {emp.wage_offer.mean():.3f}"
    )
    # log.debug(f"Detailed wage offers:\n"
    #           f"{np.array2string(emp.wage_offer, precision=2)}")


# ---------------------------------------------------------------------
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

    hiring = np.where(emp.n_vacancies > 0)[0]  # hiring ids
    unemp = np.where(wrk.employed == 0)[0]  # unemployed ids

    # ── fast exits ──────────────────────────────────────────────────────
    if unemp.size == 0:
        log.info("No unemployed workers; skipping application phase.")
        wrk.job_apps_head.fill(-1)
        return

    if hiring.size == 0:
        log.info("No firm is hiring this period – all application queues cleared.")
        wrk.job_apps_head[unemp] = -1
        wrk.job_apps_targets[unemp, :].fill(-1)
        return

    # ── sample M random hiring firms per worker (with replacement) ─────
    M_eff = min(max_M, hiring.size)
    sample = rng.choice(
        hiring, size=(unemp.size, M_eff), replace=True
    )  # shape = (U, M)

    # ── loyalty rule ----------------------------------------------------
    loyal_mask = (
        (wrk.contract_expired[unemp] == 1)
        & (wrk.fired[unemp] == 0)
        & np.isin(wrk.employer_prev[unemp], hiring)  # only if prev firm is hiring
    )
    if loyal_mask.any():
        sample[loyal_mask, 0] = wrk.employer_prev[unemp[loyal_mask]]

    # ── wage-descending partial sort ------------------------------------
    topk = select_top_k_indices(emp.wage_offer[sample], k=max_M, descending=True)
    sorted_sample = np.take_along_axis(sample, topk, axis=1)

    # loyalty: ensure prev-employer sits in column-0 after sort
    if loyal_mask.any():
        loyal_rows = np.where(loyal_mask)[0]
        for r in loyal_rows:
            prev = wrk.employer_prev[unemp[r]]
            row = sorted_sample[r]
            if row[0] != prev:
                j = np.where(row == prev)[0][0]  # guaranteed to exist
                row[0], row[j] = row[j], row[0]

    # ── write buffers ----------------------------------------------------
    stride = max_M
    for k, w in enumerate(unemp):
        wrk.job_apps_targets[w, :M_eff] = sorted_sample[k]
        # pad any remaining columns with -1
        if M_eff < max_M:
            wrk.job_apps_targets[w, M_eff:max_M] = -1
        wrk.job_apps_head[w] = w * stride

    # reset flags
    wrk.contract_expired[unemp] = 0
    wrk.fired[unemp] = 0

    log.info(f"{unemp.size} unemployed workers send {M_eff} applications each.")


# ---------------------------------------------------------------------
def workers_send_one_round(
    wrk: Worker, emp: Employer, rng: Generator = default_rng()
) -> None:
    """A single round of job applications being sent and received."""
    stride = wrk.job_apps_targets.shape[1]
    unemp_ids = np.where(wrk.employed == 0)[0]
    rng.shuffle(unemp_ids)
    for w in unemp_ids:
        h = wrk.job_apps_head[w]
        if h < 0:
            continue

        row, col = divmod(h, stride)
        firm_idx = wrk.job_apps_targets[row, col]
        if firm_idx < 0:  # exhausted list
            wrk.job_apps_head[w] = -1
            continue

        # bounded queue
        ptr = emp.recv_job_apps_head[firm_idx] + 1
        if ptr >= emp.recv_job_apps.shape[1]:
            continue  # queue full – drop
        emp.recv_job_apps_head[firm_idx] = ptr
        emp.recv_job_apps[firm_idx, ptr] = w

        # advance pointer & clear slot
        wrk.job_apps_head[w] = h + 1
        wrk.job_apps_targets[row, col] = -1


# ---------------------------------------------------------------------
def _check_labor_consistency(tag: str, i: int, wrk: Worker, emp: Employer) -> bool:
    """
    Compare firm‐side bookkeeping (`emp.current_labor[i]`)
    with the ground truth reconstructed from the Worker table.
    """
    true_headcount = np.count_nonzero((wrk.employed == 1) & (wrk.employer == i))
    recorded = int(emp.current_labor[i])

    if true_headcount != recorded and log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"[{tag}] \t firm={i:4d} \t recorded={recorded:4d} \t"
            f"true={true_headcount:4d} \t Δ={true_headcount-recorded:+d}"
        )
        return False

    return True


def _safe_bincount_employed(wrk: Worker, n_firms: int) -> Int1D:
    """
    Return head-counts per firm, *ignoring* any corrupted rows where
    wrk.employed == 1 but wrk.employer < 0.
    Also logs those rows so you can trace them later.
    """
    mask_good = (wrk.employed == 1) & (wrk.employer >= 0)
    mask_bad = (wrk.employed == 1) & (wrk.employer < 0)

    if mask_bad.any() and log.isEnabledFor(logging.DEBUG):
        bad_idx = np.where(mask_bad)[0]
        log.debug(
            f"[CORRUPT] {bad_idx.size} worker rows have "
            f"employed=1 but employer<0; indices={bad_idx.tolist()}"
        )

    return np.bincount(
        wrk.employer[mask_good],
        minlength=n_firms,
    )


def _clean_queue(slice_: Idx1D, wrk: Worker) -> Idx1D:
    """
    Return a *unique* array of still-unemployed worker ids
    from the raw queue slice (may contain -1 sentinels and duplicates).
    """
    # drop -1 sentinels
    slice_ = slice_[slice_ >= 0]
    if slice_.size == 0:
        return slice_

    # uniqueness first (cheaper than masking twice)
    slice_ = np.unique(slice_)

    # keep only unemployed
    unemployed_mask = wrk.employed[slice_] == 0
    return cast(Idx1D, slice_[unemployed_mask])


def firms_hire_workers(
    wrk: Worker,
    emp: Employer,
    *,
    theta: int,
    rng: Generator = default_rng(),
) -> None:
    """Match firms with queued applicants and update all related state."""
    hiring_ids = np.where(emp.n_vacancies > 0)[0]
    rng.shuffle(hiring_ids)

    for i in hiring_ids:
        # ── PRE–hire sanity check ───────────────────────────────────────
        _check_labor_consistency("PRE-hire", i, wrk, emp)

        n_recv = emp.recv_job_apps_head[i] + 1  # queue length (−1 ⇒ 0)
        if n_recv <= 0:
            continue

        queue = emp.recv_job_apps[i, :n_recv]
        hires = _clean_queue(queue, wrk)  # <— all the cleaning

        if hires.size == 0:
            # nothing useful in the queue → just flush it
            emp.recv_job_apps_head[i] = -1
            emp.recv_job_apps[i, :n_recv] = -1
            continue

        # cap by remaining vacancies
        hires = hires[: emp.n_vacancies[i]]

        # ---- worker‑side updates ----------------------------------------
        wrk.employed[hires] = 1
        wrk.employer[hires] = i
        wrk.wage[hires] = emp.wage_offer[i]
        wrk.periods_left[hires] = theta
        wrk.contract_expired[hires] = 0
        wrk.fired[hires] = 0
        wrk.job_apps_head[hires] = -1
        wrk.job_apps_targets[hires, :] = -1

        # ---- firm‑side updates ------------------------------------------
        emp.current_labor[i] += hires.size
        emp.n_vacancies[i] -= hires.size

        # flush inbound queue
        emp.recv_job_apps_head[i] = -1
        emp.recv_job_apps[i, :n_recv] = -1

        # ── POST-hire sanity check ──────────────────────────────────────
        _check_labor_consistency("POST-hire", i, wrk, emp)

    # -------- global cross-check -----------------------------------------
    if log.isEnabledFor(logging.DEBUG):
        true = _safe_bincount_employed(wrk, emp.current_labor.size)
        bad = np.flatnonzero(emp.current_labor != true)
        if bad.size:
            log.debug(
                f"[SUMMARY] {bad.size} firm(s) out of sync after hiring: "
                f"indices={bad.tolist()}"
            )
            for i in bad:
                log.debug(
                    f"  firm {i}: recorded {emp.current_labor[i]}, true {true[i]}"
                )


def firms_calc_wage_bill(emp: Employer) -> None:
    """
    W_i = L_i · w_i
    """
    np.multiply(emp.current_labor, emp.wage_offer, out=emp.wage_bill)
    log.debug("Hiring complete")
    # log.debug(f"  Current Labor after hiring (L_i):\n{emp.current_labor}")
    # log.debug(
    #     f"  Wage bills after hiring:\n{np.array2string(emp.wage_bill, precision=2)}"
    # )
