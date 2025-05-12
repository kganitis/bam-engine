import logging

import numpy as np
from numpy.random import Generator

from bamengine.components.economy import Economy
from bamengine.components.employer import Employer
from bamengine.components.worker import Worker
from bamengine.typing import Float1D, Idx1D

log = logging.getLogger(__name__)


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

    p_now = ec.avg_mkt_price_history[-2]  # price of period t-1
    p_prev = ec.avg_mkt_price_history[-m - 1]  # price of period t-m
    inflation = (p_now - p_prev) / p_prev

    ec.min_wage *= 1.0 + inflation


def firms_decide_wage_offer(
    emp: Employer,
    *,
    w_min: float,
    h_xi: float,
    rng: Generator,
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


# --------------------------------------------------------------------------- #
def _topk_indices_desc(values: Float1D, k: int) -> Idx1D:
    """
    Return indices of the *k* largest elements along the last axis
    (unsorted, descending).

    Complexity
    ----------
    argpartition : O(n)   – finds the split position,
                          - ascending -> we call it on **‑values**
    slicing      : O(k)   – keeps only the first k
    Total        : O(n + k)  vs. full argsort O(n logn)
    """
    if k >= values.shape[-1]:  # degenerate: keep all
        return np.argpartition(-values, kth=0, axis=-1)
    part = np.argpartition(-values, kth=k - 1, axis=-1)  # top‑k to the left
    return part[..., :k]  # [:, :k] for 2‑D case, same ndim as input


# ---------------------------------------------------------------------
def workers_decide_firms_to_apply(
    wrk: Worker,
    emp: Employer,
    *,
    max_M: int,
    rng: Generator,
) -> None:
    n_firms = emp.wage_offer.size
    unem = np.where(wrk.employed == 0)[0]  # unemployed ids

    if unem.size == 0:  # early-exit → nothing to do
        wrk.job_apps_head.fill(-1)
        return

    # -------- sample M random emp per worker -----------------------
    sample = rng.integers(0, n_firms, size=(unem.size, max_M), dtype=np.int64)

    loyal = (
        (wrk.contract_expired[unem] == 1)
        & (wrk.fired[unem] == 0)
        & (wrk.employer_prev[unem] >= 0)
    )
    if loyal.any():
        sample[loyal, 0] = wrk.employer_prev[unem[loyal]]

    # -------- wage‑descending *partial* sort ----------------------------
    topk = _topk_indices_desc(emp.wage_offer[sample], k=max_M)
    sorted_sample = np.take_along_axis(sample, topk, axis=1)

    #
    # -------- loyalty: ensure previous employer is always in column 0 ---
    if loyal.any():
        # indices of loyal workers in the `unem` array
        loyal_rows = np.where(loyal)[0]

        # swap previous‑employer into col 0 when it got shuffled away
        for r in loyal_rows:
            prev = wrk.employer_prev[unem[r]]
            row = sorted_sample[r]

            if row[0] != prev:  # not covered by tests
                # find where prev employer ended up (guaranteed to exist)
                j = np.where(row == prev)[0][0]
                row[0], row[j] = row[j], row[0]

    stride = max_M
    for k, w in enumerate(unem):
        wrk.job_apps_targets[w, :stride] = sorted_sample[k]
        wrk.job_apps_head[w] = w * stride  # first slot of that row

    # reset flags
    wrk.contract_expired[unem] = 0
    wrk.fired[unem] = 0


# ---------------------------------------------------------------------
def workers_send_one_round(wrk: Worker, emp: Employer) -> None:
    stride = wrk.job_apps_targets.shape[1]

    for w in np.where(wrk.employed == 0)[0]:
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
def firms_hire_workers(
    wrk: Worker,
    emp: Employer,
    *,
    theta: int,
) -> None:
    """Match firms with queued applicants and update all related state."""
    for i in np.where(emp.n_vacancies > 0)[0]:
        n_recv = emp.recv_job_apps_head[i] + 1  # queue length (−1 ⇒ 0)
        if n_recv <= 0:
            continue

        n_hire = int(min(n_recv, emp.n_vacancies[i]))
        hires = emp.recv_job_apps[i, :n_hire]
        hires = hires[hires >= 0]  # drop sentinel slots
        if hires.size == 0:
            continue

        # ---- worker‑side updates ----------------------------------------
        wrk.employed[hires] = 1
        wrk.employer[hires] = i  # update contract

        # if wages become worker-specific replace with np.put(…) / gather logic
        wrk.wage[hires] = emp.wage_offer[i]

        wrk.periods_left[hires] = theta
        wrk.contract_expired[hires] = 0
        wrk.fired[hires] = 0

        wrk.job_apps_head[hires] = -1  # clear queue
        wrk.job_apps_targets[hires, :] = -1

        # ---- firm‑side updates ------------------------------------------
        emp.current_labor[i] += hires.size
        emp.n_vacancies[i] -= hires.size

        emp.recv_job_apps_head[i] = -1  # clear queue
        emp.recv_job_apps[i, :n_recv] = -1


def firms_calc_wage_bill(emp: Employer) -> None:
    """
    W_i = L_i · w_i
    """
    np.multiply(emp.current_labor, emp.wage_offer, out=emp.wage_bill)
