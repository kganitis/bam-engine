import logging

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from bamengine.components.economy import Economy
from bamengine.components.firm_labor import FirmHiring, FirmWageOffer
from bamengine.components.worker_labor import WorkerJobSearch

log = logging.getLogger(__name__)


def adjust_minimum_wage(ec: Economy) -> None:
    """
    Every `min_wage_rev_period` periods update ŵ_t by realised inflation:

        π = (P_{t-1} - P_{t-m}) / P_{t-m}
        ŵ_t = ŵ_{t-1} * (1 + π)
    """
    m = ec.min_wage_rev_period
    if ec.avg_mrkt_price_history.size <= m:
        return  # not enough data yet
    if (ec.avg_mrkt_price_history.size - 1) % m != 0:
        return  # not a revision step

    p_now = ec.avg_mrkt_price_history[-2]  # price of period t-1
    p_prev = ec.avg_mrkt_price_history[-m - 1]  # price of period t-m
    inflation = (p_now - p_prev) / p_prev

    ec.min_wage *= 1.0 + inflation

    log.debug(
        "adjust_minimum_wage: m=%d  π=%.4f  new_ŵ=%.3f",
        m,
        inflation,
        ec.min_wage,
    )


def firms_decide_wage_offer(
    fw: FirmWageOffer,
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
    # Draw one shock per firm, then mask where V_i==0.
    shock = rng.uniform(0.0, h_xi, size=fw.wage_prev.shape)
    shock[fw.n_vacancies == 0] = 0.0

    np.multiply(fw.wage_prev, 1.0 + shock, out=fw.wage_offer)
    np.maximum(fw.wage_offer, w_min, out=fw.wage_offer)

    log.debug(
        "decide_wage_offer: n=%d  w_min=%.3f  h_xi=%.3f  "
        "mean_w_prev=%.3f  mean_w_offer=%.3f",
        fw.wage_prev.size,
        w_min,
        h_xi,
        fw.wage_prev.mean(),
        fw.wage_offer.mean(),
    )


# --------------------------------------------------------------------------- #
def _topk_indices_desc(values: NDArray[np.float64], k: int) -> NDArray[np.intp]:
    """
    Indices of the *k* largest elements along the last axis, **unsorted**.

    Complexity
    ----------
    * argpartition  → O(n)  (find the split point)
    * slicing k     → O(k)
    Total           → O(n + k)          vs.   full argsort O(n log n)
    """
    if k >= values.shape[-1]:  # degenerate: keep all
        return np.argpartition(values, kth=0, axis=-1)
    part = np.argpartition(values, kth=k - 1, axis=-1)  # top‑k to the left
    return part[..., :k]  # [:, :k] for 2‑D case


# ---------------------------------------------------------------------
def workers_prepare_applications(
    ws: WorkerJobSearch,
    fw: FirmWageOffer,
    *,
    max_M: int,
    rng: Generator,
) -> None:
    n_firms = fw.wage_offer.size
    unem = np.where(ws.employed == 0)[0]  # unemployed ids

    if unem.size == 0:  # early-exit → nothing to do
        ws.apps_head.fill(-1)
        return

    # -------- sample M random firms per worker -----------------------
    sample = rng.integers(0, n_firms, size=(unem.size, max_M), dtype=np.int64)

    loyal = (
        (ws.contract_expired[unem] == 1)
        & (ws.fired[unem] == 0)
        & (ws.employer_prev[unem] >= 0)
    )
    if loyal.any():
        sample[loyal, 0] = ws.employer_prev[unem[loyal]]

    # -------- wage‑descending *partial* sort ----------------------------
    topk = _topk_indices_desc(fw.wage_offer[sample], k=max_M)
    sorted_sample = np.take_along_axis(sample, topk, axis=1)

    #
    # -------- loyalty: ensure previous employer is always in column 0 ---
    if loyal.any():
        # indices of loyal workers in the `unem` array
        loyal_rows = np.where(loyal)[0]

        # swap previous‑employer into col 0 when it got shuffled away
        for r in loyal_rows:
            prev = ws.employer_prev[unem[r]]
            row = sorted_sample[r]

            if row[0] != prev:
                # find where prev employer ended up (guaranteed to exist)
                j = np.where(row == prev)[0][0]
                row[0], row[j] = row[j], row[0]

    # -------- write to global buffers --------------------------------
    stride = max_M
    ws.apps_targets.fill(-1)
    ws.apps_head.fill(-1)

    for k, w in enumerate(unem):
        ws.apps_targets[w, :stride] = sorted_sample[k]
        ws.apps_head[w] = w * stride  # first slot of that row

    # reset flags
    ws.contract_expired[unem] = 0
    ws.fired[unem] = 0

    # -------- logging ------------------------------------------------
    log.debug(
        "workers_prepare_applications: U=%d  loyal=%d  avg_apps_per_U=%.1f",
        unem.size,
        int(loyal.sum()),
        float((ws.apps_head[unem] >= 0).sum()) / unem.size * max_M,
    )


# ---------------------------------------------------------------------
def workers_send_one_round(ws: WorkerJobSearch, fh: FirmHiring) -> None:
    stride = ws.apps_targets.shape[1]
    sent = 0

    for w in np.where(ws.employed == 0)[0]:
        h = ws.apps_head[w]
        if h < 0:
            continue
        row, col = divmod(h, stride)
        firm_idx = ws.apps_targets[row, col]
        if firm_idx < 0:  # exhausted list
            ws.apps_head[w] = -1
            continue

        # bounded queue
        ptr = fh.recv_apps_head[firm_idx] + 1
        if ptr >= fh.recv_apps.shape[1]:
            continue  # queue full – drop
        fh.recv_apps_head[firm_idx] = ptr
        fh.recv_apps[firm_idx, ptr] = w
        sent += 1

        # advance pointer & clear slot
        ws.apps_head[w] = h + 1
        ws.apps_targets[row, col] = -1

    log.debug(
        "workers_send_one_round: sent=%d  firms_receiving=%d",
        sent,
        int((fh.recv_apps_head >= 0).sum()),
    )


# ---------------------------------------------------------------------
def firms_hire_workers(
    ws: WorkerJobSearch,
    fh: FirmHiring,
    *,
    contract_theta: int,  # not used yet but kept for future extension
) -> None:
    """Match firms with queued applicants and update all related state.

    Side‑effects
    ------------
    * `ws.employed`, `ws.apps_head`, `ws.employer_prev`
    * `fh.n_vacancies`, `fh.recv_apps_head` / `recv_apps`
    * `fh.current_labor` ← increments by the number of hires
    """
    total_hires = 0

    for i in np.where(fh.n_vacancies > 0)[0]:
        n_recv = fh.recv_apps_head[i] + 1  # queue length (−1 ⇒ 0)
        if n_recv <= 0:
            continue

        n_hire = int(min(n_recv, fh.n_vacancies[i]))
        hires = fh.recv_apps[i, :n_hire]
        hires = hires[hires >= 0]  # drop sentinel slots
        if hires.size == 0:
            continue

        # ---- worker‑side updates ----------------------------------------
        ws.employed[hires] = 1
        ws.apps_head[hires] = -1
        ws.employer_prev[hires] = i
        # (wage / contract arrays would be updated here)

        # ---- firm‑side updates ------------------------------------------
        fh.current_labor[i] += hires.size  # NEW: keep labour stock
        fh.n_vacancies[i] -= hires.size

        fh.recv_apps_head[i] = -1  # clear queue
        fh.recv_apps[i, :n_recv] = -1
        total_hires += hires.size

    log.debug("firms_hire: hires=%d", total_hires)
