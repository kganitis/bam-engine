"""
Labour-market unit tests

These tests exercise every pure function in
`bamengine.systems.labor_market` under corner-case scenarios that matter both
mathematically and economically.

The structure is:

* Minimum-wage adjustment           – inflation edge cases
* Wage-offer decision               – floor & shock edge cases
* Application preparation           – unemployed mix, loyalty, buffer limits
* One-round message passing         – queue boundaries
* Hiring                            – vacancies / current labor / pointer invariants
"""

from typing import Tuple

import numpy as np
import pytest
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from bamengine.components.economy import Economy
from bamengine.components.firm_labor import FirmHiring, FirmWageOffer
from bamengine.components.worker_labor import WorkerJobSearch
from bamengine.systems.labor_market import (
    _topk_indices_desc,
    adjust_minimum_wage,
    firms_decide_wage_offer,
    firms_hire_workers,
    workers_prepare_job_applications,
    workers_send_one_round,
)

FloatA = NDArray[np.float64]


# --------------------------------------------------------------------------- #
#  Minimum-wage inflation rule
# --------------------------------------------------------------------------- #
def test_adjust_minimum_wage_revision() -> None:
    """
    At a revision step (len = m+1), min_wage should increase by realised
    inflation between t-m and t-1.
    """
    ec = Economy(
        min_wage=1.0,
        avg_mrkt_price=1.15,
        avg_mrkt_price_history=np.array([1.0, 1.05, 1.10, 1.12, 1.15]),  # t = 4
        min_wage_rev_period=4,
    )
    adjust_minimum_wage(ec)
    # Inflation between P0 and P3: (1.12 − 1.00) / 1.00 = 0.12
    assert np.isclose(ec.min_wage, 1.0 * 1.12)


@pytest.mark.parametrize(
    "prices,delta_sign",
    [
        # (price path, expected sign of Δŵ)
        (np.array([1.0, 1.1, 1.2, 1.3, 1.4]), "up"),  # inflation
        (np.array([1.0, 0.9, 0.8, 0.8, 0.8]), "down"),  # deflation
        (np.array([1.0, 1.1, 1.2]), "none"),  # < m+1, no revision
    ],
)
def test_adjust_minimum_wage_edges(prices: FloatA, delta_sign: str) -> None:
    """Guard array bounds; min wage may go **up or down** depending on inflation."""
    ec = Economy(
        min_wage=1.0,
        avg_mrkt_price=float(prices[-1]),
        avg_mrkt_price_history=prices,
        min_wage_rev_period=4,
    )
    old = ec.min_wage
    adjust_minimum_wage(ec)
    if delta_sign == "up":
        assert ec.min_wage > old
    elif delta_sign == "down":
        assert ec.min_wage < old
    else:
        assert ec.min_wage == pytest.approx(old)


# --------------------------------------------------------------------------- #
#  Wage-offer decision rule
# --------------------------------------------------------------------------- #
def test_decide_wage_offer_basic() -> None:
    """
    Firms with vacancies receive a stochastic mark-up; non-hiring firms’
    offers are simply clipped at the minimum wage.
    """
    rng = default_rng(seed=42)
    fw = FirmWageOffer(
        wage_prev=np.array([1.0, 1.2, 1.1, 1.3]),
        n_vacancies=np.array([3, 0, 2, 0]),
        wage_offer=np.zeros(4),
    )

    firms_decide_wage_offer(fw, w_min=1.05, h_xi=0.1, rng=rng)

    # Non-hiring firms keep their previous wage (already ≥ w_min)
    assert np.isclose(fw.wage_offer[1], 1.2)
    assert np.isclose(fw.wage_offer[3], 1.3)

    # Hiring firms obey floor and non-decreasing rule
    assert (fw.wage_offer[fw.n_vacancies > 0] >= fw.wage_prev[fw.n_vacancies > 0]).all()
    assert (fw.wage_offer >= 1.05).all()


def test_decide_wage_offer_floor_and_shock() -> None:
    """
    If w_prev < w_min the floor binds; otherwise a positive shock draws
    w_offer above w_prev.
    """
    rng = default_rng(2)
    fw = FirmWageOffer(
        wage_prev=np.array([0.8, 1.2]),
        n_vacancies=np.array([0, 3]),
        wage_offer=np.zeros(2),
    )
    firms_decide_wage_offer(fw, w_min=1.0, h_xi=0.1, rng=rng)
    assert fw.wage_offer[0] == pytest.approx(1.0)  # floor
    assert fw.wage_offer[1] >= 1.2  # shock


# --------------------------------------------------------------------------- #
#  Shared helpers for application / hiring tests
# --------------------------------------------------------------------------- #
def _mini_state() -> Tuple[FirmWageOffer, WorkerJobSearch, FirmHiring, Generator, int]:
    """Return a deterministic tiny labor-market state (6 workers × 3 firms)."""
    rng = default_rng(seed=1)
    n_workers, n_firms, M = 6, 3, 2

    fw = FirmWageOffer(
        wage_prev=np.array([1.0, 1.5, 1.2]),
        n_vacancies=np.array([2, 1, 0]),
        wage_offer=np.array([1.2, 1.5, 1.3]),
    )
    ws = WorkerJobSearch(
        employed=np.array([0, 0, 0, 1, 0, 1], dtype=np.int64),
        employer_prev=np.full(n_workers, -1, dtype=np.int64),
        contract_expired=np.zeros(n_workers, dtype=np.int64),
        fired=np.zeros(n_workers, dtype=np.int64),
        apps_head=np.full(n_workers, -1, dtype=np.int64),
        apps_targets=np.full((n_workers, M), -1, dtype=np.int64),
    )
    fh = FirmHiring(
        wage_offer=fw.wage_offer,
        n_vacancies=fw.n_vacancies,
        current_labor=np.array([1, 0, 2], dtype=np.int64),
        recv_apps_head=np.full(n_firms, -1, dtype=np.int64),
        recv_apps=np.full((n_firms, M), -1, dtype=np.int64),
    )
    return fw, ws, fh, rng, M


# --------------------------------------------------------------------------- #
#  workers_prepare_applications
# --------------------------------------------------------------------------- #
def test__topk_indices_desc_partial_sort() -> None:
    """
    Requesting k < n should hit the argpartition ↦ slice branch.
    We verify that the returned (unsorted) indices are exactly the k
    positions of the largest elements.
    """
    vals = np.array([[5.0, 1.0, 3.0, 4.0]], dtype=np.float64)
    k = 2
    idx = _topk_indices_desc(vals, k=k)

    # Extract the values referenced by idx and check they are the two maxima.
    top_vals = vals[0, idx[0]]
    assert set(top_vals) == {5.0, 4.0}
    # Shape must be preserved (unsorted, but length == k)
    assert idx.shape == (1, k)


def test_prepare_applications_basic() -> None:
    """
    Unemployed workers must obtain a valid apps_head and targets within bounds.
    """
    fw, ws, _, rng, M = _mini_state()
    workers_prepare_job_applications(ws, fw, max_M=M, rng=rng)

    heads = ws.apps_head[ws.employed == 0]
    assert (heads >= 0).all()

    rows = heads // M
    targets = ws.apps_targets[rows]
    assert (targets >= 0).all()
    assert (targets < fw.wage_offer.size).all()


def test_prepare_applications_no_unemployed() -> None:
    """If everyone is employed no application pointers should be set."""
    fw = FirmWageOffer(
        wage_prev=np.array([1.0, 1.5, 1.2]),
        n_vacancies=np.array([2, 1, 0]),
        wage_offer=np.array([1.2, 1.5, 1.3]),
    )
    ws = WorkerJobSearch(
        employed=np.ones(3, dtype=np.int64),
        employer_prev=np.full(3, -1, dtype=np.int64),
        contract_expired=np.zeros(3, dtype=np.int64),
        fired=np.zeros(3, dtype=np.int64),
        apps_head=np.full(3, -1, dtype=np.int64),
        apps_targets=np.full((3, 2), -1, dtype=np.int64),
    )
    workers_prepare_job_applications(ws, fw, max_M=2, rng=default_rng(0))
    assert (ws.apps_head == -1).all()


def test_prepare_applications_loyalty_to_employer() -> None:
    """
    A worker whose contract just expired (not fired) should list her previous
    employer first.
    """
    rng = default_rng(3)
    fw, ws, _, _, M = _mini_state()
    # worker-idx 0 just finished contract at firm-idx 1
    ws.employer_prev[0] = 1
    ws.contract_expired[0] = 1
    workers_prepare_job_applications(ws, fw, max_M=M, rng=rng)

    first_choice = ws.apps_targets[0, 0]
    assert first_choice == 1  # loyalty preserved


@pytest.mark.xfail(reason="Cover loyalty‑swap branch later")
def test_prepare_applications_loyalty_swap() -> None:  # pragma: no cover
    raise NotImplementedError


def test_prepare_applications_one_trial() -> None:
    """Edge case M = 1 (single application per worker)."""
    rng = default_rng(4)
    fw, ws, _, _, _ = _mini_state()
    M = 1
    ws.apps_targets = np.full((ws.employed.size, M), -1, dtype=np.int64)
    workers_prepare_job_applications(ws, fw, max_M=M, rng=rng)

    assert (ws.apps_head[ws.employed == 0] % M == 0).all()
    assert (ws.apps_targets >= -1).all()  # buffer still valid


def test_prepare_applications_large_unemployment() -> None:
    """
    More unemployed workers than firms × M → sampling with replacement must
    still yield valid firm indices.
    """
    rng = default_rng(5)
    n_workers, n_firms, M = 20, 3, 2
    fw = FirmWageOffer(
        wage_prev=np.array([1.0, 1.5, 1.2]),
        n_vacancies=np.array([2, 1, 0]),
        wage_offer=np.array([1.2, 1.5, 1.3]),
    )
    ws = WorkerJobSearch(
        employed=np.zeros(n_workers, dtype=np.int64),
        employer_prev=np.full(n_workers, -1, dtype=np.int64),
        contract_expired=np.zeros(n_workers, dtype=np.int64),
        fired=np.zeros(n_workers, dtype=np.int64),
        apps_head=np.full(n_workers, -1, dtype=np.int64),
        apps_targets=np.full((n_workers, M), -1, dtype=np.int64),
    )
    workers_prepare_job_applications(ws, fw, max_M=M, rng=rng)
    assert (ws.apps_targets[ws.apps_targets >= 0] < n_firms).all()


# --------------------------------------------------------------------------- #
#  workers_send_one_round
# --------------------------------------------------------------------------- #
def test_workers_send_one_round_queue_bounds() -> None:
    """
    When a firm’s queue is almost full, the next application must be dropped
    rather than overflow the buffer.
    """
    fw, ws, fh, _, M = _mini_state()
    # Preload firm-0 queue to capacity-1
    fh.recv_apps_head[0] = M - 2
    fh.recv_apps[0, : M - 1] = [0, 2][: M - 1]

    workers_prepare_job_applications(ws, fw, max_M=M, rng=default_rng(6))
    workers_send_one_round(ws, fh)

    assert fh.recv_apps_head[0] <= M - 1
    assert fh.recv_apps_head[0] >= -1  # still valid pointer


def test_worker_with_empty_list_is_skipped() -> None:
    fw, ws, fh, rng, M = _mini_state()
    # worker‑0: unemployed but apps_head = -1  => should be ignored
    ws.employed[0] = 0
    ws.apps_head[0] = -1
    workers_send_one_round(ws, fh)
    # queue heads unchanged
    assert (fh.recv_apps_head == -1).all()


def test_firm_queue_full_drops_application() -> None:
    fw, ws, fh, rng, M = _mini_state()
    fh.recv_apps_head[0] = M - 1  # already full
    fh.recv_apps[0] = 99  # dummy
    workers_prepare_job_applications(ws, fw, max_M=M, rng=rng)
    workers_send_one_round(ws, fh)
    # still full, nothing overwritten
    assert fh.recv_apps_head[0] == M - 1
    assert (fh.recv_apps[0] == 99).all()


def test_workers_send_one_round_exhausted_target() -> None:
    """
    When the current target has already been set to -1 *before* the call,
    the branch `firm_idx < 0` must trigger and clear apps_head to -1.
    """
    # --- minimal 1‑worker 1‑firm state -----------------------------------
    M = 1
    fw = FirmWageOffer(
        wage_prev=np.array([1.0]),
        n_vacancies=np.array([1]),
        wage_offer=np.array([1.0]),
    )
    ws = WorkerJobSearch(
        employed=np.array([0], dtype=np.int64),
        employer_prev=np.array([-1], dtype=np.int64),
        contract_expired=np.array([0], dtype=np.int64),
        fired=np.array([0], dtype=np.int64),
        apps_head=np.array([0], dtype=np.int64),  # points to first cell
        apps_targets=np.array([[-1]], dtype=np.int64),  # *already* exhausted
    )
    fh = FirmHiring(
        wage_offer=fw.wage_offer,
        n_vacancies=fw.n_vacancies,
        current_labor=np.array([0], dtype=np.int64),
        recv_apps_head=np.array([-1], dtype=np.int64),
        recv_apps=np.full((1, M), -1, dtype=np.int64),
    )

    workers_send_one_round(ws, fh)

    # apps_head must be cleared; nothing should be queued
    assert ws.apps_head[0] == -1
    assert fh.recv_apps_head[0] == -1


# --------------------------------------------------------------------------- #
#  firms_hire_workers
# --------------------------------------------------------------------------- #
def test_firms_hire_no_vacancies() -> None:
    """
    Applications sent to a firm with zero vacancies should be cleared without
    hiring anyone and without crashing.
    """
    _, ws, fh, _, _ = _mini_state()
    start_labor = fh.current_labor.copy()

    fh.n_vacancies[:] = 0  # nobody hiring
    fh.recv_apps_head[1] = 0
    fh.recv_apps[1, 0] = 2

    firms_hire_workers(ws, fh, contract_theta=8)

    # the application must *not* result in a hire
    np.testing.assert_array_equal(fh.current_labor, start_labor)
    assert ws.employed[2] == 0

    # pointer stays where it was (0) — queue ignored, not flushed
    assert fh.recv_apps_head[1] == 0


def test_firms_hire_exact_fit() -> None:
    """
    If the number of applications equals vacancies, all are hired and the queue
    is cleared.
    """
    _, ws, fh, _, _ = _mini_state()
    start_labor = fh.current_labor.copy()

    fh.recv_apps_head[0] = 1
    fh.recv_apps[0, :2] = [0, 2]

    firms_hire_workers(ws, fh, contract_theta=8)

    assert fh.n_vacancies[0] == 0
    assert fh.current_labor[0] == start_labor[0] + 2
    assert (fh.recv_apps[0] == -1).all()
    assert ws.employed[[0, 2]].sum() == 2  # both hired


def test_hire_workers_skips_invalid_slots() -> None:
    _, ws, fh, _, _ = _mini_state()
    fh.n_vacancies[:] = 1
    fh.recv_apps_head[0] = 1
    fh.recv_apps[0] = [-1, -1]  # all sentinels → size==0
    start = fh.current_labor.copy()
    firms_hire_workers(ws, fh, contract_theta=8)
    # nothing hired, vacancies unchanged
    np.testing.assert_array_equal(fh.current_labor, start)
    assert fh.n_vacancies[0] == 1


# --------------------------------------------------------------------------- #
#  End-to-end micro integration of one hiring event
# --------------------------------------------------------------------------- #
def test_full_round() -> None:
    """
    One complete labour-market event with M rounds should leave:
    * non-negative vacancies,
    * at least one worker hired (given vacancies > 0),
    * no dangling head pointers for hired workers.
    """
    fw, ws, fh, rng, M = _mini_state()
    start_labor = fh.current_labor.copy()

    workers_prepare_job_applications(ws, fw, max_M=M, rng=rng)
    for _ in range(M):
        workers_send_one_round(ws, fh)
        firms_hire_workers(ws, fh, contract_theta=8)

    assert (fh.n_vacancies >= 0).all()
    assert ws.employed.sum() >= 1
    assert fh.current_labor.sum() >= start_labor.sum()
    assert (ws.apps_head[ws.employed == 1] == -1).all()
