# tests/unit/systems/test_labor_market.py
"""
Labour-market unit tests

These tests exercise every pure function in `bamengine.systems.labor_market`
under the extreme or economically-interesting scenarios that are possible
after the component refactor.

The tiny helpers from  `tests.helpers.factories`  hide all boiler-plate
array construction so each test can focus on *behaviour* instead of setup.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from bamengine.components import Employer, Worker
from bamengine.systems.labor_market import (  # system under test
    _topk_indices_desc,
    adjust_minimum_wage,
    firms_decide_wage_offer,
    firms_hire_workers,
    workers_decide_firms_to_apply,
    workers_send_one_round,
)
from tests.helpers.factories import (
    mock_economy,
    mock_employer,
    mock_worker,
)

# --------------------------------------------------------------------------- #
#  deterministic micro-scenario helper                                        #
# --------------------------------------------------------------------------- #
def _mini_state(
    *,
    n_workers: int = 6,
    n_employers: int = 3,
    M: int = 2,
    seed: int = 1,
) -> tuple[Employer, Worker, Generator, int]:
    """
    Build **one** fully-featured Employer and one Worker component,
    plus an RNG and queue width *M*.

    * Firms: wage-offers [1.0, 1.5, 1.2], vacancies [2, 1, 0],
             current labour [1, 0, 2]  ⇒ mix of hiring / non-hiring.
    * Workers: 4 unemployed, 2 employed – the unemployed span every branch
      of the application logic.
    """
    assert M > 0
    rng = default_rng(seed)
    emp = mock_employer(
        n=n_employers,
        queue_w=M,
        wage_offer=np.array([1.0, 1.5, 1.2]),
        n_vacancies=np.array([2, 1, 0]),
        current_labor=np.array([1, 0, 2], dtype=np.int64),
    )
    wrk = mock_worker(
        n=n_workers,
        queue_w=M,
        employed=np.array([False, False, False, True, False, True]),
        employer_prev=np.full(n_workers, -1, dtype=np.intp),
        contract_expired=np.zeros(n_workers, dtype=np.bool_),
        fired=np.zeros(n_workers, dtype=np.bool_),
    )

    return emp, wrk, rng, M

# --------------------------------------------------------------------------- #
#  Minimum-wage inflation rule
# --------------------------------------------------------------------------- #
def test_adjust_minimum_wage_revision() -> None:
    """Exact revision step (len = m+1) – floor must move by realised inflation."""
    ec = mock_economy(
        min_wage=1.0,
        avg_mkt_price_history=np.array([1.00, 1.05, 1.10, 1.15, 1.20]),  # t = 4 (len = 5)
        min_wage_rev_period=4,
    )
    adjust_minimum_wage(ec)
    assert ec.min_wage == pytest.approx(1.20)  # +20 % inflation


@pytest.mark.parametrize(
    ("prices", "direction"),
    [
        (np.array([1.00, 1.05, 1.10, 1.15, 1.20]), "up"),  # inflation
        (np.array([1.00, 0.95, 0.90, 0.85, 0.80]), "down"),  # deflation
        (np.array([1.00, 1.10, 1.20, 1.30]), "flat"),  # = m  → no change
        (np.array([1.00, 1.10, 1.20]), "flat"),  # < m  → no change
    ],
)
def test_adjust_minimum_wage_edges(prices: NDArray[np.float64], direction: str) -> None:
    """Guard array bounds; min wage may go **up or down** depending on inflation."""
    ec = mock_economy(
        min_wage=2.0,
        avg_mkt_price_history=prices,
        min_wage_rev_period=4,
    )
    old = ec.min_wage
    adjust_minimum_wage(ec)
    if direction == "up":
        assert ec.min_wage > old
    elif direction == "down":
        assert ec.min_wage < old
    else:  # no revision
        assert ec.min_wage == pytest.approx(old)

# --------------------------------------------------------------------------- #
#  firms_decide_wage_offer                                                    #
# --------------------------------------------------------------------------- #
def test_decide_wage_offer_basic() -> None:
    """Hiring firms get a stochastic mark-up, non-hiring firms keep their offer."""
    rng = default_rng(0)
    emp = mock_employer(
        n=4,
        n_vacancies=np.array([3, 0, 2, 0]),
        wage_offer=np.array([1.0, 1.2, 1.1, 1.3]),
    )
    prev = emp.wage_offer.copy()
    firms_decide_wage_offer(emp, w_min=1.05, h_xi=0.1, rng=rng)

    # bound checks
    assert (emp.wage_offer >= 1.05).all()
    # non-hiring unchanged (were already above floor)
    np.testing.assert_allclose(emp.wage_offer[[1, 3]], prev[[1, 3]])
    # hiring firms cannot decrease
    assert (emp.wage_offer[emp.n_vacancies > 0] >= prev[emp.n_vacancies > 0]).all()


def test_decide_wage_offer_floor_and_shock() -> None:
    """
    If w_prev < w_min the floor binds;
    otherwise a positive shock draws w_offer above w_prev.
    """
    rng = default_rng(2)
    emp = mock_employer(
        n=2,
        n_vacancies=np.array([0, 3]),
        wage_offer=np.array([0.8, 1.2]),
    )
    firms_decide_wage_offer(emp, w_min=1.0, h_xi=0.1, rng=rng)
    assert emp.wage_offer[0] == pytest.approx(1.0)  # floor binds
    assert emp.wage_offer[1] >= 1.2  # positive shock


def test_decide_wage_offer_reuses_scratch() -> None:
    emp = mock_employer(n=3, n_vacancies=[1, 0, 2])
    firms_decide_wage_offer(emp, w_min=1.0, h_xi=0.1, rng=default_rng(0))
    buf0 = emp.wage_shock
    firms_decide_wage_offer(emp, w_min=1.1, h_xi=0.1, rng=default_rng(0))
    buf1 = emp.wage_shock
    assert buf0 is buf1                # same object
    assert buf1 is not None and buf1.flags.writeable


# --------------------------------------------------------------------------- #
#  _topk_indices_desc helper                                                  #
# --------------------------------------------------------------------------- #
def test_topk_indices_desc_partial_sort() -> None:
    """
    Requesting k < n should hit the argpartition ↦ slice branch.
    We verify that the returned (unsorted) indices are exactly
    the k positions of the largest elements.
    """
    vals = np.array([[5.0, 1.0, 3.0, 4.0]])
    k = 2
    idx = _topk_indices_desc(vals, k=k)
    # Extract the values referenced by idx and check they are the two maxima.
    top_vals = vals[0, idx[0]]
    assert set(top_vals) == {5.0, 4.0}
    # Shape must be preserved (unsorted, but length == k)
    assert idx.shape == (1, k)

# --------------------------------------------------------------------------- #
#  workers_decide_firms_to_apply                                              #
# --------------------------------------------------------------------------- #
def test_prepare_applications_basic() -> None:
    """
    Unemployed workers must obtain a valid job_apps_head and targets within bounds.
    """
    emp, wrk, rng, M = _mini_state()
    workers_decide_firms_to_apply(wrk, emp, max_M=M, rng=rng)

    heads = wrk.job_apps_head[~wrk.employed]
    assert (heads >= 0).all()

    rows = heads // M
    targets = wrk.job_apps_targets[rows]
    assert ((0 <= targets) & (targets < emp.wage_offer.size)).all()


def test_prepare_applications_no_unemployed() -> None:
    """If everyone is employed no application pointers should be set."""
    emp = mock_employer(
        n=3,
        wage_offer=np.array([1.0, 1.5, 1.2]),
        n_vacancies=np.array([0, 0, 0]),
    )
    wrk = mock_worker(n=3, employed=np.ones(3, dtype=np.bool_))
    workers_decide_firms_to_apply(wrk, emp, max_M=2, rng=default_rng(0))
    assert (wrk.job_apps_head == -1).all()


def test_prepare_applications_loyalty_to_employer() -> None:
    """
    A worker whose contract just expired (not fired)
    should list her previous employer first.
    """
    emp, wrk, rng, M = _mini_state()
    # worker-idx 0 just finished contract at firm-idx 1
    wrk.employer_prev[0] = 1
    wrk.contract_expired[0] = True
    workers_decide_firms_to_apply(wrk, emp, max_M=M, rng=rng)
    assert wrk.job_apps_targets[0, 0] == 1  # loyalty kept


@pytest.mark.xfail(reason="Cover loyalty‑swap branch later")
def test_prepare_applications_loyalty_swap() -> None:  # pragma: no cover
    raise NotImplementedError


def test_prepare_applications_one_trial() -> None:
    """Edge case M = 1 (single application per worker)."""
    emp, wrk, rng, _ = _mini_state()
    M = 1
    wrk.job_apps_targets = np.full((wrk.employed.size, M), -1, dtype=np.intp)
    workers_decide_firms_to_apply(wrk, emp, max_M=M, rng=rng)
    assert (wrk.job_apps_head[~wrk.employed] % M == 0).all()  # buffer still valid


def test_prepare_applications_large_unemployment() -> None:
    """
    More unemployed workers than emp × M.
    Sampling with replacement must still yield valid firm indices.
    """
    rng = default_rng(5)
    n_wrk, n_emp, M = 20, 3, 2
    fw = mock_employer(
        n=n_emp,
        queue_w=M,
        wage_offer=np.array([1.0, 1.5, 1.2]),
        n_vacancies=np.array([2, 1, 0]),
    )
    ws = mock_worker(n=n_wrk, queue_w=M)
    workers_decide_firms_to_apply(ws, fw, max_M=M, rng=rng)
    assert (ws.job_apps_targets[ws.job_apps_targets >= 0] < n_emp).all()

# --------------------------------------------------------------------------- #
#  workers_send_one_round
# --------------------------------------------------------------------------- #
def test_workers_send_one_round_queue_bounds() -> None:
    """
    When a firm’s queue is almost full, the next application must be dropped
    rather than overflow the buffer.
    """
    emp, wrk, _, M = _mini_state()
    # Preload firm-0 queue to capacity-1
    emp.recv_job_apps_head[0] = M - 2
    emp.recv_job_apps[0, : M - 1] = [0, 2][: M - 1]

    workers_decide_firms_to_apply(wrk, emp, max_M=M, rng=default_rng(6))
    workers_send_one_round(wrk, emp)

    # still valid pointer
    np.testing.assert_array_less(emp.recv_job_apps_head, M)
    assert (emp.recv_job_apps_head >= -1).all()


def test_worker_with_empty_list_is_skipped() -> None:
    emp, wrk, _, _ = _mini_state()
    # worker‑0: unemployed but job_apps_head = -1  => should be ignored
    wrk.employed[0] = False
    wrk.job_apps_head[0] = -1
    workers_send_one_round(wrk, emp)
    # queue heads unchanged
    assert (emp.recv_job_apps_head == -1).all()


def test_firm_queue_full_drops_application() -> None:
    emp, wrk, rng, M = _mini_state()
    emp.recv_job_apps_head[0] = M - 1  # already full
    emp.recv_job_apps[0] = 99  # dummy
    workers_decide_firms_to_apply(wrk, emp, max_M=M, rng=rng)
    workers_send_one_round(wrk, emp)
    # still full, nothing overwritten
    assert emp.recv_job_apps_head[0] == M - 1
    assert (emp.recv_job_apps[0] == 99).all()


def test_workers_send_one_round_exhausted_target() -> None:
    """
    When the current target has already been set to -1 *before* the call,
    the branch `firm_idx < 0` must trigger and clear job_apps_head to -1.
    """
    # --- minimal 1‑worker 1‑firm state -----------------------------------
    M = 1
    wrk = mock_worker(
        n=1,
        queue_w=M,
        employed=np.array([False]),
        job_apps_head=np.array([0]),  # points to first cell
        job_apps_targets=np.array([[-1]], dtype=np.intp),  # *already* exhausted
    )
    emp = mock_employer(
        n=1,
        queue_w=M,
        wage_offer=np.array([1.0]),
        n_vacancies=np.array([1]),
        current_labor=np.zeros(1, dtype=np.int64),
    )
    workers_send_one_round(wrk, emp)

    # job_apps_head must be cleared; nothing should be queued
    assert wrk.job_apps_head[0] == -1
    assert emp.recv_job_apps_head[0] == -1

# --------------------------------------------------------------------------- #
#  firms_hire_workers
# --------------------------------------------------------------------------- #
def test_firms_hire_no_vacancies() -> None:
    """
    Applications sent to a firm with zero vacancies should be cleared
    without hiring anyone and without crashing.
    """
    emp, wrk, _, _ = _mini_state()
    start = emp.current_labor.copy()

    emp.n_vacancies.fill(0)  # nobody hiring
    emp.recv_job_apps_head[1] = 0
    emp.recv_job_apps[1, 0] = 2

    firms_hire_workers(wrk, emp, theta=8)

    # the application must *not* result in a hire
    np.testing.assert_array_equal(emp.current_labor, start)
    assert not wrk.employed[2]

    # pointer stays where it was (0) — queue ignored, not flushed
    assert emp.recv_job_apps_head[1] == 0


def test_firms_hire_exact_fit() -> None:
    """
    If the number of applications equals vacancies,
    all are hired and the queue is cleared.
    """
    emp, wrk, _, _ = _mini_state()
    start = emp.current_labor.copy()

    emp.recv_job_apps_head[0] = 1
    emp.recv_job_apps[0, :2] = [0, 2]

    firms_hire_workers(wrk, emp, theta=8)

    assert emp.n_vacancies[0] == 0
    assert emp.current_labor[0] == start[0] + 2
    assert (emp.recv_job_apps[0] == -1).all()
    assert wrk.employed[[0, 2]].sum() == 2  # both hired


def test_hire_workers_skips_invalid_slots() -> None:
    emp, wrk, _, _ = _mini_state()
    emp.n_vacancies.fill(1)
    emp.recv_job_apps_head[0] = 1
    emp.recv_job_apps[0].fill(-1)  # all sentinels → size==0
    start = emp.current_labor.copy()
    firms_hire_workers(wrk, emp, theta=8)
    # nothing hired, vacancies unchanged
    np.testing.assert_array_equal(emp.current_labor, start)
    assert emp.n_vacancies[0] == 1


# --------------------------------------------------------------------------- #
#  End-to-end micro integration of one hiring event
# --------------------------------------------------------------------------- #
def test_full_round() -> None:
    """
    One complete labor-market event with M rounds should leave:
    * non-negative vacancies,
    * at least one worker hired (given vacancies > 0),
    * no dangling head pointers for hired workers.
    """
    emp, wrk, rng, M = _mini_state()
    start_total_labor = emp.current_labor.sum()

    workers_decide_firms_to_apply(wrk, emp, max_M=M, rng=rng)
    for _ in range(M):
        workers_send_one_round(wrk, emp)
        firms_hire_workers(wrk, emp, theta=8)

    assert (emp.n_vacancies >= 0).all()
    assert wrk.employed.any()
    assert emp.current_labor.sum() >= start_total_labor
    assert (wrk.job_apps_head[wrk.employed] == -1).all()
