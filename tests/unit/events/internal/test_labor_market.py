"""
Labor-market events internal implementation unit tests.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from bamengine import Rng, make_rng
from bamengine.events._internal.labor_market import (
    adjust_minimum_wage,
    calc_annual_inflation_rate,
    firms_decide_wage_offer,
    firms_hire_workers,
    workers_decide_firms_to_apply,
    workers_send_one_round,
)
from bamengine.roles import Employer, Worker
from tests.helpers.factories import mock_economy, mock_employer, mock_worker
from tests.helpers.fixed_rng import FixedRNG


def _mini_state(
    *,
    n_workers: int = 6,
    n_employers: int = 3,
    M: int = 2,
    seed: int = 1,
) -> tuple[Employer, Worker, Rng, int]:
    """
    Build **one** fully-featured Employer and one Worker component,
    plus an RNG and queue width *M*.

    * Firms: wage-offers [1.0, 1.5, 1.2], vacancies [2, 1, 0],
             current labor [1, 0, 2]  ⇒ mix of hiring / non-hiring.
    * Workers: 4 unemployed, 2 employed – the unemployed span every branch
      of the application logic.
    """
    assert M > 0
    rng = make_rng(seed)
    emp = mock_employer(
        n=n_employers,
        queue_m=M,
        wage_offer=np.array([1.0, 1.5, 1.2]),
        n_vacancies=np.array([2, 1, 0]),
        current_labor=np.array([1, 0, 2], dtype=np.int64),
    )
    wrk = mock_worker(
        n=n_workers,
        queue_m=M,
        employer=np.array([-1, -1, -1, 0, -1, 1], dtype=np.intp),
        employer_prev=np.full(n_workers, -1, dtype=np.intp),
        contract_expired=np.zeros(n_workers, dtype=np.bool_),
        fired=np.zeros(n_workers, dtype=np.bool_),
    )

    return emp, wrk, rng, M


def test_calc_annual_inflation_rate_min_history() -> None:
    from bamengine.events._internal.labor_market import calc_annual_inflation_rate

    ec = mock_economy(avg_mkt_price_history=np.array([1.0, 1.1, 1.2, 1.3]))
    calc_annual_inflation_rate(ec)
    assert ec.inflation_history[-1] == 0.0


def test_calc_annual_inflation_rate_prev_nonpositive() -> None:
    from bamengine.events._internal.labor_market import calc_annual_inflation_rate

    # t = 4 (len=5); p_{t-4} is index 0 -> set to 0.0 to trigger branch
    ec = mock_economy(avg_mkt_price_history=np.array([0.0, 1.0, 1.1, 1.2, 1.3]))
    calc_annual_inflation_rate(ec)
    assert ec.inflation_history[-1] == 0.0


@pytest.mark.parametrize(
    ("prices", "direction"),
    [
        (np.array([1.00, 1.05, 1.10, 1.15, 1.20]), "up"),  # inflation
        (np.array([1.00, 0.95, 0.90, 0.85, 0.80]), "down"),  # deflation
        (np.array([1.00, 1.10, 1.20, 1.30]), "flat"),  # = m  → no change
        (np.array([1.00, 1.05, 1.10, 1.15, 1.20, 1.30]), "flat"),  # < m  → no change
    ],
)
def test_adjust_minimum_wage_edges(prices: NDArray[np.float64], direction: str) -> None:
    """Guard array bounds; min wage may go **up or down** depending on inflation."""
    ec = mock_economy(
        min_wage=2.0,
        avg_mkt_price_history=prices,
        min_wage_rev_period=4,
    )
    wrk = mock_worker(
        n=2, employer=np.array([0, -1], dtype=np.intp), wage=np.array([1.5, 0.0])
    )
    old = ec.min_wage
    calc_annual_inflation_rate(ec)
    adjust_minimum_wage(ec, wrk)
    if direction == "up":
        assert ec.min_wage > old
    elif direction == "down":
        assert ec.min_wage < old
    else:  # no revision
        assert ec.min_wage == pytest.approx(old)


def test_adjust_minimum_wage_revision() -> None:
    """Exact revision step (len = m+1) – floor must move by realised inflation."""
    ec = mock_economy(
        min_wage=1.0,
        avg_mkt_price_history=np.array(
            [1.00, 1.05, 1.10, 1.15, 1.20]
        ),  # t = 4 (len = 5)
        min_wage_rev_period=4,
    )
    wrk = mock_worker(
        n=2, employer=np.array([0, -1], dtype=np.intp), wage=np.array([0.9, 0.0])
    )
    calc_annual_inflation_rate(ec)
    adjust_minimum_wage(ec, wrk)
    assert ec.min_wage == pytest.approx(1.20)  # +20 % inflation
    # Employed worker wage should be updated to new minimum
    assert wrk.wage[0] == pytest.approx(1.20)


def test_adjust_minimum_wage_from_history_revision() -> None:
    ec = mock_economy(
        min_wage=1.0,
        avg_mkt_price_history=np.array([1, 1, 1, 1, 1]),  # len=5, m=4 -> revision
        min_wage_rev_period=4,
    )
    ec.inflation_history = np.array([0.10])  # +10%
    wrk = mock_worker(
        n=2, employer=np.array([0, -1], dtype=np.intp), wage=np.array([0.9, 0.0])
    )
    adjust_minimum_wage(ec, wrk)
    assert ec.min_wage == pytest.approx(1.1)
    # Employed worker wage should be updated to new minimum
    assert wrk.wage[0] == pytest.approx(1.1)


def test_adjust_minimum_wage_skips_when_not_revision() -> None:
    ec = mock_economy(
        min_wage=1.0,
        avg_mkt_price_history=np.array(
            [1, 1, 1, 1, 1, 1]
        ),  # len=6, not revision (5 % 4 != 0)
        min_wage_rev_period=4,
    )
    ec.inflation_history = np.array([0.25])
    wrk = mock_worker(
        n=2, employer=np.array([0, -1], dtype=np.intp), wage=np.array([0.9, 0.0])
    )
    old = ec.min_wage
    old_wage = wrk.wage[0]
    adjust_minimum_wage(ec, wrk)
    assert ec.min_wage == pytest.approx(old)
    # Worker wage should remain unchanged since no revision happened
    assert wrk.wage[0] == pytest.approx(old_wage)


def test_decide_wage_offer_basic() -> None:
    """Hiring firms get a stochastic mark-up, non-hiring firms keep their offer."""
    rng = make_rng(0)
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
    rng = make_rng(2)
    emp = mock_employer(
        n=2,
        n_vacancies=np.array([0, 3]),
        wage_offer=np.array([0.8, 1.2]),
    )
    firms_decide_wage_offer(emp, w_min=1.0, h_xi=0.1, rng=rng)
    assert emp.wage_offer[0] == pytest.approx(1.0)  # floor binds
    assert emp.wage_offer[1] >= 1.2  # positive shock


def test_decide_wage_offer_reuses_scratch() -> None:
    emp = mock_employer(n=3, n_vacancies=np.array([1, 0, 2]))
    firms_decide_wage_offer(emp, w_min=1.0, h_xi=0.1, rng=make_rng(0))
    buf0 = emp.wage_shock
    firms_decide_wage_offer(emp, w_min=1.1, h_xi=0.1, rng=make_rng(0))
    buf1 = emp.wage_shock
    assert buf0 is buf1  # same object
    assert buf1 is not None and buf1.flags.writeable


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
    assert ((targets >= 0) & (targets < emp.wage_offer.size)).all()


def test_prepare_applications_no_unemployed() -> None:
    """If everyone is employed no application pointers should be set."""
    emp = mock_employer(
        n=3,
        wage_offer=np.array([1.0, 1.5, 1.2]),
        n_vacancies=np.array([0, 0, 0]),
    )
    wrk = mock_worker(n=3, employer=np.array([0, 1, 2], dtype=np.intp))  # all employed
    workers_decide_firms_to_apply(wrk, emp, max_M=2, rng=make_rng(0))
    assert np.all(wrk.job_apps_head == -1)


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


def test_prepare_applications_loyalty_swap_branch() -> None:
    """
    Drive the branch that swaps columns when the previous employer is pushed
    away from column-0 by the wage-descending partial sort.

    Conditions required by the code:

    • loyal == True                      → worker had employer_prev ≥ 0
    • filled > 1                         → max_M ≥ 2  and we drew ≥ 1 other firm
    • row[0] != prev after the sort      → prev's wage < another sampled firm
    """

    emp = mock_employer(
        n=2,
        queue_m=2,
        wage_offer=np.array([1.0, 10.0]),  # firm-1 much higher wage
        n_vacancies=np.array([1, 1]),
    )
    wrk = mock_worker(
        n=1,
        queue_m=2,
        employer=np.array([-1], dtype=np.intp),  # unemployed
        contract_expired=np.array([True]),
        fired=np.array([False]),
        employer_prev=np.array([0]),  # loyalty to firm-0
    )

    # Fixed buffer will force the sample [[1, 1]]
    stub_rng = FixedRNG(np.array([[1, 1]], dtype=np.int64))

    # Cast keeps the production code’s type hints intact
    workers_decide_firms_to_apply(
        wrk, emp, max_M=2, rng=cast(Rng, cast(object, stub_rng))
    )

    # assertions
    assert wrk.job_apps_targets[0, 0] == 0  # previous employer in col-0
    assert wrk.job_apps_targets[0, 1] == 1  # other firm in col-1


def test_prepare_applications_loyalty_noop_when_already_first() -> None:
    emp = mock_employer(
        n=2,
        queue_m=2,
        wage_offer=np.array([10.0, 1.0]),  # prev employer (0) has higher wage
        n_vacancies=np.array([1, 1]),
    )
    wrk = mock_worker(
        n=1,
        queue_m=2,
        employer=np.array([-1], dtype=np.intp),  # unemployed
        contract_expired=np.array([True]),
        fired=np.array([False]),
        employer_prev=np.array([0]),
    )
    # Sample [0,1] so sort keeps 0 in col-0
    stub_rng = FixedRNG(np.array([[0, 1]], dtype=np.int64))
    workers_decide_firms_to_apply(
        wrk, emp, max_M=2, rng=cast(Rng, cast(object, stub_rng))
    )
    np.testing.assert_array_equal(wrk.job_apps_targets[0], np.array([0, 1]))


def test_prepare_applications_one_trial() -> None:
    """Edge case M = 1 (single application per worker)."""
    emp, wrk, rng, _ = _mini_state()
    M = 1
    wrk.job_apps_targets = np.full((wrk.employed.size, M), -1, dtype=np.intp)
    workers_decide_firms_to_apply(wrk, emp, max_M=M, rng=rng)
    assert np.all(wrk.job_apps_head[~wrk.employed] % M == 0)  # buffer still valid


def test_prepare_applications_large_unemployment() -> None:
    """
    More unemployed workers than emp × M.
    Sampling with replacement must still yield valid firm indices.
    """
    rng = make_rng(5)
    n_wrk, n_emp, M = 20, 3, 2
    fw = mock_employer(
        n=n_emp,
        queue_m=M,
        wage_offer=np.array([1.0, 1.5, 1.2]),
        n_vacancies=np.array([2, 1, 0]),
    )
    ws = mock_worker(n=n_wrk, queue_m=M)
    workers_decide_firms_to_apply(ws, fw, max_M=M, rng=rng)
    assert (ws.job_apps_targets[ws.job_apps_targets >= 0] < n_emp).all()


def test_prepare_applications_no_hiring_but_unemployed() -> None:
    emp = mock_employer(
        n=3, wage_offer=np.array([1.0, 1.5, 1.2]), n_vacancies=np.array([0, 0, 0])
    )
    wrk = mock_worker(
        n=4, queue_m=2, employer=np.array([-1, -1, 0, -1], dtype=np.intp)
    )  # worker 2 employed, others unemployed
    workers_decide_firms_to_apply(wrk, emp, max_M=2, rng=make_rng(0))
    unemp = np.where(wrk.employed == 0)[0]
    assert np.all(wrk.job_apps_head[unemp] == -1)
    assert np.all(wrk.job_apps_targets[unemp] == -1)


def test_prepare_applications_job_search_method_vacancies_only() -> None:
    """
    With job_search_method='vacancies_only', workers should only sample
    from firms with vacancies (the default behavior).
    """
    emp = mock_employer(
        n=4,
        queue_m=2,
        wage_offer=np.array([1.0, 1.5, 1.2, 2.0]),
        n_vacancies=np.array([2, 0, 1, 0]),  # only firms 0 and 2 have vacancies
    )
    wrk = mock_worker(
        n=3,
        queue_m=2,
        employer=np.array([-1, -1, -1], dtype=np.intp),  # all unemployed
    )
    workers_decide_firms_to_apply(
        wrk, emp, max_M=2, job_search_method="vacancies_only", rng=make_rng(0)
    )
    # All targets should only be firms 0 or 2 (those with vacancies)
    unemp = np.where(wrk.employed == 0)[0]
    targets = wrk.job_apps_targets[unemp]
    valid_targets = targets[targets >= 0]
    assert np.all(np.isin(valid_targets, [0, 2]))


def test_prepare_applications_job_search_method_all_firms() -> None:
    """
    With job_search_method='all_firms', workers can sample from ANY firm,
    including those without vacancies.
    """
    emp = mock_employer(
        n=4,
        queue_m=4,
        wage_offer=np.array([1.0, 1.5, 1.2, 2.0]),
        n_vacancies=np.array([0, 0, 0, 0]),  # NO firm has vacancies
    )
    wrk = mock_worker(
        n=3,
        queue_m=4,
        employer=np.array([-1, -1, -1], dtype=np.intp),  # all unemployed
    )
    # With vacancies_only, this would result in empty queues
    # With all_firms, workers should still have targets
    workers_decide_firms_to_apply(
        wrk, emp, max_M=4, job_search_method="all_firms", rng=make_rng(0)
    )
    unemp = np.where(wrk.employed == 0)[0]
    # Workers should have valid targets even though no firm has vacancies
    assert np.any(wrk.job_apps_head[unemp] >= 0)
    targets = wrk.job_apps_targets[unemp]
    valid_targets = targets[targets >= 0]
    assert valid_targets.size > 0
    # Targets can be any firm (0, 1, 2, or 3)
    assert np.all((valid_targets >= 0) & (valid_targets < 4))


def test_workers_send_one_round() -> None:
    """
    One unemployed worker should place a single application in the only
    firm's inbound queue and advance her head pointer by exactly +1.
    """
    M = 2
    # build minimal roles
    emp = mock_employer(
        n=1,
        queue_m=M,
        wage_offer=np.array([1.0]),
        n_vacancies=np.array([2]),
    )
    wrk = mock_worker(n=1, queue_m=M)  # 1 unemployed worker

    # prepare targets & head pointer
    workers_decide_firms_to_apply(wrk, emp, max_M=M, rng=make_rng(0))
    head_before = int(wrk.job_apps_head[0])

    workers_send_one_round(wrk, emp)

    # firm-side
    assert emp.recv_job_apps_head[0] == 0  # one message
    assert emp.recv_job_apps[0, 0] == 0  # worker-id stored
    np.testing.assert_array_less(emp.recv_job_apps_head, M)

    # worker-side
    assert wrk.job_apps_head[0] == head_before + 1
    assert wrk.job_apps_targets[0, 0] == -1  # slot cleared


def test_workers_send_one_round_queue_bounds() -> None:
    """
    When a firm's queue is full, the next application must be dropped
    rather than overflow the buffer. Also ensure heads remain within
    [-1, capacity-1] across all firms.
    """
    emp, wrk, _, _M = _mini_state()
    cap = emp.recv_job_apps.shape[1]

    # Fill firm-0 queue to capacity (head points at the last valid slot)
    emp.recv_job_apps_head[0] = cap - 1
    prefill = np.arange(cap) % wrk.employed.size  # any valid worker ids
    emp.recv_job_apps[0, :cap] = prefill

    # Prepare exactly one unemployed worker to apply to firm 0 next
    # (bypass the planner to avoid randomness)
    target_worker = int(np.flatnonzero(wrk.employed == 0)[0])
    wrk.job_apps_targets[target_worker, :] = -1
    wrk.job_apps_targets[target_worker, 0] = 0  # firm 0
    wrk.job_apps_head[target_worker] = 0

    # Snapshot state to verify no writes past capacity
    head_before = emp.recv_job_apps_head.copy()
    queue_before = emp.recv_job_apps.copy()

    workers_send_one_round(wrk, emp)  # rng irrelevant with one applicant

    # Pointers stay valid w.r.t. actual queue capacity (not M)
    np.testing.assert_array_less(emp.recv_job_apps_head, cap)
    assert (emp.recv_job_apps_head >= -1).all()

    # Firm-0 remained full; the extra application was dropped
    assert emp.recv_job_apps_head[0] == cap - 1
    np.testing.assert_array_equal(emp.recv_job_apps[0], queue_before[0])

    # Other firms unaffected
    np.testing.assert_array_equal(emp.recv_job_apps_head[1:], head_before[1:])
    np.testing.assert_array_equal(emp.recv_job_apps[1:], queue_before[1:])

    # The worker advanced past the attempted application and it was cleared
    assert wrk.job_apps_head[target_worker] == 1
    assert wrk.job_apps_targets[target_worker, 0] == -1


def test_worker_with_empty_list_is_skipped() -> None:
    emp, wrk, _, _ = _mini_state()
    # worker‑0: unemployed but job_apps_head = -1  => should be ignored
    wrk.employed[0] = False
    wrk.job_apps_head[0] = -1
    workers_send_one_round(wrk, emp)
    # queue heads unchanged
    assert np.all(emp.recv_job_apps_head == -1)


def test_firm_queue_full_drops_application() -> None:
    emp, wrk, rng, M = _mini_state()
    emp.recv_job_apps_head[0] = M - 1  # already full
    emp.recv_job_apps[0] = 99  # dummy
    workers_decide_firms_to_apply(wrk, emp, max_M=M, rng=rng)
    workers_send_one_round(wrk, emp)
    # still full, nothing overwritten
    assert emp.recv_job_apps_head[0] == M - 1
    assert np.all(emp.recv_job_apps[0] == 99)


def test_workers_send_one_round_exhausted_target() -> None:
    """
    When the current target has already been set to -1 *before* the call,
    the branch `firm_idx < 0` must trigger and clear job_apps_head to -1.
    """
    # minimal 1‑worker 1‑firm state
    M = 1
    wrk = mock_worker(
        n=1,
        queue_m=M,
        employer=np.array([-1], dtype=np.intp),  # unemployed
        job_apps_head=np.array([0]),  # points to first cell
        job_apps_targets=np.array([[-1]], dtype=np.intp),  # *already* exhausted
    )
    emp = mock_employer(
        n=1,
        queue_m=M,
        wage_offer=np.array([1.0]),
        n_vacancies=np.array([1]),
        current_labor=np.zeros(1, dtype=np.int64),
    )
    workers_send_one_round(wrk, emp)

    # job_apps_head must be cleared; nothing should be queued
    assert wrk.job_apps_head[0] == -1
    assert emp.recv_job_apps_head[0] == -1


def test_workers_send_one_round_drops_due_to_no_vacancy() -> None:
    M = 2
    emp = mock_employer(
        n=1, queue_m=M, wage_offer=np.array([1.0]), n_vacancies=np.array([0])
    )
    wrk = mock_worker(
        n=1,
        queue_m=M,
        employer=np.array([-1], dtype=np.intp),  # unemployed
        job_apps_head=np.array([0]),
        job_apps_targets=np.array([[0]], dtype=np.intp),
    )
    workers_send_one_round(wrk, emp)
    assert wrk.job_apps_head[0] == 1
    assert wrk.job_apps_targets[0, 0] == -1
    assert emp.recv_job_apps_head[0] == -1  # nothing queued


def test_send_one_job_app_head_negative_after_shuffle_branch() -> None:
    """
    Reach the branch:
        if head < 0:  # worker considered active but head flipped negative
            ... continue
    We simulate an external mutation by providing a custom RNG whose shuffle()
    both shuffles and sets one worker's head to -1 *after* unemp_ids_applying
    is constructed but *before* the loop reads head.
    """
    # One firm with capacity; one unemployed worker with a valid head/target
    M = 1
    emp = mock_employer(
        n=1,
        queue_m=M,
        wage_offer=np.array([1.0]),
        n_vacancies=np.array([1]),
    )
    wrk = mock_worker(
        n=1,
        queue_m=M,
        employer=np.array([-1], dtype=np.intp),  # unemployed
        job_apps_head=np.array([0]),  # would otherwise proceed
        job_apps_targets=np.array([[0]], dtype=np.intp),  # firm 0
    )

    class EvilRng:
        @staticmethod
        def shuffle(arr: np.ndarray) -> None:
            # Flip the head of the first worker in the shuffled list to -1
            if arr.size > 0:
                idx = int(arr[0])
                wrk.job_apps_head[idx] = -1
            # deterministic "shuffle"
            arr[:] = arr[::-1]

    workers_send_one_round(wrk, emp, rng=cast(Rng, cast(object, EvilRng())))

    # Since the only worker got head < 0 right before the loop,
    # nothing should have been queued and head should remain -1.
    assert emp.recv_job_apps_head[0] == -1
    assert wrk.job_apps_head[0] == -1
    assert np.all(emp.recv_job_apps[0] == -1)


def test_send_one_job_app_exhausted_then_cleanup_next_round() -> None:
    """
    Cover the branch: head >= (j + 1) * stride  → mark worker done (head = -1).
    Let a worker consume all their slots; on the *next* call they should be
    detected as exhausted and cleaned up without queue writes.
    """
    # One worker, one firm, stride = 2 (two application slots)
    M = 2
    emp = mock_employer(
        n=1,
        queue_m=M,
        wage_offer=np.array([1.0]),
        n_vacancies=np.array([2]),  # allow both apps to be queued
    )
    wrk = mock_worker(
        n=1,
        queue_m=M,
        employer=np.array([-1], dtype=np.intp),  # unemployed
        job_apps_head=np.array([0]),  # start at first cell
        job_apps_targets=np.array([[0, 0]], dtype=np.intp),  # same firm twice is fine
    )

    # First call → consumes col=0, advances head to 1
    workers_send_one_round(wrk, emp)
    assert wrk.job_apps_head[0] == 1
    assert emp.recv_job_apps_head[0] == 0  # one application landed

    # Second call → consumes col=1, advances head to 2 == (j+1)*stride
    workers_send_one_round(wrk, emp)
    assert wrk.job_apps_head[0] == 2
    assert emp.recv_job_apps_head[0] == 1  # two applications landed

    # Third call → triggers the exhausted-branch and cleans up (no extra writes)
    queue_before = emp.recv_job_apps.copy()
    head_before = int(emp.recv_job_apps_head[0])

    workers_send_one_round(wrk, emp)

    assert wrk.job_apps_head[0] == -1
    assert emp.recv_job_apps_head[0] == head_before  # unchanged
    np.testing.assert_array_equal(emp.recv_job_apps, queue_before)


def test_firms_hire_workers_basic() -> None:
    """
    A firm with vacancies hires the first two applicants in its queue
    and updates all related state consistently.
    """
    M = 3
    theta = 8
    emp = mock_employer(
        n=1,
        queue_m=M,
        n_vacancies=np.array([3]),
        current_labor=np.array([0], dtype=np.int64),
    )
    wrk = mock_worker(n=3, queue_m=M)

    rng = make_rng(0)
    rng_check = make_rng(0)
    expected_extra = rng_check.poisson(10)

    # preload queue with worker-ids 0 and 1
    emp.recv_job_apps_head[0] = 1  # queue length = 2
    emp.recv_job_apps[0, :2] = [0, 1]

    L_before = emp.current_labor[0]
    V_before = emp.n_vacancies[0]

    firms_hire_workers(wrk, emp, theta=theta, rng=rng)

    # firm-side checks
    assert emp.current_labor[0] == L_before + 2
    assert emp.n_vacancies[0] == V_before - 2
    assert emp.recv_job_apps_head[0] == -1
    assert np.all(emp.recv_job_apps[0] == -1)

    # worker-side checks
    assert wrk.employed[[0, 1]].all()  # both hired
    assert np.all(wrk.employer[[0, 1]] == 0)  # employer set

    # contract length: same scalar for both hires
    expected_periods = theta + expected_extra
    assert np.all(wrk.periods_left[[0, 1]] == expected_periods)

    # queues cleared for those workers
    assert np.all(wrk.job_apps_head[[0, 1]] == -1)


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
    assert np.all(emp.recv_job_apps[0] == -1)
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


def test_firms_hire_capped_by_vacancies() -> None:
    emp = mock_employer(
        n=1,
        queue_m=3,
        n_vacancies=np.array([1]),
        current_labor=np.array([0], dtype=np.int64),
    )
    wrk = mock_worker(n=3, queue_m=3)
    emp.recv_job_apps_head[0] = 2
    emp.recv_job_apps[0, :3] = [0, 1, 2]
    firms_hire_workers(wrk, emp, theta=8)
    assert emp.current_labor[0] == 1
    assert wrk.employed.sum() == 1
    assert emp.n_vacancies[0] == 0


def _queues_with_dupes(
    *,
    n_firms: int,
    M: int,
    n_workers: int,
) -> st.SearchStrategy[list[list[int]]]:
    """
    Strategy: per firm build a raw queue (list[int]) of length ≤ M that may
    contain duplicates, cross–firm overlaps and “-1” sentinels.
    """
    return st.lists(
        st.lists(
            st.integers(-1, n_workers - 1),
            min_size=0,
            max_size=M,
        ),
        min_size=n_firms,
        max_size=n_firms,
    )


@settings(max_examples=300, deadline=None)
@given(
    st.integers(1, 6).flatmap(  # number of firms
        lambda n_firms: st.tuples(
            st.integers(1, 5),  # M
            st.lists(st.integers(0, 5), min_size=n_firms, max_size=n_firms),  # V_i
            st.integers(n_firms + 10, n_firms + 60),  # workers
        ).flatmap(
            lambda t: st.tuples(
                st.just(n_firms),
                st.just(t[0]),
                st.just(np.asarray(t[1], dtype=np.int64)),  # vacancies
                st.just(t[2]),  # n_workers
                _queues_with_dupes(
                    n_firms=n_firms,
                    M=t[0],
                    n_workers=t[2],
                ),
            )
        )
    )
)
def test_hire_invariants_with_duplicates(
    random_case: tuple[int, int, NDArray[np.int64], int, list[list[int]]],
) -> None:
    """
    After `firms_hire_workers` :

        • 0 ≤ ΔLᵢ ≤ min(Uᵢ , Vᵢ)
        • ΔVᵢ == −ΔLᵢ
        • global hired-once guarantee  (Σ ΔL == Σ hired workers)

    with
        Uᵢ … #unique, *still unemployed* applicants in firm-i queue *before* call
        Vᵢ … vacancies before call
    """
    # unpack
    n_firms, M, V_in, n_workers, q_raw = random_case
    vacancies: NDArray[np.int64] = V_in

    # build minimal roles
    emp = mock_employer(
        n=n_firms,
        queue_m=M,
        n_vacancies=vacancies.copy(),  # keep original untouched
    )
    wrk = mock_worker(n=n_workers, queue_m=M)
    wrk.employed[:] = False  # everyone unemployed at t₀

    # load raw queues into the employer buffers
    for i in range(n_firms):
        q: list[int] = q_raw[i]
        q_arr: NDArray[np.intp] = np.asarray(q[:M], dtype=np.intp)  # trim to M
        emp.recv_job_apps_head[i] = q_arr.size - 1  # −1 ⇒ empty
        if q_arr.size:
            emp.recv_job_apps[i, : q_arr.size] = q_arr

    # oracle values BEFORE hiring
    def _U_before(queue: NDArray[np.intp]) -> int:
        queue = queue[queue >= 0]  # drop sentinels
        if queue.size == 0:
            return 0
        return int(np.unique(queue).size)  # all unemployed at this point

    U_before: NDArray[np.int64] = np.array(
        [_U_before(emp.recv_job_apps[i]) for i in range(n_firms)],
        dtype=np.int64,
    )
    L_before = emp.current_labor.copy()
    V_before = emp.n_vacancies.copy()

    # run system under test
    firms_hire_workers(wrk, emp, theta=8)

    # deltas
    delta_L = emp.current_labor - L_before  # hires per firm
    delta_V = emp.n_vacancies - V_before

    # assertions
    # bounds: 0 ≤ ΔLᵢ ≤ min(Uᵢ , Vᵢ)
    assert (delta_L >= 0).all()
    assert (delta_L <= np.minimum(U_before, V_before)).all()

    # vacancies mirror labour changes
    np.testing.assert_array_equal(delta_V, -delta_L, err_msg="Δvacancies mismatch")

    # every worker hired at most once  (global sanity)
    assert emp.current_labor.sum() == wrk.employed.sum()


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
    assert np.all(wrk.job_apps_head[wrk.employed] == -1)


def test_firms_calc_wage_bill_basic() -> None:
    from bamengine.events._internal.labor_market import firms_calc_wage_bill

    emp = mock_employer(n=3)
    wrk = mock_worker(n=5)
    wrk.employed[:] = np.array([1, 1, 0, 1, 1], dtype=np.bool_)
    wrk.employer[:] = np.array([0, 0, -1, 2, 2], dtype=np.intp)
    wrk.wage[:] = np.array([1.0, 1.5, 9.9, 2.0, 3.0])
    firms_calc_wage_bill(emp, wrk)
    np.testing.assert_allclose(emp.wage_bill, np.array([2.5, 0.0, 5.0]))


# ============================================================================
# Labor Consistency Check Coverage
# ============================================================================


def test_labor_consistency_check_with_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _check_labor_consistency logs warning on mismatch."""
    from bamengine import logging
    from bamengine.events._internal.labor_market import _check_labor_consistency

    emp = mock_employer(n=2, n_vacancies=np.array([1, 0]))
    wrk = mock_worker(n=3, employer=np.array([0, 0, -1], dtype=np.intp))
    # 2 workers at firm 0, but recorded as 1 (mismatch)
    emp.current_labor[0] = 1

    # Call the check function
    with caplog.at_level(logging.WARNING, logger="bamengine"):
        result = _check_labor_consistency("TEST", 0, wrk, emp)

    assert result is False
    assert "LABOR INCONSISTENCY" in caplog.text


def test_labor_consistency_check_when_consistent() -> None:
    """Test _check_labor_consistency returns True when consistent."""
    from bamengine.events._internal.labor_market import _check_labor_consistency

    emp = mock_employer(n=2, n_vacancies=np.array([1, 0]))
    wrk = mock_worker(n=3, employer=np.array([0, 0, -1], dtype=np.intp))
    emp.current_labor[0] = 2  # Matches actual count (2 workers at firm 0)
    emp.current_labor[1] = 0  # Matches actual count (0 workers at firm 1)

    result = _check_labor_consistency("TEST", 0, wrk, emp)
    assert result is True


def test_firms_hire_workers_with_debug_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test firms_hire_workers exercises consistency checks with DEBUG logging."""
    from bamengine import logging

    M = 2
    emp = mock_employer(
        n=1,
        queue_m=M,
        n_vacancies=np.array([2]),
        current_labor=np.array([0], dtype=np.int64),
    )
    wrk = mock_worker(n=2, queue_m=M)

    # Set up queue with one application
    emp.recv_job_apps_head[0] = 0
    emp.recv_job_apps[0, 0] = 0

    # Run with DEBUG logging enabled
    with caplog.at_level(logging.DEBUG, logger="bamengine"):
        firms_hire_workers(wrk, emp, theta=8)

    # Verify hiring happened
    assert wrk.employed[0]
    assert emp.current_labor[0] == 1


# ============================================================================
# Simultaneous Matching Edge Cases
# ============================================================================


def test_workers_send_one_round_simultaneous_basic() -> None:
    """Test simultaneous matching basic flow with valid workers."""
    from bamengine.events._internal.labor_market import (
        _workers_send_one_round_simultaneous,
    )

    M = 2
    emp = mock_employer(
        n=1,
        queue_m=M,
        wage_offer=np.array([1.0]),
        n_vacancies=np.array([2]),  # Has vacancies
    )
    wrk = mock_worker(
        n=1,
        queue_m=M,
        employer=np.array([-1], dtype=np.intp),  # Unemployed
        job_apps_head=np.array([0]),  # Points to first slot
        job_apps_targets=np.array([[0, -1]], dtype=np.intp),  # Target firm 0
    )

    _workers_send_one_round_simultaneous(wrk, emp, rng=make_rng(42))

    # Worker should have applied and head advanced
    assert wrk.job_apps_head[0] == 1
    # Firm should have received application
    assert emp.recv_job_apps_head[0] >= 0


def test_workers_send_one_round_simultaneous_no_vacancy() -> None:
    """Test simultaneous matching drops application when no vacancy."""
    from bamengine.events._internal.labor_market import (
        _workers_send_one_round_simultaneous,
    )

    M = 2
    emp = mock_employer(
        n=1,
        queue_m=M,
        wage_offer=np.array([1.0]),
        n_vacancies=np.array([0]),  # NO vacancies
    )
    wrk = mock_worker(
        n=1,
        queue_m=M,
        employer=np.array([-1], dtype=np.intp),  # Unemployed
        job_apps_head=np.array([0]),  # Points to first slot
        job_apps_targets=np.array([[0, -1]], dtype=np.intp),  # Target firm 0
    )

    _workers_send_one_round_simultaneous(wrk, emp, rng=make_rng(42))

    # Worker's head should advance (skipped firm with no vacancies)
    assert wrk.job_apps_head[0] == 1
    # Firm should NOT have received application
    assert emp.recv_job_apps_head[0] == -1


def test_workers_send_one_round_simultaneous_queue_full() -> None:
    """Test simultaneous matching handles full firm queue (application dropped)."""
    from bamengine.events._internal.labor_market import (
        _workers_send_one_round_simultaneous,
    )

    # Use larger queue for valid application, but fill it first
    emp = mock_employer(
        n=1,
        queue_m=2,
        wage_offer=np.array([1.0]),
        n_vacancies=np.array([5]),  # Has vacancies
    )
    # Fill the queue completely
    emp.recv_job_apps_head[0] = emp.recv_job_apps.shape[1] - 1
    emp.recv_job_apps[0, :] = np.arange(emp.recv_job_apps.shape[1])

    wrk = mock_worker(
        n=1,
        queue_m=2,
        employer=np.array([-1], dtype=np.intp),  # Unemployed
        job_apps_head=np.array([0]),
        job_apps_targets=np.array([[0, -1]], dtype=np.intp),
    )

    head_before = emp.recv_job_apps_head[0]
    _workers_send_one_round_simultaneous(wrk, emp, rng=make_rng(42))

    # Queue should still be full (no additional apps added beyond capacity)
    assert emp.recv_job_apps_head[0] == head_before


def test_workers_send_one_round_simultaneous_exhausted_list() -> None:
    """Test simultaneous matching handles worker exhausting their application list."""
    from bamengine.events._internal.labor_market import (
        _workers_send_one_round_simultaneous,
    )

    M = 2  # stride = 2 slots per worker
    emp = mock_employer(
        n=1,
        queue_m=M,
        wage_offer=np.array([1.0]),
        n_vacancies=np.array([5]),
    )
    wrk = mock_worker(
        n=1,
        queue_m=M,
        employer=np.array([-1], dtype=np.intp),  # Unemployed
        # head = 2 means we're at position 2, which is (0+1)*2 = 2
        # This triggers head >= (j + 1) * stride
        job_apps_head=np.array([2]),
        job_apps_targets=np.array([[0, 0]], dtype=np.intp),
    )

    _workers_send_one_round_simultaneous(wrk, emp, rng=make_rng(42))

    # Worker should be marked as exhausted (head = -1)
    assert wrk.job_apps_head[0] == -1
    # No application should have been queued
    assert emp.recv_job_apps_head[0] == -1
