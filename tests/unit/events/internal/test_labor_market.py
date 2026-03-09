"""
Labor-market events internal implementation unit tests.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from bamengine import Rng, make_rng
from bamengine.events._internal.labor_market import (
    adjust_minimum_wage,
    calc_inflation_rate,
    firms_decide_wage_offer,
    workers_decide_firms_to_apply,
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


def test_calc_inflation_rate_yoy_min_history() -> None:
    """YoY method requires 5 periods of history, returns 0 if less."""
    ec = mock_economy(avg_mkt_price_history=np.array([1.0, 1.1, 1.2, 1.3]))
    calc_inflation_rate(ec)
    assert ec.inflation_history[-1] == 0.0


def test_calc_inflation_rate_yoy_prev_nonpositive() -> None:
    """YoY method returns 0 if previous price is non-positive."""
    # t = 4 (len=5); p_{t-4} is index 0 -> set to 0.0 to trigger branch
    ec = mock_economy(avg_mkt_price_history=np.array([0.0, 1.0, 1.1, 1.2, 1.3]))
    calc_inflation_rate(ec)
    assert ec.inflation_history[-1] == 0.0


@pytest.mark.parametrize(
    ("prices", "direction"),
    [
        (np.array([1.00, 1.05, 1.10, 1.15, 1.20]), "up"),  # inflation → increases
        (np.array([1.00, 0.95, 0.90, 0.85, 0.80]), "down"),  # deflation → decreases
        (np.array([1.00, 1.10, 1.20, 1.30]), "flat"),  # = m  → no revision
        (np.array([1.00, 1.05, 1.10, 1.15, 1.20, 1.30]), "flat"),  # < m  → no revision
    ],
)
def test_adjust_minimum_wage_edges(prices: NDArray[np.float64], direction: str) -> None:
    """Guard array bounds; min wage moves with inflation or stays flat."""
    ec = mock_economy(
        min_wage=2.0,
        avg_mkt_price_history=prices,
        min_wage_rev_period=4,
    )
    wrk = mock_worker(
        n=2, employer=np.array([0, -1], dtype=np.intp), wage=np.array([1.5, 0.0])
    )
    old = ec.min_wage
    calc_inflation_rate(ec)
    adjust_minimum_wage(ec, wrk)
    if direction == "up":
        assert ec.min_wage > old
    elif direction == "down":
        assert ec.min_wage < old
    else:  # flat (no revision triggered)
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
    calc_inflation_rate(ec)
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


def test_firms_calc_wage_bill_basic() -> None:
    from bamengine.events._internal.labor_market import firms_calc_wage_bill

    emp = mock_employer(n=3)
    wrk = mock_worker(n=5)
    wrk.employed[:] = np.array([1, 1, 0, 1, 1], dtype=np.bool_)
    wrk.employer[:] = np.array([0, 0, -1, 2, 2], dtype=np.intp)
    wrk.wage[:] = np.array([1.0, 1.5, 9.9, 2.0, 3.0])
    firms_calc_wage_bill(emp, wrk)
    np.testing.assert_allclose(emp.wage_bill, np.array([2.5, 0.0, 5.0]))
