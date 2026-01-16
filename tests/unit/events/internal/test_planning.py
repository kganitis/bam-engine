"""
Planning-system unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from bamengine import make_rng
from bamengine.events._internal.planning import (
    EPS,
    firms_adjust_price,
    firms_calc_breakeven_price,
    firms_decide_desired_labor,
    firms_decide_desired_production,
    firms_decide_vacancies,
    firms_fire_excess_workers,
)
from bamengine.typing import FloatA
from tests.helpers.factories import mock_employer, mock_producer, mock_worker


@pytest.mark.parametrize(
    "price, inventory, p_avg, branch",
    [
        (2.0, 0.0, 1.5, "up"),  # sold-out, competitive price
        (1.0, 0.0, 1.5, "none"),  # sold-out, under-priced
        (2.0, 5.0, 1.5, "none"),  # leftovers, expensive
        (1.0, 5.0, 1.5, "down"),  # leftovers, cheap
    ],
    ids=["up", "sold_low_price", "stock_expensive", "down"],
)
def test_production_branches(
    price: FloatA, inventory: FloatA, p_avg: float, branch: str
) -> None:
    """
    Check that each branch of the BAM rule moves Yd in the correct direction.
    """
    rng = make_rng(5)
    prod = mock_producer(inventory=np.array([inventory]), price=np.array([price]))
    firms_decide_desired_production(prod, p_avg=p_avg, h_rho=0.1, rng=rng)
    yd = prod.desired_production[0]

    if branch == "up":
        assert yd > 10.0
    elif branch == "down":
        assert yd < 10.0
    else:  # "none"
        assert np.isclose(yd, 10.0)


def test_decide_desired_production_vector() -> None:
    """
    Compare full-vector output against a hand-calculated expectation
    (identical RNG seed).
    """
    rng = make_rng(1)
    prod = mock_producer(
        n=5,
        inventory=np.array([0.0, 0.0, 5.0, 0.0, 5.0]),
        price=np.array([2.0, 1.5, 1.0, 1.0, 2.0]),
    )

    firms_decide_desired_production(prod, p_avg=1.5, h_rho=0.1, rng=rng)

    # shapes must be preserved
    assert prod.expected_demand.shape == (5,)
    assert prod.desired_production.shape == (5,)

    # _tests reference with same seed
    shocks = make_rng(1).uniform(0.0, 0.1, 5)
    expected = np.array(
        [
            10.0 * (1 + shocks[0]),  # cond_up
            10.0 * (1 + shocks[1]),  # cond_up
            10.0 * (1 - shocks[2]),  # cond_down
            10.0,  # unchanged
            10.0,  # unchanged
        ]
    )
    np.testing.assert_allclose(prod.desired_production, expected, rtol=1e-6)
    # guard against negative output
    assert (prod.desired_production >= 0).all()


def test_shock_off_no_change() -> None:
    """
    With h_rho = 0 the rule must leave Yd unchanged, regardless of conditions.
    Uses production_prev as the baseline for expected demand.
    """
    prod = mock_producer(production_prev=np.array([5.0]))
    firms_decide_desired_production(prod, p_avg=1.5, h_rho=0.0, rng=make_rng(7))
    assert np.isclose(prod.desired_production[0], 5.0)


def test_planning_zeros_production_at_start() -> None:
    """
    Planning must zero out production at the start.
    production_prev retains previous period's value for use as planning signal.
    """
    prod = mock_producer(
        n=3,
        production=np.array([5.0, 10.0, 15.0]),  # Will be zeroed
        production_prev=np.array([5.0, 10.0, 15.0]),  # Retained for planning
    )
    firms_decide_desired_production(prod, p_avg=1.5, h_rho=0.0, rng=make_rng(0))

    # production should be zeroed
    np.testing.assert_array_equal(prod.production, [0.0, 0.0, 0.0])
    # production_prev unchanged - used as baseline for expected_demand
    np.testing.assert_array_equal(prod.production_prev, [5.0, 10.0, 15.0])


def test_reuses_internal_buffers() -> None:
    rng = make_rng(0)
    prod = mock_producer(n=2, alloc_scratch=False)

    # first call allocates the scratch arrays
    firms_decide_desired_production(prod, p_avg=1.5, h_rho=0.05, rng=rng)

    # capture the objects *after* they’ve been created
    buf0 = prod.prod_shock
    mask_up0 = prod.prod_mask_up
    mask_dn0 = prod.prod_mask_dn

    # second call must reuse the very same objects
    firms_decide_desired_production(prod, p_avg=1.5, h_rho=0.05, rng=rng)

    # same *object*
    assert buf0 is prod.prod_shock
    assert mask_up0 is prod.prod_mask_up
    assert mask_dn0 is prod.prod_mask_dn

    # scratch buffer is still write-able
    assert buf0 is not None
    assert buf0.flags.writeable


def test_decide_desired_labor_vector() -> None:
    """Labor demand must equal ceil(Yd / a) element-wise."""
    prod = mock_producer(
        n=5,
        desired_production=np.full(5, 10.0),
        labor_productivity=np.array([1.0, 0.8, 1.2, 0.5, 2.0]),
    )
    emp = mock_employer(n=5)
    firms_decide_desired_labor(prod, emp)
    expected = np.array([10, 13, 9, 20, 5])
    np.testing.assert_array_equal(emp.desired_labor, expected)


def test_zero_productivity_guard() -> None:
    """
    Productivity a ≤ 0 is invalid – the rule must fail fast to avoid NaNs.
    """
    prod = mock_producer(labor_productivity=np.array([0.0]))
    emp = mock_employer()
    firms_decide_desired_labor(prod, emp)
    np.testing.assert_array_equal(prod.labor_productivity, [EPS])


def test_decide_vacancies_vector() -> None:
    emp = mock_employer(
        n=4,
        desired_labor=np.array([10, 5, 3, 1], dtype=np.int64),
        current_labor=np.array([7, 5, 4, 0], dtype=np.int64),
    )
    firms_decide_vacancies(emp)
    np.testing.assert_array_equal(emp.n_vacancies, [3, 0, 0, 1])


@settings(max_examples=300, deadline=None)
@given(
    st.integers(min_value=1, max_value=40).flatmap(  # random N
        lambda n: st.tuples(
            # desired production  Yd  > 0
            st.lists(st.floats(0.01, 500.0), min_size=n, max_size=n),
            # labor productivity a  (can be ≤ 0 to hit the CAP branch)
            st.lists(
                st.floats(-5.0, 20.0).filter(  # allow a ≤ 0 so we exercise the guard
                    lambda x: not np.isnan(x) and not np.isinf(x)
                ),
                min_size=n,
                max_size=n,
            ),
            # current labor      L ≥ 0
            st.lists(st.integers(0, 800), min_size=n, max_size=n),
        )
    )
)
def test_labor_and_vacancy_properties(data) -> None:  # type: ignore[no-untyped-def]
    """
    Invariant (vector form) that **must** hold after running the two rules:
        V == max( ceil(Yd / â) − L , 0 )
    where
        â = a               if  a > 0
        CAP_LAB_PROD        otherwise
    """
    # build random, *typed* numpy vectors
    yd_raw, a_raw, L_raw = map(np.asarray, data)
    assume((yd_raw > 0).all())  # guard against rounding to zero
    prod = mock_producer(
        n=yd_raw.size,
        desired_production=yd_raw.astype(np.float64),
        labor_productivity=a_raw.astype(np.float64),
    )
    emp = mock_employer(
        n=yd_raw.size,
        desired_labor=np.zeros_like(yd_raw, dtype=np.int64),
        current_labor=L_raw.astype(np.int64),
    )

    firms_decide_desired_labor(prod, emp)
    firms_decide_vacancies(emp)

    # reproduce the algorithm in pure NumPy
    a_eff = np.where(a_raw > EPS, a_raw, EPS)
    desired_labor_expected = np.ceil(yd_raw / a_eff)
    desired_labor_expected = np.clip(
        desired_labor_expected, 0, np.iinfo(np.int64).max
    ).astype(np.int64)
    vacancies_expected = np.maximum(desired_labor_expected - L_raw, 0)

    np.testing.assert_array_equal(emp.desired_labor, desired_labor_expected)
    np.testing.assert_array_equal(emp.n_vacancies, vacancies_expected)


def test_breakeven_no_cap_equals_raw() -> None:
    """
    If no cap_factor is provided (or cap_factor <= 1), the breakeven floor
    equals raw (wage_bill + interest) / max(projected_production, _EPS).
    projected_production = labor_productivity * current_labor.
    """
    n = 3
    prod = mock_producer(n=n)
    prod.price[:] = 1.0  # irrelevant when no cap
    # labor_productivity defaults to 1.0, so projected_production = current_labor

    emp = mock_employer(n=n)
    emp.wage_bill[:] = np.array([10.0, 5.0, 8.0])
    emp.current_labor[:] = np.array([10, 5, 2], dtype=np.int64)  # projected_production

    class DummyLoanBook:
        @staticmethod
        def interest_per_borrower(nfirms: int) -> np.ndarray:
            assert nfirms == n
            return np.array([1.0, 0.0, 1.0])

    lb = DummyLoanBook()

    # noinspection PyTypeChecker
    firms_calc_breakeven_price(prod, emp, lb, cap_factor=None)

    # projected_production = labor_productivity * current_labor = 1.0 * [10, 5, 2]
    projected_production = prod.labor_productivity * emp.current_labor
    expected = (emp.wage_bill + np.array([1.0, 0.0, 1.0])) / np.maximum(
        projected_production, EPS
    )
    np.testing.assert_allclose(prod.breakeven_price, expected, rtol=1e-12, atol=0.0)


def test_breakeven_capped_by_price_times_factor() -> None:
    """
    With a cap_factor > 1, breakeven is min(raw_breakeven, price * cap_factor).
    projected_production = labor_productivity * current_labor.
    """
    n = 3
    prod = mock_producer(n=n)
    prod.price[:] = np.array([2.0, 1.0, 0.5])  # caps: [4.0, 2.0, 1.0]
    # labor_productivity defaults to 1.0, so projected_production = current_labor

    emp = mock_employer(n=n)
    emp.wage_bill[:] = np.array([100.0, 1.0, 1.0])  # raw: [100, 1, 1] when proj_prod=1
    emp.current_labor[:] = np.array(
        [1, 1, 1], dtype=np.int64
    )  # projected_production = 1

    class DummyLoanBook:
        @staticmethod
        def interest_per_borrower(nfirms: int) -> np.ndarray:
            return np.zeros(nfirms)

    lb = DummyLoanBook()

    # noinspection PyTypeChecker
    firms_calc_breakeven_price(prod, emp, lb, cap_factor=2)

    # projected_production = labor_productivity * current_labor = 1.0 * [1, 1, 1]
    projected_production = prod.labor_productivity * emp.current_labor
    raw = (emp.wage_bill + 0.0) / np.maximum(projected_production, EPS)
    cap = prod.price * 2
    expected = np.minimum(raw, cap)

    np.testing.assert_allclose(prod.breakeven_price, expected, rtol=1e-12, atol=0.0)


def test_price_adjust_raise_branch() -> None:
    """
    Inventory == 0 and price < p_avg → raise by (1 + shock) and respect breakeven floor.
    """
    rng = make_rng(0)
    prod = mock_producer(n=1)
    prod.inventory[:] = 0.0
    prod.price[:] = 1.0
    prod.breakeven_price[:] = 0.1  # low floor so it doesn't bind

    old = prod.price.copy()
    firms_adjust_price(prod, p_avg=2.0, h_eta=0.1, rng=rng)

    # replicate the single shock deterministically
    shock = make_rng(0).uniform(0.0, 0.1, 1)[0]
    expected = max(float(old[0] * (1.0 + shock)), float(prod.breakeven_price[0]))

    assert prod.price[0] == pytest.approx(expected)
    assert prod.price[0] >= old[0]


def test_price_adjust_cut_branch_with_floor_increase() -> None:
    """
    Inventory > 0 and price >= p_avg → cut by (1 - shock) but *not below* breakeven.
    If breakeven > old price, the price should increase to the floor (warning case).
    """
    rng = make_rng(1)
    prod = mock_producer(n=2)

    # both firms in "cut" set (inventory>0 & price >= p_avg)
    prod.inventory[:] = np.array([5.0, 5.0])
    prod.price[:] = np.array([2.0, 1.0])
    prod.breakeven_price[:] = np.array([1.5, 1.2])  # second > old price

    old = prod.price.copy()
    firms_adjust_price(prod, p_avg=1.0, h_eta=0.1, rng=rng)

    # first: normal cut (floor 1.5 below plausible cut target)
    assert prod.price[0] <= old[0]

    # second: floor-induced increase (breakeven above old)
    assert prod.price[1] == pytest.approx(1.2)
    assert prod.price[1] > old[1]


def test_price_adjust_noop_when_masks_empty() -> None:
    """
    If a firm is neither in mask_up nor mask_dn, its price must remain unchanged.
    Cases:
      - inventory == 0 and price >= p_avg  (not mask_up since price !< p_avg)
      - inventory > 0 and price < p_avg   (not mask_dn since price !>= p_avg)
    """
    rng = make_rng(3)
    prod = mock_producer(n=2)
    prod.inventory[:] = np.array([0.0, 10.0])
    prod.price[:] = np.array([2.0, 0.5])
    prod.breakeven_price[:] = np.array([0.1, 0.4])

    old = prod.price.copy()
    firms_adjust_price(prod, p_avg=2.0, h_eta=0.1, rng=rng)

    np.testing.assert_array_equal(prod.price, old)


# ============================================================================
# FirmsFireExcessWorkers Tests
# ============================================================================


def test_fire_excess_workers_fires_correct_number() -> None:
    """Firms with excess workers fire until current_labor == desired_labor."""
    n_firms = 3
    n_workers = 10

    desired_labor = np.array([2, 3, 1], dtype=np.int64)
    current_labor = np.array([4, 3, 3], dtype=np.int64)  # excess: [2, 0, 2]

    emp = mock_employer(
        n=n_firms,
        desired_labor=desired_labor,
        current_labor=current_labor,
    )
    wrk = mock_worker(n=n_workers)
    # Assign workers to firms: workers 0-3 to firm 0, 4-6 to firm 1, 7-9 to firm 2
    wrk.employer[:] = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.intp)
    wrk.wage[:] = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])

    # employed is a computed property based on employer >= 0, no need to set it

    firms_fire_excess_workers(emp, wrk, method="random", rng=make_rng(42))

    # Check that current_labor now equals desired_labor
    np.testing.assert_array_equal(emp.current_labor, emp.desired_labor)

    # Verify total fired is 4 (2 from firm 0, 0 from firm 1, 2 from firm 2)
    assert wrk.fired.sum() == 4


def test_fire_excess_workers_expensive_first() -> None:
    """With method='expensive', highest-wage workers are fired first."""
    n_firms = 1
    n_workers = 4

    desired_labor = np.array([2], dtype=np.int64)
    current_labor = np.array([4], dtype=np.int64)  # excess: 2

    emp = mock_employer(
        n=n_firms,
        desired_labor=desired_labor,
        current_labor=current_labor,
    )
    wrk = mock_worker(n=n_workers)
    wrk.employer[:] = np.array([0, 0, 0, 0], dtype=np.intp)
    wrk.wage[:] = np.array([1.0, 4.0, 2.0, 3.0])  # worker 1 and 3 are most expensive

    # employed is a computed property based on employer >= 0, no need to set it

    firms_fire_excess_workers(emp, wrk, method="expensive", rng=make_rng(42))

    # Workers 1 (wage=4.0) and 3 (wage=3.0) should be fired
    assert wrk.fired[1] == 1  # highest wage
    assert wrk.fired[3] == 1  # second highest wage
    assert wrk.fired[0] == 0  # lowest wage - kept
    assert wrk.fired[2] == 0  # second lowest - kept


def test_fire_excess_workers_no_excess() -> None:
    """Firms with current_labor <= desired_labor don't fire anyone."""
    n_firms = 2
    n_workers = 6

    desired_labor = np.array([3, 4], dtype=np.int64)
    current_labor = np.array([2, 4], dtype=np.int64)  # no excess

    emp = mock_employer(
        n=n_firms,
        desired_labor=desired_labor,
        current_labor=current_labor,
    )
    wrk = mock_worker(n=n_workers)
    wrk.employer[:] = np.array([0, 0, 1, 1, 1, 1], dtype=np.intp)
    wrk.wage[:] = np.ones(n_workers)

    # employed is a computed property based on employer >= 0, no need to set it

    firms_fire_excess_workers(emp, wrk, method="random", rng=make_rng(42))

    # No workers should be fired
    assert wrk.fired.sum() == 0
    np.testing.assert_array_equal(emp.current_labor, [2, 4])


def test_fire_excess_workers_updates_worker_state() -> None:
    """Fired workers have correct state: employer=-1, fired=1, wage=0, periods_left=0."""
    n_firms = 1
    n_workers = 3

    desired_labor = np.array([1], dtype=np.int64)
    current_labor = np.array([3], dtype=np.int64)  # excess: 2

    emp = mock_employer(
        n=n_firms,
        desired_labor=desired_labor,
        current_labor=current_labor,
    )
    wrk = mock_worker(n=n_workers)
    wrk.employer[:] = np.array([0, 0, 0], dtype=np.intp)
    wrk.wage[:] = np.array([1.0, 2.0, 3.0])
    wrk.periods_left[:] = np.array([5, 10, 15], dtype=np.int64)

    # employed is a computed property based on employer >= 0, no need to set it

    firms_fire_excess_workers(emp, wrk, method="expensive", rng=make_rng(42))

    # Workers 1 and 2 (highest wages) should be fired
    fired_mask = wrk.fired == 1

    # Verify all fired workers have correct state
    assert (wrk.employer[fired_mask] == -1).all()
    assert (wrk.wage[fired_mask] == 0.0).all()
    assert (wrk.periods_left[fired_mask] == 0).all()

    # Remaining worker (worker 0) should be unchanged
    assert wrk.employer[0] == 0
    assert wrk.wage[0] == 1.0
    assert wrk.periods_left[0] == 5


def test_fire_excess_workers_updates_firm_state() -> None:
    """Firm's current_labor is decremented correctly after firing."""
    n_firms = 2
    n_workers = 7

    desired_labor = np.array([1, 2], dtype=np.int64)
    current_labor = np.array([3, 4], dtype=np.int64)  # excess: [2, 2]

    emp = mock_employer(
        n=n_firms,
        desired_labor=desired_labor,
        current_labor=current_labor,
    )
    wrk = mock_worker(n=n_workers)
    wrk.employer[:] = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.intp)
    wrk.wage[:] = np.ones(n_workers)

    # employed is a computed property based on employer >= 0, no need to set it

    firms_fire_excess_workers(emp, wrk, method="random", rng=make_rng(42))

    # Check current_labor matches desired_labor after firing
    np.testing.assert_array_equal(emp.current_labor, [1, 2])


def test_fire_excess_workers_sets_employer_prev() -> None:
    """Fired workers have employer_prev set to their former employer."""
    n_firms = 1
    n_workers = 3

    desired_labor = np.array([1], dtype=np.int64)
    current_labor = np.array([3], dtype=np.int64)

    emp = mock_employer(
        n=n_firms,
        desired_labor=desired_labor,
        current_labor=current_labor,
    )
    wrk = mock_worker(n=n_workers)
    wrk.employer[:] = np.array([0, 0, 0], dtype=np.intp)
    wrk.employer_prev[:] = np.array([-1, -1, -1], dtype=np.intp)
    wrk.wage[:] = np.array([1.0, 2.0, 3.0])

    # employed is a computed property based on employer >= 0, no need to set it

    firms_fire_excess_workers(emp, wrk, method="expensive", rng=make_rng(42))

    # Fired workers should have employer_prev = 0 (their former employer)
    fired_mask = wrk.fired == 1
    assert (wrk.employer_prev[fired_mask] == 0).all()
