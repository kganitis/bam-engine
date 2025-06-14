# tests/unit/systems/test_planning.py
"""
Planning-system unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.random import default_rng

from bamengine.systems.planning import (
    _EPS,
    firms_decide_desired_labor,
    firms_decide_desired_production,
    firms_decide_vacancies,
)
from bamengine.typing import FloatA
from tests.helpers.factories import mock_employer, mock_producer


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
    rng = default_rng(5)
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
    rng = default_rng(1)
    prod = mock_producer(
        n=5,
        inventory=np.array([0.0, 0.0, 5.0, 0.0, 5.0]),
        price=np.array([2.0, 1.5, 1.0, 1.0, 2.0]),
    )

    firms_decide_desired_production(prod, p_avg=1.5, h_rho=0.1, rng=rng)

    # shapes must be preserved
    assert prod.expected_demand.shape == (5,)
    assert prod.desired_production.shape == (5,)

    # manual reference with same seed
    shocks = default_rng(1).uniform(0.0, 0.1, 5)
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
    """
    prod = mock_producer(production=np.array([5.0]))
    firms_decide_desired_production(prod, p_avg=1.5, h_rho=0.0, rng=default_rng(7))
    assert np.isclose(prod.desired_production[0], 5.0)


def test_reuses_internal_buffers() -> None:
    rng = default_rng(0)
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
    np.testing.assert_array_equal(prod.labor_productivity, [_EPS])


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
    a_eff = np.where(a_raw > _EPS, a_raw, _EPS)
    desired_labor_expected = np.ceil(yd_raw / a_eff)
    desired_labor_expected = np.clip(
        desired_labor_expected, 0, np.iinfo(np.int64).max
    ).astype(np.int64)
    vacancies_expected = np.maximum(desired_labor_expected - L_raw, 0)

    np.testing.assert_array_equal(emp.desired_labor, desired_labor_expected)
    np.testing.assert_array_equal(emp.n_vacancies, vacancies_expected)
