# src/tests/unit/systems/test_planning.py
"""
Planning-system unit tests

These tests verify that the three pure functions in
`bamengine.systems.planning` behave correctly for all economically
relevant branches:
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.random import default_rng
from numpy.typing import NDArray

from bamengine.components import Producer, Employer
from bamengine.systems.planning import (
    firms_decide_desired_labor,
    firms_decide_desired_production,
    firms_decide_vacancies,
    CAP_LAB_PROD,
)
from bamengine.typing import FloatA, IntA


# --------------------------------------------------------------------------- #
# 1. desired production – explicit branch table                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "price, inventory, p_avg, branch",
    [
        (2.0, 0.0, 1.5, "up"),  # A: sold-out, competitive price
        (1.0, 0.0, 1.5, "none"),  # B: sold-out, under-priced
        (2.0, 5.0, 1.5, "none"),  # C: leftovers, expensive
        (1.0, 5.0, 1.5, "down"),  # D: leftovers, cheap
    ],
)
def test_production_branches(
    price: FloatA, inventory: FloatA, p_avg: float, branch: str
) -> None:
    """
    Check that each branch of the BAM rule moves Yd in the correct direction.
    """
    rng = default_rng(5)
    prod = Producer(
        production=np.array([10.0]),
        inventory=np.array([inventory]),
        expected_demand=np.zeros(1),
        desired_production=np.zeros(1),
        labor_productivity=np.ones(1),
        price=np.array([price]),
    )
    firms_decide_desired_production(prod, p_avg=p_avg, h_rho=0.1, rng=rng)
    yd = prod.desired_production[0]

    if branch == "up":
        assert yd > 10.0
    elif branch == "down":
        assert yd < 10.0
    else:  # "none"
        assert np.isclose(yd, 10.0)


# --------------------------------------------------------------------------- #
# 2. desired production – deterministic vector test                           #
# --------------------------------------------------------------------------- #
def test_decide_desired_production_vector() -> None:
    """
    Compare full-vector output against a hand-calculated expectation
    (identical RNG seed).
    """
    rng = default_rng(1)
    prod = Producer(
        production=np.full(5, 10.0),
        inventory=np.array([0.0, 0.0, 5.0, 0.0, 5.0]),
        expected_demand=np.zeros(5),
        desired_production=np.zeros(5),
        labor_productivity=np.ones(5),
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
    With hᵨ = 0 the rule must leave Yd unchanged, regardless of conditions.
    """
    prod = Producer(
        production=np.array([8.0]),
        inventory=np.array([0.0]),
        expected_demand=np.zeros(1),
        desired_production=np.zeros(1),
        labor_productivity=np.ones(1),
        price=np.array([1.6]),
    )
    firms_decide_desired_production(prod, p_avg=1.5, h_rho=0.0, rng=default_rng(7))
    assert np.isclose(prod.desired_production[0], 8.0)


def test_reuses_internal_buffers() -> None:
    rng = default_rng(0)
    prod = Producer(
        production=np.ones(2),
        inventory=np.zeros(2),
        expected_demand=np.zeros(2),
        desired_production=np.zeros(2),
        labor_productivity=np.ones(2),
        price=np.array([2.0, 2.0]),
    )

    # --- first call allocates the scratch arrays ----------------------
    firms_decide_desired_production(prod, p_avg=1.5, h_rho=0.05, rng=rng)

    # capture the objects *after* they’ve been created
    buf0 = prod.prod_shock
    mask_up0 = prod.prod_mask_up
    mask_dn0 = prod.prod_mask_dn

    # --- second call must reuse the very same objects -----------------
    firms_decide_desired_production(prod, p_avg=1.5, h_rho=0.05, rng=rng)

    assert buf0 is prod.prod_shock
    assert mask_up0 is prod.prod_mask_up
    assert mask_dn0 is prod.prod_mask_dn


# --------------------------------------------------------------------------- #
# 3. desired labour – deterministic test & zero-productivity guard           #
# --------------------------------------------------------------------------- #
def test_decide_desired_labor_vector() -> None:
    """Labour demand must equal ceil(Yd / aᵢ) element-wise."""
    prod = Producer(
        production=np.ones(5),
        inventory=np.zeros(5),
        expected_demand=np.zeros(5),
        desired_production=np.full(5, 10.0),
        labor_productivity=np.array([1.0, 0.8, 1.2, 0.5, 2.0]),
        price=np.ones(5),
    )
    emp = Employer(
        desired_labor=np.zeros(5, dtype=np.int64),
        current_labor=np.ones(5, dtype=np.int64),
        wage_offer=np.ones(5),
        wage_bill=np.full(5, 5.0),
        n_vacancies=np.zeros(5, dtype=np.int64),
        total_funds=np.full(5, 5.0),
        recv_job_apps_head=np.full(5, -1, dtype=np.int64),
        recv_job_apps=np.full((5, 4), -1, dtype=np.int64),
    )
    firms_decide_desired_labor(prod, emp)
    expected = np.array([10, 13, 9, 20, 5])
    np.testing.assert_array_equal(emp.desired_labor, expected)


def test_zero_productivity_guard() -> None:
    """
    Productivity aᵢ ≤ 0 is invalid – the rule must fail fast to avoid NaNs.
    """
    prod = Producer(
        production=np.ones(1),
        inventory=np.zeros(1),
        expected_demand=np.zeros(1),
        desired_production=np.array([10.0]),
        labor_productivity=np.array([0.0]),
        price=np.ones(1),
    )
    emp = Employer(
        desired_labor=np.zeros(1, dtype=np.int64),
        current_labor=np.ones(1, dtype=np.int64),
        wage_offer=np.ones(1),
        wage_bill=np.full(1, 5.0),
        n_vacancies=np.zeros(1, dtype=np.int64),
        total_funds=np.full(1, 5.0),
        recv_job_apps_head=np.full(1, -1, dtype=np.int64),
        recv_job_apps=np.full((1, 4), -1, dtype=np.int64),
    )
    firms_decide_desired_labor(prod, emp)
    np.testing.assert_array_equal(prod.labor_productivity, [CAP_LAB_PROD])

# --------------------------------------------------------------------------- #
# 4. vacancies – simple deterministic check                                  #
# --------------------------------------------------------------------------- #
def test_decide_vacancies_vector() -> None:
    emp = Employer(
        desired_labor=np.array([10, 5, 3, 1], dtype=np.int64),
        current_labor=np.array([7, 5, 4, 0], dtype=np.int64),
        wage_offer=np.ones(4),
        wage_bill=np.full(4, 5.0),
        n_vacancies=np.zeros(4, dtype=np.int64),
        total_funds=np.full(4, 5.0),
        recv_job_apps_head=np.full(4, -1, dtype=np.int64),
        recv_job_apps=np.full((4, 4), -1, dtype=np.int64),
    )
    firms_decide_vacancies(emp)
    np.testing.assert_array_equal(emp.n_vacancies, [3, 0, 0, 1])


# --------------------------------------------------------------------------- #
# 5. Property-based check (random vectors)                                    #
# --------------------------------------------------------------------------- #
@given(
    st.integers(min_value=1, max_value=40).flatmap(          # random N
        lambda n: st.tuples(
            # desired production  Ydᵢ  > 0
            st.lists(st.floats(0.01, 500.0), min_size=n, max_size=n),
            # labour productivity aᵢ  (can be ≤ 0 to hit the CAP branch)
            st.lists(
                st.floats(-5.0, 20.0)  # allow aᵢ ≤ 0 so we exercise the guard
                .filter(lambda x: not np.isnan(x) and not np.isinf(x)),
                min_size=n,
                max_size=n,
            ),
            # current labour      Lᵢ ≥ 0
            st.lists(st.integers(0, 800), min_size=n, max_size=n),
        )
    )
)
def test_labor_and_vacancy_properties(data) -> None:  # type: ignore[no-untyped-def]
    """
    Invariant (vector form) that **must** hold after running the two rules:

        Vᵢ == max( ceil(Ydᵢ / âᵢ) − Lᵢ , 0 )

    where

        âᵢ = aᵢ               if  aᵢ > 0
             CAP_LAB_PROD      otherwise
    """
    # ------------------------------------------------------------------ #
    # 1.  build random, *typed* numpy vectors                             #
    # ------------------------------------------------------------------ #
    yd_raw, a_raw, L_raw = map(np.asarray, data)

    _f64 = lambda x: np.asarray(x, dtype=np.float64)
    _i64 = lambda x: np.asarray(x, dtype=np.int64)

    prod_cmp = _ProducerCMP(
        desired_production=_f64(yd_raw),
        labor_productivity=_f64(a_raw),
    )
    emp_cmp = _EmployerCMP(
        desired_labor=_i64(np.zeros_like(yd_raw)),
        current_labor=_i64(L_raw),
        n_vacancies=_i64(np.zeros_like(L_raw)),
    )

    # ------------------------------------------------------------------ #
    # 2.  run the two systems                                            #
    # ------------------------------------------------------------------ #
    firms_decide_desired_labor(prod_cmp, emp_cmp)
    firms_decide_vacancies(emp_cmp)

    # ------------------------------------------------------------------ #
    # 3.  reproduce the algorithm in pure NumPy                          #
    # ------------------------------------------------------------------ #
    a_eff = np.where(a_raw > 0.0, a_raw, CAP_LAB_PROD)
    desired_labor_expected = np.ceil(yd_raw / a_eff).astype(np.int64)
    vacancies_expected = np.maximum(desired_labor_expected - L_raw, 0)

    # ------------------------------------------------------------------ #
    # 4.  assertions                                                     #
    # ------------------------------------------------------------------ #
    np.testing.assert_array_equal(
        emp_cmp.desired_labor, desired_labor_expected, err_msg="Ld mismatch"
    )
    np.testing.assert_array_equal(
        emp_cmp.n_vacancies, vacancies_expected, err_msg="V mismatch"
    )
