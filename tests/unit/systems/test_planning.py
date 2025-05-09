"""
Planning-system unit tests

These tests verify that the three pure functions in
`bamengine.systems.planning` behave correctly for all economically
relevant branches:

* desired production (Event-1 rule)
* desired labour     (ceil(Yd / aᵢ))
* vacancies          (max(Ld − L, 0))

Notation used in comments:
  Sᵢ inventory, Pᵢ price, p̄ average price, Ydᵢ desired production,
  Ldᵢ desired labour, Vᵢ vacancies
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.random import default_rng
from numpy.typing import NDArray

from bamengine.components.firm_plan import (
    FirmLaborPlan,
    FirmProductionPlan,
    FirmVacancies,
)
from bamengine.systems.planning import (
    firms_decide_desired_labor as decide_desired_labor,
)
from bamengine.systems.planning import (
    firms_decide_desired_production as decide_desired_production,
)
from bamengine.systems.planning import (
    firms_decide_vacancies as decide_vacancies,
)

FloatA = NDArray[np.float64]
IntA = NDArray[np.int64]


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
    prod = FirmProductionPlan(
        price=np.array([price]),
        inventory=np.array([inventory]),
        prev_production=np.array([10.0]),
        expected_demand=np.zeros(1),
        desired_production=np.zeros(1),
    )
    decide_desired_production(prod, p_avg=p_avg, h_rho=0.1, rng=rng)
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
    prod = FirmProductionPlan(
        price=np.array([2.0, 1.5, 1.0, 1.0, 2.0]),
        inventory=np.array([0.0, 0.0, 5.0, 0.0, 5.0]),
        prev_production=np.full(5, 10.0),
        expected_demand=np.zeros(5),
        desired_production=np.zeros(5),
    )
    decide_desired_production(prod, p_avg=1.5, h_rho=0.1, rng=rng)

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
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(prod.desired_production, expected, rtol=1e-6)
    # guard against negative output
    assert (prod.desired_production >= 0).all()


def test_shock_off_no_change() -> None:
    """
    With hᵨ = 0 the rule must leave Yd unchanged, regardless of conditions.
    """
    prod = FirmProductionPlan(
        price=np.array([1.6]),
        inventory=np.array([0.0]),
        prev_production=np.array([8.0]),
        expected_demand=np.zeros(1),
        desired_production=np.zeros(1),
    )
    decide_desired_production(prod, p_avg=1.5, h_rho=0.0, rng=default_rng(7))
    assert np.isclose(prod.desired_production[0], 8.0)


def test_reuses_internal_buffers() -> None:
    rng = default_rng(0)
    prod = FirmProductionPlan(
        price=np.array([2.0, 2.0]),
        inventory=np.zeros(2),
        prev_production=np.ones(2),
        expected_demand=np.zeros(2),
        desired_production=np.zeros(2),
    )

    # --- first call allocates the scratch arrays ----------------------
    decide_desired_production(prod, p_avg=1.5, h_rho=0.05, rng=rng)

    # capture the objects *after* they’ve been created
    buf0 = prod.prod_shock
    mask_up0 = prod.prod_mask_up
    mask_dn0 = prod.prod_mask_dn

    # --- second call must reuse the very same objects -----------------
    decide_desired_production(prod, p_avg=1.5, h_rho=0.05, rng=rng)

    assert buf0 is prod.prod_shock
    assert mask_up0 is prod.prod_mask_up
    assert mask_dn0 is prod.prod_mask_dn


# --------------------------------------------------------------------------- #
# 3. desired labour – deterministic test & zero-productivity guard           #
# --------------------------------------------------------------------------- #
def test_decide_desired_labor_vector() -> None:
    """Labour demand must equal ceil(Yd / aᵢ) element-wise."""
    lab = FirmLaborPlan(
        desired_production=np.full(5, 10.0),
        labor_productivity=np.array([1.0, 0.8, 1.2, 0.5, 2.0]),
        desired_labor=np.zeros(5, dtype=np.int64),
    )
    decide_desired_labor(lab)
    expected = np.array([10, 13, 9, 20, 5])
    np.testing.assert_array_equal(lab.desired_labor, expected)


def test_zero_productivity_guard() -> None:
    """
    Productivity aᵢ ≤ 0 is invalid – the rule must fail fast to avoid NaNs.
    """
    lab = FirmLaborPlan(
        desired_production=np.array([10.0]),
        labor_productivity=np.array([0.0]),
        desired_labor=np.zeros(1, dtype=np.int64),
    )
    with pytest.raises(ValueError):
        decide_desired_labor(lab)


# --------------------------------------------------------------------------- #
# 4. vacancies – simple deterministic check                                  #
# --------------------------------------------------------------------------- #
def test_decide_vacancies_vector() -> None:
    vac = FirmVacancies(
        desired_labor=np.array([10, 5, 3, 1]),
        current_labor=np.array([7, 5, 4, 0]),
        n_vacancies=np.zeros(4, dtype=np.int64),
    )
    decide_vacancies(vac)
    np.testing.assert_array_equal(vac.n_vacancies, [3, 0, 0, 1])


# --------------------------------------------------------------------------- #
# 5. Property-based checks (random vectors)                                   #
# --------------------------------------------------------------------------- #
@given(
    st.integers(min_value=1, max_value=30).flatmap(
        lambda n: st.tuples(
            # production > 0
            st.lists(st.floats(0.01, 200.0), min_size=n, max_size=n),
            # productivity > 0
            st.lists(st.floats(0.01, 10.0), min_size=n, max_size=n),
            # current labour ≥ 0
            st.lists(st.integers(0, 400), min_size=n, max_size=n),
        )
    )
)
def test_labor_and_vacancy_properties(data) -> None:  # type: ignore[no-untyped-def]
    """
    Property-level invariant:

      Vᵢ = max( ceil(Ydᵢ / aᵢ) − Lᵢ , 0 )   for all i
    """
    desired, prod, current = map(np.asarray, data)
    lab = FirmLaborPlan(
        desired_production=desired.astype(np.float64),
        labor_productivity=prod.astype(np.float64),
        desired_labor=np.zeros_like(desired, dtype=np.int64),
    )
    decide_desired_labor(lab)

    vac = FirmVacancies(
        desired_labor=lab.desired_labor,
        current_labor=current.astype(np.int64),
        n_vacancies=np.zeros_like(current, dtype=np.int64),
    )
    decide_vacancies(vac)

    expected = np.maximum(lab.desired_labor - current, 0)
    assert (vac.n_vacancies == expected).all()
