"""
Build a synthetic economy in an arbitrary mid-simulation state, run ONE period,
and verify that high-level invariants still hold.

•  Works even though only Event 1 is implemented.
•  Uses a fixed RNG seed so the test is fully reproducible.
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng

from bamengine.components.firm_plan import (
    FirmProductionPlan,
    FirmLaborPlan,
    FirmVacancies,
)
from bamengine.events.firms_planning import firms_planning


N_FIRMS = 17  # prime → catches shape mix-ups
SEED = 5678
H_RHO = 0.1
P_AVG = 1.4


def _make_random_state(n: int, seed: int = 0):
    """Return prod, lab, vac filled with *plausible* random data."""
    rng = default_rng(seed)

    # --- core arrays -------------------------------------------------------
    price = rng.uniform(1.0, 2.0, n)
    inventory = rng.integers(0, 8, n).astype(float)
    prev_prod = rng.uniform(5.0, 15.0, n)
    expected_demand = rng.uniform(5.0, 15.0, n)
    desired_production = rng.uniform(5.0, 15.0, n)

    labour_productivity = rng.uniform(0.5, 1.5, n)
    desired_labor = np.ceil(desired_production / labour_productivity).astype(np.int64)
    current_labor = rng.integers(0, desired_labor + 3, n, dtype=np.int64)
    n_vacancies = np.maximum(desired_labor - current_labor, 0, dtype=np.int64)

    # --- wrap into components ---------------------------------------------
    prod = FirmProductionPlan(
        price=price,
        inventory=inventory,
        prev_production=prev_prod,
        expected_demand=expected_demand,
        desired_production=desired_production,
    )
    lab = FirmLaborPlan(
        desired_production=desired_production,  # share!
        labor_productivity=labour_productivity,
        desired_labor=desired_labor,
    )
    vac = FirmVacancies(
        desired_labor=desired_labor,  # share!
        current_labor=current_labor,
        n_vacancies=n_vacancies,
    )
    return prod, lab, vac, rng


# ---------------------------------------------------------------------------


def test_random_state_one_step() -> None:
    prod, lab, vac, rng = _make_random_state(N_FIRMS, SEED)

    firms_planning(prod, lab, vac, p_avg=P_AVG, h_rho=H_RHO, rng=rng)

    # ---------- invariants -------------------------------------------------
    # shapes / dtypes
    assert prod.desired_production.shape == (N_FIRMS,)
    assert lab.desired_labor.dtype == np.int64
    assert vac.n_vacancies.dtype == np.int64

    assert (prod.expected_demand > 0).all()
    # expected_demand updated in-place and matches desired_production
    np.testing.assert_array_equal(prod.expected_demand, prod.desired_production)

    assert (lab.desired_labor > 0).all()

    # vacancies respect bounds: 0 ≤ V_i ≤ Ld_i
    assert (vac.n_vacancies >= 0).all()
    assert (vac.n_vacancies <= lab.desired_labor).all()

    # at least one firm has a positive vacancy (with high probability)
    assert (vac.n_vacancies > 0).any()
