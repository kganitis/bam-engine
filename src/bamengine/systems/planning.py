from __future__ import annotations

import logging
from typing import Optional, cast

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from bamengine.components.firm_plan import (
    FirmLaborPlan,
    FirmProductionPlan,
    FirmVacancies,
)

log = logging.getLogger(__name__)


FloatA = NDArray[np.float64]
BoolA = NDArray[np.bool_]


def firms_decide_desired_production(  # noqa: C901  (still quite short)
    prod: FirmProductionPlan,
    p_avg: float,
    h_rho: float,
    rng: Generator,
    *,
    shock_buf: Optional[FloatA] = None,
    up_mask: Optional[BoolA] = None,
    dn_mask: Optional[BoolA] = None,
) -> None:
    """
    Update `prod.expected_demand` and `prod.desired_production` **in‑place**.

    Rule
    ----
      if S_i == 0 and P_i ≥ p̄   → raise   by (1 + shock)
      if S_i  > 0 and P_i < p̄   → cut     by (1 − shock)
      otherwise                 → keep previous level

    Performance hooks
    -----------------
    `shock_buf`, `up_mask`, `dn_mask` let the caller pass pre‑allocated
    arrays that **must** have ``shape == prod.price.shape`` and matching
    dtypes:

    * ``shock_buf``  – float64 (same as `prod.price`)
    * ``up_mask``    – bool
    * ``dn_mask``    – bool

    When *None* (default) a fresh temporary array is created, preserving the
    old behaviour.

    Notes
    -----
    Arrays mutated **in‑place** (NumPy ``out=`` semantics):

    * ``prod.expected_demand   ← out``
    * ``prod.desired_production ← out``
    """
    n: int = prod.price.size
    shape = prod.price.shape

    # ── 1. re‑use or fall back to caller‑provided work buffers ---------------
    if shock_buf is None or shock_buf.shape != shape:
        shock_buf = np.empty(shape, dtype=np.float64)
    shock_buf[:] = rng.uniform(0.0, h_rho, size=shape)

    if up_mask is None or up_mask.shape != shape:
        up_mask = np.empty(shape, dtype=np.bool_)
    np.logical_and(prod.inventory == 0.0, prod.price >= p_avg, out=up_mask)

    if dn_mask is None or dn_mask.shape != shape:
        dn_mask = np.empty(shape, dtype=np.bool_)
    np.logical_and(prod.inventory > 0.0, prod.price < p_avg, out=dn_mask)

    # Tell mypy the optionals are now concrete ndarrays
    shock_buf = cast(FloatA, shock_buf)
    up_mask = cast(BoolA, up_mask)
    dn_mask = cast(BoolA, dn_mask)

    # ── 2. core computation ---------------------------------------------------
    prod.expected_demand[:] = prod.prev_production
    prod.expected_demand[up_mask] *= 1.0 + shock_buf[up_mask]
    prod.expected_demand[dn_mask] *= 1.0 - shock_buf[dn_mask]
    prod.desired_production[:] = prod.expected_demand

    # ── 3. logging ------------------------------------------------------------
    log.debug(
        "decide_desired_production: n=%d  p̄=%.3f  up=%d  down=%d  mean_shock=%.4f",
        n,
        p_avg,
        int(up_mask.sum()),
        int(dn_mask.sum()),
        float(shock_buf.mean()),
    )


def firms_decide_desired_labor(lab: FirmLaborPlan) -> None:
    """
    Desired labor demand (vectorised):

        Ld_i = ceil(Yd_i / a_i)
    """
    if (lab.labor_productivity <= 0).any():
        raise ValueError("labor_productivity must be > 0")
    ratio = lab.desired_production / lab.labor_productivity
    np.ceil(ratio, out=ratio)
    lab.desired_labor[:] = ratio.astype(np.int64)

    log.debug(
        "decide_desired_labor: n=%d  avg_prod=%.2f  avg_a=%.2f  "
        "avg_Ld=%.2f  max_Ld=%d",
        lab.desired_production.size,
        lab.desired_production.mean(),
        lab.labor_productivity.mean(),
        lab.desired_labor.mean(),
        int(lab.desired_labor.max()),
    )


def firms_decide_vacancies(vac: FirmVacancies) -> None:
    """
    Vector rule: V_i = max( Ld_i – L_i , 0 )
    """
    # in-place subtraction → tmp array not allocated
    np.subtract(
        vac.desired_labor, vac.current_labor, out=vac.n_vacancies, dtype=np.int64
    )
    np.maximum(vac.n_vacancies, 0, out=vac.n_vacancies)

    log.debug(
        "decide_vacancies: n=%d  mean_V=%.2f  openings=%d",
        vac.n_vacancies.size,
        vac.n_vacancies.mean(),
        int((vac.n_vacancies > 0).sum()),
    )
