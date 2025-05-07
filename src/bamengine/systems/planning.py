from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator

from bamengine.components.firm_plan import (
    FirmLaborPlan,
    FirmProductionPlan,
    FirmVacancies,
)

log = logging.getLogger(__name__)


def firms_decide_desired_production(  # noqa: C901  (still quite short)
    fp: FirmProductionPlan,
    *,
    p_avg: float,
    h_rho: float,
    rng: Generator,
) -> None:
    """
    Update `prod.expected_demand` and `prod.desired_production` **in‑place**.

    Rule
    ----
      if S_i == 0 and P_i ≥ p̄   → raise   by (1 + shock)
      if S_i  > 0 and P_i < p̄   → cut     by (1 − shock)
      otherwise                 → keep previous level
    """
    shape = fp.price.shape

    # ── 1. permanent scratches ---------------
    shock = fp.prod_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64)

    up_mask = fp.prod_mask_up
    if up_mask is None or up_mask.shape != shape:
        up_mask = np.empty(shape, dtype=np.bool_)

    dn_mask = fp.prod_mask_dn
    if dn_mask is None or dn_mask.shape != shape:
        dn_mask = np.empty(shape, dtype=np.bool_)

    # ── 2. fill buffers in‑place ---------------
    shock[:] = rng.uniform(0.0, h_rho, size=shape)
    np.logical_and(fp.inventory == 0.0, fp.price >= p_avg, out=up_mask)
    np.logical_and(fp.inventory > 0.0, fp.price < p_avg, out=dn_mask)

    # ── 3. core rule ----------------------------------
    fp.expected_demand[:] = fp.prev_production
    fp.expected_demand[up_mask] *= 1.0 + shock[up_mask]
    fp.expected_demand[dn_mask] *= 1.0 - shock[dn_mask]
    fp.desired_production[:] = fp.expected_demand

    # ── 3. logging ------------------------------------------------------------
    log.debug(
        "decide_desired_production: n=%d  p̄=%.3f  up=%d  down=%d  mean_shock=%.4f",
        fp.price.size,
        p_avg,
        int(up_mask.sum()),
        int(dn_mask.sum()),
        float(shock.mean()),
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
