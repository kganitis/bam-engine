import logging

import numpy as np
from numpy.random import Generator

from bamengine.components.firm_plan import (
    FirmLaborPlan,
    FirmProductionPlan,
    FirmVacancies,
)

log = logging.getLogger(__name__)


def firms_decide_desired_production(
    prod: FirmProductionPlan,
    p_avg: float,
    h_rho: float,
    rng: Generator,
) -> None:
    """
    Update expected and desired production for every firm (vectorised).

        if S_i == 0 and P_i >= p̄ → raise by (1+shock)
        if S_i > 0 and P_i < p̄ → cut   by (1−shock)
        otherwise                → keep previous level
    """
    # ── draw one idiosyncratic shock per firm ────────────────────────────────
    shock = rng.uniform(0.0, h_rho, size=prod.price.shape)

    cond_up = (prod.inventory == 0.0) & (prod.price >= p_avg)
    cond_dn = (prod.inventory > 0.0) & (prod.price < p_avg)

    # start with base = last period’s output
    prod.expected_demand[:] = prod.prev_production

    # apply positive / negative adjustments
    prod.expected_demand[cond_up] *= 1.0 + shock[cond_up]
    prod.expected_demand[cond_dn] *= 1.0 - shock[cond_dn]

    # desired production equals expected demand
    prod.desired_production[:] = prod.expected_demand

    # ── logging (vector summary) ─────────────────────────────────────────────
    log.debug(
        "decide_desired_production: n=%d  p̄=%.3f  up=%d  down=%d  mean_shock=%.4f",
        prod.price.size,
        p_avg,
        int(cond_up.sum()),
        int(cond_dn.sum()),
        shock.mean(),
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
