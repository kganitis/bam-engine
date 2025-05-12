# src/bamengine/systems/planning.py
from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator

from bamengine.components.employer import Employer
from bamengine.components.producer import Producer

log = logging.getLogger(__name__)

CAP_LAB_PROD = 1.0e-6  # labor productivity cap if below from or equal to zero


def firms_decide_desired_production(  # noqa: C901  (still quite short)
    prod: Producer,
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
    shape = prod.price.shape

    # ── 1. permanent scratches ---------------
    shock = prod.prod_shock
    if shock is None or shock.shape != shape:
        shock = np.empty(shape, dtype=np.float64)
        prod.prod_shock = shock

    up_mask = prod.prod_mask_up
    if up_mask is None or up_mask.shape != shape:
        up_mask = np.empty(shape, dtype=np.bool_)
        prod.prod_mask_up = up_mask

    dn_mask = prod.prod_mask_dn
    if dn_mask is None or dn_mask.shape != shape:
        dn_mask = np.empty(shape, dtype=np.bool_)
        prod.prod_mask_dn = dn_mask

    # ── 2. fill buffers in‑place ---------------
    shock[:] = rng.uniform(0.0, h_rho, size=shape)
    np.logical_and(prod.inventory == 0.0, prod.price >= p_avg, out=up_mask)
    np.logical_and(prod.inventory > 0.0, prod.price < p_avg, out=dn_mask)

    # ── 3. core rule ----------------------------------
    prod.expected_demand[:] = prod.production
    prod.expected_demand[up_mask] *= 1.0 + shock[up_mask]
    prod.expected_demand[dn_mask] *= 1.0 - shock[dn_mask]
    prod.desired_production[:] = prod.expected_demand


def firms_decide_desired_labor(prod: Producer, emp: Employer) -> None:
    """
    Desired labor demand (vectorised):

        Ld_i = ceil(Yd_i / a_i)
    """
    if (prod.labor_productivity <= 0).any():
        prod.labor_productivity[:] = CAP_LAB_PROD
    ratio = prod.desired_production / prod.labor_productivity
    np.ceil(ratio, out=ratio)
    emp.desired_labor[:] = ratio.astype(np.int64)


def firms_decide_vacancies(emp: Employer) -> None:
    """
    Vector rule: V_i = max( Ld_i – L_i , 0 )
    """
    np.subtract(
        emp.desired_labor, emp.current_labor, out=emp.n_vacancies, dtype=np.int64
    )
    np.maximum(emp.n_vacancies, 0, out=emp.n_vacancies)
