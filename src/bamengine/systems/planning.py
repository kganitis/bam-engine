# src/bamengine/systems/planning.py
from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator

from bamengine.components.employer import Employer
from bamengine.components.producer import Producer

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)

CAP_LAB_PROD = 1.0e-6  # labor productivity cap if below from or equal to zero


def firms_decide_desired_production(  # noqa: C901
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
    log.info("Firms deciding desired production...")
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

    n_up = np.sum(up_mask)
    n_dn = np.sum(dn_mask)
    n_keep = len(prod.price) - n_up - n_dn
    log.info(f"Production changes: {n_up} firms ↑, {n_dn} firms ↓, {n_keep} firms ↔.")

    # Add detailed vector logging only if DEBUG is enabled to avoid performance hit
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Avg market price (p_avg): {p_avg:.4f}")
        log.debug(
            f"  Inventories (S_i):\n{np.array2string(prod.inventory, precision=2)}"
        )
        log.debug(f"  Prices (P_i):\n{np.array2string(prod.price, precision=2)}")
        log.debug(f"  Up mask:\n{up_mask}")
        log.debug(f"  Down mask:\n{dn_mask}")
        log.debug(
            f"  Previous Production (Y_{{t-1}}):\n"
            f"{np.array2string(prod.production, precision=2)}"
        )

    # ── 3. core rule ----------------------------------
    prod.expected_demand[:] = prod.production
    prod.expected_demand[up_mask] *= 1.0 + shock[up_mask]
    prod.expected_demand[dn_mask] *= 1.0 - shock[dn_mask]
    prod.desired_production[:] = prod.expected_demand

    log.debug(
        f"  Desired Production (Yd_i):\n"
        f"{np.array2string(prod.desired_production, precision=2)}"
    )


def firms_decide_desired_labor(prod: Producer, emp: Employer) -> None:
    """
    Desired labor demand (vectorised):

        Ld_i = ceil(Yd_i / a_i)
    """
    log.info("Firms deciding desired labor...")
    invalid = (~np.isfinite(prod.labor_productivity)) | (
        prod.labor_productivity <= CAP_LAB_PROD
    )
    if invalid.any():
        n_invalid = np.sum(invalid)
        log.warning(
            f"{n_invalid} firms have too low/non-finite labor productivity; clamping."
        )
        prod.labor_productivity[invalid] = CAP_LAB_PROD

    # core rule -----------------------------------------------------------
    ratio = prod.desired_production / prod.labor_productivity
    np.ceil(ratio, out=ratio)  # in-place ceiling

    # clip to int64 range to avoid overflow warnings
    int64_max = np.iinfo(np.int64).max
    np.clip(ratio, 0, int64_max, out=ratio)

    emp.desired_labor[:] = ratio.astype(np.int64)  # safe int cast
    log.info(f"Total desired labor across all firms: {emp.desired_labor.sum()}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"  Labor Productivity (a_i):\n"
            f"{np.array2string(prod.labor_productivity, precision=2)}"
        )
        log.debug(f"  Desired Labor (Ld_i):\n{emp.desired_labor}")


def firms_decide_vacancies(emp: Employer) -> None:
    """
    Vector rule: V_i = max( Ld_i – L_i , 0 )
    """
    log.info("Firms deciding vacancies...")
    np.subtract(
        emp.desired_labor,
        emp.current_labor,
        out=emp.n_vacancies,
        dtype=np.int64,
        casting="unsafe",  # makes MyPy/NumPy on Windows happy
    )
    np.maximum(emp.n_vacancies, 0, out=emp.n_vacancies)
    total_vacancies = emp.n_vacancies.sum()
    log.info(f"Total open vacancies in the economy: {total_vacancies}")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"  Current Labor (L_i):\n{emp.current_labor}")
        log.debug(f"  Vacancies (V_i):\n{emp.n_vacancies}")
