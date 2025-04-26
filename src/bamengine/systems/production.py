import logging

from numpy.random import Generator

from bamengine.components.firm_production import FirmProduction

logger = logging.getLogger(__name__)


def decide_desired_production(
    prod: FirmProduction,
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
    logger.debug(
        "decide_desired_production: n=%d  p̄=%.3f  up=%d  down=%d  mean_shock=%.4f",
        prod.price.size,
        p_avg,
        int(cond_up.sum()),
        int(cond_dn.sum()),
        shock.mean(),
    )
