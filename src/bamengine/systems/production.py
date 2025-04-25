from numpy.random import Generator
from bamengine.components.firm_production import FirmProduction


def decide_desired_production(
    prod: FirmProduction,
    p_avg: float,
    h_rho: float,
    rng: Generator,
) -> None:
    """
    Vectorised rule:

        shock ~ U(0, h_rho)

        if S_i == 0 and P_i >= p_avg:
            D̂_i = Y_{i,t-1} · (1 + shock)
        elif S_i > 0 and P_i < p_avg:
            D̂_i = Y_{i,t-1} · (1 - shock)

        Yd_i = D̂_i
    """
    shock = rng.uniform(0.0, h_rho, size=prod.price.shape)

    cond_up = (prod.inventory == 0.0) & (prod.price >= p_avg)
    cond_dn = (prod.inventory > 0.0) & (prod.price < p_avg)

    prod.expected_demand[:] = prod.production_prev  # default copy
    prod.expected_demand[cond_up] *= 1.0 + shock[cond_up]
    prod.expected_demand[cond_dn] *= 1.0 - shock[cond_dn]

    prod.desired_production[:] = prod.expected_demand
