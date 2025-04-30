import logging

import numpy as np
from numpy.random import Generator

from bamengine.components.economy import Economy
from bamengine.components.firm_labor import FirmWageOffer

log = logging.getLogger(__name__)


def adjust_minimum_wage(ec: Economy) -> None:
    """
    Every `min_wage_rev_period` periods update ŵ_t by realised inflation:

        π = (P̅_{t-1} - P̅_{t-m}) / P̅_{t-m}
        ŵ_t = ŵ_{t-1} * (1 + π)
    """
    m = ec.min_wage_rev_period
    if ec.avg_mrkt_price_history.size <= m:
        return  # not enough data yet
    if (ec.avg_mrkt_price_history.size - 1) % m != 0:
        return  # not a revision step

    p_now = ec.avg_mrkt_price_history[-2]  # price of period t-1
    p_prev = ec.avg_mrkt_price_history[-m - 1]  # price of period t-m
    inflation = (p_now - p_prev) / p_prev

    ec.min_wage *= 1.0 + inflation

    log.debug(
        "adjust_minimum_wage: m=%d  π=%.4f  new_ŵ=%.3f",
        m,
        inflation,
        ec.min_wage,
    )


def decide_wage_offer(
    fw: FirmWageOffer,
    *,
    w_min: float,
    h_xi: float,
    rng: Generator,
) -> None:
    """
    Vector rule:

        shock_i ~ U(0, h_xi)  if V_i>0 else 0
        w_i^b   = max( w_min , w_{i,t-1} * (1 + shock_i) )

    Works fully in-place, no temporary allocations.
    """
    # Draw one shock per firm, then mask where V_i==0.
    shock = rng.uniform(0.0, h_xi, size=fw.wage_prev.shape)
    shock[fw.n_vacancies == 0] = 0.0

    np.multiply(fw.wage_prev, 1.0 + shock, out=fw.wage_offer)
    np.maximum(fw.wage_offer, w_min, out=fw.wage_offer)

    log.debug(
        "decide_wage_offer: n=%d  w_min=%.3f  h_xi=%.3f  "
        "mean_w_prev=%.3f  mean_w_offer=%.3f",
        fw.wage_prev.size,
        w_min,
        h_xi,
        fw.wage_prev.mean(),
        fw.wage_offer.mean(),
    )
