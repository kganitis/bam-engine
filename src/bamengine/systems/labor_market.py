import logging

from bamengine.components.economy import Economy

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
