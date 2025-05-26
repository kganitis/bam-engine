# src/diagnostics.py
import logging

import numpy as np

from bamengine.scheduler import Scheduler

log = logging.getLogger(__name__)


def log_firm_strategy_distribution(sched: Scheduler) -> None:
    """
    Analyzes and logs the distribution of firms across the four strategic cases
    defined by the BAM model.
    """
    p_avg = sched.ec.avg_mkt_price
    inventory = sched.prod.inventory
    price = sched.prod.price

    # Case (a): Reduce Price (Inventory > 0, Price >= Avg)
    case_a = (inventory > 0) & (price >= p_avg)
    n_a = np.sum(case_a)

    # Case (b): Increase Price (Inventory == 0, Price < Avg)
    case_b = (inventory == 0) & (price < p_avg)
    n_b = np.sum(case_b)

    # Case (c): Reduce Quantity (Inventory > 0, Price < Avg)
    case_c = (inventory > 0) & (price < p_avg)
    n_c = np.sum(case_c)

    # Case (d): Increase Quantity (Inventory == 0, Price >= Avg)
    case_d = (inventory == 0) & (price >= p_avg)
    n_d = np.sum(case_d)

    total = n_a + n_b + n_c + n_d
    if total != sched.n_firms:
        log.error(
            f"Strategy case summation mismatch! "
            f"Sum is {total}, expected {sched.n_firms}"
        )

    log.info(
        f"[STRATEGY MATRIX] "
        f"Reduce Price (a): {n_a:3d} | "
        f"Increase Price (b): {n_b:3d} | "
        f"Reduce Quantity (c): {n_c:3d} | "
        f"Increase Quantity (d): {n_d:3d}"
    )
