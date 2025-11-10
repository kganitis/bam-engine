"""Economy statistics events for aggregate metrics calculation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bamengine import logging
from bamengine.core.decorators import event
from bamengine.utils import trimmed_weighted_mean

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@event
class UpdateAvgMktPrice:
    """
    Update exponentially smoothed average market price and update economy state.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute average market price update."""
        log = self.get_logger()
        ec = sim.ec
        prod = sim.prod
        alpha = 1.0
        trim_pct = 0.0

        log.info("--- Updating Average Market Price ---")

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"  Price update parameters: alpha={alpha:.3f}, "
                f"trim_pct={trim_pct:.3f}"
            )

        # calculate trimmed weighted mean
        p_avg_trimmed = trimmed_weighted_mean(prod.price, trim_pct=trim_pct)
        previous_price = ec.avg_mkt_price

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"  Price calculation: trimmed_mean={p_avg_trimmed:.4f}, "
                f"previous_avg={previous_price:.4f}"
            )

        # update economy state
        ec.avg_mkt_price = alpha * p_avg_trimmed + (1.0 - alpha) * ec.avg_mkt_price
        ec.avg_mkt_price_history = np.append(ec.avg_mkt_price_history, ec.avg_mkt_price)

        log.info(f"  Average market price updated: {ec.avg_mkt_price:.4f}")
        log.info("--- Average Market Price Update complete ---")


# TODO Not unit tested yet
@event
class CalcUnemploymentRate:
    """
    Calculate unemployment rate and update economy history.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute unemployment rate calculation."""
        log = self.get_logger()
        ec = sim.ec
        wrk = sim.wrk

        log.info("--- Calculating Unemployment Rate ---")

        n_workers = wrk.employed.size
        unemployed_count = n_workers - wrk.employed.sum()
        rate = unemployed_count / n_workers

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                f"  Unemployment calculation: {unemployed_count} unemployed "
                f"out of {n_workers} total workers"
            )

        log.info(f"  Current unemployment rate: {rate * 100:.2f}%")

        # update economy state
        ec.unemp_rate_history = np.append(ec.unemp_rate_history, rate)

        log.info("--- Unemployment Rate Calculation complete ---")
