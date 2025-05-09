# src/_testing/__init__.py
"""
Private helpers used by the test‑suite only.
NOT part of the public API.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bamengine.scheduler import Scheduler


def advance_stub_state(sched: "Scheduler") -> None:
    """
    Temporary helper used while future events are not implemented.

    It performs three minimal tasks so the simulation can run multiple
    periods and the tests have fresh – but *plausible* – arrays:

    1. Append the realised average market price to the global series.
    2. Roll `prev_production` forward to last period’s `desired_production`.
    3. Jitter inventory and price so future periods don’t see constants.
    """

    # store wage
    sched.fw.wage_prev[:] = sched.wb.wage

    # mock production
    sched.prod.prev_production[:] *= sched.rng.uniform(
        0.7, 1.0, size=sched.prod.desired_production.shape
    )

    # mock new prices
    sched.prod.price[:] *= sched.rng.uniform(0.95, 1.05, size=sched.prod.price.shape)
    sched.ec.avg_mkt_price = float(sched.prod.price.mean())
    sched.ec.avg_mkt_price_history = np.append(
        sched.ec.avg_mkt_price_history, sched.ec.avg_mkt_price
    )

    # mock goods market
    sched.prod.inventory[:] = sched.rng.integers(0, 6, size=sched.prod.inventory.shape)
