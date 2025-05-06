"""
Private helpers used by the test‑suite only.
NOT part of the public API.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bamengine.scheduler import Scheduler


def advance_stub_state(sched: "Scheduler", p_avg: float) -> None:
    """
    Temporary helper used while future events are not implemented.

    It performs three minimal tasks so the simulation can run multiple
    periods and the tests have fresh – but *plausible* – arrays:

    1. Append the realised average market price to the global series.
    2. Roll `prev_production` forward to last period’s `desired_production`.
    3. Jitter inventory and price so future periods don’t see constants.
    """
    # add price to history (needed for minimum-wage inflation rule)
    sched.ec.avg_mrkt_price_history = np.append(sched.ec.avg_mrkt_price_history, p_avg)

    # carry forward production level
    sched.prod.prev_production[:] = sched.prod.desired_production

    # stub: random new inventory and mild price noise
    sched.prod.inventory[:] = sched.rng.integers(0, 6, size=sched.prod.inventory.shape)
    sched.prod.price[:] *= sched.rng.uniform(0.98, 1.02, size=sched.prod.price.shape)
