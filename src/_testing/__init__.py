# src/_testing/__init__.py
"""
Private helpers used *only* by the test-suite.
They should disappear once all real events are implemented.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bamengine.scheduler import Scheduler


def advance_stub_state(sched: "Scheduler") -> None:
    """
    One-shot placeholder that nudges the state forward so multi-period
    tests have fresh, non-degenerate arrays.
    """

    # 1. --- mock production ---------------------------------------
    sched.prod.production[:] *= sched.rng.uniform(
        0.5, 2.0, size=sched.prod.desired_production.shape
    )

    # 2. --- mock prices -------------------------------------------
    sched.prod.price[:] *= sched.rng.uniform(0.95, 1.05, size=sched.prod.price.shape)
    sched.ec.avg_mkt_price = float(sched.prod.price.mean())
    sched.ec.avg_mkt_price_history = np.append(
        sched.ec.avg_mkt_price_history, sched.ec.avg_mkt_price
    )

    # ------ mock goods market -------------------------------------
    sched.prod.inventory[:] = sched.rng.integers(0, 6, size=sched.prod.inventory.shape)
