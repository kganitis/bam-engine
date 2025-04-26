from __future__ import annotations

import logging

import numpy as np
from numpy.random import default_rng

from bamengine.components.firm_production import FirmProduction
from bamengine.components.firm_labor import FirmLabor
from bamengine.systems.production import decide_desired_production
from bamengine.systems.labor import decide_desired_labor


# ───────────────────────── Logging ──────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ───────────────────────── Parameters ───────────────────────
N_FIRMS = 10
N_STEPS = 3
H_RHO = 0.10  # max growth rate of quantities
SEED = 42


def main() -> None:
    rng = default_rng(SEED)

    # ── 1. create shared arrays ─────────────────────────────
    price = np.full(N_FIRMS, 1.5)
    inventory = rng.integers(0, 6, N_FIRMS).astype(float)
    prev_production = np.full(N_FIRMS, 10.0)

    # one shared array for desired production
    desired_prod = np.zeros(N_FIRMS)

    prod = FirmProduction(
        price=price,
        inventory=inventory,
        prev_production=prev_production,
        expected_demand=np.zeros(N_FIRMS),
        desired_production=desired_prod,  # shared
    )

    lab = FirmLabor(
        desired_production=desired_prod,  # same ndarray
        labor_productivity=np.ones(N_FIRMS),  # a_i = 1 for now
        desired_labor=np.zeros(N_FIRMS, dtype=np.int64),
    )

    # ── 2. run a tiny loop ───────────────────────────────────
    for t in range(1, N_STEPS + 1):
        logger.info("=== PERIOD %d ===", t)

        p_avg = prod.price.mean()

        decide_desired_production(prod, p_avg, H_RHO, rng)
        decide_desired_labor(lab)

        # Example: advance state for next step (simplified)
        prod.prev_production[:] = prod.desired_production
        prod.inventory[:] = rng.integers(0, 6, N_FIRMS)  # stub
        prod.price[:] *= rng.uniform(0.98, 1.02, N_FIRMS)  # stub inflation

        # quick human-readable stats
        logger.info(
            "mean Yd=%.2f | mean Ld=%.2f | max Ld=%d",
            prod.desired_production.mean(),
            lab.desired_labor.mean(),
            int(lab.desired_labor.max()),
        )

    logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
