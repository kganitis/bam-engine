import logging

import numpy as np

from bamengine.components.firm_labor import FirmLabor

logger = logging.getLogger(__name__)


def decide_desired_labor(lab: FirmLabor) -> None:
    """
    Desired labor demand (vectorised):

        Ld_i = ceil(Yd_i / a_i)
    """
    ratio = lab.desired_production / lab.labor_productivity
    np.ceil(ratio, out=ratio)
    lab.desired_labor[:] = ratio.astype(np.int64)

    logger.debug(
        "decide_desired_labor: n=%d  avg_prod=%.2f  avg_a=%.2f  "
        "avg_Ld=%.2f  max_Ld=%d",
        lab.desired_production.size,
        lab.desired_production.mean(),
        lab.labor_productivity.mean(),
        lab.desired_labor.mean(),
        int(lab.desired_labor.max()),
    )
