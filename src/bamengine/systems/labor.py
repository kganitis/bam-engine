import numpy as np
from bamengine.components.firm_labor import FirmLabor


def decide_desired_labor(lab: FirmLabor) -> None:
    """
    Ld_i = ceil( Yd_i / a_i )

    Vectorised over all firms.
    """
    ratio = lab.desired_production / lab.labor_productivity
    lab.desired_labor[:] = np.ceil(ratio).astype(np.int64)
