from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatA = NDArray[np.float64]


@dataclass(slots=True)
class FirmProduction:
    """Dense state needed for production-planning."""

    price: FloatA  # current price P_i
    inventory: FloatA  # inventory S_i  (carried from t-1)
    prev_production: FloatA  # Y_{i,t-1}
    expected_demand: FloatA  # D̂_i  (output, same size)
    desired_production: FloatA  # Yd_i (output, same size)
