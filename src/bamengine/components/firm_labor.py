from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

FloatA = NDArray[np.float64]
IntA = NDArray[np.int64]


@dataclass(slots=True)
class FirmLabor:
    """Dense state needed for labor-demand decisions."""

    desired_production: FloatA  # Yd_i  (read-only here)
    labor_productivity: FloatA  # a_i   (can change with R&D later)
    desired_labor: IntA  # Ld_i  (output)
