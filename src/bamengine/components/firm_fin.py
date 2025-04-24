from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

FloatA = NDArray[np.float64]


@dataclass(slots=True)
class FirmFin:
    net_worth: FloatA
    wage_bill: FloatA
    credit_demand: FloatA
