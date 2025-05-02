from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatA = NDArray[np.float64]


@dataclass(slots=True)
class Economy:
    """Global, mutable scalars & time-series."""

    avg_mrkt_price: float
    avg_mrkt_price_history: FloatA  # P_0 … P_t   (append-only)
    min_wage: float  # ŵ_t
    min_wage_rev_period: int  # constant (e.g. 4)

    # helper (no allocations)
    def last_prices(self, k: int) -> FloatA:  # view, not copy
        return self.avg_mrkt_price_history[-k:]
