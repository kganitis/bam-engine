from dataclasses import dataclass

from bamengine.typing import FloatA


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
