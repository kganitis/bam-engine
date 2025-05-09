from dataclasses import dataclass

from bamengine.typing import Float1D


@dataclass(slots=True)
class Economy:
    """Global, mutable scalars & time-series."""

    avg_mkt_price: float
    avg_mkt_price_history: Float1D  # P_0 … P_t   (append-only)
    min_wage: float  # ŵ_t
    min_wage_rev_period: int  # constant (e.g. 4)
    r_bar: float  # base interest rate
    v: float  # capital requirement coefficient
