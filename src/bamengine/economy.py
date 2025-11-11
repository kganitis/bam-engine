from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bamengine.typing import Float1D, Idx1D


@dataclass(slots=True)
class Economy:
    """
    Pure *state* container for economy-wide scalar parameters,
    time series, transient exit lists and flags.
    """

    # policy / structural scalars
    avg_mkt_price: float
    min_wage: float
    min_wage_rev_period: int
    # TODO move r_bar and v out of the economy component and move to the simulation level
    r_bar: float  # base interest-rate
    v: float  # capital-adequacy coefficient

    # time-series
    avg_mkt_price_history: Float1D  # shape  (t+1,)
    unemp_rate_history: Float1D  # shape  (t+1,)
    inflation_history: Float1D  # shape  (t+1,)

    # transient exit lists (flushed each Entry event)
    exiting_firms: Idx1D = field(default_factory=lambda: np.empty(0, np.intp))
    exiting_banks: Idx1D = field(default_factory=lambda: np.empty(0, np.intp))

    # Termination flag
    destroyed: bool = False
