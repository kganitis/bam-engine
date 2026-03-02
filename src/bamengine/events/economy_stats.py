"""
Economy statistics events for aggregate metrics calculation.

This module defines economy-level statistics events that calculate and track
aggregate economic indicators like average prices.

Examples
--------
>>> import bamengine as be
>>> sim = be.Simulation.init(seed=42)
>>> sim.step()  # Stats events run as part of default pipeline
>>> sim.ec.avg_mkt_price  # doctest: +SKIP
1.05
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:  # pragma: no cover
    from bamengine.simulation import Simulation


@event
class UpdateAvgMktPrice:
    """
    Update exponentially smoothed average market price.

    The average market price is calculated from all firm prices and tracked
    in economy history for inflation calculations.

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> event = sim.get_event("update_avg_mkt_price")
    >>> event.execute(sim)
    >>> sim.ec.avg_mkt_price  # doctest: +SKIP
    1.02

    See Also
    --------
    CalcInflationRate : Uses price history for inflation
    bamengine.events._internal.production.update_avg_mkt_price : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.production import update_avg_mkt_price

        update_avg_mkt_price(sim.ec, sim.prod)
