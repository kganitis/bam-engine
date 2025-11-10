"""Economy statistics events for aggregate metrics calculation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@event
class UpdateAvgMktPrice:
    """
    Update exponentially smoothed average market price and update economy state.
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.production import update_avg_mkt_price

        update_avg_mkt_price(sim.ec, sim.prod)


@event
class CalcUnemploymentRate:
    """
    Calculate unemployment rate and update economy history.
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.production import calc_unemployment_rate

        calc_unemployment_rate(sim.ec, sim.wrk)
