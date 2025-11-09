"""Economy statistics events for aggregate metrics calculation.

This module contains Event classes that calculate economy-wide statistics
such as average market price and unemployment rate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bamengine.core.decorators import event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@event
class UpdateAvgMktPrice:
    """
    Update average market price based on firm prices.

    Calculates economy-wide average price:
        P̄ = mean(p_i) for all firms

    This is used for inflation calculations and firm price adjustments.

    This event wraps `bamengine.systems.production.update_avg_mkt_price`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute average market price update."""
        from bamengine.systems.production import update_avg_mkt_price

        update_avg_mkt_price(sim.ec, sim.prod)


@event
class CalcUnemploymentRate:
    """
    Calculate unemployment rate at end of period.

    Unemployment rate is:
        u = (N_unemployed / N_total) · 100

    where N_unemployed counts workers with expired or no contracts.

    This event wraps `bamengine.systems.production.calc_unemployment_rate`.
    """

    def execute(self, sim: Simulation) -> None:
        """Execute unemployment rate calculation."""
        from bamengine.systems.production import calc_unemployment_rate

        calc_unemployment_rate(sim.ec, sim.wrk)
