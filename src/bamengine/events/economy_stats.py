"""Economy statistics events for aggregate metrics calculation.

This module contains Event classes that calculate economy-wide statistics
such as average market price and unemployment rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bamengine.core import Event

if TYPE_CHECKING:
    from bamengine.simulation import Simulation


@dataclass(slots=True)
class UpdateAvgMktPrice(Event):
    """
    Update average market price based on firm prices.

    Calculates economy-wide average price:
        P̄ = mean(p_i) for all firms

    This is used for inflation calculations and firm price adjustments.

    This event wraps `bamengine.systems.production.update_avg_mkt_price`.

    Dependencies
    ------------
    - firms_adjust_price : Uses updated firm prices

    See Also
    --------
    bamengine.systems.production.update_avg_mkt_price : Underlying logic
    """

    def execute(self, sim: Simulation) -> None:
        """Execute average market price update."""
        from bamengine.systems.production import update_avg_mkt_price

        update_avg_mkt_price(sim.ec, sim.prod)


@dataclass(slots=True)
class CalcUnemploymentRate(Event):
    """
    Calculate unemployment rate at end of period.

    Unemployment rate is:
        u = (N_unemployed / N_total) · 100

    where N_unemployed counts workers with expired or no contracts.

    This event wraps `bamengine.systems.production.calc_unemployment_rate`.

    Dependencies
    ------------
    - spawn_replacement_banks : Last event before period end

    See Also
    --------
    bamengine.systems.production.calc_unemployment_rate : Underlying logic
    """

    def execute(self, sim: Simulation) -> None:
        """Execute unemployment rate calculation."""
        from bamengine.systems.production import calc_unemployment_rate

        calc_unemployment_rate(sim.ec, sim.wrk)
