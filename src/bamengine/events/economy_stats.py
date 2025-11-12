"""
Economy statistics events for aggregate metrics calculation.

This module defines economy-level statistics events that calculate and track
aggregate economic indicators like average prices and unemployment.

Examples
--------
>>> import bamengine as be
>>> sim = be.Simulation.init(seed=42)
>>> sim.step()  # Stats events run as part of default pipeline
>>> sim.ec.avg_mkt_price  # doctest: +SKIP
1.05
>>> sim.ec.unemployment_rate_history[-1]  # doctest: +SKIP
0.04
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
    CalcAnnualInflationRate : Uses price history for inflation
    bamengine.events._internal.production.update_avg_mkt_price : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.production import update_avg_mkt_price

        update_avg_mkt_price(sim.ec, sim.prod)


@event
class CalcUnemploymentRate:
    """
    Calculate unemployment rate from worker employment status.

    Unemployment rate = (unemployed workers / total workers). Tracked in
    economy history for analysis.

    Examples
    --------
    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_households=500, seed=42)
    >>> event = sim.get_event("calc_unemployment_rate")
    >>> event.execute(sim)
    >>> sim.ec.unemployment_rate_history[-1]  # doctest: +SKIP
    0.04

    See Also
    --------
    Worker : Employment status
    bamengine.events._internal.production.calc_unemployment_rate : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.production import calc_unemployment_rate

        calc_unemployment_rate(sim.ec, sim.wrk)
