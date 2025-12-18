"""
Economy statistics events for aggregate metrics calculation.

This module defines economy-level statistics events that calculate and track
aggregate economic indicators like average prices.

Note: The CalcUnemploymentRate event has been DEPRECATED. Unemployment rate
is now calculated directly from Worker.employed data in SimulationResults.
The event is kept for backward compatibility but is no longer in the default
pipeline.

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

    .. deprecated::
        This event is DEPRECATED and no longer included in the default pipeline.
        Unemployment rate should now be calculated directly from Worker.employed
        data in SimulationResults::

            employed = results.role_data["Worker"]["employed"]  # (n_periods, n_workers)
            unemployment_rate = 1 - np.mean(employed, axis=1)

        The event is kept for backward compatibility. To re-enable, add
        ``calc_unemployment_rate`` to your custom pipeline YAML.

    Unemployment rate = (unemployed workers / total workers). Tracked in
    economy history for analysis.

    See Also
    --------
    Worker : Employment status
    bamengine.events._internal.production.calc_unemployment_rate : Implementation
    """

    def execute(self, sim: Simulation) -> None:
        from bamengine.events._internal.production import calc_unemployment_rate

        calc_unemployment_rate(sim.ec, sim.wrk)
