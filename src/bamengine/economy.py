"""
Economy-wide state container for BAM Engine.

This module provides the Economy class, which stores global economy state
including policy parameters, market statistics, and time-series histories.
Unlike roles which store per-agent arrays, Economy stores single values
or time series affecting the entire economy.

Classes
-------
Economy
    Container for economy-wide scalars and time series.

See Also
--------
bamengine.roles : Per-agent state components (Producer, Worker, etc.)
bamengine.simulation : Main simulation facade accessing Economy via ec attribute

Notes
-----
Economy is not a Role - it does not inherit from the Role base class because
it represents economy-wide state rather than per-agent arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bamengine.typing import Float1D, Idx1D


@dataclass(slots=True)
class Economy:
    """
    Economy-wide state container for scalar parameters and time series.

    Stores global economy state including policy parameters (min wage),
    market statistics (average price, unemployment), and time-series histories.
    Unlike roles which store per-agent state, Economy stores single values
    or time series affecting the entire economy.

    Attributes
    ----------
    avg_mkt_price : float
        Current average market price across all firms (P̄).
    min_wage : float
        Minimum wage floor enforced by policy.
    min_wage_rev_period : int
        Period when min wage was last revised.
    avg_mkt_price_history : Float1D
        Time series of average market prices, shape (t+1,).
    unemp_rate_history : Float1D
        Time series of raw unemployment rates, shape (t+1,).
        Apply rolling mean for smoothing if needed.
    inflation_history : Float1D
        Time series of inflation rates, shape (t+1,).
    exiting_firms : Idx1D
        Transient list of firms exiting this period (flushed each period).
    exiting_banks : Idx1D
        Transient list of banks exiting this period (flushed each period).
    destroyed : bool
        Termination flag indicating simulation should stop.

    Examples
    --------
    Access economy state in simulation:

    >>> import bamengine as bam
    >>> sim = bam.Simulation.init(n_firms=100, seed=42)
    >>> sim.ec.avg_mkt_price
    1.0
    >>> sim.ec.min_wage
    1.0
    >>> len(sim.ec.avg_mkt_price_history)
    1

    Check time series after running:

    >>> sim.run(n_periods=10)
    >>> len(sim.ec.avg_mkt_price_history)
    11
    >>> sim.ec.unemp_rate_history[-1]  # Latest unemployment rate
    0.0...

    Notes
    -----
    Economy is not a Role - it doesn't inherit from the Role base class
    because it stores economy-wide state, not per-agent arrays.

    See Also
    --------
    :class:`~bamengine.core.Role` : Base class for per-agent state components
    :class:`~bamengine.Simulation` : Main simulation facade with ec attribute
    """

    # policy / structural scalars
    avg_mkt_price: float
    min_wage: float
    min_wage_rev_period: int

    # time-series
    avg_mkt_price_history: Float1D  # shape  (t+1,)
    unemp_rate_history: Float1D  # shape  (t+1,) -- raw unemployment rate
    inflation_history: Float1D  # shape  (t+1,)

    # transient exit lists (flushed each Entry event)
    exiting_firms: Idx1D = field(default_factory=lambda: np.empty(0, np.intp))
    exiting_banks: Idx1D = field(default_factory=lambda: np.empty(0, np.intp))

    # Termination flag
    destroyed: bool = False
