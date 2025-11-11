"""
Configuration dataclass for simulation parameters.

This module defines the Config dataclass, which groups all simulation
hyperparameters in one immutable object. Config instances are created
by Simulation.init() after merging defaults, user config, and kwargs.

Design Notes
------------
- Immutable (frozen=True) to prevent accidental modification
- Memory-efficient (slots=True)
- All required parameters listed explicitly (no defaults except optional ones)
- Simple dataclass, no methods - validation happens in ConfigValidator

See Also
--------
ConfigValidator : Centralized validation for configuration parameters
bamengine.simulation.Simulation.init : Creates Config from merged parameters
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Config:
    """
    Immutable configuration for BAM simulation parameters.

    This dataclass groups all simulation hyperparameters in one place,
    making them easier to manage and pass around. Config instances are
    created internally by Simulation.init() after parameter validation.

    Parameters
    ----------
    h_rho : float
        Maximum production-growth shock (0 to 1).
    h_xi : float
        Maximum wage-growth shock (0 to 1).
    h_phi : float
        Maximum bank operational costs shock (0 to 1).
    h_eta : float
        Maximum price-growth shock (0 to 1).
    max_M : int
        Maximum job applications per unemployed worker (positive).
    max_H : int
        Maximum loan applications per firm (positive).
    max_Z : int
        Maximum firm visits per consumer (positive).
    theta : int
        Job contract base duration in periods (positive).
    beta : float
        Propensity to consume exponent (positive).
    delta : float
        Dividend payout ratio (0 to 1).
    r_bar : float
        Base interest rate (0 to 1).
    v : float
        Bank capital requirement ratio (0 to 1).
    cap_factor : float or None, optional
        Cap factor for breakeven price calculation (>= 1.0 if specified).
        If None, no cap is applied.

    Examples
    --------
    Config instances are typically created by Simulation.init():

    >>> import bamengine as be
    >>> sim = be.Simulation.init(n_firms=100, seed=42)
    >>> cfg = sim.config
    >>> cfg.h_rho
    0.1
    >>> cfg.max_M
    4

    Manual creation (rarely needed):

    >>> from bamengine.config import Config
    >>> cfg = Config(
    ...     h_rho=0.1, h_xi=0.1, h_phi=0.1, h_eta=0.1,
    ...     max_M=4, max_H=2, max_Z=3,
    ...     theta=3, beta=2.5, delta=0.4,
    ...     r_bar=0.02, v=0.1,
    ...     cap_factor=None
    ... )
    >>> cfg.beta
    2.5

    Config is immutable:

    >>> cfg.beta = 3.0  # doctest: +SKIP
    FrozenInstanceError: cannot assign to field 'beta'

    Notes
    -----
    Config is a simple data container with no validation logic. All validation
    happens in ConfigValidator before Config creation. This separation keeps
    concerns cleanly separated: Config holds data, ConfigValidator ensures
    data validity.

    See Also
    --------
    ConfigValidator : Validates parameters before Config creation
    bamengine.simulation.Simulation.init : Factory for creating configured simulations
    """

    # Shock parameters
    h_rho: float
    h_xi: float
    h_phi: float
    h_eta: float

    # Market queue parameters
    max_M: int
    max_H: int
    max_Z: int

    # Economy-level parameters
    theta: int
    beta: float
    delta: float
    r_bar: float
    v: float

    # Optional parameters
    cap_factor: float | None = None
