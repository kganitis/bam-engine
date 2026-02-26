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
    labor_productivity : float
        Labor productivity (goods per worker, positive).
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
    max_loan_to_net_worth : float, optional
        Maximum loan as multiple of borrower's net worth (default: 2.0).
        Caps individual loans at X × net_worth. Set to 0 for no limit.
    max_leverage : float, optional
        Maximum leverage ratio (fragility cap) for interest rate calculation.
        Caps the fragility used in rate formula to prevent extreme rates.
        Set to 0 for no limit. Default: 10.0.
    cap_factor : float or None, optional
        Cap factor for breakeven price calculation (>= 1.0 if specified).
        If None, no cap is applied.
    matching_method : str, optional
        How workers and firms are matched in labor market:
        "sequential" = workers shuffled, apply one at a time (efficient)
        "simultaneous" = all workers apply at once, creating crowding at
        high-wage firms (creates natural unemployment). Default: "sequential".
    job_search_method : str, optional
        How unemployed workers sample firms for job applications:
        "vacancies_only" = sample only from firms with open vacancies
        "all_firms" = sample from ALL firms (applications to non-hiring firms
        are wasted) (default). Default: "all_firms".
    price_cut_allow_increase : bool, optional
        Whether to allow price increase when trying to cut due to
        breakeven floor. Default: True.
    new_firm_size_factor : float, optional
        Scale factor for new firm net worth vs survivor mean. Default: 0.5.
    new_firm_production_factor : float, optional
        Scale factor for new firm production vs survivor mean. Default: 0.5.
    new_firm_wage_factor : float, optional
        Scale factor for new firm wage offer vs survivor mean. Default: 0.5.
    new_firm_price_markup : float, optional
        Price markup for new firms vs avg market price. Default: 1.15.

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
    ...     h_rho=0.1,
    ...     h_xi=0.1,
    ...     h_phi=0.1,
    ...     h_eta=0.1,
    ...     max_M=4,
    ...     max_H=2,
    ...     max_Z=3,
    ...     labor_productivity=0.5,
    ...     theta=8,
    ...     beta=2.5,
    ...     delta=0.10,
    ...     r_bar=0.02,
    ...     v=0.10,
    ...     cap_factor=None,
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
    """Maximum production-growth shock (0 to 1)."""
    h_xi: float
    """Maximum wage-growth shock (0 to 1)."""
    h_phi: float
    """Maximum bank operational costs shock (0 to 1)."""
    h_eta: float
    """Maximum price-growth shock (0 to 1)."""

    # Market queue parameters
    max_M: int
    """Maximum job applications per unemployed worker."""
    max_H: int
    """Maximum loan applications per firm."""
    max_Z: int
    """Maximum firm visits per consumer."""

    # Economy-level parameters
    labor_productivity: float
    """Labor productivity in goods per worker."""
    theta: int
    """Job contract base duration in periods."""
    beta: float
    """Propensity to consume exponent."""
    delta: float
    """Dividend payout ratio (0 to 1)."""
    r_bar: float
    """Base interest rate (0 to 1)."""
    v: float
    """Bank capital requirement ratio (0 to 1)."""
    max_loan_to_net_worth: float = 2.0
    """Maximum loan as multiple of borrower's net worth (0 = no limit)."""
    max_leverage: float = 10.0
    """Maximum leverage ratio (fragility cap) for interest rate calculation (0 = no cap)."""

    # Optional parameters
    cap_factor: float | None = None
    """Cap factor for breakeven price calculation (>= 1.0 if set, None = no cap)."""

    # Implementation variant parameters
    matching_method: str = "sequential"
    """Labor market matching method: ``"sequential"`` or ``"simultaneous"``."""
    job_search_method: str = "all_firms"
    """How unemployed workers sample firms: ``"vacancies_only"`` or ``"all_firms"``."""

    # New firm entry parameters
    new_firm_size_factor: float = 0.5
    """Scale factor for new firm net worth vs survivor mean."""
    new_firm_production_factor: float = 0.5
    """Scale factor for new firm production vs survivor mean."""
    new_firm_wage_factor: float = 0.5
    """Scale factor for new firm wage offer vs survivor mean."""
    new_firm_price_markup: float = 1.15
    """Price markup for new firms vs average market price."""

    # Pricing phase
    pricing_phase: str = "planning"
    """Pricing phase: ``"planning"`` or ``"production"``."""

    # Consumer matching strategy
    consumer_matching: str = "loyalty"
    """Consumer matching strategy: ``"loyalty"`` or ``"random"``."""

    # === DEPRECATED (may be removed in future) ===
    price_cut_allow_increase: bool = True
    """Whether to allow price increase when cutting due to breakeven floor. *Deprecated.*"""
    inflation_method: str = "yoy"
    """Inflation calculation method: ``"yoy"`` or ``"annualized"``. *Deprecated.*"""
    labor_matching: str = "interleaved"
    """Labor market matching mode (always interleaved). *Deprecated.*"""
    credit_matching: str = "interleaved"
    """Credit market matching mode (always interleaved). *Deprecated.*"""
    min_wage_ratchet: bool = False
    """Whether minimum wage can only increase. *Deprecated.*"""
