# src/bamengine/config.py
"""Configuration dataclass for simulation parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Config:
    """
    Immutable configuration for BAM simulation parameters.

    This dataclass groups all simulation hyperparameters in one place,
    making them easier to manage and pass around.

    Attributes
    ----------
    h_rho : float
        Max production-growth shock
    h_xi : float
        Max wage-growth shock
    h_phi : float
        Max bank operational costs shock
    h_eta : float
        Max price-growth shock
    max_M : int
        Max job applications per unemployed worker
    max_H : int
        Max loan applications per firm
    max_Z : int
        Max firm visits per consumer
    theta : int
        Job contract length (base duration)
    beta : float
        Propensity to consume exponent
    delta : float
        Dividend payout ratio (DPR)
    cap_factor : int | None
        Cap factor for breakeven price calculation (optional)
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

    # Optional parameters
    cap_factor: int | None = None
