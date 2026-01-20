"""Parameter space definition for calibration.

This module defines the parameter grid for calibration and the default
values used as a baseline for sensitivity analysis.

Supports multiple scenarios:
    - baseline: Standard BAM model (Section 3.9.1)
    - growth_plus: Endogenous productivity growth via R&D (Section 3.8)
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Any

# =============================================================================
# Scenario-specific parameter grids
# =============================================================================

PARAMETER_GRIDS: dict[str, dict[str, list[Any]]] = {
    "baseline": {
        # New firm entry parameters
        "new_firm_size_factor": [0.5, 0.7, 0.8, 0.9],
        "new_firm_production_factor": [0.5, 0.7, 0.8, 0.9, 1.0],
        "new_firm_wage_factor": [0.5, 0.7, 0.8, 0.9, 1.0],
        "new_firm_price_markup": [1.0, 1.25, 1.50],
    },
    "growth_plus": {
        # R&D extension parameter (Growth+ specific)
        "sigma_decay": [-2.0, -1.5, -1.0, -0.5],
        # New firm entry parameters
        "new_firm_size_factor": [0.25, 0.5, 0.7, 0.8, 0.9],
        "new_firm_production_factor": [0.25, 0.5, 0.7, 0.8, 0.9, 1.0],
        "new_firm_wage_factor": [0.25, 0.5, 0.7, 0.8, 0.9, 1.0],
        "new_firm_price_markup": [1.0, 1.25, 1.50, 1.75, 2.0],
    },
}

# =============================================================================
# Scenario-specific default values
# =============================================================================

DEFAULT_VALUES_BY_SCENARIO: dict[str, dict[str, Any]] = {
    "baseline": {
        "new_firm_size_factor": 0.9,
        "new_firm_production_factor": 0.9,
        "new_firm_wage_factor": 0.9,
        "new_firm_price_markup": 1.0,
    },
    "growth_plus": {
        "sigma_decay": -1.0,
        "new_firm_size_factor": 0.5,
        "new_firm_production_factor": 0.5,
        "new_firm_wage_factor": 0.5,
        "new_firm_price_markup": 1.5,
    },
}

# =============================================================================
# Backwards compatibility aliases
# =============================================================================

# Keep old names for backwards compatibility with existing code
PARAMETER_GRID: dict[str, list[Any]] = PARAMETER_GRIDS["baseline"]
DEFAULT_VALUES: dict[str, Any] = DEFAULT_VALUES_BY_SCENARIO["baseline"]


# =============================================================================
# Helper functions
# =============================================================================


def get_parameter_grid(scenario: str = "baseline") -> dict[str, list[Any]]:
    """Get the parameter grid for a scenario.

    Parameters
    ----------
    scenario : str
        Scenario name ("baseline" or "growth_plus").

    Returns
    -------
    dict
        Parameter grid for the scenario.

    Raises
    ------
    ValueError
        If scenario is not recognized.
    """
    if scenario not in PARAMETER_GRIDS:
        raise ValueError(
            f"Unknown scenario: {scenario}. Available: {list(PARAMETER_GRIDS.keys())}"
        )
    return PARAMETER_GRIDS[scenario]


def get_default_values(scenario: str = "baseline") -> dict[str, Any]:
    """Get the default values for a scenario.

    Parameters
    ----------
    scenario : str
        Scenario name ("baseline" or "growth_plus").

    Returns
    -------
    dict
        Default values for the scenario.

    Raises
    ------
    ValueError
        If scenario is not recognized.
    """
    if scenario not in DEFAULT_VALUES_BY_SCENARIO:
        raise ValueError(
            f"Unknown scenario: {scenario}. "
            f"Available: {list(DEFAULT_VALUES_BY_SCENARIO.keys())}"
        )
    return DEFAULT_VALUES_BY_SCENARIO[scenario]


def generate_combinations(
    grid: dict[str, list[Any]] | None = None,
    scenario: str = "baseline",
) -> Iterator[dict[str, Any]]:
    """Generate all parameter combinations from grid.

    Parameters
    ----------
    grid : dict, optional
        Parameter grid to use. If None, uses scenario-specific grid.
    scenario : str
        Scenario name (used if grid is None).

    Yields
    ------
    dict
        Dictionary mapping parameter names to values.
    """
    if grid is None:
        grid = get_parameter_grid(scenario)
    keys = list(grid.keys())
    for values in product(*grid.values()):
        yield dict(zip(keys, values, strict=True))


def count_combinations(
    grid: dict[str, list[Any]] | None = None,
    scenario: str = "baseline",
) -> int:
    """Count total combinations in grid.

    Parameters
    ----------
    grid : dict, optional
        Parameter grid to use. If None, uses scenario-specific grid.
    scenario : str
        Scenario name (used if grid is None).

    Returns
    -------
    int
        Number of combinations in the grid.
    """
    if grid is None:
        grid = get_parameter_grid(scenario)
    count = 1
    for values in grid.values():
        count *= len(values)
    return count
