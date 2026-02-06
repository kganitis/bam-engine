"""Parameter space definition for calibration.

This module defines the parameter grid for calibration and the default
values used as a baseline for sensitivity analysis.

Supports multiple scenarios:
    - baseline: Standard BAM model (Section 3.9.1)
    - growth_plus: Endogenous productivity growth via R&D (Section 3.9.2)
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
        "new_firm_size_factor": [0.5, 0.7, 0.8, 0.9],
        "new_firm_production_factor": [0.5, 0.7, 0.8, 0.9, 1.0],
        "new_firm_wage_factor": [0.5, 0.7, 0.8, 0.9, 1.0],
        "new_firm_price_markup": [1.0, 1.10, 1.25, 1.50],
        "max_loan_to_net_worth": [0, 2, 5],
        "firing_method": ["random", "expensive"],
        "matching_method": ["sequential", "simultaneous"],
        "job_search_method": ["vacancies_only", "all_firms"],
    },
    "growth_plus": {
        # R&D extension parameter (Growth+ specific)
        # "sigma_decay": [-2.0, -1.5, -1.0, -0.5],
        # New firm entry parameters
        "new_firm_size_factor": [0.5, 0.7, 0.8, 0.9],
        "new_firm_production_factor": [0.5, 0.7, 0.8, 0.9, 1.0],
        "new_firm_wage_factor": [0.5, 0.7, 0.8, 0.9, 1.0],
        "new_firm_price_markup": [1.0, 1.10, 1.25, 1.50],
        "max_loan_to_net_worth": [0, 2, 5],
        "firing_method": ["random", "expensive"],
        "matching_method": ["sequential", "simultaneous"],
        "job_search_method": ["vacancies_only", "all_firms"],
    },
}

# =============================================================================
# Scenario-specific parameter overrides
# =============================================================================

# For baseline: empty dict means engine defaults.yml are used for all params.
# For growth_plus: explicit overrides for R&D extension and different starting points.
SCENARIO_OVERRIDES: dict[str, dict[str, Any]] = {
    "baseline": {},  # Use engine defaults from defaults.yml
    "growth_plus": {
        "sigma_decay": -1.0,  # R&D extension param (not in engine defaults)
        # different defaults for new firm entry parameters
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
DEFAULT_VALUES: dict[str, Any] = SCENARIO_OVERRIDES["baseline"]


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
    """Get the scenario-specific parameter overrides.

    Parameters
    ----------
    scenario : str
        Scenario name ("baseline" or "growth_plus").

    Returns
    -------
    dict
        Scenario overrides. For baseline, returns empty dict (engine defaults).
        For growth_plus, returns R&D and new firm parameter overrides.

    Raises
    ------
    ValueError
        If scenario is not recognized.
    """
    if scenario not in SCENARIO_OVERRIDES:
        raise ValueError(
            f"Unknown scenario: {scenario}. "
            f"Available: {list(SCENARIO_OVERRIDES.keys())}"
        )
    return SCENARIO_OVERRIDES[scenario]


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
