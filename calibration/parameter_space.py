"""Parameter space definition for calibration.

This module defines the parameter grid for calibration and the default
values used as a baseline for sensitivity analysis.

Supports multiple scenarios:
    - baseline: Standard BAM model (Section 3.9.1)
    - growth_plus: Endogenous productivity growth via R&D (Section 3.9.2)
    - buffer_stock: Buffer-stock consumption with R&D (Section 3.9.4)
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Any

# =============================================================================
# Common parameter grid (shared by all scenarios)
# =============================================================================

_COMMON_GRID: dict[str, list[Any]] = {
    # Initial conditions
    "price_init": [0.5, 1.0, 1.5, 2.0, 3.0],
    "min_wage_ratio": [0.50, 0.67, 0.80, 1.0],
    "net_worth_ratio": [1.0, 2.0, 4.0, 6.0, 10.0],
    "equity_base_init": [1.0, 3.0, 5.0, 10.0],
    "savings_init": [1.0, 3.0, 5.0, 10.0],
    # New firm entry
    "new_firm_size_factor": [0.5, 0.7, 0.8, 0.9],
    "new_firm_production_factor": [0.5, 0.7, 0.8, 0.9, 1.0],
    "new_firm_wage_factor": [0.5, 0.7, 0.8, 0.9, 1.0],
    "new_firm_price_markup": [1.0, 1.05, 1.10, 1.25, 1.50],
    # Economy-wide
    "beta": [0.5, 1.0, 2.5, 5.0, 10.0],
    # Search frictions
    "max_M": [2, 4],
    # Implementation variants
    "max_loan_to_net_worth": [0, 2, 5, 10],
    "max_leverage": [0, 5, 10, 20],
    "job_search_method": ["vacancies_only", "all_firms"],
}

# =============================================================================
# Scenario-specific parameter grids
# =============================================================================

PARAMETER_GRIDS: dict[str, dict[str, list[Any]]] = {
    "baseline": {**_COMMON_GRID},
    "growth_plus": {
        **_COMMON_GRID,
        # R&D extension parameters
        "sigma_decay": [-2.0, -1.5, -1.0, -0.5],
    },
    "buffer_stock": {
        **_COMMON_GRID,
        # Buffer-stock extension parameter
        "buffer_stock_h": [1.0, 1.5, 2.0, 3.0, 4.0],
        # R&D extension parameters (buffer_stock includes R&D)
        "sigma_decay": [-2.0, -1.5, -1.0, -0.5],
    },
}

# =============================================================================
# Scenario-specific parameter overrides (sensitivity baseline)
# =============================================================================

# For baseline: empty dict means engine defaults.yml are used for all params.
# For extensions: explicit overrides for extension-specific defaults.
SCENARIO_OVERRIDES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "growth_plus": {
        "new_firm_price_markup": 1.15,
        "sigma_decay": -1.0,
        "sigma_max": 0.1,
    },
    "buffer_stock": {
        "buffer_stock_h": 2.0,
        "sigma_decay": -1.0,
        "sigma_max": 0.1,
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
        Scenario name ("baseline", "growth_plus", or "buffer_stock").

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
        Scenario name ("baseline", "growth_plus", or "buffer_stock").

    Returns
    -------
    dict
        Scenario overrides. For baseline, returns empty dict (engine defaults).
        For extensions, returns extension-specific parameter defaults.

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
