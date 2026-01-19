"""Parameter space definition for calibration.

This module defines the parameter grid for calibration and the default
values used as a baseline for sensitivity analysis.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Any

# Full parameter grid definition
PARAMETER_GRID: dict[str, list[Any]] = {
    # New firm entry parameters
    "new_firm_size_factor": [0.5, 0.7, 0.9],
    "new_firm_production_factor": [0.5, 0.75, 1.0],
    "new_firm_wage_factor": [0.5, 0.75, 1.0],
    "new_firm_price_markup": [1.0, 1.1, 1.25, 1.50],
    # Credit market parameters
    "max_loan_to_net_worth": [2, 10, 100],
    "max_leverage": [2, 10, 100],
    # Behavioral variants
    "firing_method": ["expensive", "random"],
    "matching_method": ["sequential", "simultaneous"],
    "job_search_method": ["vacancies_only", "all_firms"],
    # Price dynamics
    "cap_factor": [1.1, 2, 10, 100],
}

# Default values (from defaults.yml) for sensitivity baseline
DEFAULT_VALUES: dict[str, Any] = {
    "new_firm_size_factor": 0.9,
    "new_firm_production_factor": 0.9,
    "new_firm_wage_factor": 0.5,
    "new_firm_price_markup": 1.50,
    "max_loan_to_net_worth": 10,
    "max_leverage": 10,
    "firing_method": "expensive",
    "matching_method": "simultaneous",
    "job_search_method": "vacancies_only",
    "cap_factor": 10,
}


def generate_combinations(
    grid: dict[str, list[Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Generate all parameter combinations from grid.

    Parameters
    ----------
    grid : dict, optional
        Parameter grid to use. Defaults to PARAMETER_GRID.

    Yields
    ------
    dict
        Dictionary mapping parameter names to values.
    """
    if grid is None:
        grid = PARAMETER_GRID
    keys = list(grid.keys())
    for values in product(*grid.values()):
        yield dict(zip(keys, values, strict=True))


def count_combinations(grid: dict[str, list[Any]] | None = None) -> int:
    """Count total combinations in grid.

    Parameters
    ----------
    grid : dict, optional
        Parameter grid to use. Defaults to PARAMETER_GRID.

    Returns
    -------
    int
        Number of combinations in the grid.
    """
    if grid is None:
        grid = PARAMETER_GRID
    count = 1
    for values in grid.values():
        count *= len(values)
    return count
