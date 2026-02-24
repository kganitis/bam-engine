"""Grid building, loading, validation, and combination generation.

This module handles parameter grid operations:
- Building focused grids from sensitivity analysis results
- Loading grids from YAML/JSON files
- Validating grid structure
- Generating and counting parameter combinations
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from calibration.parameter_space import get_parameter_grid
from calibration.sensitivity import SensitivityResult


def build_focused_grid(
    sensitivity: SensitivityResult,
    full_grid: dict[str, list[Any]] | None = None,
    scenario: str = "baseline",
    sensitivity_threshold: float = 0.02,
    pruning_threshold: float | None = 0.04,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Build focused grid from sensitivity analysis.

    Parameters
    ----------
    sensitivity : SensitivityResult
        Result from run_sensitivity_analysis().
    full_grid : dict, optional
        Full parameter grid. Defaults to scenario-specific grid.
    scenario : str
        Scenario name.
    sensitivity_threshold : float
        Minimum sensitivity (delta) for inclusion in grid search.
    pruning_threshold : float or None
        Maximum score gap from best value for keeping a grid value.
        ``None`` disables pruning.

    Returns
    -------
    tuple[dict, dict]
        (grid_to_search, fixed_params)
        - INCLUDE params (delta > threshold): all grid values (pruned if enabled)
        - FIX params (delta <= threshold): fix at best value
    """
    if full_grid is None:
        full_grid = get_parameter_grid(scenario)

    included, _ = sensitivity.get_important(sensitivity_threshold)
    param_best = {p.name: p.best_value for p in sensitivity.parameters}

    grid_to_search: dict[str, list[Any]] = {}
    fixed_params: dict[str, Any] = {}

    for name, values in full_grid.items():
        if name in included:
            grid_to_search[name] = values
        else:
            fixed_params[name] = param_best[name]

    grid_to_search = sensitivity.prune_grid(grid_to_search, pruning_threshold)

    return grid_to_search, fixed_params


def load_grid(path: Path) -> dict[str, list[Any]]:
    """Load parameter grid from YAML/JSON file.

    Light validation: check dict-of-lists structure, warn about empty values.
    Supports both .yaml/.yml and .json extensions.

    Parameters
    ----------
    path : Path
        Path to grid file.

    Returns
    -------
    dict[str, list[Any]]
        Parameter grid (param_name -> list of values).

    Raises
    ------
    ValueError
        If the file contents are not a dict-of-lists structure.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Grid file not found: {path}")

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
    else:
        with open(path) as f:
            data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Grid file must contain a dict, got {type(data).__name__}")

    # Validate and normalize: ensure all values are lists
    grid: dict[str, list[Any]] = {}
    for key, values in data.items():
        if not isinstance(values, list):
            raise ValueError(
                f"Grid values for '{key}' must be a list, got {type(values).__name__}"
            )
        grid[key] = values

    warnings = validate_grid(grid)
    for w in warnings:
        print(f"  Warning: {w}")

    return grid


def validate_grid(grid: dict[str, list[Any]]) -> list[str]:
    """Light validation of grid structure.

    Parameters
    ----------
    grid : dict[str, list[Any]]
        Parameter grid to validate.

    Returns
    -------
    list[str]
        List of warnings (empty = OK).
    """
    warnings: list[str] = []
    for name, values in grid.items():
        if not values:
            warnings.append(f"Parameter '{name}' has empty values list")
        elif len(values) == 1:
            warnings.append(f"Parameter '{name}' has only one value: {values[0]}")
    return warnings


def count_combinations(grid: dict[str, list[Any]]) -> int:
    """Count total combinations in grid.

    Parameters
    ----------
    grid : dict[str, list[Any]]
        Parameter grid.

    Returns
    -------
    int
        Number of combinations in the grid.
    """
    count = 1
    for values in grid.values():
        count *= len(values)
    return count


def generate_combinations(
    grid: dict[str, list[Any]],
    fixed: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Generate all parameter combinations, merged with fixed params.

    Parameters
    ----------
    grid : dict[str, list[Any]]
        Parameter grid to generate combinations from.
    fixed : dict, optional
        Fixed parameter values to merge into each combination.

    Yields
    ------
    dict[str, Any]
        Dictionary mapping parameter names to values.
    """
    keys = list(grid.keys())
    fixed = fixed or {}
    for values in product(*grid.values()):
        combo = dict(zip(keys, values, strict=True))
        if fixed:
            yield {**fixed, **combo}
        else:
            yield combo
