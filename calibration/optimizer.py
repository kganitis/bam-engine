"""Main optimization logic using validation infrastructure.

This module provides the core calibration functionality, including
grid building based on sensitivity analysis and focused grid search.

Supports multiple scenarios:
    - baseline: Standard BAM model (Section 3.9.1)
    - growth_plus: Endogenous productivity growth via R&D (Section 3.9.2)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from calibration.parameter_space import (
    count_combinations,
    generate_combinations,
    get_parameter_grid,
)
from calibration.sensitivity import SensitivityResult
from validation import (
    DEFAULT_STABILITY_SEEDS,
    StabilityResult,
    compute_combined_score,
    get_validation_funcs,
)


@dataclass
class CalibrationResult:
    """Result from calibration optimization.

    Attributes
    ----------
    params : dict
        Parameter configuration.
    single_score : float
        Validation score from single-seed run.
    n_pass : int
        Number of metrics that passed.
    n_warn : int
        Number of metrics with warnings.
    n_fail : int
        Number of metrics that failed.
    mean_score : float, optional
        Mean score across stability seeds.
    std_score : float, optional
        Standard deviation of scores across seeds.
    pass_rate : float, optional
        Fraction of seeds that passed (no FAIL metrics).
    combined_score : float, optional
        Combined score balancing accuracy and stability.
    stability_result : StabilityResult, optional
        Full stability test result.
    """

    params: dict[str, Any]
    single_score: float
    n_pass: int
    n_warn: int
    n_fail: int
    mean_score: float | None = None
    std_score: float | None = None
    pass_rate: float | None = None
    combined_score: float | None = None
    stability_result: StabilityResult | None = None


def build_focused_grid(
    sensitivity: SensitivityResult,
    full_grid: dict[str, list[Any]] | None = None,
    scenario: str = "baseline",
    high_threshold: float = 0.05,
    medium_threshold: float = 0.02,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Build focused grid from sensitivity analysis.

    Parameters
    ----------
    sensitivity : SensitivityResult
        Result from run_sensitivity_analysis().
    full_grid : dict, optional
        Full parameter grid. Defaults to scenario-specific grid.
    scenario : str
        Scenario name ("baseline" or "growth_plus").
    high_threshold : float
        Sensitivity threshold for HIGH importance.
    medium_threshold : float
        Sensitivity threshold for MEDIUM importance.

    Returns
    -------
    tuple[dict, dict]
        (grid_to_search, fixed_params)
        - HIGH sensitivity params: Keep all values in grid
        - MEDIUM sensitivity params: Keep only min/max values
        - LOW sensitivity params: Fix at best value
    """
    if full_grid is None:
        full_grid = get_parameter_grid(scenario)

    high, medium, _low = sensitivity.get_important(high_threshold, medium_threshold)
    param_best = {p.name: p.best_value for p in sensitivity.parameters}

    grid_to_search: dict[str, list[Any]] = {}
    fixed_params: dict[str, Any] = {}

    for name, values in full_grid.items():
        if name in high:
            grid_to_search[name] = values
        elif name in medium:
            # Keep only min and max
            grid_to_search[name] = [values[0], values[-1]]
        else:  # low
            fixed_params[name] = param_best[name]

    return grid_to_search, fixed_params


def screen_single_seed(
    params: dict[str, Any],
    scenario: str,
    seed: int,
    n_periods: int,
) -> CalibrationResult:
    """Run single-seed validation for quick screening.

    Parameters
    ----------
    params : dict
        Parameter configuration.
    scenario : str
        Scenario name ("baseline" or "growth_plus").
    seed : int
        Random seed.
    n_periods : int
        Number of simulation periods.

    Returns
    -------
    CalibrationResult
        Result with single-seed metrics.
    """
    validate, _ = get_validation_funcs(scenario)
    result = validate(seed=seed, n_periods=n_periods, **params)
    return CalibrationResult(
        params=params,
        single_score=result.total_score,
        n_pass=result.n_pass,
        n_warn=result.n_warn,
        n_fail=result.n_fail,
    )


def evaluate_stability(
    params: dict[str, Any],
    scenario: str,
    seeds: list[int],
    n_periods: int,
) -> CalibrationResult:
    """Run multi-seed stability test for full evaluation.

    Parameters
    ----------
    params : dict
        Parameter configuration.
    scenario : str
        Scenario name ("baseline" or "growth_plus").
    seeds : list[int]
        List of random seeds to test.
    n_periods : int
        Number of simulation periods.

    Returns
    -------
    CalibrationResult
        Result with stability metrics and combined score.
    """
    _, run_stability = get_validation_funcs(scenario)
    stability = run_stability(seeds=seeds, n_periods=n_periods, **params)
    combined = compute_combined_score(stability)
    return CalibrationResult(
        params=params,
        single_score=stability.seed_results[0].total_score,
        n_pass=stability.seed_results[0].n_pass,
        n_warn=stability.seed_results[0].n_warn,
        n_fail=stability.seed_results[0].n_fail,
        mean_score=stability.mean_score,
        std_score=stability.std_score,
        pass_rate=stability.pass_rate,
        combined_score=combined,
        stability_result=stability,
    )


def run_focused_calibration(
    grid: dict[str, list[Any]],
    fixed_params: dict[str, Any],
    scenario: str = "baseline",
    top_k: int = 20,
    n_workers: int = 10,
    stability_seeds: list[int] | None = None,
    n_periods: int = 1000,
) -> list[CalibrationResult]:
    """Run calibration on focused grid with fixed params.

    Parameters
    ----------
    grid : dict
        Parameter grid to search (from build_focused_grid).
    fixed_params : dict
        Fixed parameter values (from build_focused_grid).
    scenario : str
        Scenario name ("baseline" or "growth_plus").
    top_k : int
        Number of top configurations to stability test.
    n_workers : int
        Number of parallel workers.
    stability_seeds : list[int], optional
        Seeds for stability testing. Defaults to DEFAULT_STABILITY_SEEDS.
    n_periods : int
        Number of simulation periods.

    Returns
    -------
    list[CalibrationResult]
        Results sorted by combined_score (best first).
    """
    if stability_seeds is None:
        stability_seeds = DEFAULT_STABILITY_SEEDS

    total = count_combinations(grid)
    print(f"\n[{scenario}] Focused Grid Search: {total} combinations")
    print(f"Fixed params: {fixed_params}")

    # Generate all combinations (merging fixed params)
    combinations = []
    for combo in generate_combinations(grid):
        full_params = {**fixed_params, **combo}
        combinations.append(full_params)

    # Screening phase
    print(f"\nScreening {total} combinations with {n_workers} workers...")
    screening_results: list[CalibrationResult] = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(screen_single_seed, p, scenario, 0, n_periods): p
            for p in combinations
        }
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            screening_results.append(result)
            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"  Screened {i + 1}/{total}")

    # Sort and get top_k
    screening_results.sort(key=lambda r: r.single_score, reverse=True)
    top_results = screening_results[:top_k]

    # Stability testing phase
    print(f"\nStability testing top {top_k} configurations...")
    final_results: list[CalibrationResult] = []

    for i, screened in enumerate(top_results):
        print(f"  Testing {i + 1}/{top_k}: score={screened.single_score:.3f}")
        result = evaluate_stability(
            screened.params, scenario, stability_seeds, n_periods
        )
        final_results.append(result)

    final_results.sort(key=lambda r: r.combined_score or 0, reverse=True)
    return final_results
