"""One-At-a-Time (OAT) sensitivity analysis.

This module provides sensitivity analysis functionality to identify which
parameters have the most impact on validation scores.

Supports multiple scenarios:
    - baseline: Standard BAM model (Section 3.9.1)
    - growth_plus: Endogenous productivity growth via R&D (Section 3.8)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from calibration.parameter_space import get_default_values, get_parameter_grid
from validation import get_validation_func


@dataclass
class ParameterSensitivity:
    """Sensitivity result for a single parameter.

    Attributes
    ----------
    name : str
        Parameter name.
    values : list
        All values tested for this parameter.
    scores : list[float]
        Validation scores for each value.
    best_value : Any
        Value that produced the highest score.
    best_score : float
        Highest score achieved.
    sensitivity : float
        Score range (max - min), indicating parameter importance.
    """

    name: str
    values: list[Any]
    scores: list[float]
    best_value: Any
    best_score: float
    sensitivity: float


@dataclass
class SensitivityResult:
    """Full sensitivity analysis result.

    Attributes
    ----------
    parameters : list[ParameterSensitivity]
        Sensitivity results for all parameters.
    baseline_score : float
        Score with all default values.
    scenario : str
        The scenario that was analyzed.
    """

    parameters: list[ParameterSensitivity]
    baseline_score: float
    scenario: str = "baseline"

    @property
    def ranked(self) -> list[ParameterSensitivity]:
        """Parameters ranked by sensitivity (highest first)."""
        return sorted(self.parameters, key=lambda p: p.sensitivity, reverse=True)

    def get_important(
        self,
        high_threshold: float = 0.05,
        medium_threshold: float = 0.02,
    ) -> tuple[list[str], list[str], list[str]]:
        """Categorize parameters by sensitivity.

        Parameters
        ----------
        high_threshold : float
            Minimum sensitivity for HIGH category.
        medium_threshold : float
            Minimum sensitivity for MEDIUM category.

        Returns
        -------
        tuple[list[str], list[str], list[str]]
            (high, medium, low) parameter name lists.
        """
        high, medium, low = [], [], []
        for p in self.parameters:
            if p.sensitivity > high_threshold:
                high.append(p.name)
            elif p.sensitivity > medium_threshold:
                medium.append(p.name)
            else:
                low.append(p.name)
        return high, medium, low


def _evaluate_param_value(
    param_name: str,
    value: Any,
    baseline: dict[str, Any],
    scenario: str,
    seed: int,
    n_periods: int,
) -> tuple[Any, float]:
    """Evaluate a single parameter value.

    This is a standalone function for use with ProcessPoolExecutor.
    """
    config = baseline.copy()
    config[param_name] = value
    validate = get_validation_func(scenario)
    result = validate(seed=seed, n_periods=n_periods, **config)
    return value, result.total_score


def run_sensitivity_analysis(
    scenario: str = "baseline",
    grid: dict[str, list[Any]] | None = None,
    baseline: dict[str, Any] | None = None,
    seed: int = 0,
    n_periods: int = 1000,
    n_workers: int = 10,
) -> SensitivityResult:
    """Run OAT sensitivity analysis.

    Tests each parameter independently while holding others at baseline values.

    Parameters
    ----------
    scenario : str
        Scenario to calibrate ("baseline" or "growth_plus").
    grid : dict, optional
        Parameter grid. Defaults to scenario-specific grid.
    baseline : dict, optional
        Baseline parameter values. Defaults to scenario-specific defaults.
    seed : int
        Random seed for reproducibility.
    n_periods : int
        Number of simulation periods.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    SensitivityResult
        Sensitivity ranking of all parameters.
    """
    if grid is None:
        grid = get_parameter_grid(scenario)
    if baseline is None:
        baseline = get_default_values(scenario).copy()

    validate = get_validation_func(scenario)

    # Get baseline score first
    baseline_result = validate(seed=seed, n_periods=n_periods, **baseline)
    baseline_score = baseline_result.total_score
    print(f"[{scenario}] Baseline score: {baseline_score:.4f}")

    # Test each parameter
    results: list[ParameterSensitivity] = []

    total_runs = sum(len(values) for values in grid.values())
    completed = 0

    for param_name, values in grid.items():
        print(f"Testing {param_name}...")

        # Run all values for this parameter in parallel
        param_scores: list[tuple[Any, float]] = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_param_value,
                    param_name,
                    value,
                    baseline,
                    scenario,
                    seed,
                    n_periods,
                )
                for value in values
            ]
            for future in as_completed(futures):
                val, score = future.result()
                param_scores.append((val, score))
                completed += 1
                print(f"  [{completed}/{total_runs}] {param_name}={val}: {score:.4f}")

        # Sort by value order
        param_scores.sort(key=lambda x: values.index(x[0]))
        vals = [x[0] for x in param_scores]
        scores = [x[1] for x in param_scores]

        best_idx = scores.index(max(scores))
        sensitivity = max(scores) - min(scores)

        results.append(
            ParameterSensitivity(
                name=param_name,
                values=vals,
                scores=scores,
                best_value=vals[best_idx],
                best_score=scores[best_idx],
                sensitivity=sensitivity,
            )
        )

    return SensitivityResult(
        parameters=results,
        baseline_score=baseline_score,
        scenario=scenario,
    )


def print_sensitivity_report(result: SensitivityResult) -> None:
    """Print formatted sensitivity analysis report.

    Parameters
    ----------
    result : SensitivityResult
        Result from run_sensitivity_analysis().
    """
    print("\n" + "=" * 70)
    print(f"SENSITIVITY ANALYSIS RESULTS ({result.scenario})")
    print("=" * 70)
    print(f"\nBaseline score: {result.baseline_score:.4f}")

    print(
        f"\n{'Parameter':<30} {'Sensitivity':>12} {'Best Value':>12} {'Best Score':>10}"
    )
    print("-" * 70)

    for p in result.ranked:
        print(
            f"{p.name:<30} {p.sensitivity:>12.4f} {p.best_value!s:>12} {p.best_score:>10.4f}"
        )

    high, medium, low = result.get_important()

    print("\n" + "=" * 70)
    print("PARAMETER IMPORTANCE")
    print("=" * 70)
    print(f"HIGH (Δ > 0.05):    {', '.join(high) or 'None'}")
    print(f"MEDIUM (0.02-0.05): {', '.join(medium) or 'None'}")
    print(f"LOW (Δ ≤ 0.02):     {', '.join(low) or 'None'}")
    print("=" * 70 + "\n")
