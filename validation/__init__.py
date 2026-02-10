"""Validation package for BAM Engine.

This package provides tools for validating simulation results against
targets derived from Delli Gatti et al. (2011).

Subpackages:
    scenarios/: Scenario definitions and visualizations
        - baseline.py: Baseline scenario (Section 3.9.1)
        - growth_plus.py: Growth+ scenario (Section 3.9.2)
        - *_viz.py: Visualization modules
    targets/: YAML files defining target values

Modules:
    types: Core types, dataclasses, and enums
    scoring: Scoring and status check functions
    engine: Generic validation engine
    reporting: Report printing functions

Usage:
    from validation import run_validation, run_stability_test

    # Compare different configurations (baseline)
    score_a = run_validation(seed=0)
    score_b = run_validation(seed=0, h_rho=0.15)
    print(f"Default: {score_a.total_score:.3f}")
    print(f"Modified: {score_b.total_score:.3f}")

    # Test stability across seeds
    result = run_stability_test(seeds=[0, 42, 123, 456, 789])
    print(f"Mean score: {result.mean_score:.3f} ± {result.std_score:.3f}")

    # Growth+ scenario
    from validation import run_growth_plus_validation
    score = run_growth_plus_validation(seed=0)
    print(f"Growth+ score: {score.total_score:.3f}")

    # Buffer-stock scenario
    from validation import run_buffer_stock_validation
    score = run_buffer_stock_validation(seed=0)
    print(f"Buffer-stock score: {score.total_score:.3f}")

    # Run scenarios with visualization
    from validation import run_baseline_scenario, run_growth_plus_scenario
    run_baseline_scenario(seed=0, show_plot=True)
    run_growth_plus_scenario(seed=2, show_plot=True)

    # For extensions, import from extensions package:
    from extensions.rnd import RnD, RND_EVENTS
    from extensions.buffer_stock import BufferStock, BUFFER_STOCK_EVENTS
"""

from __future__ import annotations

from typing import Any

# Engine functions
from validation.engine import evaluate_metric, load_targets, stability_test, validate

# Reporting functions
from validation.reporting import (
    print_baseline_stability_report,
    print_buffer_stock_report,
    print_buffer_stock_stability_report,
    print_growth_plus_report,
    print_growth_plus_stability_report,
    print_report,
    print_stability_report,
    print_validation_report,
)

# Scenario-specific exports (lazy loaded to avoid circular imports)
from validation.scenarios.baseline import COLLECT_CONFIG as BASELINE_COLLECT_CONFIG
from validation.scenarios.baseline import DEFAULT_CONFIG as BASELINE_DEFAULT_CONFIG
from validation.scenarios.baseline import METRIC_SPECS as BASELINE_METRIC_SPECS
from validation.scenarios.baseline import SCENARIO as BASELINE_SCENARIO
from validation.scenarios.baseline import (
    BaselineMetrics,
    compute_baseline_metrics,
    load_baseline_targets,
)

# Buffer-stock scenario
from validation.scenarios.buffer_stock import (
    COLLECT_CONFIG as BUFFER_STOCK_COLLECT_CONFIG,
)
from validation.scenarios.buffer_stock import (
    DEFAULT_CONFIG as BUFFER_STOCK_DEFAULT_CONFIG,
)
from validation.scenarios.buffer_stock import METRIC_SPECS as BUFFER_STOCK_METRIC_SPECS
from validation.scenarios.buffer_stock import SCENARIO as BUFFER_STOCK_SCENARIO
from validation.scenarios.buffer_stock import (
    BufferStockMetrics,
    compute_buffer_stock_metrics,
    load_buffer_stock_targets,
)
from validation.scenarios.growth_plus import (
    COLLECT_CONFIG as GROWTH_PLUS_COLLECT_CONFIG,
)
from validation.scenarios.growth_plus import (
    DEFAULT_CONFIG as GROWTH_PLUS_DEFAULT_CONFIG,
)
from validation.scenarios.growth_plus import METRIC_SPECS as GROWTH_PLUS_METRIC_SPECS
from validation.scenarios.growth_plus import SCENARIO as GROWTH_PLUS_SCENARIO
from validation.scenarios.growth_plus import (
    GrowthPlusMetrics,
    compute_growth_plus_metrics,
    load_growth_plus_targets,
)

# Scoring functions
from validation.scoring import (
    check_mean_tolerance,
    check_outlier_penalty,
    check_pct_within_target,
    check_range,
    compute_combined_score,
    score_mean_tolerance,
    score_outlier_penalty,
    score_pct_within_target,
    score_range,
)

# Core types
from validation.types import (
    DEFAULT_STABILITY_SEEDS,
    CheckType,
    MetricFormat,
    MetricGroup,
    MetricResult,
    MetricSpec,
    MetricStats,
    Scenario,
    StabilityResult,
    Status,
    ValidationScore,
)

# =============================================================================
# Weights (derived from MetricSpecs for backwards compatibility)
# =============================================================================


def _derive_weights(specs: list[MetricSpec]) -> dict[str, float]:
    """Derive weights dictionary from MetricSpecs."""
    return {spec.name: spec.weight for spec in specs}


BASELINE_WEIGHTS = _derive_weights(BASELINE_METRIC_SPECS)
GROWTH_PLUS_WEIGHTS = _derive_weights(GROWTH_PLUS_METRIC_SPECS)
BUFFER_STOCK_WEIGHTS = _derive_weights(BUFFER_STOCK_METRIC_SPECS)


# =============================================================================
# Thin Wrapper Functions (for backwards compatibility)
# =============================================================================


def run_validation(
    *,
    seed: int = 0,
    n_periods: int = 1000,
    **config_overrides: Any,
) -> ValidationScore:
    """Run baseline validation and return scored result.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_periods : int
        Number of simulation periods.
    **config_overrides
        Any simulation config parameters to override.

    Returns
    -------
    ValidationScore
        Validation result with total score and per-metric results.
    """
    return validate(
        BASELINE_SCENARIO,
        seed=seed,
        n_periods=n_periods,
        **config_overrides,
    )


def run_stability_test(
    seeds: list[int] | int = 5,
    n_periods: int = 1000,
    **config_overrides: Any,
) -> StabilityResult:
    """Run baseline validation across multiple seeds.

    Parameters
    ----------
    seeds : list[int] or int
        List of seeds or number of seeds to test.
    n_periods : int
        Number of simulation periods per seed.
    **config_overrides
        Any simulation config parameters to override.

    Returns
    -------
    StabilityResult
        Aggregated results across all seeds.
    """
    return stability_test(
        BASELINE_SCENARIO,
        seeds=seeds,
        n_periods=n_periods,
        **config_overrides,
    )


def run_growth_plus_validation(
    *,
    seed: int = 0,
    n_periods: int = 1000,
    **config_overrides: Any,
) -> ValidationScore:
    """Run Growth+ validation and return scored result.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_periods : int
        Number of simulation periods.
    **config_overrides
        Any simulation config parameters to override.

    Returns
    -------
    ValidationScore
        Validation result with total score and per-metric results.
    """
    return validate(
        GROWTH_PLUS_SCENARIO,
        seed=seed,
        n_periods=n_periods,
        **config_overrides,
    )


def run_growth_plus_stability_test(
    seeds: list[int] | int = 5,
    n_periods: int = 1000,
    **config_overrides: Any,
) -> StabilityResult:
    """Run Growth+ validation across multiple seeds.

    Parameters
    ----------
    seeds : list[int] or int
        List of seeds or number of seeds to test.
    n_periods : int
        Number of simulation periods per seed.
    **config_overrides
        Any simulation config parameters to override.

    Returns
    -------
    StabilityResult
        Aggregated results across all seeds.
    """
    return stability_test(
        GROWTH_PLUS_SCENARIO,
        seeds=seeds,
        n_periods=n_periods,
        **config_overrides,
    )


# =============================================================================
# Scenario Runner Functions (with visualization)
# =============================================================================


def run_baseline_scenario(**kwargs: Any) -> BaselineMetrics:
    """Run baseline scenario with visualization.

    See validation.scenarios.baseline.run_scenario for parameters.
    """
    from validation.scenarios.baseline import run_scenario

    return run_scenario(**kwargs)


def run_buffer_stock_validation(
    *,
    seed: int = 0,
    n_periods: int = 1000,
    **config_overrides: Any,
) -> ValidationScore:
    """Run buffer-stock validation and return scored result.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_periods : int
        Number of simulation periods.
    **config_overrides
        Any simulation config parameters to override.

    Returns
    -------
    ValidationScore
        Validation result with total score and per-metric results.
    """
    return validate(
        BUFFER_STOCK_SCENARIO,
        seed=seed,
        n_periods=n_periods,
        **config_overrides,
    )


def run_buffer_stock_stability_test(
    seeds: list[int] | int = 5,
    n_periods: int = 1000,
    **config_overrides: Any,
) -> StabilityResult:
    """Run buffer-stock validation across multiple seeds.

    Parameters
    ----------
    seeds : list[int] or int
        List of seeds or number of seeds to test.
    n_periods : int
        Number of simulation periods per seed.
    **config_overrides
        Any simulation config parameters to override.

    Returns
    -------
    StabilityResult
        Aggregated results across all seeds.
    """
    return stability_test(
        BUFFER_STOCK_SCENARIO,
        seeds=seeds,
        n_periods=n_periods,
        **config_overrides,
    )


def run_growth_plus_scenario(**kwargs: Any) -> GrowthPlusMetrics:
    """Run Growth+ scenario with visualization.

    See validation.scenarios.growth_plus.run_scenario for parameters.
    """
    from validation.scenarios.growth_plus import run_scenario

    return run_scenario(**kwargs)


def run_buffer_stock_scenario(**kwargs: Any) -> BufferStockMetrics:
    """Run buffer-stock scenario with visualization.

    See validation.scenarios.buffer_stock.run_scenario for parameters.
    """
    from validation.scenarios.buffer_stock import run_scenario

    return run_scenario(**kwargs)


# =============================================================================
# Calibration Package Support
# =============================================================================


def get_validation_funcs(
    scenario: str = "baseline",
) -> tuple[Any, Any, Any, Any]:
    """Get validation functions for a scenario (for calibration).

    Parameters
    ----------
    scenario : str
        Either "baseline" or "growth_plus".

    Returns
    -------
    tuple
        (run_validation_func, run_stability_func, print_report_func, print_stability_func)
    """
    if scenario == "baseline":
        return (
            run_validation,
            run_stability_test,
            print_validation_report,
            print_baseline_stability_report,
        )
    elif scenario == "growth_plus":
        return (
            run_growth_plus_validation,
            run_growth_plus_stability_test,
            print_growth_plus_report,
            print_growth_plus_stability_report,
        )
    elif scenario == "buffer_stock":
        return (
            run_buffer_stock_validation,
            run_buffer_stock_stability_test,
            print_buffer_stock_report,
            print_buffer_stock_stability_report,
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def get_validation_func(scenario: str = "baseline") -> Any:
    """Get the validation function for a scenario.

    Parameters
    ----------
    scenario : str
        Either "baseline" or "growth_plus".

    Returns
    -------
    Callable
        The run_validation function for the scenario.
    """
    return get_validation_funcs(scenario)[0]


__all__ = [
    # Core types
    "Status",
    "CheckType",
    "MetricGroup",
    "MetricFormat",
    "MetricSpec",
    "MetricResult",
    "ValidationScore",
    "MetricStats",
    "StabilityResult",
    "Scenario",
    # Constants
    "DEFAULT_STABILITY_SEEDS",
    "BASELINE_WEIGHTS",
    "GROWTH_PLUS_WEIGHTS",
    # Scoring functions
    "score_mean_tolerance",
    "score_range",
    "score_pct_within_target",
    "score_outlier_penalty",
    "check_mean_tolerance",
    "check_range",
    "check_pct_within_target",
    "check_outlier_penalty",
    "compute_combined_score",
    # Engine functions
    "validate",
    "stability_test",
    "evaluate_metric",
    "load_targets",
    # Report functions
    "print_report",
    "print_stability_report",
    "print_validation_report",
    "print_baseline_stability_report",
    "print_growth_plus_report",
    "print_growth_plus_stability_report",
    "print_buffer_stock_report",
    "print_buffer_stock_stability_report",
    # Baseline scenario
    "BASELINE_SCENARIO",
    "BASELINE_COLLECT_CONFIG",
    "BASELINE_DEFAULT_CONFIG",
    "BASELINE_METRIC_SPECS",
    "BaselineMetrics",
    "compute_baseline_metrics",
    "load_baseline_targets",
    # Growth+ scenario
    "GROWTH_PLUS_SCENARIO",
    "GROWTH_PLUS_COLLECT_CONFIG",
    "GROWTH_PLUS_DEFAULT_CONFIG",
    "GROWTH_PLUS_METRIC_SPECS",
    "GrowthPlusMetrics",
    "compute_growth_plus_metrics",
    "load_growth_plus_targets",
    # Buffer-stock scenario
    "BUFFER_STOCK_SCENARIO",
    "BUFFER_STOCK_COLLECT_CONFIG",
    "BUFFER_STOCK_DEFAULT_CONFIG",
    "BUFFER_STOCK_METRIC_SPECS",
    "BUFFER_STOCK_WEIGHTS",
    "BufferStockMetrics",
    "compute_buffer_stock_metrics",
    "load_buffer_stock_targets",
    # Wrapper functions
    "run_validation",
    "run_stability_test",
    "run_growth_plus_validation",
    "run_growth_plus_stability_test",
    "run_buffer_stock_validation",
    "run_buffer_stock_stability_test",
    # Scenario visualization
    "run_baseline_scenario",
    "run_growth_plus_scenario",
    "run_buffer_stock_scenario",
    # Calibration support
    "get_validation_funcs",
    "get_validation_func",
]
