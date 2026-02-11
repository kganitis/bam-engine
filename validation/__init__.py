"""Validation package for BAM Engine.

This package provides tools for validating simulation results against
targets derived from Delli Gatti et al. (2011).

Subpackages:
    scenarios/: Scenario definitions and visualizations
        - baseline/: Baseline scenario (Section 3.9.1)
        - growth_plus/: Growth+ scenario (Section 3.9.2)
        - buffer_stock/: Buffer-stock scenario (Section 3.9.4)

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
    from extensions.rnd import RnD, RND_EVENTS, RND_CONFIG
    from extensions.buffer_stock import BufferStock, BUFFER_STOCK_EVENTS, BUFFER_STOCK_CONFIG
"""

from __future__ import annotations

from functools import partial
from typing import Any

# Engine functions
from validation.engine import evaluate_metric, load_targets, stability_test, validate

# Reporting functions (generic)
from validation.reporting import print_report, print_stability_report

# Scenario registry
from validation.scenarios import get_scenario

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


BASELINE_WEIGHTS = _derive_weights(get_scenario("baseline").metric_specs)
GROWTH_PLUS_WEIGHTS = _derive_weights(get_scenario("growth_plus").metric_specs)
BUFFER_STOCK_WEIGHTS = _derive_weights(get_scenario("buffer_stock").metric_specs)


# =============================================================================
# Factory-generated wrapper functions (backwards-compatible)
# =============================================================================


def _make_validate(scenario_name: str):  # type: ignore[no-untyped-def]
    def _validate(
        *, seed: int = 0, n_periods: int = 1000, **config_overrides: Any
    ) -> ValidationScore:
        return validate(
            get_scenario(scenario_name),
            seed=seed,
            n_periods=n_periods,
            **config_overrides,
        )

    _validate.__doc__ = f"Run {scenario_name} validation and return scored result."
    return _validate


def _make_stability(scenario_name: str):  # type: ignore[no-untyped-def]
    def _stability(
        seeds: list[int] | int = 5, n_periods: int = 1000, **config_overrides: Any
    ) -> StabilityResult:
        return stability_test(
            get_scenario(scenario_name),
            seeds=seeds,
            n_periods=n_periods,
            **config_overrides,
        )

    _stability.__doc__ = f"Run {scenario_name} validation across multiple seeds."
    return _stability


# Backwards-compatible names
run_validation = _make_validate("baseline")
run_stability_test = _make_stability("baseline")
run_growth_plus_validation = _make_validate("growth_plus")
run_growth_plus_stability_test = _make_stability("growth_plus")
run_buffer_stock_validation = _make_validate("buffer_stock")
run_buffer_stock_stability_test = _make_stability("buffer_stock")

# =============================================================================
# Report printers (generated via partial from Scenario.title)
# =============================================================================

print_validation_report = partial(print_report, title=get_scenario("baseline").title)
print_baseline_stability_report = partial(
    print_stability_report, title=get_scenario("baseline").stability_title
)
print_growth_plus_report = partial(
    print_report, title=get_scenario("growth_plus").title
)
print_growth_plus_stability_report = partial(
    print_stability_report, title=get_scenario("growth_plus").stability_title
)
print_buffer_stock_report = partial(
    print_report, title=get_scenario("buffer_stock").title
)
print_buffer_stock_stability_report = partial(
    print_stability_report, title=get_scenario("buffer_stock").stability_title
)


# =============================================================================
# Scenario Runner Functions (with visualization — lazy imports)
# =============================================================================


def run_baseline_scenario(**kwargs: Any) -> Any:
    """Run baseline scenario with visualization.

    See validation.scenarios.baseline.run_scenario for parameters.
    """
    from validation.scenarios.baseline import run_scenario

    return run_scenario(**kwargs)


def run_growth_plus_scenario(**kwargs: Any) -> Any:
    """Run Growth+ scenario with visualization.

    See validation.scenarios.growth_plus.run_scenario for parameters.
    """
    from validation.scenarios.growth_plus import run_scenario

    return run_scenario(**kwargs)


def run_buffer_stock_scenario(**kwargs: Any) -> Any:
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
        Scenario name (e.g. "baseline", "growth_plus", "buffer_stock").

    Returns
    -------
    tuple
        (run_validation_func, run_stability_func, print_report_func, print_stability_func)
    """
    s = get_scenario(scenario)
    return (
        _make_validate(scenario),
        _make_stability(scenario),
        partial(print_report, title=s.title),
        partial(print_stability_report, title=s.stability_title),
    )


def get_validation_func(scenario: str = "baseline") -> Any:
    """Get the validation function for a scenario.

    Parameters
    ----------
    scenario : str
        Scenario name (e.g. "baseline", "growth_plus", "buffer_stock").

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
    "BUFFER_STOCK_WEIGHTS",
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
    # Registry
    "get_scenario",
]
