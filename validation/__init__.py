"""Validation package for BAM Engine.

This package provides tools for validating simulation results against
targets derived from Delli Gatti et al. (2011).

Subpackages:
    metrics/: Metric computation for validation (baseline.py, growth_plus.py)
    scenarios/: Detailed scenario visualizations (baseline.py, growth_plus.py)
    targets/: YAML files defining target values for different scenarios

Modules:
    core: Shared types, dataclasses, and utility functions
    runners: Validation runner functions for all scenarios

Scenarios:
    - Baseline (Section 3.9.1): Standard BAM model behavior
    - Growth+ (Section 3.8): Endogenous productivity growth via R&D

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

    # Run scenarios with visualization
    from validation import run_baseline_scenario, run_growth_plus_scenario
    run_baseline_scenario(seed=0, show_plot=True)
    run_growth_plus_scenario(seed=2, show_plot=True)

    # For Growth+ extension (RnD role and events), import directly:
    from validation.scenarios.growth_plus_extension import RnD
"""

from __future__ import annotations

from typing import Any

# Core types and utilities
from validation.core import (
    BASELINE_WEIGHTS,
    DEFAULT_STABILITY_SEEDS,
    GROWTH_PLUS_WEIGHTS,
    MetricResult,
    MetricStats,
    StabilityResult,
    Status,
    ValidationScore,
    check_mean_tolerance,
    check_range,
    compute_combined_score,
    get_validation_func,
    get_validation_funcs,
    score_mean_tolerance,
    score_range,
)

# Metrics classes and functions
from validation.metrics import (
    BASELINE_COLLECT_CONFIG,
    GROWTH_PLUS_COLLECT_CONFIG,
    BaselineMetrics,
    GrowthPlusMetrics,
    compute_baseline_metrics,
    compute_growth_plus_metrics,
    load_baseline_targets,
    load_growth_plus_targets,
)

# Validation runners
from validation.runners import (
    print_growth_plus_report,
    print_growth_plus_stability_report,
    print_stability_report,
    print_validation_report,
    run_growth_plus_stability_test,
    run_growth_plus_validation,
    run_stability_test,
    run_validation,
)

# Scenario visualization modules
# Note: These are imported lazily to avoid registering Growth+ events globally.
# Import directly from the modules when needed:
#   from validation.scenarios.growth_plus_extension import RnD
#   from validation.scenarios.baseline import visualize_baseline_results
#   from validation.scenarios.growth_plus import visualize_growth_plus_results


def run_baseline_scenario(**kwargs: Any) -> Any:
    """Run baseline scenario with visualization. See scenarios.baseline.run_scenario."""
    from validation.scenarios.baseline import run_scenario

    return run_scenario(**kwargs)


def run_growth_plus_scenario(**kwargs: Any) -> Any:
    """Run Growth+ scenario with visualization. See scenarios.growth_plus.run_scenario."""
    from validation.scenarios.growth_plus import run_scenario

    return run_scenario(**kwargs)


__all__ = [
    # Core types
    "Status",
    "MetricResult",
    "ValidationScore",
    "MetricStats",
    "StabilityResult",
    # Constants
    "DEFAULT_STABILITY_SEEDS",
    "BASELINE_WEIGHTS",
    "GROWTH_PLUS_WEIGHTS",
    # Scoring functions
    "score_mean_tolerance",
    "score_range",
    "check_mean_tolerance",
    "check_range",
    # Scenario helpers
    "get_validation_funcs",
    "get_validation_func",
    "compute_combined_score",
    # Metrics classes and functions
    "BaselineMetrics",
    "compute_baseline_metrics",
    "BASELINE_COLLECT_CONFIG",
    "GrowthPlusMetrics",
    "compute_growth_plus_metrics",
    "GROWTH_PLUS_COLLECT_CONFIG",
    # Target loading
    "load_baseline_targets",
    "load_growth_plus_targets",
    # Baseline scenario runners
    "run_validation",
    "print_validation_report",
    "run_stability_test",
    "print_stability_report",
    # Growth+ scenario runners
    "run_growth_plus_validation",
    "print_growth_plus_report",
    "run_growth_plus_stability_test",
    "print_growth_plus_stability_report",
    # Scenario visualization (lazy imports - see module docs for direct imports)
    "run_baseline_scenario",
    "run_growth_plus_scenario",
]
