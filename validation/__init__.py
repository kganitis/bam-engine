"""Validation package for BAM Engine.

This package provides tools for validating simulation results against
targets derived from Delli Gatti et al. (2011).

Modules:
    metrics: Compute validation metrics from simulation results
    targets/: YAML files defining target values for different scenarios

Usage:
    from validation import run_validation

    # Compare different configurations
    score_a = run_validation(seed=0)
    score_b = run_validation(seed=0, h_rho=0.15)
    print(f"Default: {score_a.total_score:.3f}")
    print(f"Modified: {score_b.total_score:.3f}")
"""

from validation.metrics import BaselineMetrics, compute_baseline_metrics

# Import validation runner (lazy import to avoid circular deps)
# These are imported from tests/validation but are useful for programmatic use


def run_validation(**kwargs):
    """Run validation with scoring. See tests.validation.test_baseline_scenario."""
    from tests.validation.test_baseline_scenario import run_validation as _run

    return _run(**kwargs)


def print_validation_report(result):
    """Print formatted validation report."""
    from tests.validation.test_baseline_scenario import (
        print_validation_report as _print,
    )

    return _print(result)


__all__ = [
    "BaselineMetrics",
    "compute_baseline_metrics",
    "run_validation",
    "print_validation_report",
]
