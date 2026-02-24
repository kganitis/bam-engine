"""Validation tests for baseline scenario against book targets.

This module validates that the BAM Engine simulation reproduces the qualitative
and quantitative behavior documented in Delli Gatti et al. (2011), Section 3.9.1.

The test compares simulation results against targets defined in:
    validation/targets/baseline.yaml

Test behavior:
    - PASS: Metric within acceptable range
    - WARN: Metric outside target but within tolerance (calibration needed)
    - FAIL: Metric significantly outside acceptable range

The test fails if ANY metric has FAIL status. WARN is acceptable.

Scoring:
    Each metric receives a 0-1 score based on how close it is to the target.
    A weighted total score enables comparison between parameter configurations.
"""

from __future__ import annotations

import pytest

from validation import (
    DEFAULT_STABILITY_SEEDS,
    print_stability_report,
    print_validation_report,
    run_stability_test,
    run_validation,
)

# =============================================================================
# Main Validation Test
# =============================================================================


@pytest.mark.slow
def test_baseline_scenario_validation() -> None:
    """Validate baseline scenario results against book targets.

    This test runs a full 1000-period simulation and compares the results
    against targets derived from Delli Gatti et al. (2011), Section 3.9.1.

    The test FAILS if any metric has FAIL status.
    WARN status is acceptable (indicates calibration work needed).
    """
    result = run_validation(seed=0, n_periods=1000)
    print_validation_report(result)

    # Assert no failures
    if not result.passed:
        failure_names = [r.name for r in result.metric_results if r.status == "FAIL"]
        pytest.fail(f"{result.n_fail} metric(s) failed validation: {failure_names}")


# =============================================================================
# Seed Stability Testing
# =============================================================================


@pytest.mark.slow
def test_baseline_seed_stability() -> None:
    """Test that baseline passes consistently across multiple seeds.

    This test runs validation with 20 different seeds and checks:
    1. At least 95% of seeds pass (no FAIL metrics)
    2. Score standard deviation is reasonable (< 0.15)
    """
    result = run_stability_test(seeds=DEFAULT_STABILITY_SEEDS)
    print_stability_report(result)

    # Assert stability criteria
    assert result.pass_rate >= 0.95, (
        f"Pass rate too low: {result.pass_rate:.0%} (expected >= 95%)"
    )
    assert result.std_score <= 0.15, (
        f"Score too variable: std={result.std_score:.3f} (expected <= 0.15)"
    )
