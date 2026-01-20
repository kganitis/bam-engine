"""Validation tests for Growth+ scenario against book targets.

This module validates that the BAM Engine simulation with the Growth+ extension
reproduces the qualitative and quantitative behavior documented in
Delli Gatti et al. (2011), Section 3.8.

The Growth+ scenario extends the baseline with endogenous productivity growth
through R&D investment. Key differences from baseline:
- Productivity and real wage grow over time (non-stationary)
- Phillips correlation is stronger (-0.19 vs -0.10)
- Firm size distribution has larger values due to productivity growth

The test compares simulation results against targets defined in:
    validation/targets/growth_plus.yaml

Test behavior:
    - PASS: Metric within acceptable range
    - WARN: Metric outside target but within tolerance (calibration needed)
    - FAIL: Metric significantly outside acceptable range

The test fails if ANY metric has FAIL status. WARN is acceptable.
"""

from __future__ import annotations

import pytest

from validation import (
    DEFAULT_STABILITY_SEEDS,
    print_growth_plus_report,
    print_growth_plus_stability_report,
    run_growth_plus_stability_test,
    run_growth_plus_validation,
)

# =============================================================================
# Main Validation Test
# =============================================================================


@pytest.mark.slow
def test_growth_plus_scenario_validation() -> None:
    """Validate Growth+ scenario results against book targets.

    This test runs a full 1000-period simulation with the RnD extension
    and compares the results against targets derived from
    Delli Gatti et al. (2011), Section 3.8.

    The test FAILS if any metric has FAIL status.
    WARN status is acceptable (indicates calibration work needed).
    """
    result = run_growth_plus_validation(seed=0, n_periods=1000)
    print_growth_plus_report(result)

    # Assert no failures
    if not result.passed:
        failure_names = [r.name for r in result.metric_results if r.status == "FAIL"]
        pytest.fail(f"{result.n_fail} metric(s) failed validation: {failure_names}")


# =============================================================================
# Seed Stability Testing
# =============================================================================


@pytest.mark.slow
def test_growth_plus_seed_stability() -> None:
    """Test that Growth+ passes consistently across multiple seeds.

    This test runs validation with 5 different seeds and checks:
    1. ALL seeds pass (no FAIL metrics)
    2. Score standard deviation is reasonable (< 0.15)
    """
    result = run_growth_plus_stability_test(seeds=DEFAULT_STABILITY_SEEDS)
    print_growth_plus_stability_report(result)

    # Assert stability criteria
    assert result.pass_rate >= 1.0, (
        f"Pass rate too low: {result.pass_rate:.0%} (expected = 100%)"
    )
    assert result.std_score <= 0.15, (
        f"Score too variable: std={result.std_score:.3f} (expected <= 0.15)"
    )
