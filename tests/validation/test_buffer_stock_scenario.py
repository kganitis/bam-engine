"""Validation tests for buffer-stock scenario against book targets.

This module validates that the BAM Engine simulation with the buffer-stock
extension reproduces the qualitative behavior documented in
Delli Gatti et al. (2011), Section 3.9.3.

The buffer-stock scenario replaces the baseline mean-field MPC with an
individual adaptive rule. Key validation targets:
- Wealth distribution fitted with heavy-tailed distributions (SM, Dagum, GB2)
- CCDF on log-log axes matching Figure 3.8
- Baseline macro dynamics preserved (Phillips, Okun, Beveridge)

The test compares simulation results against targets defined in:
    validation/targets/buffer_stock.yaml

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
    print_buffer_stock_report,
    print_buffer_stock_stability_report,
    run_buffer_stock_stability_test,
    run_buffer_stock_validation,
)

# =============================================================================
# Main Validation Test
# =============================================================================


@pytest.mark.validation
def test_buffer_stock_scenario_validation() -> None:
    """Validate buffer-stock scenario results against book targets.

    This test runs a full 1000-period simulation with the buffer-stock
    extension and compares the results against targets derived from
    Delli Gatti et al. (2011), Section 3.9.3.

    The test FAILS if any metric has FAIL status.
    WARN status is acceptable (indicates calibration work needed).
    """
    result = run_buffer_stock_validation(seed=0, n_periods=1000)
    print_buffer_stock_report(result)

    # Assert no failures
    if not result.passed:
        failure_names = [r.name for r in result.metric_results if r.status == "FAIL"]
        pytest.fail(f"{result.n_fail} metric(s) failed validation: {failure_names}")


# =============================================================================
# Seed Stability Testing
# =============================================================================


@pytest.mark.slow
@pytest.mark.validation
def test_buffer_stock_seed_stability() -> None:
    """Test that buffer-stock passes consistently across multiple seeds.

    This test runs validation with 20 different seeds and checks:
    1. At least 95% of seeds pass (no FAIL metrics)
    2. Score standard deviation is reasonable (< 0.15)
    """
    result = run_buffer_stock_stability_test(seeds=DEFAULT_STABILITY_SEEDS)
    print_buffer_stock_stability_report(result)

    # Assert stability criteria
    assert result.pass_rate >= 0.95, (
        f"Pass rate too low: {result.pass_rate:.0%} (expected >= 95%)"
    )
    assert result.std_score <= 0.15, (
        f"Score too variable: std={result.std_score:.3f} (expected <= 0.15)"
    )
