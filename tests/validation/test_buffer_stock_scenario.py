"""Validation tests for buffer-stock scenario against book targets.

This module validates that the BAM Engine simulation with the buffer-stock
extension reproduces the qualitative behavior documented in
Delli Gatti et al. (2011), Section 3.9.4.

The buffer-stock scenario validates:
1. Unique metrics (per-seed): Wealth distribution fitting (Figure 3.8) and
   MPC behavior — these determine per-seed PASS/FAIL.
2. Improvement over Growth+ (aggregate): Mean score deltas across all seeds
   checked after stability testing completes.

The test fails if any of the 8 unique metrics has FAIL status.
"""

from __future__ import annotations

import math

import pytest

from validation import (
    DEFAULT_STABILITY_SEEDS,
    BufferStockValidationScore,
    print_buffer_stock_report,
    print_buffer_stock_stability_report,
    run_buffer_stock_stability_test,
    run_buffer_stock_validation,
    run_growth_plus_validation,
)

# =============================================================================
# Main Validation Test
# =============================================================================


@pytest.mark.slow
def test_buffer_stock_scenario_validation() -> None:
    """Validate buffer-stock scenario results against book targets.

    Runs a full 1000-period simulation with the buffer-stock extension.
    FAILS if any of the 8 unique metrics has FAIL status.
    """
    result = run_buffer_stock_validation(seed=0, n_periods=1000)

    assert isinstance(result, BufferStockValidationScore)
    assert result.baseline_score is not None
    assert len(result.improvement_deltas) > 0

    print_buffer_stock_report(result)

    # Assert no failures
    if not result.passed:
        failure_names = [r.name for r in result.metric_results if r.status == "FAIL"]
        pytest.fail(f"{result.n_fail} metric(s) failed validation: {failure_names}")


@pytest.mark.slow
def test_buffer_stock_with_reuse() -> None:
    """Validate that pre-computed Growth+ results can be reused.

    Runs Growth+ first, then passes the result to buffer-stock validation
    to avoid re-running the Growth+ simulation.
    """
    gp_result = run_growth_plus_validation(seed=0, n_periods=1000)

    result = run_buffer_stock_validation(
        seed=0, n_periods=1000, growth_plus_result=gp_result
    )

    assert isinstance(result, BufferStockValidationScore)
    assert result.baseline_score is gp_result
    assert len(result.improvement_deltas) > 0


# =============================================================================
# Result Structure Tests
# =============================================================================


@pytest.mark.slow
def test_buffer_stock_result_structure() -> None:
    """Verify the BufferStockValidationScore structure."""
    result = run_buffer_stock_validation(seed=42, n_periods=200)

    assert isinstance(result, BufferStockValidationScore)
    assert not math.isnan(result.total_score)
    assert 0.0 <= result.blend_alpha <= 1.0

    # Per-seed metric_results contains only the 8 unique metrics
    assert len(result.metric_results) == 8

    # Improvement deltas are informational (not in metric_results)
    assert len(result.improvement_deltas) > 60  # Growth+ has ~67 metrics

    # Baseline score is stored
    assert result.baseline_score is not None


# =============================================================================
# Seed Stability Testing
# =============================================================================


@pytest.mark.slow
def test_buffer_stock_seed_stability() -> None:
    """Test that buffer-stock passes consistently across 100 seeds.

    Runs validation with 100 seeds using 10 parallel workers and checks:
    1. At least 95% of seeds pass (unique metrics only)
    2. Score standard deviation is reasonable (< 0.15)
    """
    result = run_buffer_stock_stability_test(
        seeds=DEFAULT_STABILITY_SEEDS, n_workers=10
    )
    print_buffer_stock_stability_report(result)

    assert result.pass_rate >= 0.95, (
        f"Pass rate too low: {result.pass_rate:.0%} (expected >= 95%)"
    )
    assert result.std_score <= 0.15, (
        f"Score too variable: std={result.std_score:.3f} (expected <= 0.15)"
    )
