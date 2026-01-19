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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
import yaml

import bamengine as bam
from validation.metrics import BASELINE_COLLECT_CONFIG, compute_baseline_metrics

# =============================================================================
# Validation Result Types
# =============================================================================


Status = Literal["PASS", "WARN", "FAIL"]


@dataclass
class MetricResult:
    """Result of validating a single metric."""

    name: str
    status: Status
    actual: float
    target_desc: str
    message: str = ""


def check_mean_tolerance(
    actual: float,
    target: float,
    tolerance: float,
    warn_multiplier: float = 2.0,
) -> Status:
    """Check if actual value is within tolerance of target.

    Returns:
        PASS if within tolerance
        WARN if within warn_multiplier * tolerance
        FAIL otherwise
    """
    diff = abs(actual - target)
    if diff <= tolerance:
        return "PASS"
    elif diff <= tolerance * warn_multiplier:
        return "WARN"
    return "FAIL"


def check_range(
    actual: float,
    min_val: float,
    max_val: float,
    warn_buffer: float = 0.5,
) -> Status:
    """Check if actual value is within range.

    Returns:
        PASS if within [min_val, max_val]
        WARN if within extended range (buffer applied)
        FAIL otherwise
    """
    range_size = max_val - min_val
    if min_val <= actual <= max_val:
        return "PASS"
    elif (
        (min_val - warn_buffer * range_size)
        <= actual
        <= (max_val + warn_buffer * range_size)
    ):
        return "WARN"
    return "FAIL"


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
    # Load targets
    targets_path = (
        Path(__file__).parent.parent.parent / "validation/targets/baseline.yaml"
    )
    with open(targets_path) as f:
        targets = yaml.safe_load(f)

    # Run simulation
    sim = bam.Simulation.init(
        n_firms=300,
        n_households=3000,
        n_banks=10,
        n_periods=1000,
        seed=0,
        logging={"default_level": "ERROR"},
    )

    results = sim.run(collect=BASELINE_COLLECT_CONFIG)

    # Compute metrics using shared module
    burn_in = targets["metadata"]["validation"]["burn_in_periods"]
    firm_threshold = targets["distributions"]["firm_size"]["targets"]["threshold_small"]
    metrics = compute_baseline_metrics(
        sim, results, burn_in=burn_in, firm_size_threshold=firm_threshold
    )

    # Validate each metric
    validation_results: list[MetricResult] = []

    # --- Time Series Metrics ---
    ts = targets["time_series"]

    # Unemployment
    u = ts["unemployment_rate"]["targets"]
    status = check_mean_tolerance(
        metrics.unemployment_mean, u["mean_target"], u["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "unemployment_rate_mean",
            status,
            metrics.unemployment_mean,
            f"target: {u['mean_target']:.4f} ± {u['mean_tolerance']:.4f}",
        )
    )

    # Inflation
    i = ts["inflation_rate"]["targets"]
    status = check_mean_tolerance(
        metrics.inflation_mean, i["mean_target"], i["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "inflation_rate_mean",
            status,
            metrics.inflation_mean,
            f"target: {i['mean_target']:.4f} ± {i['mean_tolerance']:.4f}",
        )
    )

    # Log GDP
    g = ts["log_gdp"]["targets"]
    status = check_mean_tolerance(
        metrics.log_gdp_mean, g["mean_target"], g["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "log_gdp_mean",
            status,
            metrics.log_gdp_mean,
            f"target: {g['mean_target']:.4f} ± {g['mean_tolerance']:.4f}",
        )
    )

    # Real wage
    w = ts["real_wage"]["targets"]
    status = check_mean_tolerance(
        metrics.real_wage_mean, w["mean_target"], w["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "real_wage_mean",
            status,
            metrics.real_wage_mean,
            f"target: {w['mean_target']:.4f} ± {w['mean_tolerance']:.4f}",
        )
    )

    # Vacancy rate
    v = targets["distributions"]["vacancy_rate"]["targets"]
    status = check_mean_tolerance(
        metrics.vacancy_rate_mean, v["mean_target"], v["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "vacancy_rate_mean",
            status,
            metrics.vacancy_rate_mean,
            f"target: {v['mean_target']:.4f} ± {v['mean_tolerance']:.4f}",
        )
    )

    # --- Curve Correlations ---
    curves = targets["curves"]

    # Phillips
    p = curves["phillips"]["targets"]
    status = check_range(
        metrics.phillips_corr, p["correlation_min"], p["correlation_max"]
    )
    validation_results.append(
        MetricResult(
            "phillips_correlation",
            status,
            metrics.phillips_corr,
            f"range: [{p['correlation_min']:.2f}, {p['correlation_max']:.2f}]",
        )
    )

    # Okun
    o = curves["okun"]["targets"]
    status = check_range(metrics.okun_corr, o["correlation_min"], o["correlation_max"])
    validation_results.append(
        MetricResult(
            "okun_correlation",
            status,
            metrics.okun_corr,
            f"range: [{o['correlation_min']:.2f}, {o['correlation_max']:.2f}]",
        )
    )

    # Beveridge
    b = curves["beveridge"]["targets"]
    status = check_range(
        metrics.beveridge_corr, b["correlation_min"], b["correlation_max"]
    )
    validation_results.append(
        MetricResult(
            "beveridge_correlation",
            status,
            metrics.beveridge_corr,
            f"range: [{b['correlation_min']:.2f}, {b['correlation_max']:.2f}]",
        )
    )

    # --- Distribution Metrics ---
    d = targets["distributions"]["firm_size"]["targets"]

    # Skewness
    status = check_range(
        metrics.firm_size_skewness, d["skewness_min"], d["skewness_max"]
    )
    validation_results.append(
        MetricResult(
            "firm_size_skewness",
            status,
            metrics.firm_size_skewness,
            f"range: [{d['skewness_min']:.1f}, {d['skewness_max']:.1f}]",
        )
    )

    # Percentile threshold
    status = check_range(
        metrics.firm_size_pct_below_threshold,
        d["pct_below_small_min"],
        d["pct_below_small_max"],
    )
    validation_results.append(
        MetricResult(
            "firm_size_pct_below",
            status,
            metrics.firm_size_pct_below_threshold,
            f"range: [{d['pct_below_small_min'] * 100:.0f}%, {d['pct_below_small_max'] * 100:.0f}%]",
        )
    )

    # --- Print Summary ---
    print("\n" + "=" * 70)
    print("BASELINE SCENARIO VALIDATION")
    print("=" * 70)

    print("\nTIME SERIES:")
    for r in validation_results[:5]:
        print(f"  {r.name:<28} {r.status:<6} {r.actual:.4f}  ({r.target_desc})")

    print("\nCURVES:")
    for r in validation_results[5:8]:
        print(f"  {r.name:<28} {r.status:<6} {r.actual:.4f}  ({r.target_desc})")

    print("\nDISTRIBUTION:")
    for r in validation_results[8:]:
        if "pct" in r.name:
            print(
                f"  {r.name:<28} {r.status:<6} {r.actual * 100:.1f}%    ({r.target_desc})"
            )
        else:
            print(f"  {r.name:<28} {r.status:<6} {r.actual:.4f}  ({r.target_desc})")

    # Count results
    n_pass = sum(1 for r in validation_results if r.status == "PASS")
    n_warn = sum(1 for r in validation_results if r.status == "WARN")
    n_fail = sum(1 for r in validation_results if r.status == "FAIL")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")
    print("=" * 70 + "\n")

    # Assert no failures
    failures = [r for r in validation_results if r.status == "FAIL"]
    if failures:
        failure_names = [r.name for r in failures]
        pytest.fail(f"{n_fail} metric(s) failed validation: {failure_names}")
