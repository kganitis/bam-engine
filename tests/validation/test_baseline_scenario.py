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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml

import bamengine as bam
from validation.metrics import BASELINE_COLLECT_CONFIG, compute_baseline_metrics

# =============================================================================
# Validation Result Types
# =============================================================================


Status = Literal["PASS", "WARN", "FAIL"]

# Default weights for each metric (higher = more important)
DEFAULT_WEIGHTS: dict[str, float] = {
    "unemployment_rate_mean": 1.5,  # Key macroeconomic indicator
    "inflation_rate_mean": 1.5,  # Key macroeconomic indicator
    "log_gdp_mean": 1.0,  # Scale-dependent
    "real_wage_mean": 1.0,  # Labor market
    "vacancy_rate_mean": 0.5,  # Secondary indicator
    "phillips_correlation": 1.0,  # Classic curve
    "okun_correlation": 1.5,  # Strongest relationship
    "beveridge_correlation": 1.0,  # Classic curve
    "firm_size_skewness": 0.5,  # Distribution shape
    "firm_size_pct_below": 0.5,  # Distribution shape
}


@dataclass
class MetricResult:
    """Result of validating a single metric."""

    name: str
    status: Status
    actual: float
    target_desc: str
    score: float  # 0-1 score (1 = perfect match)
    weight: float = 1.0  # Weight for total score calculation
    message: str = ""


@dataclass
class ValidationScore:
    """Overall validation result with scoring for comparison."""

    metric_results: list[MetricResult]
    total_score: float  # Weighted average of all metric scores
    n_pass: int
    n_warn: int
    n_fail: int
    config: dict[str, Any] = field(default_factory=dict)  # Config used for this run

    @property
    def passed(self) -> bool:
        """True if no metrics failed validation."""
        return self.n_fail == 0

    def __str__(self) -> str:
        return (
            f"ValidationScore(total={self.total_score:.3f}, "
            f"pass={self.n_pass}, warn={self.n_warn}, fail={self.n_fail})"
        )


# =============================================================================
# Scoring Functions
# =============================================================================


def score_mean_tolerance(actual: float, target: float, tolerance: float) -> float:
    """Score from 0-1 based on distance from target.

    Returns 1.0 if exactly on target, decreasing linearly.
    Returns 0.0 if distance >= 2 * tolerance.
    """
    distance = abs(actual - target) / tolerance
    return max(0.0, 1.0 - distance / 2.0)


def score_range(actual: float, min_val: float, max_val: float) -> float:
    """Score from 0-1 based on position relative to range.

    Returns 0.75-1.0 if inside range (higher near center).
    Returns 0.0-0.75 if outside range (decays with distance).
    """
    range_size = max_val - min_val
    if range_size == 0:
        return 1.0 if actual == min_val else 0.0

    if min_val <= actual <= max_val:
        # Inside range: score based on distance to center
        center = (min_val + max_val) / 2
        half_range = range_size / 2
        distance_from_center = abs(actual - center) / half_range
        return 1.0 - 0.25 * distance_from_center  # 0.75-1.0
    else:
        # Outside range: decay toward 0
        if actual < min_val:
            overshoot = (min_val - actual) / range_size
        else:
            overshoot = (actual - max_val) / range_size
        return max(0.0, 0.75 - overshoot)  # 0.0-0.75


# =============================================================================
# Status Check Functions
# =============================================================================


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
# Validation Logic
# =============================================================================


def run_validation(
    *,
    seed: int = 0,
    n_periods: int = 1000,
    weights: dict[str, float] | None = None,
    **config_overrides: Any,
) -> ValidationScore:
    """Run validation and return a scored result.

    This function allows programmatic comparison of different configurations.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_periods : int
        Number of simulation periods.
    weights : dict, optional
        Custom weights for metrics. Defaults to DEFAULT_WEIGHTS.
    **config_overrides
        Any simulation config parameters to override (e.g., h_rho=0.15).

    Returns
    -------
    ValidationScore
        Validation result with total score and per-metric results.

    Example
    -------
    >>> score_a = run_validation(seed=0)
    >>> score_b = run_validation(seed=0, h_rho=0.15)
    >>> print(f"Default: {score_a.total_score:.3f}")
    >>> print(f"Modified: {score_b.total_score:.3f}")
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Load targets
    targets_path = (
        Path(__file__).parent.parent.parent / "validation/targets/baseline.yaml"
    )
    with open(targets_path) as f:
        targets = yaml.safe_load(f)

    # Build config
    config = {
        "n_firms": 300,
        "n_households": 3000,
        "n_banks": 10,
        "n_periods": n_periods,
        "seed": seed,
        "logging": {"default_level": "ERROR"},
        **config_overrides,
    }

    # Run simulation
    sim = bam.Simulation.init(**config)
    results = sim.run(collect=BASELINE_COLLECT_CONFIG)

    # Compute metrics
    burn_in = targets["metadata"]["validation"]["burn_in_periods"]
    firm_threshold = targets["distributions"]["firm_size"]["targets"]["threshold_small"]
    metrics = compute_baseline_metrics(
        sim, results, burn_in=burn_in, firm_size_threshold=firm_threshold
    )

    # Validate each metric
    validation_results: list[MetricResult] = []
    ts = targets["time_series"]
    curves = targets["curves"]

    # --- Time Series Metrics ---

    # Unemployment
    u = ts["unemployment_rate"]["targets"]
    status = check_mean_tolerance(
        metrics.unemployment_mean, u["mean_target"], u["mean_tolerance"]
    )
    score = score_mean_tolerance(
        metrics.unemployment_mean, u["mean_target"], u["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "unemployment_rate_mean",
            status,
            metrics.unemployment_mean,
            f"target: {u['mean_target']:.4f} ± {u['mean_tolerance']:.4f}",
            score,
            weights.get("unemployment_rate_mean", 1.0),
        )
    )

    # Inflation
    i = ts["inflation_rate"]["targets"]
    status = check_mean_tolerance(
        metrics.inflation_mean, i["mean_target"], i["mean_tolerance"]
    )
    score = score_mean_tolerance(
        metrics.inflation_mean, i["mean_target"], i["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "inflation_rate_mean",
            status,
            metrics.inflation_mean,
            f"target: {i['mean_target']:.4f} ± {i['mean_tolerance']:.4f}",
            score,
            weights.get("inflation_rate_mean", 1.0),
        )
    )

    # Log GDP
    g = ts["log_gdp"]["targets"]
    status = check_mean_tolerance(
        metrics.log_gdp_mean, g["mean_target"], g["mean_tolerance"]
    )
    score = score_mean_tolerance(
        metrics.log_gdp_mean, g["mean_target"], g["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "log_gdp_mean",
            status,
            metrics.log_gdp_mean,
            f"target: {g['mean_target']:.4f} ± {g['mean_tolerance']:.4f}",
            score,
            weights.get("log_gdp_mean", 1.0),
        )
    )

    # Real wage
    w = ts["real_wage"]["targets"]
    status = check_mean_tolerance(
        metrics.real_wage_mean, w["mean_target"], w["mean_tolerance"]
    )
    score = score_mean_tolerance(
        metrics.real_wage_mean, w["mean_target"], w["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "real_wage_mean",
            status,
            metrics.real_wage_mean,
            f"target: {w['mean_target']:.4f} ± {w['mean_tolerance']:.4f}",
            score,
            weights.get("real_wage_mean", 1.0),
        )
    )

    # Vacancy rate
    v = targets["distributions"]["vacancy_rate"]["targets"]
    status = check_mean_tolerance(
        metrics.vacancy_rate_mean, v["mean_target"], v["mean_tolerance"]
    )
    score = score_mean_tolerance(
        metrics.vacancy_rate_mean, v["mean_target"], v["mean_tolerance"]
    )
    validation_results.append(
        MetricResult(
            "vacancy_rate_mean",
            status,
            metrics.vacancy_rate_mean,
            f"target: {v['mean_target']:.4f} ± {v['mean_tolerance']:.4f}",
            score,
            weights.get("vacancy_rate_mean", 1.0),
        )
    )

    # --- Curve Correlations ---

    # Phillips
    p = curves["phillips"]["targets"]
    status = check_range(
        metrics.phillips_corr, p["correlation_min"], p["correlation_max"]
    )
    score = score_range(
        metrics.phillips_corr, p["correlation_min"], p["correlation_max"]
    )
    validation_results.append(
        MetricResult(
            "phillips_correlation",
            status,
            metrics.phillips_corr,
            f"range: [{p['correlation_min']:.2f}, {p['correlation_max']:.2f}]",
            score,
            weights.get("phillips_correlation", 1.0),
        )
    )

    # Okun
    o = curves["okun"]["targets"]
    status = check_range(metrics.okun_corr, o["correlation_min"], o["correlation_max"])
    score = score_range(metrics.okun_corr, o["correlation_min"], o["correlation_max"])
    validation_results.append(
        MetricResult(
            "okun_correlation",
            status,
            metrics.okun_corr,
            f"range: [{o['correlation_min']:.2f}, {o['correlation_max']:.2f}]",
            score,
            weights.get("okun_correlation", 1.0),
        )
    )

    # Beveridge
    b = curves["beveridge"]["targets"]
    status = check_range(
        metrics.beveridge_corr, b["correlation_min"], b["correlation_max"]
    )
    score = score_range(
        metrics.beveridge_corr, b["correlation_min"], b["correlation_max"]
    )
    validation_results.append(
        MetricResult(
            "beveridge_correlation",
            status,
            metrics.beveridge_corr,
            f"range: [{b['correlation_min']:.2f}, {b['correlation_max']:.2f}]",
            score,
            weights.get("beveridge_correlation", 1.0),
        )
    )

    # --- Distribution Metrics ---
    d = targets["distributions"]["firm_size"]["targets"]

    # Skewness
    status = check_range(
        metrics.firm_size_skewness, d["skewness_min"], d["skewness_max"]
    )
    score = score_range(
        metrics.firm_size_skewness, d["skewness_min"], d["skewness_max"]
    )
    validation_results.append(
        MetricResult(
            "firm_size_skewness",
            status,
            metrics.firm_size_skewness,
            f"range: [{d['skewness_min']:.1f}, {d['skewness_max']:.1f}]",
            score,
            weights.get("firm_size_skewness", 1.0),
        )
    )

    # Percentile threshold
    status = check_range(
        metrics.firm_size_pct_below_threshold,
        d["pct_below_small_min"],
        d["pct_below_small_max"],
    )
    score = score_range(
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
            score,
            weights.get("firm_size_pct_below", 1.0),
        )
    )

    # Compute totals
    n_pass = sum(1 for r in validation_results if r.status == "PASS")
    n_warn = sum(1 for r in validation_results if r.status == "WARN")
    n_fail = sum(1 for r in validation_results if r.status == "FAIL")

    # Compute weighted total score
    total_weight = sum(r.weight for r in validation_results)
    total_score = sum(r.score * r.weight for r in validation_results) / total_weight

    return ValidationScore(
        metric_results=validation_results,
        total_score=total_score,
        n_pass=n_pass,
        n_warn=n_warn,
        n_fail=n_fail,
        config=config,
    )


def print_validation_report(result: ValidationScore) -> None:
    """Print formatted validation report to stdout."""
    print("\n" + "=" * 78)
    print("BASELINE SCENARIO VALIDATION")
    print("=" * 78)

    print("\nTIME SERIES:")
    print(f"  {'Metric':<28} {'Status':<6} {'Actual':>8}  {'Score':>6}  Target")
    print("  " + "-" * 74)
    for r in result.metric_results[:5]:
        print(
            f"  {r.name:<28} {r.status:<6} {r.actual:>8.4f}  {r.score:>6.3f}  ({r.target_desc})"
        )

    print("\nCURVES:")
    print(f"  {'Metric':<28} {'Status':<6} {'Actual':>8}  {'Score':>6}  Target")
    print("  " + "-" * 74)
    for r in result.metric_results[5:8]:
        print(
            f"  {r.name:<28} {r.status:<6} {r.actual:>8.4f}  {r.score:>6.3f}  ({r.target_desc})"
        )

    print("\nDISTRIBUTION:")
    print(f"  {'Metric':<28} {'Status':<6} {'Actual':>8}  {'Score':>6}  Target")
    print("  " + "-" * 74)
    for r in result.metric_results[8:]:
        if "pct" in r.name:
            print(
                f"  {r.name:<28} {r.status:<6} {r.actual * 100:>7.1f}%  {r.score:>6.3f}  ({r.target_desc})"
            )
        else:
            print(
                f"  {r.name:<28} {r.status:<6} {r.actual:>8.4f}  {r.score:>6.3f}  ({r.target_desc})"
            )

    print("\n" + "=" * 78)
    print(
        f"SUMMARY: {result.n_pass} PASS, {result.n_warn} WARN, {result.n_fail} FAIL  |  "
        f"TOTAL SCORE: {result.total_score:.3f}"
    )
    print("=" * 78 + "\n")


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
