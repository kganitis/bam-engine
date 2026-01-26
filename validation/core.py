"""Core types and utilities for validation.

This module contains shared types, dataclasses, and utility functions used
across the validation and calibration packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# =============================================================================
# Constants
# =============================================================================

# Default seeds for stability testing across validation/calibration
DEFAULT_STABILITY_SEEDS: list[int] = list(range(20))

# Type alias for validation status
Status = Literal["PASS", "WARN", "FAIL"]

# =============================================================================
# Validation Result Types
# =============================================================================


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


@dataclass
class MetricStats:
    """Statistics for a single metric across multiple seeds."""

    name: str
    mean_value: float
    std_value: float
    mean_score: float
    std_score: float
    pass_rate: float  # Fraction of seeds where this metric passed (not FAIL)


@dataclass
class StabilityResult:
    """Result of multi-seed stability testing."""

    seed_results: list[ValidationScore]  # Individual seed results

    # Aggregate score metrics
    mean_score: float  # Mean total score across seeds
    std_score: float  # Standard deviation of scores
    min_score: float  # Worst seed
    max_score: float  # Best seed

    pass_rate: float  # Fraction of seeds that passed (no FAILs)
    n_seeds: int  # Number of seeds tested

    # Per-metric stability
    metric_stats: dict[str, MetricStats]  # Stats for each metric

    @property
    def is_stable(self) -> bool:
        """True if pass_rate >= 80% and std_score <= 0.1."""
        return self.pass_rate >= 0.8 and self.std_score <= 0.1

    def __str__(self) -> str:
        return (
            f"StabilityResult(mean={self.mean_score:.3f}±{self.std_score:.3f}, "
            f"pass_rate={self.pass_rate:.0%}, seeds={self.n_seeds})"
        )


# =============================================================================
# Metric Weights
# =============================================================================

# Default weights for baseline scenario (higher = more important)
BASELINE_WEIGHTS: dict[str, float] = {
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

# Default weights for Growth+ scenario (higher weights for productivity metrics)
GROWTH_PLUS_WEIGHTS: dict[str, float] = {
    "unemployment_rate_mean": 1.5,  # Key macroeconomic indicator
    "inflation_rate_mean": 1.5,  # Key macroeconomic indicator
    "log_gdp_mean": 1.0,  # Growing over time
    "vacancy_rate_mean": 0.5,  # Secondary indicator
    "phillips_correlation": 1.5,  # Stronger in Growth+ (-0.19)
    "okun_correlation": 1.5,  # Strongest relationship
    "beveridge_correlation": 1.0,  # Classic curve
    "firm_size_skewness": 0.5,  # Distribution shape
    "firm_size_pct_below": 0.5,  # Distribution shape
    # Growth-specific metrics
    "productivity_growth": 1.5,  # Key Growth+ metric
    "real_wage_growth": 1.0,  # Should track productivity
    "productivity_trend": 1.0,  # Trend coefficient
    # Financial dynamics metrics
    "real_interest_rate_mean": 0.5,  # Mean real interest rate
    "real_interest_rate_std": 0.5,  # Volatility of real interest rate
    "financial_fragility_mean": 0.5,  # Mean financial fragility
    "financial_fragility_std": 0.5,  # Volatility of financial fragility
    "price_ratio_mean": 0.5,  # Mean price ratio (P / P*)
    "price_ratio_std": 0.5,  # Volatility of price ratio
    "price_dispersion_mean": 0.5,  # Mean price dispersion (CV)
    "price_dispersion_std": 0.5,  # Volatility of price dispersion
    "equity_dispersion_mean": 0.5,  # Mean equity dispersion (CV)
    "equity_dispersion_std": 0.5,  # Volatility of equity dispersion
    "sales_dispersion_mean": 0.5,  # Mean sales dispersion (CV)
    "sales_dispersion_std": 0.5,  # Volatility of sales dispersion
    # Minsky classification
    "minsky_hedge_pct": 0.5,  # Hedge firm percentage
    "minsky_ponzi_pct": 0.5,  # Ponzi firm percentage
    # Growth rate distribution metrics (tiered validation)
    "output_growth_pct_tight": 1.0,  # % within tight range
    "output_growth_pct_normal": 0.5,  # % within normal range
    "output_growth_outliers": 1.5,  # Outlier penalty (higher weight)
    "networth_growth_pct_tight": 1.0,  # % within tight range
    "networth_growth_pct_normal": 0.5,  # % within normal range
    "networth_growth_outliers": 1.5,  # Outlier penalty (higher weight)
}


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


def score_pct_within_target(
    actual_pct: float, target_pct: float, min_pct: float
) -> float:
    """Score 0-1 for percentage meeting target.

    Returns 1.0 if actual >= target, scores proportionally if >= min,
    and penalizes below min.
    """
    if actual_pct >= target_pct:
        return 1.0
    elif actual_pct >= min_pct:
        progress = (actual_pct - min_pct) / (target_pct - min_pct)
        return 0.75 + 0.25 * progress
    else:
        if min_pct > 0:
            shortfall = (min_pct - actual_pct) / min_pct
            return max(0.0, 0.75 * (1 - shortfall))
        return 0.0


def score_outlier_penalty(
    outlier_pct: float, max_outlier_pct: float, penalty_weight: float = 2.0
) -> float:
    """Score 0-1 with exponential penalty for excessive outliers.

    Returns 1.0 if outlier_pct <= max_outlier_pct, else exponentially
    decays based on how much the actual exceeds the maximum allowed.
    """
    import math

    if outlier_pct <= max_outlier_pct:
        return 1.0
    excess = outlier_pct - max_outlier_pct
    return max(0.0, math.exp(-penalty_weight * excess / max_outlier_pct))


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


def check_pct_within_target(
    actual_pct: float, target_pct: float, min_pct: float
) -> Status:
    """Check if percentage within target meets threshold.

    Returns:
        PASS if actual >= target
        WARN if actual >= min
        FAIL otherwise
    """
    if actual_pct >= target_pct:
        return "PASS"
    elif actual_pct >= min_pct:
        return "WARN"
    return "FAIL"


def check_outlier_penalty(
    outlier_pct: float, max_outlier_pct: float, severe_multiplier: float = 2.0
) -> Status:
    """Check if outlier percentage is within acceptable limits.

    Returns:
        PASS if outlier_pct <= max_outlier_pct
        WARN if outlier_pct <= max_outlier_pct * severe_multiplier
        FAIL otherwise
    """
    if outlier_pct <= max_outlier_pct:
        return "PASS"
    elif outlier_pct <= max_outlier_pct * severe_multiplier:
        return "WARN"
    return "FAIL"


# =============================================================================
# Scenario Helpers
# =============================================================================


def get_validation_funcs(scenario: str) -> tuple[Any, Any]:
    """Get validation and stability functions for a scenario.

    Parameters
    ----------
    scenario : str
        Scenario name ("baseline" or "growth_plus").

    Returns
    -------
    tuple[callable, callable]
        (validation_func, stability_func) for the scenario.

    Raises
    ------
    ValueError
        If scenario is not recognized.
    """
    # Import here to avoid circular imports
    from validation.runners import (
        run_growth_plus_stability_test,
        run_growth_plus_validation,
        run_stability_test,
        run_validation,
    )

    if scenario == "baseline":
        return run_validation, run_stability_test
    elif scenario == "growth_plus":
        return run_growth_plus_validation, run_growth_plus_stability_test
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def get_validation_func(scenario: str) -> Any:
    """Get the validation function for a scenario.

    Parameters
    ----------
    scenario : str
        Scenario name ("baseline" or "growth_plus").

    Returns
    -------
    callable
        The validation function for the scenario.

    Raises
    ------
    ValueError
        If scenario is not recognized.
    """
    validate, _ = get_validation_funcs(scenario)
    return validate


def compute_combined_score(stability: StabilityResult) -> float:
    """Compute combined score balancing accuracy and stability.

    Formula: mean_score * pass_rate * (1 - std_score)
    - Higher mean_score is better
    - Higher pass_rate is better
    - Lower std_score is better

    Parameters
    ----------
    stability : StabilityResult
        Result from run_stability_test().

    Returns
    -------
    float
        Combined score (higher is better).
    """
    return stability.mean_score * stability.pass_rate * (1 - stability.std_score)
