"""Scoring and status check functions for validation.

This module provides functions to compute scores and check statuses
for validation metrics.
"""

from __future__ import annotations

import math

from validation.types import StabilityResult, Status

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
# Aggregate Score Functions
# =============================================================================


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


# =============================================================================
# Status Helpers
# =============================================================================

STATUS_COLORS: dict[Status, str] = {
    "PASS": "lightgreen",
    "WARN": "lightyellow",
    "FAIL": "lightcoral",
}


def worst_status(*statuses: Status) -> Status:
    """Return the most severe status from the given statuses."""
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"
