"""Scoring and status check functions for validation.

This module provides functions to compute scores and check statuses
for validation metrics.
"""

from __future__ import annotations

import math

from validation.types import StabilityResult, Status

# =============================================================================
# Fail Escalation Constants
# =============================================================================

# Weight-based escalation: clamp(INTERCEPT - SLOPE * weight, FLOOR, CEILING)
# Higher weight → lower multiplier (stricter, fails more easily)
# Lower weight  → higher multiplier (more lenient, harder to fail)
_FAIL_INTERCEPT = 5.0
_FAIL_SLOPE = 2.0
_FAIL_FLOOR = 0.5  # Strictest: high-weight metrics fail faster than normal
_FAIL_CEILING = 5.0  # Most lenient: low-weight metrics need extreme deviation


def fail_escalation_multiplier(weight: float) -> float:
    """Compute the fail-escalation multiplier from a metric's weight.

    The multiplier scales the WARN→FAIL boundary in status check functions.
    A multiplier < 1 shrinks the WARN zone (stricter), > 1 widens it (more
    lenient).

    Mapping (with default constants):
        weight 3.0 → 0.5  (FAIL at 0.5× normal threshold)
        weight 2.0 → 1.0  (normal behaviour)
        weight 1.5 → 2.0  (FAIL at 2× normal threshold)
        weight 1.0 → 3.0  (FAIL at 3× normal threshold)
        weight 0.5 → 4.0  (FAIL at 4× normal threshold)

    Parameters
    ----------
    weight : float
        Metric weight (typically 0.5–3.0).

    Returns
    -------
    float
        Escalation multiplier, clamped to [_FAIL_FLOOR, _FAIL_CEILING].
    """
    raw = _FAIL_INTERCEPT - _FAIL_SLOPE * weight
    return max(_FAIL_FLOOR, min(_FAIL_CEILING, raw))


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
    escalation: float = 1.0,
) -> Status:
    """Check if actual value is within tolerance of target.

    Returns:
        PASS if within tolerance
        WARN if within warn_multiplier * escalation * tolerance
        FAIL otherwise
    """
    diff = abs(actual - target)
    if diff <= tolerance:
        return "PASS"
    elif diff <= tolerance * warn_multiplier * escalation:
        return "WARN"
    return "FAIL"


def check_range(
    actual: float,
    min_val: float,
    max_val: float,
    warn_buffer: float = 0.5,
    escalation: float = 1.0,
) -> Status:
    """Check if actual value is within range.

    Returns:
        PASS if within [min_val, max_val]
        WARN if within extended range (buffer * escalation applied)
        FAIL otherwise
    """
    range_size = max_val - min_val
    effective_buffer = warn_buffer * escalation
    if min_val <= actual <= max_val:
        return "PASS"
    elif (
        (min_val - effective_buffer * range_size)
        <= actual
        <= (max_val + effective_buffer * range_size)
    ):
        return "WARN"
    return "FAIL"


def check_pct_within_target(
    actual_pct: float,
    target_pct: float,
    min_pct: float,
    escalation: float = 1.0,
) -> Status:
    """Check if percentage within target meets threshold.

    With escalation, the WARN zone extends below ``min_pct`` proportionally
    to the original WARN-zone width ``(target_pct - min_pct)``.

    Returns:
        PASS if actual >= target_pct
        WARN if actual >= effective_min
        FAIL otherwise
    """
    effective_min = max(0.0, min_pct - (target_pct - min_pct) * (escalation - 1.0))
    if actual_pct >= target_pct:
        return "PASS"
    elif actual_pct >= effective_min:
        return "WARN"
    return "FAIL"


def check_outlier_penalty(
    outlier_pct: float,
    max_outlier_pct: float,
    severe_multiplier: float = 2.0,
    escalation: float = 1.0,
) -> Status:
    """Check if outlier percentage is within acceptable limits.

    Returns:
        PASS if outlier_pct <= max_outlier_pct
        WARN if outlier_pct <= max_outlier_pct * severe_multiplier * escalation
        FAIL otherwise
    """
    if outlier_pct <= max_outlier_pct:
        return "PASS"
    elif outlier_pct <= max_outlier_pct * severe_multiplier * escalation:
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
