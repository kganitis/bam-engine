"""Tests for validation scoring functions, focusing on fail escalation."""

import pytest

from validation.scoring import (
    check_mean_tolerance,
    check_outlier_penalty,
    check_pct_within_target,
    check_range,
    fail_escalation_multiplier,
)

# =============================================================================
# fail_escalation_multiplier
# =============================================================================


class TestFailEscalationMultiplier:
    """Test weight-to-escalation mapping: clamp(5 - 2*weight, 0.5, 5.0)."""

    @pytest.mark.parametrize(
        "weight, expected",
        [
            (3.0, 0.5),
            (2.5, 0.5),
            (2.0, 1.0),
            (1.5, 2.0),
            (1.0, 3.0),
            (0.5, 4.0),
            (0.0, 5.0),
        ],
    )
    def test_known_values(self, weight: float, expected: float) -> None:
        assert fail_escalation_multiplier(weight) == expected

    def test_floor_clamp(self) -> None:
        """Very high weights should clamp to _FAIL_FLOOR (0.5)."""
        assert fail_escalation_multiplier(10.0) == 0.5
        assert fail_escalation_multiplier(100.0) == 0.5

    def test_ceiling_clamp(self) -> None:
        """Very low / negative weights should clamp to _FAIL_CEILING (5.0)."""
        assert fail_escalation_multiplier(-1.0) == 5.0
        assert fail_escalation_multiplier(0.0) == 5.0


# =============================================================================
# check_mean_tolerance with escalation
# =============================================================================


class TestCheckMeanToleranceEscalation:
    """PASS: diff <= tol.  WARN: diff <= tol * warn_mult * esc.  FAIL: else."""

    def test_pass_is_unaffected_by_escalation(self) -> None:
        """PASS zone depends only on tolerance, not escalation."""
        assert check_mean_tolerance(1.0, 1.0, 0.1, escalation=0.5) == "PASS"
        assert check_mean_tolerance(1.0, 1.0, 0.1, escalation=5.0) == "PASS"

    def test_default_escalation_preserves_behaviour(self) -> None:
        """escalation=1.0 matches the old code (FAIL at 2x tolerance)."""
        # Just outside tolerance → WARN (diff=0.15, tol=0.1, boundary=0.2)
        assert check_mean_tolerance(1.15, 1.0, 0.1) == "WARN"
        # Beyond 2x tolerance → FAIL (diff=0.25)
        assert check_mean_tolerance(1.25, 1.0, 0.1) == "FAIL"

    def test_high_escalation_converts_fail_to_warn(self) -> None:
        """Low-weight metric (escalation=4): FAIL boundary widens to 8x tol."""
        # diff=0.25, tol=0.1, boundary = 0.1 * 2 * 4 = 0.8 → WARN
        assert check_mean_tolerance(1.25, 1.0, 0.1, escalation=4.0) == "WARN"

    def test_low_escalation_makes_warn_become_fail(self) -> None:
        """High-weight metric (escalation=0.5): FAIL boundary shrinks to 1x tol."""
        # diff=0.15, tol=0.1, boundary = 0.1 * 2 * 0.5 = 0.1 → FAIL
        assert check_mean_tolerance(1.15, 1.0, 0.1, escalation=0.5) == "FAIL"

    @pytest.mark.parametrize(
        "escalation, expected",
        [
            (0.5, "FAIL"),  # boundary = 0.1 * 2 * 0.5 = 0.10
            (1.0, "WARN"),  # boundary = 0.1 * 2 * 1.0 = 0.20
            (2.0, "WARN"),  # boundary = 0.1 * 2 * 2.0 = 0.40
            (4.0, "WARN"),  # boundary = 0.1 * 2 * 4.0 = 0.80
        ],
    )
    def test_escalation_sweep(self, escalation: float, expected: str) -> None:
        """diff=0.15, tol=0.1 → status depends on escalation."""
        assert check_mean_tolerance(1.15, 1.0, 0.1, escalation=escalation) == expected


# =============================================================================
# check_range with escalation
# =============================================================================


class TestCheckRangeEscalation:
    """PASS: in range.  WARN: in range + buffer*esc*range_size.  FAIL: else."""

    def test_pass_is_unaffected_by_escalation(self) -> None:
        assert check_range(0.5, 0.0, 1.0, escalation=0.5) == "PASS"
        assert check_range(0.5, 0.0, 1.0, escalation=5.0) == "PASS"

    def test_default_escalation_preserves_behaviour(self) -> None:
        # Just outside range (buffer zone) → WARN
        assert check_range(1.3, 0.0, 1.0) == "WARN"
        # Far outside → FAIL
        assert check_range(2.0, 0.0, 1.0) == "FAIL"

    def test_high_escalation_converts_fail_to_warn(self) -> None:
        """Low-weight: buffer * escalation widens WARN zone."""
        # actual=2.0, range=[0,1], buffer=0.5*4=2.0, extended max=1+2*1=3 → WARN
        assert check_range(2.0, 0.0, 1.0, escalation=4.0) == "WARN"

    def test_low_escalation_makes_warn_become_fail(self) -> None:
        """High-weight: buffer * 0.5 shrinks WARN zone."""
        # actual=1.3, range=[0,1], buffer=0.5*0.5=0.25, extended max=1+0.25=1.25 → FAIL
        assert check_range(1.3, 0.0, 1.0, escalation=0.5) == "FAIL"


# =============================================================================
# check_pct_within_target with escalation
# =============================================================================


class TestCheckPctWithinTargetEscalation:
    """PASS: >= target.  WARN: >= effective_min.  FAIL: < effective_min.

    effective_min = max(0, min_pct - (target - min) * (esc - 1))
    """

    def test_pass_is_unaffected_by_escalation(self) -> None:
        assert check_pct_within_target(0.95, 0.90, 0.80, escalation=0.5) == "PASS"
        assert check_pct_within_target(0.95, 0.90, 0.80, escalation=5.0) == "PASS"

    def test_default_escalation_preserves_behaviour(self) -> None:
        # Between min and target → WARN
        assert check_pct_within_target(0.85, 0.90, 0.80) == "WARN"
        # Below min → FAIL
        assert check_pct_within_target(0.75, 0.90, 0.80) == "FAIL"

    def test_high_escalation_converts_fail_to_warn(self) -> None:
        """Low-weight: effective_min drops below original min_pct."""
        # eff_min = max(0, 0.80 - 0.10 * (4 - 1)) = max(0, 0.50) = 0.50
        assert check_pct_within_target(0.75, 0.90, 0.80, escalation=4.0) == "WARN"

    def test_low_escalation_makes_warn_become_fail(self) -> None:
        """High-weight: effective_min rises above original min_pct."""
        # eff_min = max(0, 0.80 - 0.10 * (0.5 - 1)) = max(0, 0.85) = 0.85
        assert check_pct_within_target(0.83, 0.90, 0.80, escalation=0.5) == "FAIL"

    def test_effective_min_clamps_to_zero(self) -> None:
        """Very high escalation should not produce a negative threshold."""
        # eff_min = max(0, 0.80 - 0.10 * (100 - 1)) = max(0, -9.1) = 0.0
        assert check_pct_within_target(0.01, 0.90, 0.80, escalation=100.0) == "WARN"


# =============================================================================
# check_outlier_penalty with escalation
# =============================================================================


class TestCheckOutlierPenaltyEscalation:
    """PASS: <= max.  WARN: <= max * severe * esc.  FAIL: else."""

    def test_pass_is_unaffected_by_escalation(self) -> None:
        assert check_outlier_penalty(0.01, 0.02, escalation=0.5) == "PASS"
        assert check_outlier_penalty(0.01, 0.02, escalation=5.0) == "PASS"

    def test_default_escalation_preserves_behaviour(self) -> None:
        # Within severe zone → WARN (0.03 <= 0.02 * 2 = 0.04)
        assert check_outlier_penalty(0.03, 0.02) == "WARN"
        # Beyond severe zone → FAIL (0.05 > 0.04)
        assert check_outlier_penalty(0.05, 0.02) == "FAIL"

    def test_high_escalation_converts_fail_to_warn(self) -> None:
        """Low-weight: FAIL boundary = max * severe * esc = 0.02 * 2 * 4 = 0.16."""
        assert check_outlier_penalty(0.05, 0.02, escalation=4.0) == "WARN"

    def test_low_escalation_makes_warn_become_fail(self) -> None:
        """High-weight: FAIL boundary = 0.02 * 2 * 0.5 = 0.02."""
        assert check_outlier_penalty(0.03, 0.02, escalation=0.5) == "FAIL"
