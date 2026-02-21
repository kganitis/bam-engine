"""Tests for calibration.sensitivity pairwise interaction analysis."""

from __future__ import annotations

from unittest.mock import patch

from calibration.sensitivity import (
    PairInteraction,
    PairwiseResult,
    print_pairwise_report,
    run_pairwise_analysis,
)
from validation.types import MetricGroup, MetricResult, ValidationScore


def _make_score(total: float) -> ValidationScore:
    """Create a minimal ValidationScore for testing."""
    mr = MetricResult(
        name="test_metric",
        status="PASS",
        actual=total,
        target_desc="test",
        score=total,
        weight=1.0,
        group=MetricGroup.TIME_SERIES,
    )
    return ValidationScore(
        metric_results=[mr],
        total_score=total,
        n_pass=1,
        n_warn=0,
        n_fail=0,
    )


class TestPairInteraction:
    """Tests for PairInteraction dataclass."""

    def test_creation(self):
        pi = PairInteraction(
            param_a="a",
            param_b="b",
            value_a=1,
            value_b=2,
            individual_a_score=0.8,
            individual_b_score=0.7,
            combined_score=0.9,
            baseline_score=0.6,
            interaction_strength=0.0,
        )
        assert pi.param_a == "a"
        assert pi.combined_score == 0.9


class TestPairwiseResult:
    """Tests for PairwiseResult dataclass."""

    def _make_result(self) -> PairwiseResult:
        return PairwiseResult(
            interactions=[
                PairInteraction(
                    "a",
                    "b",
                    1,
                    2,
                    individual_a_score=0.7,
                    individual_b_score=0.6,
                    combined_score=0.85,  # synergy: 0.85 > 0.7+0.6-0.5=0.8
                    baseline_score=0.5,
                    interaction_strength=0.05,
                ),
                PairInteraction(
                    "a",
                    "b",
                    1,
                    3,
                    individual_a_score=0.7,
                    individual_b_score=0.65,
                    combined_score=0.75,  # conflict: 0.75 < 0.7+0.65-0.5=0.85
                    baseline_score=0.5,
                    interaction_strength=0.10,
                ),
            ],
            scenario="baseline",
            baseline_score=0.5,
        )

    def test_ranked_by_strength(self):
        result = self._make_result()
        ranked = result.ranked
        assert ranked[0].interaction_strength >= ranked[1].interaction_strength

    def test_synergies(self):
        result = self._make_result()
        syns = result.synergies
        assert len(syns) == 1
        assert syns[0].value_b == 2

    def test_conflicts(self):
        result = self._make_result()
        cons = result.conflicts
        assert len(cons) == 1
        assert cons[0].value_b == 3


class TestRunPairwiseAnalysis:
    """Tests for run_pairwise_analysis."""

    @patch("calibration.sensitivity.get_validation_func")
    def test_basic_pairwise(self, mock_get_func):
        """Test pairwise with 2 HIGH params × 2 values each."""

        def mock_validate(**kwargs):
            # Score varies based on params
            a = kwargs.get("a", 1)
            b = kwargs.get("b", 10)
            score = 0.5 + 0.05 * a + 0.01 * b / 10
            return _make_score(min(score, 1.0))

        mock_get_func.return_value = mock_validate

        result = run_pairwise_analysis(
            params=["a", "b"],
            grid={"a": [1, 2], "b": [10, 20]},
            best_values={"a": 1, "b": 10},
            scenario="baseline",
            n_seeds=1,
            n_periods=50,
            n_workers=1,
        )

        assert result.scenario == "baseline"
        assert result.baseline_score > 0
        assert len(result.interactions) == 4  # 2×2 combinations


class TestPrintPairwiseReport:
    """Test that print doesn't crash."""

    def test_print_no_crash(self, capsys):
        result = PairwiseResult(
            interactions=[
                PairInteraction(
                    "a",
                    "b",
                    1,
                    2,
                    0.7,
                    0.6,
                    0.8,
                    0.5,
                    0.05,
                ),
            ],
            scenario="baseline",
            baseline_score=0.5,
        )
        print_pairwise_report(result)
        captured = capsys.readouterr()
        assert "PAIRWISE" in captured.out
