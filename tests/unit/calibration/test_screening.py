"""Tests for calibration.screening module."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from calibration.analysis import CalibrationResult
from calibration.screening import run_screening, screen_single_seed
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


class TestScreenSingleSeed:
    """Tests for screen_single_seed."""

    @patch("calibration.screening.get_validation_funcs")
    def test_returns_calibration_result(self, mock_get_funcs):
        def mock_validate(**kwargs):
            return _make_score(0.75)

        mock_get_funcs.return_value = (mock_validate, None, None, None)

        result = screen_single_seed({"a": 1}, "baseline", 0, 50)
        assert isinstance(result, CalibrationResult)
        assert result.single_score == pytest.approx(0.75)
        assert result.n_pass == 1
        assert result.n_fail == 0


class TestRunScreening:
    """Tests for run_screening."""

    @patch("calibration.screening.ProcessPoolExecutor", ThreadPoolExecutor)
    @patch("calibration.screening.get_validation_funcs")
    def test_basic_screening(self, mock_get_funcs):
        """Use ThreadPoolExecutor so mocks propagate to worker threads."""
        call_idx = [0]
        scores = [0.7, 0.8, 0.6]

        def mock_validate(**kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            return _make_score(scores[idx % len(scores)])

        mock_get_funcs.return_value = (mock_validate, None, None, None)

        results = run_screening(
            [{"a": 1}, {"a": 2}, {"a": 3}],
            "baseline",
            n_workers=1,
            n_periods=50,
        )
        assert len(results) == 3
        # Sorted by score, best first
        assert results[0].single_score >= results[1].single_score
