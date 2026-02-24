"""Tests for calibration.analysis module."""

from __future__ import annotations

from unittest.mock import patch

from calibration.analysis import (
    CalibrationResult,
    ComparisonResult,
    analyze_parameter_patterns,
    compare_configs,
    export_best_config,
    print_comparison,
    print_parameter_patterns,
)
from validation.types import MetricGroup, MetricResult, ValidationScore


class TestAnalyzeParameterPatterns:
    """Tests for analyze_parameter_patterns."""

    def test_basic_patterns(self):
        results = [
            CalibrationResult({"method": "expensive", "x": 1}, 0.9, 1, 0, 0),
            CalibrationResult({"method": "expensive", "x": 2}, 0.85, 1, 0, 0),
            CalibrationResult({"method": "random", "x": 1}, 0.6, 1, 0, 0),
        ]

        patterns = analyze_parameter_patterns(results, top_n=3)
        assert "method" in patterns
        assert patterns["method"]["expensive"] == 2
        assert patterns["method"]["random"] == 1

    def test_empty_results(self):
        patterns = analyze_parameter_patterns([], top_n=10)
        assert patterns == {}


class TestPrintParameterPatterns:
    """Test that print doesn't crash."""

    def test_no_crash(self, capsys):
        patterns = {"method": {"expensive": 40, "random": 10}}
        print_parameter_patterns(patterns, top_n=50)
        captured = capsys.readouterr()
        assert "method" in captured.out


class TestExportBestConfig:
    """Tests for export_best_config."""

    def test_creates_yaml_file(self, tmp_path):
        result = CalibrationResult(
            {"beta": 2.5, "job_search_method": "all_firms"},
            0.85,
            1,
            0,
            0,
            mean_score=0.83,
            std_score=0.02,
            pass_rate=1.0,
            combined_score=0.81,
        )

        path = export_best_config(result, "baseline", path=tmp_path / "test.yml")
        assert path.exists()

        with open(path) as f:
            content = f.read()
        assert "beta" in content
        assert "job_search_method" in content
        assert "Best calibration config" in content


class TestCompareConfigs:
    """Tests for compare_configs."""

    @patch("calibration.analysis.get_validation_func")
    def test_basic_comparison(self, mock_get_func):
        mr_default = MetricResult(
            name="unemployment",
            status="PASS",
            actual=0.08,
            target_desc="",
            score=0.7,
            weight=1.0,
            group=MetricGroup.TIME_SERIES,
        )
        mr_calibrated = MetricResult(
            name="unemployment",
            status="PASS",
            actual=0.05,
            target_desc="",
            score=0.9,
            weight=1.0,
            group=MetricGroup.TIME_SERIES,
        )

        def mock_validate(**kwargs):
            if "beta" in kwargs:
                return ValidationScore([mr_calibrated], 0.9, 1, 0, 0)
            return ValidationScore([mr_default], 0.7, 1, 0, 0)

        mock_get_func.return_value = mock_validate

        result = compare_configs(
            {},
            {"beta": 2.5},
            "baseline",
            seed=0,
            n_periods=50,
        )

        assert isinstance(result, ComparisonResult)
        assert result.calibrated_score > result.default_score
        assert len(result.improvements) == 1
        assert result.improvements[0][0] == "unemployment"


class TestPrintComparison:
    """Test that print doesn't crash."""

    def test_no_crash(self, capsys):
        result = ComparisonResult(
            scenario="baseline",
            default_metrics={"unemp": 0.08},
            calibrated_metrics={"unemp": 0.05},
            default_score=0.7,
            calibrated_score=0.9,
            improvements=[("unemp", 0.08, 0.05, -37.5)],
        )
        print_comparison(result)
        captured = capsys.readouterr()
        assert "BEFORE/AFTER" in captured.out
