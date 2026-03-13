"""Tests for calibration.analysis module."""

from __future__ import annotations

from unittest.mock import patch

from calibration.analysis import (
    CalibrationResult,
    ComparisonResult,
    ScenarioResult,
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


class TestScenarioResult:
    """Tests for ScenarioResult dataclass."""

    def test_basic_creation(self):
        sr = ScenarioResult(
            mean_score=0.85,
            std_score=0.02,
            combined_score=0.83,
            pass_rate=0.95,
            n_fail=1,
            seed_scores=[0.84, 0.86, 0.85],
        )
        assert sr.mean_score == 0.85
        assert sr.n_fail == 1
        assert len(sr.seed_scores) == 3


class TestCalibrationResultScenarioResults:
    """Tests for the scenario_results field on CalibrationResult."""

    def test_default_is_none(self):
        r = CalibrationResult(
            params={"beta": 5}, single_score=0.8, n_pass=10, n_warn=1, n_fail=0
        )
        assert r.scenario_results is None

    def test_can_set_scenario_results(self):
        sr = ScenarioResult(
            mean_score=0.85,
            std_score=0.02,
            combined_score=0.83,
            pass_rate=0.95,
            n_fail=1,
            seed_scores=[0.84, 0.86],
        )
        r = CalibrationResult(
            params={"beta": 5},
            single_score=0.8,
            n_pass=10,
            n_warn=1,
            n_fail=0,
            scenario_results={"baseline": sr},
        )
        assert r.scenario_results["baseline"].mean_score == 0.85

    def test_from_cross_eval_factory(self):
        sr_bl = ScenarioResult(0.85, 0.02, 0.83, 1.0, 0, [0.85])
        sr_gp = ScenarioResult(0.80, 0.03, 0.78, 0.9, 2, [0.80])
        r = CalibrationResult.from_cross_eval(
            params={"beta": 5},
            scenario_results={"baseline": sr_bl, "growth_plus": sr_gp},
        )
        assert r.n_fail == 2  # total across scenarios
        assert r.pass_rate == 0.9  # min pass rate
        assert r.combined_score == 0.78  # min combined
        assert r.scenario_results is not None

    def test_exported_from_package(self):
        from calibration import ScenarioResult as SR

        assert SR is ScenarioResult
