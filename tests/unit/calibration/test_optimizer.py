"""Tests for calibration.optimizer module."""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from calibration.optimizer import (
    CalibrationResult,
    ComparisonResult,
    analyze_parameter_patterns,
    build_focused_grid,
    compare_configs,
    export_best_config,
    parse_stability_tiers,
    print_comparison,
    print_parameter_patterns,
    run_screening,
    run_tiered_stability,
    screen_single_seed,
)
from calibration.sensitivity import ParameterSensitivity, SensitivityResult
from validation.types import MetricGroup, MetricResult, StabilityResult, ValidationScore


def _make_score(total: float, group: str = "TIME_SERIES") -> ValidationScore:
    """Create a minimal ValidationScore for testing."""
    mr = MetricResult(
        name="test_metric",
        status="PASS",
        actual=total,
        target_desc="test",
        score=total,
        weight=1.0,
        group=MetricGroup[group],
    )
    return ValidationScore(
        metric_results=[mr],
        total_score=total,
        n_pass=1,
        n_warn=0,
        n_fail=0,
    )


def _make_stability(scores: list[float]) -> StabilityResult:
    """Create a minimal StabilityResult for testing."""
    seed_results = [_make_score(s) for s in scores]
    return StabilityResult(
        seed_results=seed_results,
        mean_score=statistics.mean(scores),
        std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
        min_score=min(scores),
        max_score=max(scores),
        pass_rate=1.0,
        n_seeds=len(scores),
        metric_stats={},
    )


class TestBuildFocusedGrid:
    """Tests for build_focused_grid."""

    def _make_sensitivity(self) -> SensitivityResult:
        return SensitivityResult(
            parameters=[
                ParameterSensitivity(
                    "high_param",
                    [1, 2, 3],
                    [0.5, 0.6, 0.55],
                    2,
                    0.6,
                    0.10,
                ),
                ParameterSensitivity(
                    "medium_param",
                    [10, 20],
                    [0.5, 0.52],
                    20,
                    0.52,
                    0.03,
                ),
                ParameterSensitivity(
                    "low_param",
                    ["a", "b"],
                    [0.5, 0.505],
                    "a",
                    0.505,
                    0.005,
                ),
            ],
            baseline_score=0.5,
        )

    def test_included_params_all_values(self):
        sensitivity = self._make_sensitivity()
        grid = {
            "high_param": [1, 2, 3],
            "medium_param": [10, 20],
            "low_param": ["a", "b"],
        }
        # Both high_param (Δ=0.10) and medium_param (Δ=0.03) are > 0.02
        search, _fixed = build_focused_grid(
            sensitivity, full_grid=grid, pruning_threshold=None
        )
        assert "high_param" in search
        assert search["high_param"] == [1, 2, 3]
        assert "medium_param" in search
        assert search["medium_param"] == [10, 20]

    def test_pruning_removes_bad_values(self):
        sensitivity = self._make_sensitivity()
        grid = {
            "high_param": [1, 2, 3],
            "medium_param": [10, 20],
            "low_param": ["a", "b"],
        }
        # high_param scores: [0.5, 0.6, 0.55], best=0.6
        # With pruning_threshold=0.06: value=1 gap=0.10 → dropped
        search, _fixed = build_focused_grid(
            sensitivity, full_grid=grid, pruning_threshold=0.06
        )
        assert "high_param" in search
        assert 1 not in search["high_param"]
        assert 2 in search["high_param"]
        assert 3 in search["high_param"]

    def test_fixed_params(self):
        sensitivity = self._make_sensitivity()
        grid = {
            "high_param": [1, 2, 3],
            "medium_param": [10, 20],
            "low_param": ["a", "b"],
        }
        search, fixed = build_focused_grid(
            sensitivity, full_grid=grid, pruning_threshold=None
        )
        assert "low_param" not in search
        assert "low_param" in fixed
        assert fixed["low_param"] == "a"


class TestScreenSingleSeed:
    """Tests for screen_single_seed."""

    @patch("calibration.optimizer.get_validation_funcs")
    def test_returns_calibration_result(self, mock_get_funcs):
        def mock_validate(**kwargs):
            return _make_score(0.75)

        mock_get_funcs.return_value = (mock_validate, None)

        result = screen_single_seed({"a": 1}, "baseline", 0, 50)
        assert isinstance(result, CalibrationResult)
        assert result.single_score == pytest.approx(0.75)
        assert result.n_pass == 1
        assert result.n_fail == 0


class TestRunScreening:
    """Tests for run_screening."""

    @patch("calibration.optimizer.ProcessPoolExecutor", ThreadPoolExecutor)
    @patch("calibration.optimizer.get_validation_funcs")
    def test_basic_screening(self, mock_get_funcs):
        """Use ThreadPoolExecutor so mocks propagate to worker threads."""
        call_idx = [0]
        scores = [0.7, 0.8, 0.6]

        def mock_validate(**kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            return _make_score(scores[idx % len(scores)])

        mock_get_funcs.return_value = (mock_validate, None)

        results = run_screening(
            [{"a": 1}, {"a": 2}, {"a": 3}],
            "baseline",
            n_workers=1,
            n_periods=50,
        )
        assert len(results) == 3
        # Sorted by score, best first
        assert results[0].single_score >= results[1].single_score


class TestParseStabilityTiers:
    """Tests for parse_stability_tiers."""

    def test_basic_parse(self):
        tiers = parse_stability_tiers("100:10,50:20,10:100")
        assert tiers == [(100, 10), (50, 20), (10, 100)]

    def test_single_tier(self):
        tiers = parse_stability_tiers("20:50")
        assert tiers == [(20, 50)]

    def test_spaces_handled(self):
        tiers = parse_stability_tiers("100 : 10 , 50 : 20")
        assert tiers == [(100, 10), (50, 20)]


class TestRunTieredStability:
    """Tests for run_tiered_stability."""

    @patch("calibration.optimizer.ProcessPoolExecutor", ThreadPoolExecutor)
    @patch("calibration.optimizer.get_validation_func")
    @patch("calibration.optimizer.save_checkpoint")
    @patch("calibration.optimizer.delete_checkpoint")
    def test_tiny_tiers(self, mock_del, mock_save, mock_get_func):
        """Test with tiny tiers [(3, 2), (2, 4)].

        Uses ThreadPoolExecutor so mocks propagate to worker threads.
        """

        def mock_validate(**kwargs):
            return _make_score(0.75)

        mock_get_func.return_value = mock_validate

        candidates = [
            CalibrationResult({"a": i}, 0.5 + i * 0.1, 1, 0, 0) for i in range(5)
        ]

        results = run_tiered_stability(
            candidates,
            "baseline",
            tiers=[(3, 2), (2, 4)],
            n_workers=1,
            n_periods=50,
        )

        # After tier 2, should have ≤ 2 configs
        assert len(results) <= 2
        # Each should have seed_scores
        for r in results:
            assert r.seed_scores is not None
            assert len(r.seed_scores) >= 2


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
            {"beta": 2.5, "firing_method": "expensive"},
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
        assert "firing_method" in content
        assert "Best calibration config" in content


class TestCompareConfigs:
    """Tests for compare_configs."""

    @patch("calibration.optimizer.get_validation_func")
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
