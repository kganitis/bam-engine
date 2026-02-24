"""Tests for calibration.sensitivity module."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from calibration.sensitivity import (
    ParameterSensitivity,
    SensitivityResult,
    _evaluate_param_value,
    print_sensitivity_report,
    run_sensitivity_analysis,
)
from tests.helpers.scores import make_score as _make_score


class TestParameterSensitivity:
    """Tests for ParameterSensitivity dataclass."""

    def test_basic_creation(self):
        ps = ParameterSensitivity(
            name="beta",
            values=[1.5, 2.0, 3.0],
            scores=[0.7, 0.8, 0.75],
            best_value=2.0,
            best_score=0.8,
            sensitivity=0.1,
        )
        assert ps.name == "beta"
        assert ps.sensitivity == 0.1

    def test_group_scores_default_empty(self):
        ps = ParameterSensitivity(
            name="x",
            values=[1],
            scores=[0.5],
            best_value=1,
            best_score=0.5,
            sensitivity=0.0,
        )
        assert ps.group_scores == {}

    def test_group_scores_stored(self):
        ps = ParameterSensitivity(
            name="x",
            values=[1, 2],
            scores=[0.5, 0.6],
            best_value=2,
            best_score=0.6,
            sensitivity=0.1,
            group_scores={"TIME_SERIES": [0.4, 0.5], "CURVES": [0.1, 0.1]},
        )
        assert "TIME_SERIES" in ps.group_scores
        assert len(ps.group_scores["TIME_SERIES"]) == 2


class TestSensitivityResult:
    """Tests for SensitivityResult dataclass."""

    def _make_result(self) -> SensitivityResult:
        return SensitivityResult(
            parameters=[
                ParameterSensitivity(
                    "alpha",
                    [1, 2],
                    [0.5, 0.6],
                    2,
                    0.6,
                    0.1,
                ),
                ParameterSensitivity(
                    "beta",
                    [1, 2],
                    [0.5, 0.55],
                    2,
                    0.55,
                    0.05,
                ),
                ParameterSensitivity(
                    "gamma",
                    [1, 2],
                    [0.5, 0.51],
                    2,
                    0.51,
                    0.01,
                ),
            ],
            baseline_score=0.5,
            scenario="baseline",
            avg_time_per_run=1.5,
            n_seeds=3,
        )

    def test_ranked_by_sensitivity(self):
        result = self._make_result()
        ranked = result.ranked
        assert ranked[0].name == "alpha"
        assert ranked[1].name == "beta"
        assert ranked[2].name == "gamma"

    def test_get_important_categorization(self):
        result = self._make_result()
        included, fixed = result.get_important(sensitivity_threshold=0.02)
        # alpha (Δ=0.1) and beta (Δ=0.05) both > 0.02 → included
        assert "alpha" in included
        assert "beta" in included
        # gamma (Δ=0.01) ≤ 0.02 → fixed
        assert "gamma" in fixed

    def test_get_important_strict_threshold(self):
        result = self._make_result()
        included, fixed = result.get_important(sensitivity_threshold=0.05)
        # alpha (Δ=0.1) > 0.05 → included
        assert "alpha" in included
        # beta (Δ=0.05) NOT > 0.05 (strict) → fixed
        assert "beta" in fixed
        assert "gamma" in fixed

    def test_avg_time_stored(self):
        result = self._make_result()
        assert result.avg_time_per_run == 1.5

    def test_n_seeds_stored(self):
        result = self._make_result()
        assert result.n_seeds == 3


class TestPruneGrid:
    """Tests for SensitivityResult.prune_grid."""

    def _make_result(self) -> SensitivityResult:
        return SensitivityResult(
            parameters=[
                ParameterSensitivity(
                    "alpha",
                    [1, 2, 3],
                    [0.80, 0.75, 0.60],  # best=0.80 at value=1
                    1,
                    0.80,
                    0.20,
                ),
                ParameterSensitivity(
                    "beta",
                    [10, 20],
                    [0.70, 0.72],  # best=0.72 at value=20
                    20,
                    0.72,
                    0.02,
                ),
            ],
            baseline_score=0.5,
        )

    def test_none_threshold_returns_unchanged(self):
        result = self._make_result()
        grid = {"alpha": [1, 2, 3], "beta": [10, 20]}
        pruned = result.prune_grid(grid, None)
        assert pruned == grid

    def test_drops_bad_values(self):
        result = self._make_result()
        grid = {"alpha": [1, 2, 3]}
        # threshold=0.06: keep values within 0.06 of best (0.80)
        # value=1: gap=0.00 → keep, value=2: gap=0.05 → keep, value=3: gap=0.20 → drop
        pruned = result.prune_grid(grid, 0.06)
        assert pruned["alpha"] == [1, 2]

    def test_keeps_best_value_always(self):
        result = self._make_result()
        grid = {"alpha": [1, 2, 3]}
        # Very tight threshold: only best survives
        pruned = result.prune_grid(grid, 0.01)
        assert 1 in pruned["alpha"]  # best value kept

    def test_unknown_params_kept(self):
        result = self._make_result()
        grid = {"unknown_param": [1, 2, 3]}
        pruned = result.prune_grid(grid, 0.01)
        assert pruned["unknown_param"] == [1, 2, 3]

    def test_unknown_values_kept(self):
        result = self._make_result()
        # Value 99 not in OAT results → kept conservatively
        grid = {"alpha": [1, 99]}
        pruned = result.prune_grid(grid, 0.01)
        assert 99 in pruned["alpha"]

    def test_order_preserved(self):
        result = self._make_result()
        grid = {"alpha": [3, 2, 1]}
        pruned = result.prune_grid(grid, 0.06)
        # value=3 dropped (gap=0.20), value=2 kept (gap=0.05), value=1 kept (gap=0.00)
        assert pruned["alpha"] == [2, 1]


class TestEvaluateParamValue:
    """Tests for _evaluate_param_value worker function."""

    @patch("calibration.sensitivity.get_validation_func")
    def test_single_seed(self, mock_get_func):
        def mock_validate(**kwargs):
            return _make_score(0.75)

        mock_get_func.return_value = mock_validate

        value, score, groups, elapsed = _evaluate_param_value(
            "beta",
            2.0,
            {},
            "baseline",
            [0],
            100,
        )
        assert value == 2.0
        assert score == pytest.approx(0.75)
        assert elapsed >= 0.0
        assert isinstance(groups, dict)

    @patch("calibration.sensitivity.get_validation_func")
    def test_multi_seed_averaging(self, mock_get_func):
        call_count = [0]

        def mock_validate(**kwargs):
            call_count[0] += 1
            # Different scores for different seeds
            return _make_score(0.7 + 0.1 * (kwargs.get("seed", 0) % 2))

        mock_get_func.return_value = mock_validate

        value, score, _groups, _elapsed = _evaluate_param_value(
            "beta",
            2.0,
            {},
            "baseline",
            [0, 1],
            100,
        )
        assert value == 2.0
        # Average of 0.7 and 0.8
        assert score == pytest.approx(0.75)


class TestRunSensitivityAnalysis:
    """Tests for run_sensitivity_analysis."""

    @patch("calibration.sensitivity.ProcessPoolExecutor", ThreadPoolExecutor)
    @patch("calibration.sensitivity.get_validation_func")
    def test_basic_oat(self, mock_get_func):
        """Test OAT with tiny grid (2 params × 2 values).

        Uses ThreadPoolExecutor so mocks propagate to worker threads.
        """
        scores = {
            "a=1": 0.70,
            "a=2": 0.80,
            "b=x": 0.72,
            "b=y": 0.73,
        }

        def mock_validate(**kwargs):
            config = kwargs.copy()
            # Identify which param is being varied
            if "a" in config and config["a"] != 1:
                return _make_score(scores.get(f"a={config['a']}", 0.70))
            if "b" in config and config["b"] != "x":
                return _make_score(scores.get(f"b={config['b']}", 0.72))
            return _make_score(0.70)

        mock_get_func.return_value = mock_validate

        result = run_sensitivity_analysis(
            scenario="baseline",
            grid={"a": [1, 2], "b": ["x", "y"]},
            baseline={"a": 1, "b": "x"},
            n_seeds=1,
            n_periods=50,
            n_workers=1,
        )

        assert result.scenario == "baseline"
        assert result.baseline_score == pytest.approx(0.70)
        assert len(result.parameters) == 2
        assert result.avg_time_per_run > 0

        # "a" should be more sensitive than "b"
        ranked = result.ranked
        assert ranked[0].name == "a"
        assert ranked[0].sensitivity == pytest.approx(0.10)
        assert ranked[1].name == "b"
        assert ranked[1].sensitivity == pytest.approx(0.03)

    @patch("calibration.sensitivity.ProcessPoolExecutor", ThreadPoolExecutor)
    @patch("calibration.sensitivity.get_validation_func")
    def test_multi_seed_oat(self, mock_get_func):
        """Test multi-seed averaging."""

        def mock_validate(**kwargs):
            return _make_score(0.75)

        mock_get_func.return_value = mock_validate

        result = run_sensitivity_analysis(
            scenario="baseline",
            grid={"a": [1, 2]},
            baseline={"a": 1},
            n_seeds=2,
            n_periods=50,
            n_workers=1,
        )
        assert result.n_seeds == 2


class TestPrintSensitivityReport:
    """Test that print_sensitivity_report doesn't crash."""

    def test_print_no_crash(self, capsys):
        result = SensitivityResult(
            parameters=[
                ParameterSensitivity(
                    "alpha",
                    [1, 2],
                    [0.5, 0.6],
                    2,
                    0.6,
                    0.1,
                    group_scores={"TIME_SERIES": [0.3, 0.4]},
                ),
            ],
            baseline_score=0.5,
            scenario="baseline",
            avg_time_per_run=1.0,
            n_seeds=1,
        )
        print_sensitivity_report(result)
        captured = capsys.readouterr()
        assert "SENSITIVITY ANALYSIS RESULTS" in captured.out
        assert "alpha" in captured.out
        assert "INCLUDE" in captured.out
        assert "FIX" in captured.out
        assert "HIGH" not in captured.out
        assert "MEDIUM" not in captured.out
        assert "LOW" not in captured.out
