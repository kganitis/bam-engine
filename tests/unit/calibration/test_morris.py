"""Tests for calibration.morris module."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest

from calibration.morris import (
    MorrisParameterEffect,
    MorrisResult,
    _config_key,
    _evaluate_config,
    _generate_trajectory,
    print_morris_report,
    run_morris_screening,
)
from calibration.sensitivity import SensitivityResult
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


class TestMorrisParameterEffect:
    """Tests for MorrisParameterEffect dataclass."""

    def test_basic_creation(self):
        e = MorrisParameterEffect(
            name="beta",
            mu=0.05,
            mu_star=0.08,
            sigma=0.03,
            elementary_effects=[0.05, 0.11, -0.03],
        )
        assert e.name == "beta"
        assert e.mu_star == 0.08
        assert e.sigma == 0.03
        assert len(e.elementary_effects) == 3

    def test_value_scores_default_empty(self):
        e = MorrisParameterEffect(
            name="x", mu=0.0, mu_star=0.0, sigma=0.0, elementary_effects=[]
        )
        assert e.value_scores == {}

    def test_value_scores_stored(self):
        e = MorrisParameterEffect(
            name="x",
            mu=0.0,
            mu_star=0.0,
            sigma=0.0,
            elementary_effects=[],
            value_scores={1: [0.7, 0.8], 2: [0.6]},
        )
        assert len(e.value_scores[1]) == 2
        assert e.value_scores[2] == [0.6]


class TestMorrisResult:
    """Tests for MorrisResult dataclass."""

    def _make_result(self) -> MorrisResult:
        return MorrisResult(
            effects=[
                MorrisParameterEffect(
                    "alpha",
                    mu=0.08,
                    mu_star=0.10,
                    sigma=0.04,
                    elementary_effects=[0.08, 0.12, 0.06],
                    value_scores={1: [0.7], 2: [0.8]},
                ),
                MorrisParameterEffect(
                    "beta",
                    mu=0.02,
                    mu_star=0.03,
                    sigma=0.015,
                    elementary_effects=[0.02, 0.04, 0.0],
                    value_scores={10: [0.72], 20: [0.73]},
                ),
                MorrisParameterEffect(
                    "gamma",
                    mu=0.005,
                    mu_star=0.008,
                    sigma=0.005,
                    elementary_effects=[0.005, 0.011, -0.001],
                    value_scores={100: [0.71], 200: [0.71]},
                ),
            ],
            n_trajectories=3,
            n_evaluations=10,
            scenario="baseline",
            avg_time_per_run=1.5,
            n_seeds=3,
        )

    def test_ranked_by_mu_star(self):
        result = self._make_result()
        ranked = result.ranked
        assert ranked[0].name == "alpha"
        assert ranked[1].name == "beta"
        assert ranked[2].name == "gamma"

    def test_get_important_dual_threshold(self):
        result = self._make_result()
        included, fixed = result.get_important(
            mu_star_threshold=0.02, sigma_threshold=0.02
        )
        # alpha: mu*=0.10 > 0.02 AND sigma=0.04 > 0.02 -> INCLUDE
        assert "alpha" in included
        # beta: mu*=0.03 > 0.02 -> INCLUDE
        assert "beta" in included
        # gamma: mu*=0.008 <= 0.02 AND sigma=0.005 <= 0.02 -> FIX
        assert "gamma" in fixed

    def test_get_important_sigma_catches_interactions(self):
        """A parameter with low mu* but high sigma should be INCLUDEd."""
        result = MorrisResult(
            effects=[
                MorrisParameterEffect(
                    "interaction_prone",
                    mu=0.001,
                    mu_star=0.015,  # Below threshold
                    sigma=0.05,  # Above threshold -- interaction!
                    elementary_effects=[0.05, -0.05, 0.02, -0.02],
                    value_scores={1: [0.7], 2: [0.72]},
                ),
            ],
            n_trajectories=4,
            n_evaluations=8,
        )
        included, fixed = result.get_important(
            mu_star_threshold=0.02, sigma_threshold=0.02
        )
        assert "interaction_prone" in included
        assert "interaction_prone" not in fixed

    def test_get_important_strict_threshold(self):
        result = self._make_result()
        included, fixed = result.get_important(
            mu_star_threshold=0.05, sigma_threshold=0.05
        )
        # alpha: mu*=0.10 > 0.05 -> INCLUDE
        assert "alpha" in included
        # beta: mu*=0.03 <= 0.05 AND sigma=0.015 <= 0.05 -> FIX
        assert "beta" in fixed
        # gamma: mu*=0.008 <= 0.05 AND sigma=0.005 <= 0.05 -> FIX
        assert "gamma" in fixed

    def test_metadata_stored(self):
        result = self._make_result()
        assert result.avg_time_per_run == 1.5
        assert result.n_seeds == 3
        assert result.n_trajectories == 3
        assert result.n_evaluations == 10


class TestGenerateTrajectory:
    """Tests for _generate_trajectory."""

    def test_trajectory_length(self):
        grid = {"a": [1, 2, 3], "b": ["x", "y"], "c": [10, 20, 30]}
        rng = np.random.default_rng(42)
        traj = _generate_trajectory(grid, rng)
        # p+1 configs for p parameters
        assert len(traj) == len(grid) + 1

    def test_consecutive_differ_by_at_most_one(self):
        grid = {"a": [1, 2, 3], "b": ["x", "y"], "c": [10, 20, 30]}
        rng = np.random.default_rng(42)
        traj = _generate_trajectory(grid, rng)

        for i in range(1, len(traj)):
            diffs = sum(1 for k in grid if traj[i][k] != traj[i - 1][k])
            assert diffs <= 1

    def test_each_multi_value_param_perturbed_once(self):
        grid = {"a": [1, 2], "b": ["x", "y"], "c": [10, 20]}
        rng = np.random.default_rng(42)
        traj = _generate_trajectory(grid, rng)

        changed: set[str] = set()
        for i in range(1, len(traj)):
            for k in grid:
                if traj[i][k] != traj[i - 1][k]:
                    changed.add(k)
        # All params should have been perturbed exactly once
        assert changed == set(grid.keys())

    def test_single_value_param_handled(self):
        grid = {"a": [1, 2], "b": [99]}  # b has single value
        rng = np.random.default_rng(42)
        traj = _generate_trajectory(grid, rng)

        assert len(traj) == 3  # 2 params + 1
        # b should be 99 in all configs
        for cfg in traj:
            assert cfg["b"] == 99

    def test_deterministic_with_same_seed(self):
        grid = {"a": [1, 2, 3], "b": ["x", "y"]}
        traj1 = _generate_trajectory(grid, np.random.default_rng(123))
        traj2 = _generate_trajectory(grid, np.random.default_rng(123))
        assert traj1 == traj2

    def test_different_seeds_differ(self):
        grid = {"a": [1, 2, 3], "b": ["x", "y"], "c": [10, 20, 30]}
        traj1 = _generate_trajectory(grid, np.random.default_rng(0))
        traj2 = _generate_trajectory(grid, np.random.default_rng(999))
        # Very unlikely to be identical with 3 params x 2-3 values
        assert traj1 != traj2

    def test_values_always_from_grid(self):
        grid = {"a": [1, 2, 3], "b": ["x", "y"]}
        rng = np.random.default_rng(42)
        traj = _generate_trajectory(grid, rng)

        for cfg in traj:
            assert cfg["a"] in grid["a"]
            assert cfg["b"] in grid["b"]


class TestConfigKey:
    """Tests for _config_key."""

    def test_same_config_same_key(self):
        c1 = {"a": 1, "b": "x"}
        c2 = {"b": "x", "a": 1}
        assert _config_key(c1) == _config_key(c2)

    def test_different_config_different_key(self):
        c1 = {"a": 1, "b": "x"}
        c2 = {"a": 1, "b": "y"}
        assert _config_key(c1) != _config_key(c2)


class TestToSensitivityResult:
    """Tests for MorrisResult.to_sensitivity_result."""

    def test_produces_sensitivity_result(self):
        result = MorrisResult(
            effects=[
                MorrisParameterEffect(
                    "alpha",
                    mu=0.08,
                    mu_star=0.10,
                    sigma=0.04,
                    elementary_effects=[0.08, 0.12],
                    value_scores={1: [0.7, 0.72], 2: [0.8, 0.82]},
                ),
            ],
            n_trajectories=2,
            n_evaluations=4,
            scenario="baseline",
            avg_time_per_run=1.0,
            n_seeds=1,
        )
        sr = result.to_sensitivity_result()
        assert isinstance(sr, SensitivityResult)
        assert sr.scenario == "baseline"
        assert len(sr.parameters) == 1

    def test_mu_star_maps_to_sensitivity(self):
        result = MorrisResult(
            effects=[
                MorrisParameterEffect(
                    "alpha",
                    mu=0.08,
                    mu_star=0.10,
                    sigma=0.04,
                    elementary_effects=[0.08, 0.12],
                    value_scores={1: [0.7], 2: [0.8]},
                ),
            ],
            n_trajectories=2,
            n_evaluations=4,
        )
        sr = result.to_sensitivity_result()
        assert sr.parameters[0].sensitivity == pytest.approx(0.10)

    def test_best_value_from_highest_avg_score(self):
        result = MorrisResult(
            effects=[
                MorrisParameterEffect(
                    "x",
                    mu=0.0,
                    mu_star=0.05,
                    sigma=0.01,
                    elementary_effects=[0.05],
                    value_scores={1: [0.6, 0.7], 2: [0.8, 0.9]},
                ),
            ],
            n_trajectories=1,
            n_evaluations=2,
        )
        sr = result.to_sensitivity_result()
        assert sr.parameters[0].best_value == 2
        assert sr.parameters[0].best_score == pytest.approx(0.85)

    def test_compatible_with_get_important(self):
        result = MorrisResult(
            effects=[
                MorrisParameterEffect(
                    "high",
                    mu=0.08,
                    mu_star=0.10,
                    sigma=0.04,
                    elementary_effects=[0.08, 0.12],
                    value_scores={1: [0.7], 2: [0.8]},
                ),
                MorrisParameterEffect(
                    "low",
                    mu=0.001,
                    mu_star=0.005,
                    sigma=0.003,
                    elementary_effects=[0.001],
                    value_scores={10: [0.71], 20: [0.71]},
                ),
            ],
            n_trajectories=2,
            n_evaluations=4,
        )
        sr = result.to_sensitivity_result()
        included, fixed = sr.get_important(sensitivity_threshold=0.02)
        assert "high" in included
        assert "low" in fixed

    def test_preserves_scenario_and_metadata(self):
        result = MorrisResult(
            effects=[
                MorrisParameterEffect(
                    "x",
                    mu=0.0,
                    mu_star=0.0,
                    sigma=0.0,
                    elementary_effects=[],
                    value_scores={1: [0.7]},
                ),
            ],
            n_trajectories=5,
            n_evaluations=10,
            scenario="growth_plus",
            avg_time_per_run=2.5,
            n_seeds=3,
        )
        sr = result.to_sensitivity_result()
        assert sr.scenario == "growth_plus"
        assert sr.avg_time_per_run == 2.5
        assert sr.n_seeds == 3


class TestEvaluateConfig:
    """Tests for _evaluate_config worker function."""

    @patch("calibration.morris.get_validation_func")
    def test_single_seed(self, mock_get_func):
        def mock_validate(**kwargs):
            return _make_score(0.75)

        mock_get_func.return_value = mock_validate

        config, score, elapsed = _evaluate_config(
            {"a": 1, "b": 2}, "baseline", [0], 100
        )
        assert config == {"a": 1, "b": 2}
        assert score == pytest.approx(0.75)
        assert elapsed >= 0.0

    @patch("calibration.morris.get_validation_func")
    def test_multi_seed_averaging(self, mock_get_func):
        def mock_validate(**kwargs):
            return _make_score(0.7 + 0.1 * (kwargs.get("seed", 0) % 2))

        mock_get_func.return_value = mock_validate

        _config, score, _elapsed = _evaluate_config({"a": 1}, "baseline", [0, 1], 100)
        # Average of 0.7 and 0.8
        assert score == pytest.approx(0.75)


class TestRunMorrisScreening:
    """Tests for run_morris_screening end-to-end."""

    @patch("calibration.morris.ProcessPoolExecutor", ThreadPoolExecutor)
    @patch("calibration.morris.get_validation_func")
    def test_basic_screening(self, mock_get_func):
        """Test Morris with tiny grid: 2 params x 2 values, 3 trajectories."""

        def mock_validate(**kwargs):
            # Score depends on param "a" but not "b"
            score = 0.70 + 0.10 * (kwargs.get("a", 1) - 1)
            return _make_score(score)

        mock_get_func.return_value = mock_validate

        result = run_morris_screening(
            scenario="baseline",
            grid={"a": [1, 2], "b": ["x", "y"]},
            n_trajectories=3,
            seed=42,
            n_seeds=1,
            n_periods=50,
            n_workers=1,
        )

        assert result.scenario == "baseline"
        assert result.n_trajectories == 3
        assert len(result.effects) == 2

        # "a" should have higher mu_star than "b"
        effect_map = {e.name: e for e in result.effects}
        assert effect_map["a"].mu_star > effect_map["b"].mu_star

    @patch("calibration.morris.ProcessPoolExecutor", ThreadPoolExecutor)
    @patch("calibration.morris.get_validation_func")
    def test_deduplication(self, mock_get_func):
        """Configs should be deduplicated across trajectories."""
        call_count = [0]

        def mock_validate(**kwargs):
            call_count[0] += 1
            return _make_score(0.75)

        mock_get_func.return_value = mock_validate

        result = run_morris_screening(
            scenario="baseline",
            grid={"a": [1, 2]},  # Only 2 possible configs
            n_trajectories=5,
            seed=42,
            n_seeds=1,
            n_periods=50,
            n_workers=1,
        )

        # At most 2 unique configs even with 5 trajectories
        assert result.n_evaluations <= 2

    @patch("calibration.morris.ProcessPoolExecutor", ThreadPoolExecutor)
    @patch("calibration.morris.get_validation_func")
    def test_to_sensitivity_result_integration(self, mock_get_func):
        """Morris output can be converted and used with build_focused_grid."""

        def mock_validate(**kwargs):
            score = 0.70 + 0.10 * (kwargs.get("a", 1) - 1)
            return _make_score(score)

        mock_get_func.return_value = mock_validate

        result = run_morris_screening(
            scenario="baseline",
            grid={"a": [1, 2], "b": ["x", "y"]},
            n_trajectories=3,
            seed=42,
            n_seeds=1,
            n_periods=50,
            n_workers=1,
        )

        sr = result.to_sensitivity_result()
        assert isinstance(sr, SensitivityResult)
        assert len(sr.parameters) == 2

    @patch("calibration.morris.ProcessPoolExecutor", ThreadPoolExecutor)
    @patch("calibration.morris.get_validation_func")
    def test_single_value_param_gets_zero_effect(self, mock_get_func):
        """Parameters with single values should appear with mu*=0."""

        def mock_validate(**kwargs):
            return _make_score(0.75)

        mock_get_func.return_value = mock_validate

        result = run_morris_screening(
            scenario="baseline",
            grid={"a": [1, 2], "fixed": [99]},
            n_trajectories=3,
            seed=42,
            n_seeds=1,
            n_periods=50,
            n_workers=1,
        )

        effect_map = {e.name: e for e in result.effects}
        assert "fixed" in effect_map
        assert effect_map["fixed"].mu_star == 0.0
        assert effect_map["fixed"].sigma == 0.0

    @patch("calibration.morris.ProcessPoolExecutor", ThreadPoolExecutor)
    @patch("calibration.morris.get_validation_func")
    def test_elementary_effects_populated(self, mock_get_func):
        """Each active param should have elementary effects from trajectories."""

        def mock_validate(**kwargs):
            return _make_score(0.70 + 0.05 * kwargs.get("a", 1))

        mock_get_func.return_value = mock_validate

        result = run_morris_screening(
            scenario="baseline",
            grid={"a": [1, 2], "b": ["x", "y"]},
            n_trajectories=4,
            seed=42,
            n_seeds=1,
            n_periods=50,
            n_workers=1,
        )

        effect_map = {e.name: e for e in result.effects}
        # Each active param should have at most n_trajectories EEs
        assert 0 < len(effect_map["a"].elementary_effects) <= 4


class TestPrintMorrisReport:
    """Test that print_morris_report doesn't crash."""

    def test_print_no_crash(self, capsys):
        result = MorrisResult(
            effects=[
                MorrisParameterEffect(
                    "alpha",
                    mu=0.08,
                    mu_star=0.10,
                    sigma=0.04,
                    elementary_effects=[0.08, 0.12],
                    value_scores={1: [0.7], 2: [0.8]},
                ),
                MorrisParameterEffect(
                    "beta",
                    mu=0.001,
                    mu_star=0.005,
                    sigma=0.003,
                    elementary_effects=[0.001, 0.009],
                    value_scores={10: [0.71], 20: [0.71]},
                ),
            ],
            n_trajectories=2,
            n_evaluations=6,
            scenario="baseline",
            avg_time_per_run=1.0,
            n_seeds=1,
        )
        print_morris_report(result)
        captured = capsys.readouterr()
        assert "MORRIS" in captured.out
        assert "alpha" in captured.out
        assert "INCLUDE" in captured.out
        assert "FIX" in captured.out

    def test_zero_mu_star_no_division_error(self, capsys):
        """sigma/mu* should handle mu*=0 without crashing."""
        result = MorrisResult(
            effects=[
                MorrisParameterEffect(
                    "zero_effect",
                    mu=0.0,
                    mu_star=0.0,
                    sigma=0.0,
                    elementary_effects=[0.0, 0.0],
                ),
            ],
            n_trajectories=2,
            n_evaluations=4,
        )
        print_morris_report(result)
        captured = capsys.readouterr()
        assert "MORRIS" in captured.out

    def test_interaction_reason_shown(self, capsys):
        """Parameters included only due to sigma should show '(s)' reason."""
        result = MorrisResult(
            effects=[
                MorrisParameterEffect(
                    "interaction_only",
                    mu=0.001,
                    mu_star=0.01,  # Below default 0.02 threshold
                    sigma=0.05,  # Above default 0.02 threshold
                    elementary_effects=[0.05, -0.05],
                ),
            ],
            n_trajectories=2,
            n_evaluations=4,
        )
        print_morris_report(result)
        captured = capsys.readouterr()
        assert "(s)" in captured.out
