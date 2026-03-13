"""Tests for calibration.cross_eval module."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from calibration.analysis import CalibrationResult, ScenarioResult
from calibration.cross_eval import (
    _load_configs,
    compute_scenario_tension,
    evaluate_cross_scenario,
    rank_cross_scenario,
)


class TestRankCrossScenario:
    """Tests for cross-scenario ranking strategies."""

    def _make_results(self) -> list[CalibrationResult]:
        """Two configs with different cross-scenario profiles."""
        return [
            CalibrationResult(
                params={"id": "A"},
                single_score=0.85,
                n_pass=10,
                n_warn=1,
                n_fail=0,
                scenario_results={
                    "baseline": ScenarioResult(0.87, 0.02, 0.85, 1.0, 0, [0.87]),
                    "growth_plus": ScenarioResult(0.80, 0.05, 0.76, 0.80, 4, [0.80]),
                },
            ),
            CalibrationResult(
                params={"id": "B"},
                single_score=0.82,
                n_pass=10,
                n_warn=1,
                n_fail=0,
                scenario_results={
                    "baseline": ScenarioResult(0.83, 0.03, 0.80, 0.95, 1, [0.83]),
                    "growth_plus": ScenarioResult(0.82, 0.03, 0.79, 0.95, 1, [0.82]),
                },
            ),
        ]

    def test_stability_first_ranks_by_min_pass_rate(self):
        results = self._make_results()
        ranked = rank_cross_scenario(results, "stability-first")
        # B has min_pass=0.95, A has min_pass=0.80 -> B wins
        assert ranked[0].params["id"] == "B"

    def test_score_first_ranks_by_min_combined(self):
        results = self._make_results()
        ranked = rank_cross_scenario(results, "score-first")
        # A: min_combined=0.76, B: min_combined=0.79 -> B wins
        assert ranked[0].params["id"] == "B"

    def test_balanced_uses_geometric_mean(self):
        results = self._make_results()
        ranked = rank_cross_scenario(results, "balanced")
        # geomean(A) = sqrt(0.85*0.76)=0.804, geomean(B) = sqrt(0.80*0.79)=0.795
        # A wins on balanced
        assert ranked[0].params["id"] == "A"

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown ranking"):
            rank_cross_scenario([], "nonexistent")


class TestLoadConfigs:
    """Tests for config loading with format auto-detection."""

    def test_loads_from_json_result_file(self, tmp_path):
        path = tmp_path / "results.json"
        data = {
            "results": [
                {"params": {"beta": 5.0}, "single_score": 0.85},
                {"params": {"beta": 2.5}, "single_score": 0.80},
            ]
        }
        path.write_text(json.dumps(data))
        configs = _load_configs(path)
        assert len(configs) == 2
        assert configs[0]["beta"] == 5.0

    def test_loads_from_yaml_grid(self, tmp_path):
        path = tmp_path / "grid.yml"
        path.write_text("beta: [2.5, 5.0]\nmax_M: [2, 4]\n")
        configs = _load_configs(path)
        assert len(configs) == 4  # 2 x 2 combinations


class TestEvaluateCrossScenario:
    """Tests for cross-scenario evaluation with mocked seeds."""

    @patch("calibration.cross_eval._evaluate_single_seed")
    def test_populates_scenario_results(self, mock_eval):
        mock_eval.return_value = ({"beta": 5}, "baseline", 0.85, 0)

        results = evaluate_cross_scenario(
            configs=[{"beta": 5}],
            scenarios=["baseline", "growth_plus"],
            n_seeds=2,
            n_workers=1,
        )
        assert len(results) == 1
        assert "baseline" in results[0].scenario_results
        assert "growth_plus" in results[0].scenario_results
        assert results[0].scenario_results["baseline"].pass_rate == 1.0


class TestComputeScenarioTension:
    """Tests for scenario tension analysis."""

    def test_detects_different_preferred_values(self):
        results = [
            CalibrationResult(
                params={"beta": 5.0, "max_M": 4},
                single_score=0.85,
                n_pass=10,
                n_warn=0,
                n_fail=0,
                scenario_results={
                    "baseline": ScenarioResult(0.90, 0.01, 0.89, 1.0, 0, [0.90]),
                    "growth_plus": ScenarioResult(0.70, 0.05, 0.67, 0.8, 4, [0.70]),
                },
            ),
            CalibrationResult(
                params={"beta": 2.5, "max_M": 2},
                single_score=0.82,
                n_pass=10,
                n_warn=0,
                n_fail=0,
                scenario_results={
                    "baseline": ScenarioResult(0.80, 0.02, 0.78, 0.9, 2, [0.80]),
                    "growth_plus": ScenarioResult(0.85, 0.02, 0.83, 1.0, 0, [0.85]),
                },
            ),
        ]
        tension = compute_scenario_tension(results, ["baseline", "growth_plus"])
        # baseline prefers beta=5.0, growth_plus prefers beta=2.5
        assert "beta" in tension

    def test_no_tension_when_same_winner(self):
        sr = ScenarioResult(0.85, 0.02, 0.83, 1.0, 0, [0.85])
        results = [
            CalibrationResult(
                params={"beta": 5.0},
                single_score=0.85,
                n_pass=10,
                n_warn=0,
                n_fail=0,
                scenario_results={"baseline": sr, "growth_plus": sr},
            ),
        ]
        tension = compute_scenario_tension(results, ["baseline", "growth_plus"])
        assert tension == {}
