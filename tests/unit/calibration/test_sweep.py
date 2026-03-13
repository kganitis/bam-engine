"""Tests for calibration.sweep module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from calibration.analysis import CalibrationResult
from calibration.sweep import StageResult, parse_stage, parse_stages, run_sweep


class TestParseStage:
    """Tests for parse_stage."""

    def test_single_param(self):
        label, grid = parse_stage("A:beta=0.5,1.0,2.5")
        assert label == "A"
        assert grid == {"beta": [0.5, 1.0, 2.5]}

    def test_multiple_params(self):
        label, grid = parse_stage("behavioral:beta=0.5,1.0 max_leverage=5,10,20")
        assert label == "behavioral"
        assert grid == {"beta": [0.5, 1.0], "max_leverage": [5, 10, 20]}

    def test_string_values(self):
        label, grid = parse_stage("market:job_search_method=vacancies_only,all_firms")
        assert label == "market"
        assert grid == {"job_search_method": ["vacancies_only", "all_firms"]}

    def test_integer_values(self):
        label, grid = parse_stage("credit:max_M=2,4")
        assert label == "credit"
        assert grid == {"max_M": [2, 4]}


class TestParseStages:
    """Tests for parse_stages."""

    def test_multiple_stages(self):
        stages = parse_stages(
            [
                "A:beta=0.5,1.0",
                "B:max_leverage=5,10",
            ]
        )
        assert len(stages) == 2
        assert stages[0][0] == "A"
        assert stages[1][0] == "B"


class TestRunSweep:
    """Tests for run_sweep orchestration."""

    @patch("calibration.sweep.run_tiered_stability")
    @patch("calibration.sweep.run_screening")
    def test_winners_carry_forward(self, mock_screen, mock_stab):
        """Stage 2 should use stage 1's winner as its base."""
        stage1_winner = CalibrationResult(
            params={"beta": 1.0, "max_M": 4},
            single_score=0.85,
            n_pass=10,
            n_warn=0,
            n_fail=0,
            combined_score=0.84,
            mean_score=0.85,
            pass_rate=1.0,
        )
        stage2_winner = CalibrationResult(
            params={"beta": 1.0, "max_M": 4, "max_leverage": 10},
            single_score=0.87,
            n_pass=10,
            n_warn=0,
            n_fail=0,
            combined_score=0.86,
            mean_score=0.87,
            pass_rate=1.0,
        )
        mock_screen.return_value = [MagicMock()]
        mock_stab.side_effect = [[stage1_winner], [stage2_winner]]

        results = run_sweep(
            base_params={"beta": 2.5, "max_M": 2},
            stages=[
                ("A", {"beta": [1.0, 2.5], "max_M": [2, 4]}),
                ("B", {"max_leverage": [5, 10, 20]}),
            ],
            scenario="baseline",
            n_workers=1,
        )

        assert len(results) == 2
        assert results[0].label == "A"
        assert results[0].winner_params["beta"] == 1.0
        assert results[1].label == "B"
        # Stage 2 should have stage 1's beta=1.0 carried forward
        assert results[1].winner_params["beta"] == 1.0

    @patch("calibration.sweep.run_tiered_stability")
    @patch("calibration.sweep.run_screening")
    def test_empty_results_keeps_base(self, mock_screen, mock_stab):
        mock_screen.return_value = []
        mock_stab.return_value = []

        results = run_sweep(
            base_params={"beta": 2.5},
            stages=[("A", {"beta": [1.0, 5.0]})],
            scenario="baseline",
            n_workers=1,
        )

        assert len(results) == 1
        assert not results[0].improved
        assert results[0].winner_params["beta"] == 2.5

    @patch("calibration.sweep.run_tiered_stability")
    @patch("calibration.sweep.run_screening")
    def test_returns_stage_results(self, mock_screen, mock_stab):
        winner = CalibrationResult(
            params={"beta": 1.0},
            single_score=0.85,
            n_pass=10,
            n_warn=0,
            n_fail=0,
            combined_score=0.84,
            mean_score=0.85,
            pass_rate=1.0,
        )
        mock_screen.return_value = [MagicMock()]
        mock_stab.return_value = [winner]

        results = run_sweep(
            base_params={"beta": 2.5},
            stages=[("test", {"beta": [1.0, 2.5]})],
            scenario="baseline",
            n_workers=1,
        )

        assert isinstance(results[0], StageResult)
        assert results[0].combined_score == 0.84
        assert results[0].n_candidates == 2
