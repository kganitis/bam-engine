"""Tests for calibration.cli module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from calibration.cli import (
    _load_screening,
    _load_sensitivity,
    _save_screening,
    _save_sensitivity,
    main,
    print_results,
)
from calibration.optimizer import CalibrationResult
from calibration.sensitivity import ParameterSensitivity, SensitivityResult


class TestPrintResults:
    """Tests for print_results."""

    def test_no_crash_with_combined(self, capsys):
        results = [
            CalibrationResult(
                {"a": 1},
                0.8,
                1,
                0,
                0,
                mean_score=0.78,
                std_score=0.02,
                pass_rate=1.0,
                combined_score=0.76,
                seed_scores=[0.78, 0.80],
            ),
        ]
        print_results(results, top_n=5)
        captured = capsys.readouterr()
        assert "TOP" in captured.out
        assert "0.76" in captured.out

    def test_no_crash_without_stability(self, capsys):
        results = [CalibrationResult({"a": 1}, 0.8, 1, 0, 0)]
        print_results(results, top_n=5)
        captured = capsys.readouterr()
        assert "TOP" in captured.out


class TestSensitivitySerialization:
    """Tests for sensitivity JSON round-trip."""

    def test_round_trip(self, tmp_path):
        original = SensitivityResult(
            parameters=[
                ParameterSensitivity(
                    "beta",
                    [1.5, 2.0],
                    [0.7, 0.8],
                    2.0,
                    0.8,
                    0.1,
                    group_scores={"TIME_SERIES": [0.5, 0.6]},
                ),
            ],
            baseline_score=0.65,
            scenario="baseline",
            avg_time_per_run=12.3,
            n_seeds=3,
        )

        path = tmp_path / "test_sens.json"
        _save_sensitivity(original, path)
        assert path.exists()

        loaded = _load_sensitivity(path)
        assert loaded.scenario == "baseline"
        assert loaded.baseline_score == pytest.approx(0.65)
        assert loaded.avg_time_per_run == pytest.approx(12.3)
        assert loaded.n_seeds == 3
        assert len(loaded.parameters) == 1
        assert loaded.parameters[0].name == "beta"
        assert loaded.parameters[0].sensitivity == pytest.approx(0.1)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            _load_sensitivity(Path("/nonexistent/file.json"))


class TestScreeningSerialization:
    """Tests for screening JSON round-trip."""

    def test_round_trip(self, tmp_path):
        results = [
            CalibrationResult({"a": 1, "b": "x"}, 0.8, 1, 0, 0),
            CalibrationResult({"a": 2, "b": "y"}, 0.7, 0, 1, 0),
        ]
        sensitivity = SensitivityResult(
            parameters=[],
            baseline_score=0.5,
            scenario="baseline",
            avg_time_per_run=5.0,
        )
        grid = {"a": [1, 2]}
        fixed = {"b": "x"}
        patterns = {"a": {1: 30, 2: 20}}

        path = tmp_path / "test_screen.json"
        _save_screening(results, sensitivity, grid, fixed, patterns, "baseline", path)
        assert path.exists()

        loaded_results, avg_time = _load_screening(path)
        assert len(loaded_results) == 2
        assert loaded_results[0].single_score == pytest.approx(0.8)
        assert avg_time == pytest.approx(5.0)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            _load_screening(Path("/nonexistent/file.json"))


class TestCLIPhaseParsing:
    """Tests for CLI argument parsing and phase dispatch."""

    @patch("calibration.cli._run_sensitivity_phase")
    def test_phase_sensitivity(self, mock_sens):
        mock_sens.return_value = MagicMock()
        with patch(
            "sys.argv",
            [
                "calibration",
                "--phase",
                "sensitivity",
                "--scenario",
                "baseline",
                "--periods",
                "50",
            ],
        ):
            main()
        mock_sens.assert_called_once()

    @patch("calibration.cli._run_grid_phase")
    def test_phase_grid(self, mock_grid):
        mock_grid.return_value = []
        with patch(
            "sys.argv", ["calibration", "--phase", "grid", "--scenario", "baseline"]
        ):
            main()
        mock_grid.assert_called_once()

    @patch("calibration.cli._run_stability_phase")
    def test_phase_stability(self, mock_stab):
        mock_stab.return_value = []
        with patch(
            "sys.argv",
            ["calibration", "--phase", "stability", "--scenario", "baseline"],
        ):
            main()
        mock_stab.assert_called_once()

    @patch("calibration.cli._run_pairwise_phase")
    def test_phase_pairwise(self, mock_pair):
        with patch(
            "sys.argv", ["calibration", "--phase", "pairwise", "--scenario", "baseline"]
        ):
            main()
        mock_pair.assert_called_once()

    def test_buffer_stock_accepted(self):
        """Verify buffer_stock is a valid scenario choice."""
        from calibration.cli import main

        # Just check the parser accepts it (don't run)
        with patch(
            "sys.argv",
            ["calibration", "--scenario", "buffer_stock", "--phase", "sensitivity"],
        ):
            with patch("calibration.cli._run_sensitivity_phase"):
                main()

    @patch("calibration.cli._run_sensitivity_phase")
    @patch("calibration.cli._run_grid_phase")
    @patch("calibration.cli._run_stability_phase")
    def test_all_phases_sequential(self, mock_stab, mock_grid, mock_sens):
        """No --phase runs all phases."""
        mock_sens.return_value = MagicMock()
        mock_grid.return_value = [CalibrationResult({"a": 1}, 0.8, 1, 0, 0)]
        mock_stab.return_value = []

        with patch("sys.argv", ["calibration"]):
            main()

        mock_sens.assert_called_once()
        mock_grid.assert_called_once()
        mock_stab.assert_called_once()


class TestCLIStabilityTiersParsing:
    """Tests for --stability-tiers parsing."""

    def test_tiers_in_args(self):
        with patch("sys.argv", ["calibration", "--stability-tiers", "50:10,20:30"]):
            from calibration.optimizer import parse_stability_tiers

            tiers = parse_stability_tiers("50:10,20:30")
            assert tiers == [(50, 10), (20, 30)]


class TestPhasePrerequisiteErrors:
    """Tests for helpful error messages when prerequisite files missing."""

    def test_grid_without_sensitivity(self, tmp_path):
        """--phase grid should error if no sensitivity file."""
        import argparse

        from calibration.cli import _run_grid_phase

        args = argparse.Namespace(
            scenario="baseline",
            workers=1,
            periods=50,
            sensitivity_threshold=0.02,
            pruning_threshold="auto",
            resume=False,
        )

        # Patch OUTPUT_DIR to tmp_path so no file exists
        with patch("calibration.cli.OUTPUT_DIR", tmp_path):
            with pytest.raises(FileNotFoundError, match="sensitivity"):
                _run_grid_phase(args)

    def test_stability_without_screening(self, tmp_path):
        """--phase stability should error if no screening file."""
        import argparse

        from calibration.cli import _run_stability_phase

        args = argparse.Namespace(
            scenario="baseline",
            workers=1,
            periods=50,
            stability_tiers="100:10,50:20,10:100",
            resume=False,
        )

        with patch("calibration.cli.OUTPUT_DIR", tmp_path):
            with pytest.raises(FileNotFoundError, match="screening"):
                _run_stability_phase(args)
