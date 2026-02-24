"""Tests for calibration.cli module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from calibration.analysis import CalibrationResult
from calibration.cli import main, print_results


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


class TestCLIPhaseParsing:
    """Tests for CLI argument parsing and phase dispatch."""

    @patch("calibration.cli._run_sensitivity_phase")
    @patch("calibration.cli.create_run_dir")
    def test_phase_sensitivity(self, mock_run_dir, mock_sens, tmp_path):
        mock_run_dir.return_value = tmp_path
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
    @patch("calibration.cli.create_run_dir")
    def test_phase_grid(self, mock_run_dir, mock_grid, tmp_path):
        mock_run_dir.return_value = tmp_path
        mock_grid.return_value = []
        with patch(
            "sys.argv", ["calibration", "--phase", "grid", "--scenario", "baseline"]
        ):
            main()
        mock_grid.assert_called_once()

    @patch("calibration.cli._run_stability_phase")
    @patch("calibration.cli.create_run_dir")
    def test_phase_stability(self, mock_run_dir, mock_stab, tmp_path):
        mock_run_dir.return_value = tmp_path
        mock_stab.return_value = []
        with patch(
            "sys.argv",
            ["calibration", "--phase", "stability", "--scenario", "baseline"],
        ):
            main()
        mock_stab.assert_called_once()

    @patch("calibration.cli._run_pairwise_phase")
    @patch("calibration.cli.create_run_dir")
    def test_phase_pairwise(self, mock_run_dir, mock_pair, tmp_path):
        mock_run_dir.return_value = tmp_path
        with patch(
            "sys.argv", ["calibration", "--phase", "pairwise", "--scenario", "baseline"]
        ):
            main()
        mock_pair.assert_called_once()

    @patch("calibration.cli.create_run_dir")
    def test_buffer_stock_accepted(self, mock_run_dir, tmp_path):
        """Verify buffer_stock is a valid scenario choice."""
        mock_run_dir.return_value = tmp_path
        with patch(
            "sys.argv",
            ["calibration", "--scenario", "buffer_stock", "--phase", "sensitivity"],
        ):
            with patch("calibration.cli._run_sensitivity_phase"):
                main()

    @patch("calibration.cli._run_sensitivity_phase")
    @patch("calibration.cli._run_grid_phase")
    @patch("calibration.cli._run_stability_phase")
    @patch("calibration.cli.generate_full_report")
    @patch("calibration.cli.create_run_dir")
    def test_all_phases_sequential(
        self, mock_run_dir, mock_report, mock_stab, mock_grid, mock_sens, tmp_path
    ):
        """No --phase runs all phases."""
        mock_run_dir.return_value = tmp_path
        mock_sens.return_value = MagicMock()
        mock_grid.return_value = [CalibrationResult({"a": 1}, 0.8, 1, 0, 0)]
        mock_stab.return_value = []

        with patch("sys.argv", ["calibration"]):
            main()

        mock_sens.assert_called_once()
        mock_grid.assert_called_once()
        mock_stab.assert_called_once()


class TestCLINewFlags:
    """Tests for new CLI flags (--rank-by, --k-factor, --grid, --output-dir)."""

    @patch("calibration.cli._run_stability_phase")
    @patch("calibration.cli.create_run_dir")
    def test_rank_by_flag(self, mock_run_dir, mock_stab, tmp_path):
        mock_run_dir.return_value = tmp_path
        mock_stab.return_value = []
        with patch(
            "sys.argv",
            [
                "calibration",
                "--phase",
                "stability",
                "--rank-by",
                "stability",
                "--k-factor",
                "1.5",
            ],
        ):
            main()
        # Verify args were passed (mock captures the Namespace)
        args = mock_stab.call_args[0][0]
        assert args.rank_by == "stability"
        assert args.k_factor == 1.5

    @patch("calibration.cli._run_grid_phase")
    @patch("calibration.cli.create_run_dir")
    def test_grid_flag(self, mock_run_dir, mock_grid, tmp_path):
        mock_run_dir.return_value = tmp_path
        mock_grid.return_value = []
        with patch(
            "sys.argv",
            ["calibration", "--phase", "grid", "--grid", "custom.yaml"],
        ):
            main()
        mock_grid.assert_called_once()

    def test_output_dir_flag(self, tmp_path):
        out_dir = tmp_path / "my_output"
        with patch(
            "sys.argv",
            [
                "calibration",
                "--phase",
                "sensitivity",
                "--output-dir",
                str(out_dir),
            ],
        ):
            with patch("calibration.cli._run_sensitivity_phase"):
                main()
        assert out_dir.exists()


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
            grid=None,
            fixed=None,
        )

        with patch("calibration.cli.OUTPUT_DIR", tmp_path):
            with pytest.raises(FileNotFoundError, match="[Ss]ensitivity"):
                _run_grid_phase(args, run_dir=tmp_path)

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
            rank_by="combined",
            k_factor=1.0,
        )

        with patch("calibration.cli.OUTPUT_DIR", tmp_path):
            with pytest.raises(FileNotFoundError, match="[Ss]creening"):
                _run_stability_phase(args, run_dir=tmp_path)
