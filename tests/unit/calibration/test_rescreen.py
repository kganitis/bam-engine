"""Tests for calibration.rescreen module."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from calibration.analysis import CalibrationResult
from calibration.morris import MorrisParameterEffect, MorrisResult
from calibration.rescreen import (
    compute_sensitivity_collapse,
    load_fixed_from_result,
    resolve_params,
    run_rescreen,
)


class TestResolveParams:
    """Tests for resolve_params."""

    def test_group_name_resolves(self):
        params = resolve_params("entry")
        assert "new_firm_size_factor" in params
        assert "new_firm_price_markup" in params

    def test_comma_separated_names(self):
        params = resolve_params("beta,max_M")
        assert params == ["beta", "max_M"]

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError, match="Unknown parameter group"):
            resolve_params("nonexistent_group")

    def test_behavioral_group(self):
        params = resolve_params("behavioral")
        assert "consumer_matching" in params

    def test_single_known_param(self):
        params = resolve_params("beta")
        assert params == ["beta"]


class TestLoadFixedFromResult:
    """Tests for load_fixed_from_result."""

    def test_loads_rank_1_params(self, tmp_path):
        from calibration.io import save_stability

        results = [
            CalibrationResult(
                params={"beta": 5, "max_M": 4},
                single_score=0.85,
                n_pass=10,
                n_warn=1,
                n_fail=0,
                mean_score=0.84,
                std_score=0.02,
                pass_rate=1.0,
                combined_score=0.82,
                seed_scores=[0.84, 0.84],
            ),
            CalibrationResult(
                params={"beta": 2.5, "max_M": 2},
                single_score=0.80,
                n_pass=9,
                n_warn=2,
                n_fail=0,
            ),
        ]
        path = tmp_path / "stability.json"
        save_stability(results, "test", path)
        fixed = load_fixed_from_result(path)
        assert fixed == {"beta": 5, "max_M": 4}


class TestComputeSensitivityCollapse:
    """Tests for compute_sensitivity_collapse."""

    def test_basic_collapse(self):
        phase1 = MorrisResult(
            effects=[
                MorrisParameterEffect("a", 0.1, 0.1, 0.05, [0.1]),
                MorrisParameterEffect("b", 0.02, 0.02, 0.01, [0.02]),
            ],
            n_trajectories=10,
            n_evaluations=100,
        )
        phase2 = MorrisResult(
            effects=[
                MorrisParameterEffect("a", 0.01, 0.01, 0.005, [0.01]),
                MorrisParameterEffect("b", 0.015, 0.015, 0.008, [0.015]),
            ],
            n_trajectories=10,
            n_evaluations=100,
        )
        collapse = compute_sensitivity_collapse(phase1, phase2)
        assert collapse["a"]["phase1_mu_star"] == 0.1
        assert collapse["a"]["phase2_mu_star"] == 0.01
        assert collapse["a"]["collapse_pct"] == pytest.approx(90.0)


class TestRunRescreen:
    """Tests for run_rescreen orchestration."""

    @patch("calibration.rescreen.run_morris_screening")
    @patch("calibration.rescreen.load_fixed_from_result")
    @patch("calibration.rescreen.get_parameter_grid")
    def test_calls_morris_with_correct_grid_and_fixed(
        self, mock_grid, mock_load, mock_morris
    ):
        mock_load.return_value = {"beta": 5, "max_M": 4, "price_init": 2.0}
        mock_grid.return_value = {
            "beta": [2.5, 5.0],
            "max_M": [2, 4],
            "price_init": [1.0, 2.0, 3.0],
        }
        mock_morris.return_value = MorrisResult(
            effects=[MorrisParameterEffect("beta", 0.01, 0.01, 0.005, [0.01])],
            n_trajectories=5,
            n_evaluations=50,
        )

        _result, _collapse = run_rescreen(
            scenario="baseline",
            fix_from=Path("/fake/stability.json"),
            params=["beta"],
            n_trajectories=5,
            n_seeds=3,
        )

        call_kwargs = mock_morris.call_args[1]
        assert "beta" in call_kwargs["grid"]
        assert "max_M" not in call_kwargs["grid"]
        assert call_kwargs["fixed_params"]["max_M"] == 4
        assert call_kwargs["fixed_params"]["price_init"] == 2.0

    @patch("calibration.rescreen.run_morris_screening")
    @patch("calibration.rescreen.load_fixed_from_result")
    @patch("calibration.rescreen.get_parameter_grid")
    def test_collapse_computed_when_phase1_provided(
        self, mock_grid, mock_load, mock_morris
    ):
        mock_load.return_value = {"beta": 5}
        mock_grid.return_value = {"beta": [2.5, 5.0]}
        phase2 = MorrisResult(
            effects=[MorrisParameterEffect("beta", 0.01, 0.01, 0.005, [0.01])],
            n_trajectories=5,
            n_evaluations=50,
        )
        mock_morris.return_value = phase2

        phase1 = MorrisResult(
            effects=[MorrisParameterEffect("beta", 0.1, 0.1, 0.05, [0.1])],
            n_trajectories=10,
            n_evaluations=100,
        )

        _, collapse = run_rescreen(
            scenario="baseline",
            fix_from=Path("/fake/stability.json"),
            params=["beta"],
            phase1_morris=phase1,
        )
        assert "beta" in collapse
        assert collapse["beta"]["collapse_pct"] == pytest.approx(90.0)

    @patch("calibration.rescreen.run_morris_screening")
    @patch("calibration.rescreen.load_fixed_from_result")
    @patch("calibration.rescreen.get_parameter_grid")
    def test_collapse_empty_when_no_phase1(self, mock_grid, mock_load, mock_morris):
        mock_load.return_value = {"beta": 5}
        mock_grid.return_value = {"beta": [2.5, 5.0]}
        mock_morris.return_value = MorrisResult(
            effects=[MorrisParameterEffect("beta", 0.01, 0.01, 0.005, [0.01])],
            n_trajectories=5,
            n_evaluations=50,
        )

        _, collapse = run_rescreen(
            scenario="baseline",
            fix_from=Path("/fake/stability.json"),
            params=["beta"],
        )
        assert collapse == {}


class TestRunRescreenPhase:
    """Tests for CLI entry point validation."""

    def test_missing_fix_from_raises(self):
        from calibration.rescreen import run_rescreen_phase

        args = argparse.Namespace(fix_from=None, params="entry")
        with pytest.raises(SystemExit, match="--fix-from"):
            run_rescreen_phase(args)

    def test_missing_params_raises(self):
        from calibration.rescreen import run_rescreen_phase

        args = argparse.Namespace(fix_from="some.json", params=None)
        with pytest.raises(SystemExit, match="--params"):
            run_rescreen_phase(args)
