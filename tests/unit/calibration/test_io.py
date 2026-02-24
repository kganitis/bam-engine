"""Tests for calibration.io module."""

from __future__ import annotations

import json

import pytest

from calibration.analysis import CalibrationResult
from calibration.io import (
    create_run_dir,
    load_morris,
    load_pairwise,
    load_screening,
    load_sensitivity,
    load_stability,
    save_morris,
    save_pairwise,
    save_screening,
    save_sensitivity,
    save_stability,
)
from calibration.morris import MorrisParameterEffect, MorrisResult
from calibration.sensitivity import (
    PairInteraction,
    PairwiseResult,
    ParameterSensitivity,
    SensitivityResult,
)


class TestCreateRunDir:
    """Tests for create_run_dir."""

    def test_creates_directory(self, tmp_path):
        run_dir = create_run_dir("baseline", output_dir=tmp_path)
        assert run_dir.exists()
        assert run_dir.is_dir()
        assert "baseline" in run_dir.name

    def test_timestamp_format(self, tmp_path):
        run_dir = create_run_dir("growth_plus", output_dir=tmp_path)
        name = run_dir.name
        # Format: YYYY-MM-DD_HHMMSS_scenario
        parts = name.split("_")
        assert len(parts) >= 3
        assert "growth" in name


class TestSensitivityRoundtrip:
    """Tests for sensitivity save/load."""

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
        save_sensitivity(original, path)
        assert path.exists()

        loaded = load_sensitivity(path)
        assert loaded.scenario == "baseline"
        assert loaded.baseline_score == pytest.approx(0.65)
        assert loaded.avg_time_per_run == pytest.approx(12.3)
        assert loaded.n_seeds == 3
        assert len(loaded.parameters) == 1
        assert loaded.parameters[0].name == "beta"
        assert loaded.parameters[0].sensitivity == pytest.approx(0.1)

    def test_schema_version_present(self, tmp_path):
        result = SensitivityResult(
            parameters=[], baseline_score=0.5, scenario="baseline"
        )
        path = tmp_path / "sens.json"
        save_sensitivity(result, path)
        with open(path) as f:
            data = json.load(f)
        assert "_schema_version" in data
        assert data["_schema_version"] == 1

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_sensitivity("/nonexistent/file.json")


class TestMorrisRoundtrip:
    """Tests for Morris save/load."""

    def test_round_trip(self, tmp_path):
        original = MorrisResult(
            effects=[
                MorrisParameterEffect(
                    name="beta",
                    mu=0.05,
                    mu_star=0.08,
                    sigma=0.03,
                    elementary_effects=[0.05, -0.02, 0.12],
                    value_scores={1.0: [0.7, 0.75], 2.0: [0.8]},
                ),
            ],
            n_trajectories=5,
            n_evaluations=30,
            scenario="baseline",
            avg_time_per_run=10.0,
            n_seeds=2,
        )

        path = tmp_path / "morris.json"
        save_morris(original, path)
        assert path.exists()

        loaded = load_morris(path)
        assert loaded.scenario == "baseline"
        assert loaded.n_trajectories == 5
        assert loaded.n_evaluations == 30
        assert len(loaded.effects) == 1
        assert loaded.effects[0].name == "beta"
        assert loaded.effects[0].mu_star == pytest.approx(0.08)
        assert len(loaded.effects[0].elementary_effects) == 3

    def test_schema_version_present(self, tmp_path):
        result = MorrisResult(
            effects=[], n_trajectories=1, n_evaluations=1, scenario="baseline"
        )
        path = tmp_path / "morris.json"
        save_morris(result, path)
        with open(path) as f:
            data = json.load(f)
        assert data["_schema_version"] == 1


class TestScreeningRoundtrip:
    """Tests for screening save/load."""

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

        path = tmp_path / "screen.json"
        save_screening(results, sensitivity, grid, fixed, patterns, "baseline", path)
        assert path.exists()

        loaded_results, avg_time = load_screening(path)
        assert len(loaded_results) == 2
        assert loaded_results[0].single_score == pytest.approx(0.8)
        assert avg_time == pytest.approx(5.0)


class TestStabilityRoundtrip:
    """Tests for stability save/load."""

    def test_round_trip(self, tmp_path):
        results = [
            CalibrationResult(
                {"a": 1},
                0.85,
                1,
                0,
                0,
                mean_score=0.82,
                std_score=0.03,
                pass_rate=1.0,
                combined_score=0.79,
                seed_scores=[0.80, 0.84],
            ),
        ]

        path = tmp_path / "stability.json"
        save_stability(results, "baseline", path)
        assert path.exists()

        loaded = load_stability(path)
        assert len(loaded) == 1
        assert loaded[0].mean_score == pytest.approx(0.82)
        assert loaded[0].std_score == pytest.approx(0.03)
        assert loaded[0].seed_scores == [0.80, 0.84]


class TestPairwiseRoundtrip:
    """Tests for pairwise save/load."""

    def test_round_trip(self, tmp_path):
        original = PairwiseResult(
            interactions=[
                PairInteraction(
                    param_a="beta",
                    param_b="max_M",
                    value_a=2.5,
                    value_b=4,
                    individual_a_score=0.7,
                    individual_b_score=0.75,
                    combined_score=0.80,
                    baseline_score=0.65,
                    interaction_strength=0.05,
                ),
            ],
            scenario="baseline",
            baseline_score=0.65,
        )

        path = tmp_path / "pairwise.json"
        save_pairwise(original, "baseline", path)
        assert path.exists()

        loaded = load_pairwise(path)
        assert loaded.scenario == "baseline"
        assert loaded.baseline_score == pytest.approx(0.65)
        assert len(loaded.interactions) == 1
        assert loaded.interactions[0].param_a == "beta"
        assert loaded.interactions[0].interaction_strength == pytest.approx(0.05)
