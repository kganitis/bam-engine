"""Tests for calibration.reporting module."""

from __future__ import annotations

from calibration.analysis import CalibrationResult, ComparisonResult
from calibration.reporting import (
    generate_full_report,
    generate_screening_report,
    generate_sensitivity_report,
    generate_stability_report,
)
from calibration.sensitivity import ParameterSensitivity, SensitivityResult


def _make_sensitivity() -> SensitivityResult:
    return SensitivityResult(
        parameters=[
            ParameterSensitivity("beta", [1.0, 2.0], [0.6, 0.8], 2.0, 0.8, 0.2),
            ParameterSensitivity("max_M", [2, 4], [0.7, 0.71], 4, 0.71, 0.01),
        ],
        baseline_score=0.65,
        scenario="baseline",
        avg_time_per_run=10.0,
        n_seeds=3,
    )


class TestGenerateSensitivityReport:
    """Tests for generate_sensitivity_report."""

    def test_creates_file(self, tmp_path):
        result = _make_sensitivity()
        path = tmp_path / "report.md"
        generate_sensitivity_report(result, "morris", path)
        assert path.exists()

    def test_contains_ranking(self, tmp_path):
        result = _make_sensitivity()
        path = tmp_path / "report.md"
        generate_sensitivity_report(result, "oat", path)
        content = path.read_text()
        assert "beta" in content
        assert "Sensitivity" in content
        assert "OAT" in content


class TestGenerateScreeningReport:
    """Tests for generate_screening_report."""

    def test_creates_file(self, tmp_path):
        results = [
            CalibrationResult({"beta": 2.0}, 0.85, 1, 0, 0),
            CalibrationResult({"beta": 1.0}, 0.70, 0, 1, 0),
        ]
        path = tmp_path / "report.md"
        generate_screening_report(
            results,
            grid={"beta": [1.0, 2.0]},
            fixed={"max_M": 4},
            patterns={"beta": {2.0: 40, 1.0: 10}},
            sensitivity=_make_sensitivity(),
            scenario="baseline",
            path=path,
        )
        assert path.exists()

    def test_contains_patterns(self, tmp_path):
        results = [CalibrationResult({"beta": 2.0}, 0.85, 1, 0, 0)]
        path = tmp_path / "report.md"
        generate_screening_report(
            results,
            {"beta": [1.0, 2.0]},
            {},
            {"beta": {2.0: 1}},
            _make_sensitivity(),
            "baseline",
            path,
        )
        content = path.read_text()
        assert "Pattern" in content
        assert "beta" in content


class TestGenerateStabilityReport:
    """Tests for generate_stability_report."""

    def test_creates_file(self, tmp_path):
        results = [
            CalibrationResult(
                {"beta": 2.0},
                0.85,
                1,
                0,
                0,
                mean_score=0.82,
                std_score=0.03,
                combined_score=0.79,
                seed_scores=[0.80, 0.84],
            ),
        ]
        path = tmp_path / "report.md"
        generate_stability_report(results, "baseline", [(10, 5), (5, 10)], None, path)
        assert path.exists()

    def test_contains_comparison(self, tmp_path):
        results = [
            CalibrationResult(
                {"beta": 2.0},
                0.85,
                1,
                0,
                0,
                mean_score=0.82,
                std_score=0.03,
                combined_score=0.79,
                seed_scores=[0.80, 0.84],
            ),
        ]
        comparison = ComparisonResult(
            "baseline",
            {"unemp": 0.08},
            {"unemp": 0.05},
            0.70,
            0.85,
            [("unemp", 0.08, 0.05, -37.5)],
        )
        path = tmp_path / "report.md"
        generate_stability_report(results, "baseline", [(10, 5)], comparison, path)
        content = path.read_text()
        assert "Comparison" in content
        assert "unemp" in content


class TestGenerateFullReport:
    """Tests for generate_full_report."""

    def test_creates_file(self, tmp_path):
        sensitivity = _make_sensitivity()
        screening = [CalibrationResult({"beta": 2.0}, 0.85, 1, 0, 0)]
        stability = [
            CalibrationResult(
                {"beta": 2.0},
                0.85,
                1,
                0,
                0,
                mean_score=0.82,
                std_score=0.03,
                combined_score=0.79,
            ),
        ]
        path = tmp_path / "full_report.md"
        generate_full_report(
            sensitivity,
            screening,
            stability,
            None,
            "baseline",
            [(10, 5)],
            path,
        )
        assert path.exists()
        content = path.read_text()
        assert "Calibration Report" in content
        assert "Sensitivity Phase" in content
        assert "Screening Phase" in content
        assert "Stability Phase" in content
