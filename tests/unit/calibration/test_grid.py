"""Tests for calibration.grid module."""

from __future__ import annotations

import json

import pytest

from calibration.grid import (
    build_focused_grid,
    count_combinations,
    generate_combinations,
    load_grid,
    validate_grid,
)
from calibration.sensitivity import ParameterSensitivity, SensitivityResult


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
        # Both high_param (delta=0.10) and medium_param (delta=0.03) are > 0.02
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
        # With pruning_threshold=0.06: value=1 gap=0.10 -> dropped
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


class TestLoadGrid:
    """Tests for load_grid."""

    def test_load_yaml(self, tmp_path):
        import yaml

        grid_data = {"a": [1, 2, 3], "b": ["x", "y"]}
        path = tmp_path / "grid.yaml"
        with open(path, "w") as f:
            yaml.dump(grid_data, f)

        loaded = load_grid(path)
        assert loaded == grid_data

    def test_load_json(self, tmp_path):
        grid_data = {"a": [1, 2, 3], "b": ["x", "y"]}
        path = tmp_path / "grid.json"
        with open(path, "w") as f:
            json.dump(grid_data, f)

        loaded = load_grid(path)
        assert loaded == grid_data

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_grid("/nonexistent/grid.yaml")

    def test_non_dict_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        with open(path, "w") as f:
            json.dump([1, 2, 3], f)

        with pytest.raises(ValueError, match="dict"):
            load_grid(path)

    def test_non_list_values_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        with open(path, "w") as f:
            json.dump({"a": "not_a_list"}, f)

        with pytest.raises(ValueError, match="list"):
            load_grid(path)


class TestValidateGrid:
    """Tests for validate_grid."""

    def test_valid_grid_no_warnings(self):
        warnings = validate_grid({"a": [1, 2], "b": [3, 4]})
        assert warnings == []

    def test_empty_values_warning(self):
        warnings = validate_grid({"a": [], "b": [1, 2]})
        assert len(warnings) == 1
        assert "empty" in warnings[0]

    def test_single_value_warning(self):
        warnings = validate_grid({"a": [1], "b": [1, 2]})
        assert len(warnings) == 1
        assert "one value" in warnings[0]


class TestCountCombinations:
    """Tests for count_combinations."""

    def test_basic(self):
        assert count_combinations({"a": [1, 2], "b": [3, 4, 5]}) == 6

    def test_single_param(self):
        assert count_combinations({"a": [1, 2, 3]}) == 3

    def test_empty_grid(self):
        assert count_combinations({}) == 1


class TestGenerateCombinations:
    """Tests for generate_combinations."""

    def test_basic(self):
        grid = {"a": [1, 2], "b": ["x", "y"]}
        combos = list(generate_combinations(grid))
        assert len(combos) == 4
        assert {"a": 1, "b": "x"} in combos
        assert {"a": 2, "b": "y"} in combos

    def test_with_fixed(self):
        grid = {"a": [1, 2]}
        combos = list(generate_combinations(grid, fixed={"b": "fixed"}))
        assert len(combos) == 2
        assert all(c["b"] == "fixed" for c in combos)

    def test_count_matches_product(self):
        """Property: count always matches actual number of combinations."""
        grid = {"a": [1, 2, 3], "b": ["x", "y"], "c": [True, False]}
        assert count_combinations(grid) == len(list(generate_combinations(grid)))
