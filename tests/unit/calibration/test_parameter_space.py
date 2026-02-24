"""Tests for calibration.parameter_space module."""

from __future__ import annotations

import pytest

from calibration.parameter_space import (
    _COMMON_GRID,
    DEFAULT_VALUES,
    PARAMETER_GRID,
    PARAMETER_GRIDS,
    SCENARIO_OVERRIDES,
    count_combinations,
    generate_combinations,
    get_default_values,
    get_parameter_grid,
)


class TestCommonGrid:
    """Tests for _COMMON_GRID shared parameters."""

    def test_common_grid_has_expected_params(self):
        assert len(_COMMON_GRID) == 14

    def test_common_params_in_all_scenarios(self):
        """Every _COMMON_GRID param should appear in every scenario grid."""
        for scenario, grid in PARAMETER_GRIDS.items():
            for param in _COMMON_GRID:
                assert param in grid, f"{param} missing from {scenario}"

    def test_common_param_values_match(self):
        """Common param values should be identical across scenarios."""
        for param, values in _COMMON_GRID.items():
            for scenario, grid in PARAMETER_GRIDS.items():
                assert grid[param] == values, f"{param} values differ in {scenario}"

    def test_all_values_are_lists(self):
        for param, values in _COMMON_GRID.items():
            assert isinstance(values, list), f"{param} is not a list"
            assert len(values) >= 2, f"{param} has fewer than 2 values"


class TestScenarioGrids:
    """Tests for scenario-specific parameter grids."""

    def test_baseline_grid_exists(self):
        grid = get_parameter_grid("baseline")
        assert isinstance(grid, dict)
        assert len(grid) > 0

    def test_growth_plus_grid_has_extra_params(self):
        grid = get_parameter_grid("growth_plus")
        assert "sigma_decay" in grid

    def test_buffer_stock_grid_has_extra_params(self):
        grid = get_parameter_grid("buffer_stock")
        assert "buffer_stock_h" in grid
        assert "sigma_decay" in grid

    def test_baseline_has_no_extension_params(self):
        grid = get_parameter_grid("baseline")
        assert "sigma_decay" not in grid
        assert "sigma_max" not in grid
        assert "buffer_stock_h" not in grid

    def test_growth_plus_has_no_buffer_param(self):
        grid = get_parameter_grid("growth_plus")
        assert "buffer_stock_h" not in grid

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_parameter_grid("nonexistent")

    def test_all_three_scenarios_exist(self):
        assert set(PARAMETER_GRIDS.keys()) == {
            "baseline",
            "growth_plus",
            "buffer_stock",
        }


class TestScenarioOverrides:
    """Tests for SCENARIO_OVERRIDES."""

    def test_baseline_overrides_empty(self):
        assert get_default_values("baseline") == {}

    def test_growth_plus_has_rnd_overrides(self):
        overrides = get_default_values("growth_plus")
        assert "sigma_decay" in overrides
        assert "sigma_max" in overrides

    def test_buffer_stock_has_all_extension_overrides(self):
        overrides = get_default_values("buffer_stock")
        assert "buffer_stock_h" in overrides
        assert "sigma_decay" in overrides
        assert "sigma_max" in overrides

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_default_values("nonexistent")

    def test_all_three_scenarios_exist(self):
        assert set(SCENARIO_OVERRIDES.keys()) == {
            "baseline",
            "growth_plus",
            "buffer_stock",
        }


class TestCombinations:
    """Tests for generate_combinations() and count_combinations()."""

    def test_count_baseline(self):
        count = count_combinations(scenario="baseline")
        assert count > 0
        # Product of all value list lengths
        grid = get_parameter_grid("baseline")
        expected = 1
        for values in grid.values():
            expected *= len(values)
        assert count == expected

    def test_count_growth_plus_larger_than_baseline(self):
        """Growth+ has 2 extra params, so more combinations."""
        count_base = count_combinations(scenario="baseline")
        count_gp = count_combinations(scenario="growth_plus")
        assert count_gp > count_base

    def test_count_buffer_stock_larger_than_growth_plus(self):
        """Buffer_stock has 1 more param than growth_plus."""
        count_gp = count_combinations(scenario="growth_plus")
        count_bs = count_combinations(scenario="buffer_stock")
        assert count_bs > count_gp

    def test_generate_yields_correct_count(self):
        grid = {"a": [1, 2], "b": ["x", "y", "z"]}
        combos = list(generate_combinations(grid))
        assert len(combos) == 6

    def test_generate_all_keys_present(self):
        grid = {"a": [1, 2], "b": ["x", "y"]}
        for combo in generate_combinations(grid):
            assert "a" in combo
            assert "b" in combo

    def test_count_custom_grid(self):
        grid = {"a": [1, 2, 3], "b": [True, False]}
        assert count_combinations(grid) == 6


class TestBackwardsCompat:
    """Tests for backwards compatibility aliases."""

    def test_parameter_grid_is_baseline(self):
        assert PARAMETER_GRIDS["baseline"] == PARAMETER_GRID

    def test_default_values_is_baseline(self):
        assert SCENARIO_OVERRIDES["baseline"] == DEFAULT_VALUES
