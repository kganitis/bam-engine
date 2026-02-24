"""Tests for calibration.parameter_space module."""

from __future__ import annotations

import pytest

from calibration.parameter_space import (
    DEFAULT_VALUES,
    PARAMETER_GRID,
    PARAMETER_GRIDS,
    SCENARIO_OVERRIDES,
    get_default_values,
    get_parameter_grid,
)


class TestCommonGrid:
    """Tests for the common parameter grid shared across scenarios."""

    def test_all_values_are_lists(self):
        for scenario, grid in PARAMETER_GRIDS.items():
            for name, values in grid.items():
                assert isinstance(values, list), (
                    f"{scenario}.{name} values is not a list: {type(values)}"
                )
                assert len(values) > 0, f"{scenario}.{name} has empty values"

    def test_no_duplicate_values(self):
        for scenario, grid in PARAMETER_GRIDS.items():
            for name, values in grid.items():
                assert len(values) == len(set(str(v) for v in values)), (
                    f"{scenario}.{name} has duplicate values"
                )


class TestScenarioGrids:
    """Tests for scenario-specific parameter grids."""

    def test_baseline_exists(self):
        grid = get_parameter_grid("baseline")
        assert len(grid) > 0
        assert "beta" in grid

    def test_growth_plus_has_sigma_decay(self):
        grid = get_parameter_grid("growth_plus")
        assert "sigma_decay" in grid

    def test_buffer_stock_has_buffer_stock_h(self):
        grid = get_parameter_grid("buffer_stock")
        assert "buffer_stock_h" in grid
        assert "sigma_decay" in grid  # Also includes R&D

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_parameter_grid("nonexistent")

    def test_all_scenarios_defined(self):
        """All 3 scenarios have grid definitions."""
        for scenario in ["baseline", "growth_plus", "buffer_stock"]:
            grid = get_parameter_grid(scenario)
            assert isinstance(grid, dict)
            assert len(grid) > 0

    def test_no_deprecated_params_in_grid(self):
        """Verify no deprecated/removed params in any grid."""
        deprecated = {
            "price_cut_allow_increase",
            "inflation_method",
            "labor_matching",
            "credit_matching",
            "min_wage_ratchet",
            "pricing_phase",
            "matching_method",
            "contract_poisson_mean",
            "loan_priority_method",
            "firing_method",
        }
        for scenario, grid in PARAMETER_GRIDS.items():
            for param in grid:
                assert param not in deprecated, (
                    f"Deprecated param '{param}' in {scenario} grid"
                )


class TestScenarioOverrides:
    """Tests for scenario-specific parameter overrides."""

    def test_baseline_is_empty(self):
        """Baseline uses engine defaults — empty overrides."""
        overrides = get_default_values("baseline")
        assert overrides == {}

    def test_growth_plus_overrides(self):
        overrides = get_default_values("growth_plus")
        assert "sigma_decay" in overrides
        assert "sigma_max" in overrides

    def test_buffer_stock_overrides(self):
        overrides = get_default_values("buffer_stock")
        assert "buffer_stock_h" in overrides

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_default_values("nonexistent")


class TestBackwardsCompat:
    """Tests for backwards compatibility aliases."""

    def test_parameter_grid_alias(self):
        assert PARAMETER_GRIDS["baseline"] == PARAMETER_GRID

    def test_default_values_alias(self):
        assert SCENARIO_OVERRIDES["baseline"] == DEFAULT_VALUES
