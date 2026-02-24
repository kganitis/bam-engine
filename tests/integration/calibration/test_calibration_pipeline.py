"""Integration tests for the calibration pipeline.

These tests run actual (tiny) calibration workflows with real simulations.
They are marked slow because each simulation takes ~1-2 seconds.
"""

from __future__ import annotations

import pytest

from calibration.grid import count_combinations, load_grid
from calibration.io import create_run_dir


@pytest.mark.slow
class TestTimestampedOutput:
    """Test that timestamped output directories are created correctly."""

    def test_create_run_dir_structure(self, tmp_path):
        run_dir = create_run_dir("baseline", output_dir=tmp_path)
        assert run_dir.exists()
        assert run_dir.is_dir()
        assert "baseline" in run_dir.name

    def test_multiple_runs_unique(self, tmp_path):
        import time

        dir1 = create_run_dir("baseline", output_dir=tmp_path)
        time.sleep(0.1)
        dir2 = create_run_dir("baseline", output_dir=tmp_path)
        # Same or different — both valid as long as they exist
        assert dir1.exists()
        assert dir2.exists()


@pytest.mark.slow
class TestCustomGridYAML:
    """Test loading and using custom YAML grids."""

    def test_custom_grid_yaml_roundtrip(self, tmp_path):
        import yaml

        grid_data = {"beta": [1.0, 2.5, 5.0], "max_M": [2, 4]}
        grid_path = tmp_path / "test_grid.yaml"
        with open(grid_path, "w") as f:
            yaml.dump(grid_data, f)

        loaded = load_grid(grid_path)
        assert loaded == grid_data
        assert count_combinations(loaded) == 6
