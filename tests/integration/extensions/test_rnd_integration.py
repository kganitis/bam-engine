"""Integration tests for the R&D (Growth+) extension.

These tests run short simulations to verify the extension integrates
correctly with the full BAM engine pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

import bamengine as bam


@pytest.fixture
def rnd_sim():
    """Create a simulation with RnD extension for integration tests."""
    from extensions.rnd import RND_EVENTS, RnD

    sim = bam.Simulation.init(
        n_firms=10,
        n_households=50,
        n_banks=3,
        seed=42,
        sigma_min=0.0,
        sigma_max=0.1,
        sigma_decay=-1.0,
        logging={"default_level": "ERROR"},
    )
    sim.use_role(RnD)
    sim.use_events(*RND_EVENTS)
    return sim


class TestRnDIntegration:
    """Integration tests running short simulations."""

    def test_extension_attaches(self):
        """RnD role attaches with correct array lengths."""
        from extensions.rnd import RND_EVENTS, RnD

        sim = bam.Simulation.init(
            n_firms=5,
            n_households=10,
            n_banks=2,
            seed=0,
            sigma_min=0.0,
            sigma_max=0.1,
            sigma_decay=-1.0,
            logging={"default_level": "ERROR"},
        )
        rnd = sim.use_role(RnD)
        sim.use_events(*RND_EVENTS)
        assert rnd is not None
        assert len(rnd.sigma) == sim.n_firms
        assert len(rnd.rnd_intensity) == sim.n_firms
        assert len(rnd.productivity_increment) == sim.n_firms
        assert len(rnd.fragility) == sim.n_firms

    def test_short_simulation_runs(self, rnd_sim):
        """50-period simulation completes without errors."""
        results = rnd_sim.run(n_periods=50, collect=True)
        assert results.metadata["n_periods"] == 50

    def test_pipeline_events_inserted(self, rnd_sim):
        """Three R&D event names present in pipeline."""
        event_names = [e.name for e in rnd_sim.pipeline.events]

        assert "firms_compute_rnd_intensity" in event_names
        assert "firms_apply_productivity_growth" in event_names
        assert "firms_deduct_rn_d_expenditure" in event_names

    def test_productivity_grows_over_time(self):
        """Mean productivity after 100 periods > initial."""
        from extensions.rnd import RND_EVENTS, RnD

        sim = bam.Simulation.init(
            n_firms=10,
            n_households=50,
            n_banks=3,
            seed=42,
            sigma_min=0.0,
            sigma_max=0.1,
            sigma_decay=-1.0,
            logging={"default_level": "ERROR"},
        )
        sim.use_role(RnD)
        sim.use_events(*RND_EVENTS)

        prod = sim.get_role("Producer")
        initial_mean = prod.labor_productivity.mean()

        sim.run(n_periods=100, collect=False)

        final_mean = prod.labor_productivity.mean()
        assert final_mean > initial_mean

    def test_deterministic_with_seed(self):
        """Two runs with same seed produce identical productivity."""
        from extensions.rnd import RND_EVENTS, RnD

        productivity_runs = []
        for _ in range(2):
            sim = bam.Simulation.init(
                n_firms=10,
                n_households=50,
                n_banks=3,
                seed=42,
                sigma_min=0.0,
                sigma_max=0.1,
                sigma_decay=-1.0,
                logging={"default_level": "ERROR"},
            )
            sim.use_role(RnD)
            sim.use_events(*RND_EVENTS)
            sim.run(n_periods=50)
            prod = sim.get_role("Producer")
            productivity_runs.append(prod.labor_productivity.copy())

        np.testing.assert_array_equal(productivity_runs[0], productivity_runs[1])

    def test_composable_with_buffer_stock(self):
        """R&D + BufferStock run 50 periods together."""
        from extensions.buffer_stock import BUFFER_STOCK_EVENTS, BufferStock
        from extensions.rnd import RND_EVENTS, RnD

        sim = bam.Simulation.init(
            n_firms=10,
            n_households=50,
            n_banks=3,
            seed=42,
            sigma_min=0.0,
            sigma_max=0.1,
            sigma_decay=-1.0,
            buffer_stock_h=1.0,
            logging={"default_level": "ERROR"},
        )
        sim.use_role(RnD)
        sim.use_role(BufferStock, n_agents=sim.n_households)
        sim.use_events(*RND_EVENTS, *BUFFER_STOCK_EVENTS)

        results = sim.run(n_periods=50, collect=True)
        assert results.metadata["n_periods"] == 50

    def test_composable_with_use_config(self):
        """Both extensions attach via use_config() with correct defaults."""
        from extensions.buffer_stock import (
            BUFFER_STOCK_CONFIG,
            BUFFER_STOCK_EVENTS,
            BufferStock,
        )
        from extensions.rnd import RND_CONFIG, RND_EVENTS, RnD

        sim = bam.Simulation.init(
            n_firms=10,
            n_households=50,
            n_banks=3,
            seed=42,
            logging={"default_level": "ERROR"},
        )
        sim.use_role(RnD)
        sim.use_role(BufferStock, n_agents=sim.n_households)
        sim.use_events(*RND_EVENTS, *BUFFER_STOCK_EVENTS)
        sim.use_config(RND_CONFIG)
        sim.use_config(BUFFER_STOCK_CONFIG)

        # Extension defaults applied
        assert sim.sigma_min == 0.0
        assert sim.sigma_max == 0.1
        assert sim.sigma_decay == -1.0
        assert sim.buffer_stock_h == 2.0

        results = sim.run(n_periods=50, collect=True)
        assert results.metadata["n_periods"] == 50

    def test_use_config_user_overrides_win(self):
        """User kwargs at init override use_config() defaults."""
        from extensions.buffer_stock import (
            BUFFER_STOCK_CONFIG,
            BUFFER_STOCK_EVENTS,
            BufferStock,
        )
        from extensions.rnd import RND_CONFIG, RND_EVENTS, RnD

        sim = bam.Simulation.init(
            n_firms=10,
            n_households=50,
            n_banks=3,
            seed=42,
            buffer_stock_h=3.0,  # Override default of 2.0
            sigma_max=0.05,  # Override default of 0.1
            logging={"default_level": "ERROR"},
        )
        sim.use_role(RnD)
        sim.use_role(BufferStock, n_agents=sim.n_households)
        sim.use_events(*RND_EVENTS, *BUFFER_STOCK_EVENTS)
        sim.use_config(RND_CONFIG)
        sim.use_config(BUFFER_STOCK_CONFIG)

        # User overrides win
        assert sim.buffer_stock_h == 3.0
        assert sim.sigma_max == 0.05
        # Defaults still applied for non-overridden keys
        assert sim.sigma_min == 0.0
        assert sim.sigma_decay == -1.0
