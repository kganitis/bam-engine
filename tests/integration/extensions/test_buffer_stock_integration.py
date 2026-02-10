"""Integration tests for the buffer-stock consumption extension.

These tests run short simulations to verify the extension integrates
correctly with the full BAM engine pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

import bamengine as bam


@pytest.fixture
def buffer_stock_sim():
    """Create a simulation with BufferStock extension for integration tests."""
    from extensions.buffer_stock import BUFFER_STOCK_EVENTS, BufferStock

    sim = bam.Simulation.init(
        n_firms=10,
        n_households=50,
        n_banks=3,
        seed=42,
        buffer_stock_h=1.0,
        logging={"default_level": "ERROR"},
    )
    sim.use_role(BufferStock, n_agents=sim.n_households)
    sim.use_events(*BUFFER_STOCK_EVENTS)
    return sim


class TestBufferStockIntegration:
    """Integration tests running short simulations."""

    def test_extension_attaches(self):
        """BufferStock role attaches and events register."""
        from extensions.buffer_stock import BUFFER_STOCK_EVENTS, BufferStock

        sim = bam.Simulation.init(
            n_firms=5,
            n_households=10,
            n_banks=2,
            seed=0,
            buffer_stock_h=1.0,
            logging={"default_level": "ERROR"},
        )
        buf = sim.use_role(BufferStock, n_agents=sim.n_households)
        sim.use_events(*BUFFER_STOCK_EVENTS)
        assert buf is not None
        assert len(buf.prev_income) == sim.n_households
        assert len(buf.propensity) == sim.n_households

    def test_short_simulation_runs(self, buffer_stock_sim):
        """50-period simulation completes without errors."""
        results = buffer_stock_sim.run(n_periods=50, collect=True)
        assert results.metadata["n_periods"] == 50

    def test_pipeline_events_replaced(self, buffer_stock_sim):
        """Original propensity/spending events are replaced."""
        pipeline = buffer_stock_sim.pipeline

        event_names = [e.name for e in pipeline.events]

        # Buffer-stock events should be in pipeline
        assert "consumers_calc_buffer_stock_propensity" in event_names
        assert "consumers_decide_buffer_stock_spending" in event_names

        # Original events should NOT be in pipeline
        assert "consumers_calc_propensity" not in event_names
        assert "consumers_decide_income_to_spend" not in event_names

    def test_savings_stay_non_negative(self, buffer_stock_sim):
        """After 100 periods, no household has negative savings."""
        buffer_stock_sim.run(n_periods=100, collect=False)
        con = buffer_stock_sim.get_role("Consumer")
        assert np.all(con.savings >= 0.0)

    def test_deterministic_with_seed(self):
        """Same seed produces identical results."""
        from extensions.buffer_stock import BUFFER_STOCK_EVENTS, BufferStock

        savings_runs = []
        for _ in range(2):
            sim = bam.Simulation.init(
                n_firms=10,
                n_households=50,
                n_banks=3,
                seed=42,
                buffer_stock_h=1.0,
                logging={"default_level": "ERROR"},
            )
            sim.use_role(BufferStock, n_agents=sim.n_households)
            sim.use_events(*BUFFER_STOCK_EVENTS)
            sim.run(n_periods=50)
            con = sim.get_role("Consumer")
            savings_runs.append(con.savings.copy())

        np.testing.assert_array_equal(savings_runs[0], savings_runs[1])

    def test_composable_with_rnd(self):
        """Can combine BufferStock + RnD extensions."""
        from extensions.buffer_stock import BUFFER_STOCK_EVENTS, BufferStock
        from extensions.rnd import RND_EVENTS, RnD

        sim = bam.Simulation.init(
            n_firms=10,
            n_households=50,
            n_banks=3,
            seed=42,
            buffer_stock_h=1.0,
            sigma_min=0.0,
            sigma_max=0.1,
            sigma_decay=-1.0,
            logging={"default_level": "ERROR"},
        )
        sim.use_role(RnD)
        sim.use_role(BufferStock, n_agents=sim.n_households)
        sim.use_events(*RND_EVENTS, *BUFFER_STOCK_EVENTS)

        results = sim.run(n_periods=50, collect=True)
        assert results.metadata["n_periods"] == 50

    def test_composable_with_use_config(self):
        """Both extensions attach via use_config() and run correctly."""
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
        """User kwargs at init time override use_config() defaults."""
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
