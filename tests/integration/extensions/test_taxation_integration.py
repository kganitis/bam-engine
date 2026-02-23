"""Integration tests for the taxation extension.

These tests run short simulations to verify the extension integrates
correctly with the full BAM engine pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

import bamengine as bam


@pytest.fixture
def taxation_sim():
    """Create a simulation with taxation extension for integration tests."""
    from extensions.taxation import TAXATION_CONFIG, TAXATION_EVENTS

    sim = bam.Simulation.init(
        n_firms=10,
        n_households=50,
        n_banks=3,
        seed=42,
        profit_tax_rate=0.5,
        logging={"default_level": "ERROR"},
    )
    sim.use_events(*TAXATION_EVENTS)
    sim.use_config(TAXATION_CONFIG)
    return sim


class TestTaxationIntegration:
    """Integration tests running short simulations."""

    def test_short_simulation_runs(self, taxation_sim):
        """50-period simulation completes without errors."""
        results = taxation_sim.run(n_periods=50, collect=True)
        assert results.metadata["n_periods"] == 50

    def test_pipeline_event_inserted(self, taxation_sim):
        """Taxation event name present in pipeline."""
        event_names = [e.name for e in taxation_sim.pipeline.events]
        assert "firms_tax_profits" in event_names

    def test_tax_event_between_debt_and_dividends(self, taxation_sim):
        """Tax event is positioned between debt validation and dividends."""
        names = [e.name for e in taxation_sim.pipeline.events]
        debt_idx = names.index("firms_validate_debt_commitments")
        tax_idx = names.index("firms_tax_profits")
        dividends_idx = names.index("firms_pay_dividends")
        assert debt_idx < tax_idx < dividends_idx

    def test_high_tax_reduces_profits(self):
        """High tax rate results in lower total net profits than zero tax."""
        from extensions.taxation import TAXATION_CONFIG, TAXATION_EVENTS

        # Run with high tax
        sim_tax = bam.Simulation.init(
            n_firms=10,
            n_households=50,
            n_banks=3,
            seed=42,
            profit_tax_rate=0.9,
            logging={"default_level": "ERROR"},
        )
        sim_tax.use_events(*TAXATION_EVENTS)
        sim_tax.use_config(TAXATION_CONFIG)
        sim_tax.run(n_periods=50, collect=False)
        taxed_profit = sim_tax.get_role("Borrower").net_profit.sum()

        # Run without tax (same seed)
        sim_notax = bam.Simulation.init(
            n_firms=10,
            n_households=50,
            n_banks=3,
            seed=42,
            profit_tax_rate=0.0,
            logging={"default_level": "ERROR"},
        )
        sim_notax.use_events(*TAXATION_EVENTS)
        sim_notax.use_config(TAXATION_CONFIG)
        sim_notax.run(n_periods=50, collect=False)
        untaxed_profit = sim_notax.get_role("Borrower").net_profit.sum()

        # Fiscal drag: 90% tax should depress aggregate net profits
        assert taxed_profit < untaxed_profit

    def test_deterministic_with_seed(self):
        """Two runs with same seed produce identical Borrower arrays."""
        from extensions.taxation import TAXATION_CONFIG, TAXATION_EVENTS

        profit_runs = []
        for _ in range(2):
            sim = bam.Simulation.init(
                n_firms=10,
                n_households=50,
                n_banks=3,
                seed=42,
                profit_tax_rate=0.5,
                logging={"default_level": "ERROR"},
            )
            sim.use_events(*TAXATION_EVENTS)
            sim.use_config(TAXATION_CONFIG)
            sim.run(n_periods=50)
            bor = sim.get_role("Borrower")
            profit_runs.append(bor.net_profit.copy())

        np.testing.assert_array_equal(profit_runs[0], profit_runs[1])

    def test_composable_with_rnd(self):
        """Taxation + R&D run 50 periods together."""
        from extensions.rnd import RND_EVENTS, RnD
        from extensions.taxation import TAXATION_CONFIG, TAXATION_EVENTS

        sim = bam.Simulation.init(
            n_firms=10,
            n_households=50,
            n_banks=3,
            seed=42,
            profit_tax_rate=0.5,
            sigma_min=0.0,
            sigma_max=0.1,
            sigma_decay=-1.0,
            logging={"default_level": "ERROR"},
        )
        sim.use_role(RnD)
        sim.use_events(*RND_EVENTS, *TAXATION_EVENTS)
        sim.use_config(TAXATION_CONFIG)

        results = sim.run(n_periods=50, collect=True)
        assert results.metadata["n_periods"] == 50

    def test_composable_with_buffer_stock(self):
        """Taxation + BufferStock run 50 periods together."""
        from extensions.buffer_stock import BUFFER_STOCK_EVENTS, BufferStock
        from extensions.taxation import TAXATION_CONFIG, TAXATION_EVENTS

        sim = bam.Simulation.init(
            n_firms=10,
            n_households=50,
            n_banks=3,
            seed=42,
            profit_tax_rate=0.5,
            buffer_stock_h=1.0,
            logging={"default_level": "ERROR"},
        )
        sim.use_role(BufferStock, n_agents=sim.n_households)
        sim.use_events(*BUFFER_STOCK_EVENTS, *TAXATION_EVENTS)
        sim.use_config(TAXATION_CONFIG)

        results = sim.run(n_periods=50, collect=True)
        assert results.metadata["n_periods"] == 50

    def test_composable_with_use_config(self):
        """All three extensions via use_config(), defaults applied, sim runs."""
        from extensions.buffer_stock import (
            BUFFER_STOCK_CONFIG,
            BUFFER_STOCK_EVENTS,
            BufferStock,
        )
        from extensions.rnd import RND_CONFIG, RND_EVENTS, RnD
        from extensions.taxation import TAXATION_CONFIG, TAXATION_EVENTS

        sim = bam.Simulation.init(
            n_firms=10,
            n_households=50,
            n_banks=3,
            seed=42,
            logging={"default_level": "ERROR"},
        )
        sim.use_role(RnD)
        sim.use_role(BufferStock, n_agents=sim.n_households)
        sim.use_events(*RND_EVENTS, *BUFFER_STOCK_EVENTS, *TAXATION_EVENTS)
        sim.use_config(RND_CONFIG)
        sim.use_config(BUFFER_STOCK_CONFIG)
        sim.use_config(TAXATION_CONFIG)

        # Extension defaults applied
        assert sim.sigma_min == 0.0
        assert sim.sigma_max == 0.1
        assert sim.sigma_decay == -1.0
        assert sim.buffer_stock_h == 2.0
        assert sim.profit_tax_rate == 0.0

        results = sim.run(n_periods=50, collect=True)
        assert results.metadata["n_periods"] == 50

    def test_use_config_user_overrides_win(self):
        """User kwargs at init override use_config() default of 0.0."""
        from extensions.taxation import TAXATION_CONFIG, TAXATION_EVENTS

        sim = bam.Simulation.init(
            n_firms=10,
            n_households=50,
            n_banks=3,
            seed=42,
            profit_tax_rate=0.9,  # Override default of 0.0
            logging={"default_level": "ERROR"},
        )
        sim.use_events(*TAXATION_EVENTS)
        sim.use_config(TAXATION_CONFIG)

        # User override wins
        assert sim.profit_tax_rate == 0.9
