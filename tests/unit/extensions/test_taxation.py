"""Unit tests for the taxation extension.

Tests the FirmsTaxProfits event that deducts profit tax from firms
with positive net profits.
"""

from __future__ import annotations

import numpy as np

from bamengine import Simulation
from extensions.taxation import TAXATION_CONFIG, TAXATION_EVENTS, FirmsTaxProfits


def _make_sim(**config_overrides) -> Simulation:
    """Create a simulation with taxation extension attached."""
    sim = Simulation.init(n_firms=5, n_households=25, seed=42, **config_overrides)
    sim.use_events(*TAXATION_EVENTS)
    sim.use_config(TAXATION_CONFIG)
    return sim


class TestFirmsTaxProfits:
    """Tests for the FirmsTaxProfits event."""

    def test_zero_rate_no_change(self) -> None:
        """With rate=0, net_profit and total_funds should be unchanged."""
        sim = _make_sim(profit_tax_rate=0.0)
        bor = sim.get_role("Borrower")
        original_profit = bor.net_profit.copy()
        original_funds = bor.total_funds.copy()

        FirmsTaxProfits().execute(sim)

        np.testing.assert_array_equal(bor.net_profit, original_profit)
        np.testing.assert_array_equal(bor.total_funds, original_funds)

    def test_positive_rate_reduces_profit(self) -> None:
        """Positive tax rate should reduce net_profit for profitable firms."""
        sim = _make_sim(profit_tax_rate=0.5)
        bor = sim.get_role("Borrower")
        # Set up known profits
        bor.net_profit[:] = np.array([100.0, 200.0, -50.0, 0.0, 300.0])

        FirmsTaxProfits().execute(sim)

        # Profitable firms taxed at 50%
        assert bor.net_profit[0] == 50.0  # 100 - 50
        assert bor.net_profit[1] == 100.0  # 200 - 100
        assert bor.net_profit[4] == 150.0  # 300 - 150

    def test_negative_profit_not_taxed(self) -> None:
        """Firms with negative net_profit should not be taxed."""
        sim = _make_sim(profit_tax_rate=0.9)
        bor = sim.get_role("Borrower")
        bor.net_profit[:] = np.array([-100.0, -50.0, -10.0, -1.0, -200.0])
        original_profit = bor.net_profit.copy()

        FirmsTaxProfits().execute(sim)

        np.testing.assert_array_equal(bor.net_profit, original_profit)

    def test_zero_profit_not_taxed(self) -> None:
        """Firms with exactly zero net_profit should not be taxed."""
        sim = _make_sim(profit_tax_rate=0.5)
        bor = sim.get_role("Borrower")
        bor.net_profit[:] = 0.0
        original_profit = bor.net_profit.copy()

        FirmsTaxProfits().execute(sim)

        np.testing.assert_array_equal(bor.net_profit, original_profit)

    def test_total_funds_reduced_by_tax(self) -> None:
        """Tax should also reduce total_funds by the same amount."""
        sim = _make_sim(profit_tax_rate=0.3)
        bor = sim.get_role("Borrower")
        bor.net_profit[:] = np.array([100.0, 0.0, -50.0, 200.0, 50.0])
        bor.total_funds[:] = np.array([500.0, 300.0, 100.0, 800.0, 250.0])

        FirmsTaxProfits().execute(sim)

        # Tax amounts: 30, 0, 0, 60, 15
        np.testing.assert_allclose(
            bor.total_funds,
            [470.0, 300.0, 100.0, 740.0, 235.0],
        )

    def test_taxation_config_defaults(self) -> None:
        """TAXATION_CONFIG should default to zero tax rate."""
        assert TAXATION_CONFIG["profit_tax_rate"] == 0.0

    def test_taxation_events_list(self) -> None:
        """TAXATION_EVENTS should contain exactly FirmsTaxProfits."""
        assert len(TAXATION_EVENTS) == 1
        assert TAXATION_EVENTS[0] is FirmsTaxProfits
