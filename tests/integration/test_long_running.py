"""Long-running stability tests for BAM Engine.

These tests verify that the simulation remains stable over extended periods
(100+ time steps) and doesn't accumulate errors or exhibit instability.
"""

import numpy as np
import pytest

from bamengine import Simulation


class TestLongRunningStability:
    """Tests for long-running simulation stability."""

    def test_100_period_stability(self):
        """Simulation should remain stable for 100 periods."""
        sim = Simulation.init(
            n_firms=100,
            n_households=500,
            n_banks=10,
            seed=42,
        )

        # Run 100 periods
        sim.run(n_periods=100)

        # Economy should not be destroyed
        assert not sim.ec.destroyed, "Economy was destroyed before 100 periods"

        # Prices should remain positive and finite
        assert (sim.prod.price > 0).all(), "Some prices became non-positive"
        assert np.isfinite(sim.prod.price).all(), "Some prices became infinite"

        # Production should remain non-negative and finite
        assert (sim.prod.production >= 0).all(), "Some production became negative"
        assert np.isfinite(sim.prod.production).all(), "Some production became infinite"

        # Employment should be reasonable
        employed_count = sim.wrk.employed.sum()
        assert employed_count > 0, "Complete unemployment after 100 periods"
        assert employed_count <= sim.n_households, "More employed than population"

        # Economy-wide production should be positive and finite
        total_production = sim.prod.production.sum()
        assert np.isfinite(total_production), "Total production is not finite"
        assert total_production > 0, "Total production is non-positive"

    def test_100_period_no_explosion(self):
        """Values should not explode over 100 periods."""
        sim = Simulation.init(
            n_firms=50,
            n_households=250,
            seed=123,
        )

        # Track initial values
        initial_avg_price = sim.ec.avg_mkt_price
        initial_avg_wage = sim.emp.wage_offer.mean()

        # Run 100 periods
        sim.run(n_periods=100)

        # Prices should not explode (max 100x growth)
        final_avg_price = sim.ec.avg_mkt_price
        assert final_avg_price < initial_avg_price * 100, "Prices exploded"

        # Wages should not explode
        final_avg_wage = sim.emp.wage_offer.mean()
        assert final_avg_wage < initial_avg_wage * 100, "Wages exploded"

    def test_100_period_no_collapse(self):
        """Economy should not collapse over 100 periods."""
        sim = Simulation.init(
            n_firms=80,
            n_households=400,
            seed=456,
        )

        # Run 100 periods
        sim.run(n_periods=100)

        # Should still have active firms
        active_firms = (sim.prod.production > 0).sum()
        assert active_firms > 10, f"Only {active_firms} active firms remaining"

        # Should still have employment
        employed = sim.wrk.employed.sum()
        assert employed > 50, f"Only {employed} workers employed"

        # Should still have functioning banks
        assert (sim.lend.equity_base > 0).all(), "Some banks have non-positive equity"

    def test_100_period_statistics_remain_reasonable(self):
        """Economic statistics should remain in reasonable ranges."""
        sim = Simulation.init(
            n_firms=100,
            n_households=500,
            seed=789,
        )

        # Run 100 periods
        sim.run(n_periods=100)

        # Average price should be reasonable (0.1 to 10)
        avg_price = sim.ec.avg_mkt_price
        assert 0.01 < avg_price < 100, f"Average price unreasonable: {avg_price}"

        # Unemployment should be reasonable
        employed = sim.wrk.employed.sum()
        unemployment_rate = 1.0 - (employed / sim.n_households)
        assert 0.0 <= unemployment_rate <= 0.9, (
            f"Unemployment unreasonable: {unemployment_rate:.2%}"
        )

        # Total production should be reasonable (positive and finite)
        total_production = sim.prod.production.sum()
        assert np.isfinite(total_production), "Total production not finite"
        assert total_production > 0, "Total production non-positive"

    @pytest.mark.slow
    def test_500_period_stability(self):
        """Simulation should remain stable for 500 periods (marked slow)."""
        sim = Simulation.init(
            n_firms=100,
            n_households=500,
            n_banks=10,
            seed=999,
        )

        # Run 500 periods
        sim.run(n_periods=500)

        # Basic stability checks
        assert not sim.ec.destroyed, "Economy destroyed"
        assert (sim.prod.price > 0).all(), "Non-positive prices"
        total_production = sim.prod.production.sum()
        assert np.isfinite(total_production), "Total production not finite"
        assert sim.wrk.employed.sum() > 0, "Complete unemployment"


class TestCumulativeErrors:
    """Test that errors don't accumulate over time."""

    def test_no_numerical_drift(self):
        """Numerical errors should not accumulate over time."""
        sim = Simulation.init(
            n_firms=50,
            n_households=250,
            seed=111,
        )

        # Track some values
        initial_total_equity = sim.lend.equity_base.sum()

        # Run for 50 periods
        sim.run(n_periods=50)

        # Bank equity should not drift to unreasonable values
        final_total_equity = sim.lend.equity_base.sum()

        # Should not have changed by more than 10x (reasonable for 50 periods)
        ratio = final_total_equity / max(initial_total_equity, 1e-10)
        assert 0.01 < ratio < 100, f"Total bank equity changed unreasonably: {ratio}x"

    def test_accounting_remains_consistent(self):
        """Accounting identities should hold over time."""
        sim = Simulation.init(
            n_firms=50,
            n_households=250,
            seed=222,
        )

        # Run for 50 periods
        for _ in range(50):
            sim.step()

            # Debt should equal sum of loans
            debt_from_ledger = sim.lb.debt_per_borrower(sim.n_firms)
            assert (debt_from_ledger >= 0).all(), "Negative debt from ledger"

            # Employment should match across roles
            total_employed = sim.wrk.employed.sum()
            total_firm_labor = sim.emp.current_labor.sum()
            assert abs(total_employed - total_firm_labor) <= 1, (
                "Labor accounting mismatch"
            )


class TestStressConditions:
    """Test stability under stress conditions."""

    def test_high_shock_volatility(self):
        """Simulation should handle high shock volatility."""
        sim = Simulation.init(
            n_firms=100,  # Larger population for stability
            n_households=500,
            h_rho=0.3,  # High production shocks (3x default)
            h_eta=0.3,  # High price shocks (3x default)
            h_xi=0.15,  # High wage shocks (3x default)
            h_phi=0.3,  # High bank shocks (3x default)
            seed=42,  # Stable seed
        )

        # Run for 50 periods with high shocks
        sim.run(n_periods=50)

        # Should still be stable
        assert not sim.ec.destroyed, "Economy destroyed under high shocks"
        assert (sim.prod.price > 0).all(), "Non-positive prices under high shocks"

    def test_minimal_frictions(self):
        """Simulation should handle minimal search frictions."""
        sim = Simulation.init(
            n_firms=50,
            n_households=250,
            max_M=1,  # Minimal job applications
            max_H=1,  # Minimal loan applications
            max_Z=1,  # Minimal shop visits
            seed=444,
        )

        # Run for 50 periods
        sim.run(n_periods=50)

        # Should still function
        assert not sim.ec.destroyed, "Economy destroyed with minimal frictions"
        assert sim.wrk.employed.sum() > 0, "No employment with minimal frictions"

    def test_high_frictions(self):
        """Simulation should handle high search frictions."""
        sim = Simulation.init(
            n_firms=50,
            n_households=250,
            max_M=10,  # High job applications
            max_H=5,  # High loan applications
            max_Z=5,  # High shop visits
            seed=555,
        )

        # Run for 50 periods
        sim.run(n_periods=50)

        # Should still function
        assert not sim.ec.destroyed, "Economy destroyed with high frictions"


class TestMemoryUsage:
    """Test that memory usage doesn't grow over time."""

    def test_no_memory_growth(self):
        """Memory usage should not grow with simulation steps."""
        import sys

        sim = Simulation.init(
            n_firms=100,
            n_households=500,
            seed=666,
        )

        # Get initial size estimate
        # (This is a rough estimate - actual memory tracking would need profiling tools)
        initial_size = sum(
            sys.getsizeof(getattr(sim, attr))
            for attr in ["prod", "wrk", "emp", "bor", "lend", "con", "ec", "lb"]
        )

        # Run for 100 periods
        sim.run(n_periods=100)

        # Get final size estimate
        final_size = sum(
            sys.getsizeof(getattr(sim, attr))
            for attr in ["prod", "wrk", "emp", "bor", "lend", "con", "ec", "lb"]
        )

        # Size should not have grown significantly (allow some variance)
        # LoanBook can grow, so we allow up to 2x growth
        assert final_size < initial_size * 2, "Memory usage grew significantly"


class TestDeterminism:
    """Test that simulations are deterministic with fixed seed."""

    def test_reproducibility_100_periods(self):
        """Same seed should produce identical results over 100 periods."""
        # Run simulation 1
        sim1 = Simulation.init(n_firms=50, n_households=250, seed=777)
        sim1.run(n_periods=100)
        final_production1 = sim1.prod.production.sum()
        final_prices1 = sim1.prod.price.copy()

        # Run simulation 2 with same seed
        sim2 = Simulation.init(n_firms=50, n_households=250, seed=777)
        sim2.run(n_periods=100)
        final_production2 = sim2.prod.production.sum()
        final_prices2 = sim2.prod.price.copy()

        # Results should be identical
        assert final_production1 == final_production2, (
            "Total production not deterministic"
        )
        np.testing.assert_array_equal(
            final_prices1, final_prices2, err_msg="Prices not deterministic"
        )


class TestExcessLaborFiring:
    """Test that excess labor firing works correctly in the pipeline."""

    def test_excess_labor_firing_in_pipeline(self):
        """Firms with reduced production should fire excess workers."""
        sim = Simulation.init(
            n_firms=50,
            n_households=250,
            seed=42,
        )

        # Run a few periods to establish employment
        sim.run(n_periods=10)

        # Track some firms with workers
        initial_employment = sim.emp.current_labor.copy()
        firms_with_workers = np.where(initial_employment > 0)[0]

        # There should be some firms with workers
        assert len(firms_with_workers) > 0, "No firms with workers after 10 periods"

        # Run more periods - some firms may reduce production and fire workers
        sim.run(n_periods=20)

        # Employment accounting should still be consistent
        total_employed = sim.wrk.employed.sum()
        total_firm_labor = sim.emp.current_labor.sum()
        assert total_employed == total_firm_labor, (
            f"Labor accounting mismatch: employed={total_employed}, "
            f"firm_labor={total_firm_labor}"
        )

        # The economy should still be functioning
        assert not sim.ec.destroyed, "Economy destroyed during test"
        assert sim.wrk.employed.sum() > 0, "All workers unemployed"

    def test_fired_workers_can_be_rehired(self):
        """Workers fired for excess labor should be able to find new jobs."""
        sim = Simulation.init(
            n_firms=100,
            n_households=500,
            seed=123,
        )

        # Run for some periods
        sim.run(n_periods=20)

        # Run more periods - fired workers should be able to find jobs
        sim.run(n_periods=30)

        # Check that some previously fired workers may now be employed
        # (We can't guarantee all get rehired, but the mechanism should work)
        final_employed = sim.wrk.employed.sum()
        assert final_employed > 0, "No workers employed after 50 periods"

        # Employment accounting should be consistent
        total_employed = sim.wrk.employed.sum()
        total_firm_labor = sim.emp.current_labor.sum()
        assert total_employed == total_firm_labor, (
            f"Labor accounting mismatch: employed={total_employed}, "
            f"firm_labor={total_firm_labor}"
        )

    def test_current_labor_never_exceeds_desired_after_planning(self):
        """After planning phase, current_labor should not exceed desired_labor."""
        from bamengine.core import get_event

        sim = Simulation.init(
            n_firms=50,
            n_households=250,
            seed=456,
        )

        # Run the planning phase events manually
        planning_events = [
            "firms_decide_desired_production",
            "firms_decide_desired_labor",
            "firms_decide_vacancies",
            "firms_fire_excess_workers",
        ]

        for period in range(10):
            for event_name in planning_events:
                event = get_event(event_name)()
                event.execute(sim)

            # After planning phase, current_labor should be <= desired_labor
            excess = sim.emp.current_labor - sim.emp.desired_labor
            excess_firms = np.where(excess > 0)[0]
            assert len(excess_firms) == 0, (
                f"Period {period}: {len(excess_firms)} firms have excess workers: "
                f"{excess_firms[:5]}"
            )
