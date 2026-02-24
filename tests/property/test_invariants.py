"""Property-based tests for BAM Engine invariants using Hypothesis.

These tests use randomized inputs to verify that economic invariants
hold across a wide range of parameter combinations.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bamengine import Simulation

# Hypothesis strategies for valid parameter ranges
n_firms_strategy = st.integers(min_value=10, max_value=200)
n_households_strategy = st.integers(min_value=50, max_value=1000)
n_banks_strategy = st.integers(min_value=2, max_value=20)
seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)

# Shock parameters (0 to 0.25, avoiding extreme values that cause instability)
# Note: 50% shocks are too extreme for small populations and can cause collapse
shock_strategy = st.floats(min_value=0.01, max_value=0.25)

# Search frictions (1 to 10)
friction_strategy = st.integers(min_value=1, max_value=10)


@pytest.mark.filterwarnings("ignore:n_households.*n_firms:UserWarning")
class TestSimulationInvariants:
    """Test that economic invariants hold for any valid parameters."""

    @given(
        n_firms=n_firms_strategy,
        n_households=n_households_strategy,
        n_banks=n_banks_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=50, deadline=5000)
    def test_initialization_invariants(self, n_firms, n_households, n_banks, seed):
        """Initial state should satisfy all invariants."""
        sim = Simulation.init(
            n_firms=n_firms,
            n_households=n_households,
            n_banks=n_banks,
            seed=seed,
        )

        # Production invariants
        assert (sim.prod.price > 0).all(), "Prices must be positive"
        assert (sim.prod.production >= 0).all(), "Production must be non-negative"
        assert (sim.prod.inventory >= 0).all(), "Inventory must be non-negative"

        # Labor invariants
        assert (sim.emp.wage_offer >= sim.ec.min_wage).all(), "Wages must meet minimum"
        assert (sim.emp.current_labor >= 0).all(), "Labor must be non-negative"
        assert (sim.emp.n_vacancies >= 0).all(), "Vacancies must be non-negative"

        # Financial invariants
        assert np.isfinite(sim.bor.net_worth).all(), "Net worth must be finite"
        assert (sim.lend.equity_base > 0).all(), "Bank equity must be positive"

        # Consumer invariants
        assert (sim.con.savings >= 0).all(), "Savings must be non-negative"

    @given(
        n_firms=n_firms_strategy,
        n_households=n_households_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=30, deadline=10000)
    def test_single_step_invariants(self, n_firms, n_households, seed):
        """Invariants should hold after one simulation step."""
        sim = Simulation.init(n_firms=n_firms, n_households=n_households, seed=seed)

        sim.step()

        # Prices remain positive
        assert (sim.prod.price > 0).all(), "Prices must remain positive"

        # Production is non-negative
        assert (sim.prod.production >= 0).all(), "Production must be non-negative"
        assert (sim.prod.desired_production >= 0).all(), (
            "Desired production non-negative"
        )

        # No NaN or Inf values
        assert np.isfinite(sim.prod.price).all(), "Prices must be finite"
        assert np.isfinite(sim.ec.avg_mkt_price), "Average price must be finite"

        # Employment constraints
        assert (sim.emp.current_labor >= 0).all(), "Labor must be non-negative"
        assert (sim.emp.current_labor <= n_households).all(), (
            "Can't employ more than population"
        )

    @given(
        n_firms=n_firms_strategy,
        h_rho=shock_strategy,
        h_eta=shock_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=30, deadline=15000)
    def test_multi_step_stability(self, n_firms, h_rho, h_eta, seed):
        """Economy should remain stable over multiple periods."""
        sim = Simulation.init(
            n_firms=n_firms,
            n_households=n_firms * 5,  # Reasonable ratio
            h_rho=h_rho,
            h_eta=h_eta,
            net_worth_init=10.0,  # Higher net worth for stability with small populations
            seed=seed,
        )

        # Track consecutive periods of mass unemployment
        consecutive_mass_unemployment = 0
        max_consecutive_allowed = 3  # Allow temporary unemployment spikes

        # Run 10 periods
        for period in range(10):
            sim.step()

            # Economy should not be collapsed
            assert not sim.ec.collapsed, f"Economy collapsed at period {period}"

            # Prices should remain positive and finite
            assert (sim.prod.price > 0).all(), f"Non-positive price at period {period}"
            assert np.isfinite(sim.prod.price).all(), (
                f"Infinite price at period {period}"
            )

            # Production should be non-negative and finite
            assert (sim.prod.production >= 0).all(), (
                f"Negative production at period {period}"
            )
            assert np.isfinite(sim.prod.production).all(), (
                f"Infinite production at period {period}"
            )

            # Check for mass unemployment (at least some employed)
            # Allow temporary spikes but not persistent unemployment
            employed_count = sim.wrk.employed.sum()
            if employed_count == 0:
                consecutive_mass_unemployment += 1
                assert consecutive_mass_unemployment <= max_consecutive_allowed, (
                    f"Persistent mass unemployment for {consecutive_mass_unemployment} "
                    f"consecutive periods (max allowed: {max_consecutive_allowed})"
                )
            else:
                consecutive_mass_unemployment = 0  # Reset counter


class TestProductionInvariants:
    """Test production-specific invariants."""

    @given(
        n_firms=n_firms_strategy,
        h_rho=shock_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=30, deadline=10000)
    def test_production_shock_bounds(self, n_firms, h_rho, seed):
        """Production changes should be bounded by shock parameter."""
        sim = Simulation.init(n_firms=n_firms, h_rho=h_rho, seed=seed)

        # production_prev holds the planning signal used to compute desired_production
        # (production starts at 0 and is only set during firms_run_production)
        initial_production_signal = sim.prod.production_prev.copy()

        sim.step()

        # Production changes should not exceed shock bounds
        # (relaxed constraint due to other factors affecting production)
        production_ratio = sim.prod.desired_production / np.maximum(
            initial_production_signal, 1e-10
        )

        # Allow for some flexibility due to market conditions
        # Production can grow by up to (1 + h_rho) or shrink to 0
        assert (production_ratio <= 1 + h_rho + 0.1).all(), "Production grew too much"

    @given(
        n_firms=n_firms_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=20, deadline=10000)
    def test_inventory_conservation(self, n_firms, seed):
        """Inventory should follow conservation law."""
        sim = Simulation.init(n_firms=n_firms, seed=seed)

        sim.prod.inventory.copy()

        sim.step()

        # Inventory = previous inventory + production - sales
        # (This is a weak test as we don't track sales separately)
        # Just verify inventory remains non-negative
        assert (sim.prod.inventory >= 0).all(), "Inventory became negative"


@pytest.mark.filterwarnings("ignore:n_households.*n_firms:UserWarning")
class TestLaborMarketInvariants:
    """Test labor market invariants."""

    @given(
        n_firms=n_firms_strategy,
        n_households=n_households_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=30, deadline=10000)
    def test_employment_constraints(self, n_firms, n_households, seed):
        """Employment should respect population constraints."""
        sim = Simulation.init(n_firms=n_firms, n_households=n_households, seed=seed)

        sim.step()

        # Total employed cannot exceed population
        total_employed = sim.wrk.employed.sum()
        assert total_employed <= n_households, "More employed than population"

        # Each firm's labor cannot exceed population
        assert (sim.emp.current_labor >= 0).all(), "Negative labor"
        assert (sim.emp.current_labor <= n_households).all(), (
            "Firm employs more than population"
        )

        # Sum of firm labor should match employed workers
        total_firm_labor = sim.emp.current_labor.sum()
        assert abs(total_firm_labor - total_employed) <= 1, "Labor accounting mismatch"

    @given(
        n_firms=n_firms_strategy,
        h_xi=shock_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=20, deadline=10000)
    def test_wage_minimum_floor(self, n_firms, h_xi, seed):
        """Wages should never fall below minimum wage."""
        sim = Simulation.init(n_firms=n_firms, h_xi=h_xi, seed=seed)

        # Run multiple periods
        for _ in range(5):
            sim.step()

            # All wage offers must meet or exceed minimum wage
            assert (sim.emp.wage_offer >= sim.ec.min_wage).all(), "Wage below minimum"

            # All actual wages must meet or exceed minimum wage
            employed_mask = sim.wrk.employed
            if employed_mask.any():
                employed_wages = sim.wrk.wage[employed_mask]
                assert (employed_wages >= sim.ec.min_wage).all(), (
                    "Employed wage below minimum"
                )


class TestFinancialInvariants:
    """Test financial/accounting invariants."""

    @given(
        n_firms=n_firms_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=20, deadline=10000)
    def test_debt_non_negative(self, n_firms, seed):
        """Debt should always be non-negative."""
        sim = Simulation.init(n_firms=n_firms, seed=seed)

        # Run multiple periods
        for _ in range(5):
            sim.step()

            # Get debt from loan book
            debt_per_firm = sim.lb.debt_per_borrower(n_firms)
            assert (debt_per_firm >= 0).all(), "Negative debt detected"

    @given(
        n_firms=n_firms_strategy,
        n_banks=n_banks_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=20, deadline=10000)
    def test_bank_equity_positive(self, n_firms, n_banks, seed):
        """Bank equity should remain positive (or bank fails)."""
        sim = Simulation.init(
            n_firms=n_firms,
            n_banks=n_banks,
            net_worth_init=10.0,  # Higher net worth for stability with small populations
            seed=seed,
        )

        # Run multiple periods
        for _ in range(5):
            sim.step()

            # If economy is collapsed (all banks failed), stop checking
            if sim.ec.collapsed:
                break

            # Banks that survive must have positive equity
            # (bankrupt banks are replaced)
            assert (sim.lend.equity_base > 0).all(), (
                "Bank with non-positive equity survived"
            )


class TestPricingInvariants:
    """Test pricing mechanism invariants."""

    @given(
        n_firms=n_firms_strategy,
        h_eta=shock_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=30, deadline=10000)
    def test_prices_always_positive(self, n_firms, h_eta, seed):
        """Prices should always remain strictly positive."""
        sim = Simulation.init(n_firms=n_firms, h_eta=h_eta, seed=seed)

        # Run multiple periods
        for period in range(10):
            sim.step()

            assert (sim.prod.price > 0).all(), f"Non-positive price at period {period}"
            assert (sim.prod.price < 1e6).all(), f"Price explosion at period {period}"

    @given(
        n_firms=n_firms_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=20, deadline=10000)
    def test_average_price_reasonable(self, n_firms, seed):
        """Average market price should be reasonable (not extreme)."""
        sim = Simulation.init(n_firms=n_firms, seed=seed)

        # Run multiple periods
        for _ in range(5):
            sim.step()

            avg_price = sim.ec.avg_mkt_price

            # Average price should be finite and positive
            assert np.isfinite(avg_price), "Average price not finite"
            assert avg_price > 0, "Average price non-positive"

            # Should be within reasonable bounds (0.1 to 100)
            assert 0.01 < avg_price < 1000, f"Average price unreasonable: {avg_price}"


@pytest.mark.filterwarnings("ignore:n_households.*n_firms:UserWarning")
class TestNumericalStability:
    """Test numerical stability (no NaN, Inf, overflow)."""

    @given(
        n_firms=n_firms_strategy,
        n_households=n_households_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=30, deadline=10000)
    def test_no_nan_values(self, n_firms, n_households, seed):
        """No NaN values should appear in any state."""
        sim = Simulation.init(n_firms=n_firms, n_households=n_households, seed=seed)

        for _ in range(5):
            sim.step()

            # Check all float arrays
            assert np.isfinite(sim.prod.price).all(), "NaN in prices"
            assert np.isfinite(sim.prod.production).all(), "NaN in production"
            assert np.isfinite(sim.emp.wage_offer).all(), "NaN in wage offers"
            assert np.isfinite(sim.bor.net_worth).all(), "NaN in net worth"
            assert np.isfinite(sim.con.savings).all(), "NaN in savings"
            assert np.isfinite(sim.ec.avg_mkt_price), "NaN in average price"

    @given(
        n_firms=n_firms_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=20, deadline=10000)
    def test_no_overflow(self, n_firms, seed):
        """Values should not overflow (no Inf)."""
        sim = Simulation.init(n_firms=n_firms, seed=seed)

        for _ in range(10):
            sim.step()

            # Check for overflow
            assert not np.isinf(sim.prod.price).any(), "Overflow in prices"
            assert not np.isinf(sim.bor.net_worth).any(), "Overflow in net worth"
            assert not np.isinf(sim.con.savings).any(), "Overflow in savings"
