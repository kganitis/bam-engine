"""Unit tests for the R&D (Growth+) extension.

Tests cover the three R&D events: compute intensity, apply productivity
growth, and deduct R&D expenditure.
"""

from __future__ import annotations

import numpy as np
import pytest

import bamengine as bam
from bamengine import ops


@pytest.fixture
def small_sim():
    """Create a small simulation with RnD extension attached."""
    from extensions.rnd import RND_EVENTS, RnD

    sim = bam.Simulation.init(
        n_firms=5,
        n_households=10,
        n_banks=2,
        seed=42,
        sigma_min=0.0,
        sigma_max=0.1,
        sigma_decay=-1.0,
        logging={"default_level": "ERROR"},
    )
    sim.use_role(RnD)
    sim.use_events(*RND_EVENTS)
    return sim


class TestFirmsComputeRnDIntensity:
    """Test the R&D intensity computation event."""

    def test_fragility_computed(self, small_sim):
        """fragility = wage_bill / net_worth."""
        bor = small_sim.get_role("Borrower")
        emp = small_sim.get_role("Employer")
        rnd = small_sim.get_role("RnD")
        prod = small_sim.get_role("Producer")

        n = small_sim.n_firms
        ops.assign(emp.wage_bill, np.full(n, 50.0))
        ops.assign(bor.net_worth, np.full(n, 200.0))
        ops.assign(bor.net_profit, np.full(n, 100.0))
        ops.assign(prod.price, np.full(n, 10.0))
        ops.assign(prod.production, np.full(n, 100.0))

        evt = small_sim.get_event("firms_compute_rnd_intensity")
        evt.execute(small_sim)

        expected_fragility = 50.0 / 200.0  # = 0.25
        np.testing.assert_allclose(rnd.fragility, expected_fragility, atol=1e-10)

    def test_sigma_formula(self, small_sim):
        """sigma = sigma_min + (sigma_max - sigma_min) * exp(sigma_decay * fragility)."""
        bor = small_sim.get_role("Borrower")
        emp = small_sim.get_role("Employer")
        rnd = small_sim.get_role("RnD")
        prod = small_sim.get_role("Producer")

        n = small_sim.n_firms
        ops.assign(emp.wage_bill, np.full(n, 50.0))
        ops.assign(bor.net_worth, np.full(n, 200.0))
        ops.assign(bor.net_profit, np.full(n, 100.0))
        ops.assign(prod.price, np.full(n, 10.0))
        ops.assign(prod.production, np.full(n, 100.0))

        evt = small_sim.get_event("firms_compute_rnd_intensity")
        evt.execute(small_sim)

        sigma_min = small_sim.sigma_min
        sigma_max = small_sim.sigma_max
        sigma_decay = small_sim.sigma_decay
        fragility = 50.0 / 200.0
        expected_sigma = sigma_min + (sigma_max - sigma_min) * np.exp(
            sigma_decay * fragility
        )
        np.testing.assert_allclose(rnd.sigma, expected_sigma, atol=1e-10)

    def test_sigma_zero_for_non_positive_profit(self, small_sim):
        """net_profit <= 0 => sigma = 0."""
        bor = small_sim.get_role("Borrower")
        emp = small_sim.get_role("Employer")
        rnd = small_sim.get_role("RnD")
        prod = small_sim.get_role("Producer")

        n = small_sim.n_firms
        ops.assign(emp.wage_bill, np.full(n, 50.0))
        ops.assign(bor.net_worth, np.full(n, 200.0))
        ops.assign(prod.price, np.full(n, 10.0))
        ops.assign(prod.production, np.full(n, 100.0))

        # Mix of zero and negative profits
        profits = np.array([0.0, -10.0, -50.0, 0.0, -1.0])
        ops.assign(bor.net_profit, profits)

        evt = small_sim.get_event("firms_compute_rnd_intensity")
        evt.execute(small_sim)

        np.testing.assert_allclose(rnd.sigma, 0.0, atol=1e-10)

    def test_mu_computation(self, small_sim):
        """mu = sigma * net_profit / (price * production)."""
        bor = small_sim.get_role("Borrower")
        emp = small_sim.get_role("Employer")
        rnd = small_sim.get_role("RnD")
        prod = small_sim.get_role("Producer")

        n = small_sim.n_firms
        ops.assign(emp.wage_bill, np.full(n, 50.0))
        ops.assign(bor.net_worth, np.full(n, 200.0))
        ops.assign(bor.net_profit, np.full(n, 100.0))
        ops.assign(prod.price, np.full(n, 10.0))
        ops.assign(prod.production, np.full(n, 50.0))

        evt = small_sim.get_event("firms_compute_rnd_intensity")
        evt.execute(small_sim)

        # First compute expected sigma
        sigma_min = small_sim.sigma_min
        sigma_max = small_sim.sigma_max
        sigma_decay = small_sim.sigma_decay
        fragility = 50.0 / 200.0
        expected_sigma = sigma_min + (sigma_max - sigma_min) * np.exp(
            sigma_decay * fragility
        )
        # mu = sigma * profit / revenue
        revenue = 10.0 * 50.0  # = 500
        expected_mu = expected_sigma * 100.0 / revenue
        np.testing.assert_allclose(rnd.rnd_intensity, expected_mu, atol=1e-10)

    def test_mu_non_negative(self, small_sim):
        """mu is clamped to >= 0."""
        bor = small_sim.get_role("Borrower")
        emp = small_sim.get_role("Employer")
        rnd = small_sim.get_role("RnD")
        prod = small_sim.get_role("Producer")

        n = small_sim.n_firms
        # Firms with positive profit get sigma > 0,
        # and mu = sigma * profit / revenue > 0
        # Firms with non-positive profit get sigma = 0, so mu = 0
        ops.assign(emp.wage_bill, np.full(n, 50.0))
        ops.assign(bor.net_worth, np.full(n, 200.0))
        ops.assign(bor.net_profit, np.array([100.0, -50.0, 0.0, 200.0, -10.0]))
        ops.assign(prod.price, np.full(n, 10.0))
        ops.assign(prod.production, np.full(n, 100.0))

        evt = small_sim.get_event("firms_compute_rnd_intensity")
        evt.execute(small_sim)

        assert np.all(rnd.rnd_intensity >= 0.0)

    def test_safe_division_zero_net_worth(self, small_sim):
        """net_worth = 0 doesn't produce inf/nan."""
        bor = small_sim.get_role("Borrower")
        emp = small_sim.get_role("Employer")
        rnd = small_sim.get_role("RnD")
        prod = small_sim.get_role("Producer")

        n = small_sim.n_firms
        ops.assign(bor.net_worth, np.zeros(n))
        ops.assign(emp.wage_bill, np.full(n, 50.0))
        ops.assign(bor.net_profit, np.full(n, 100.0))
        ops.assign(prod.price, np.full(n, 10.0))
        ops.assign(prod.production, np.full(n, 100.0))

        evt = small_sim.get_event("firms_compute_rnd_intensity")
        evt.execute(small_sim)

        assert np.all(np.isfinite(rnd.fragility))
        assert np.all(np.isfinite(rnd.sigma))
        assert np.all(np.isfinite(rnd.rnd_intensity))

    def test_safe_division_zero_revenue(self, small_sim):
        """price * production = 0 doesn't produce inf/nan."""
        bor = small_sim.get_role("Borrower")
        emp = small_sim.get_role("Employer")
        rnd = small_sim.get_role("RnD")
        prod = small_sim.get_role("Producer")

        n = small_sim.n_firms
        ops.assign(bor.net_worth, np.full(n, 200.0))
        ops.assign(emp.wage_bill, np.full(n, 50.0))
        ops.assign(bor.net_profit, np.full(n, 100.0))
        ops.assign(prod.price, np.zeros(n))
        ops.assign(prod.production, np.zeros(n))

        evt = small_sim.get_event("firms_compute_rnd_intensity")
        evt.execute(small_sim)

        assert np.all(np.isfinite(rnd.rnd_intensity))


class TestFirmsApplyProductivityGrowth:
    """Test the productivity growth event."""

    def test_productivity_increases(self, small_sim):
        """labor_productivity increases for active (mu > 0) firms."""
        prod = small_sim.get_role("Producer")
        rnd = small_sim.get_role("RnD")

        n = small_sim.n_firms
        initial_prod = np.full(n, 1.0)
        ops.assign(prod.labor_productivity, initial_prod.copy())
        # Set mu > 0 for all firms
        ops.assign(rnd.rnd_intensity, np.full(n, 0.05))

        evt = small_sim.get_event("firms_apply_productivity_growth")
        evt.execute(small_sim)

        # All firms should have higher productivity
        assert np.all(prod.labor_productivity >= initial_prod)
        # At least some should have strictly increased (exponential draws > 0)
        assert np.any(prod.labor_productivity > initial_prod)

    def test_zero_mu_no_growth(self, small_sim):
        """mu = 0 => z = 0 (no productivity change)."""
        prod = small_sim.get_role("Producer")
        rnd = small_sim.get_role("RnD")

        n = small_sim.n_firms
        initial_prod = np.full(n, 1.5)
        ops.assign(prod.labor_productivity, initial_prod.copy())
        ops.assign(rnd.rnd_intensity, np.zeros(n))

        evt = small_sim.get_event("firms_apply_productivity_growth")
        evt.execute(small_sim)

        np.testing.assert_allclose(prod.labor_productivity, initial_prod, atol=1e-10)
        np.testing.assert_allclose(rnd.productivity_increment, 0.0, atol=1e-10)

    def test_increments_stored(self, small_sim):
        """productivity_increment array populated correctly."""
        prod = small_sim.get_role("Producer")
        rnd = small_sim.get_role("RnD")

        n = small_sim.n_firms
        initial_prod = np.full(n, 1.0)
        ops.assign(prod.labor_productivity, initial_prod.copy())
        ops.assign(rnd.rnd_intensity, np.full(n, 0.1))

        evt = small_sim.get_event("firms_apply_productivity_growth")
        evt.execute(small_sim)

        # z = new_productivity - initial
        expected_z = prod.labor_productivity - initial_prod
        np.testing.assert_allclose(rnd.productivity_increment, expected_z, atol=1e-10)

    def test_deterministic_with_seed(self, small_sim):
        """Same seed => same z values."""
        from extensions.rnd import RND_EVENTS, RnD

        z_runs = []
        for _ in range(2):
            sim = bam.Simulation.init(
                n_firms=5,
                n_households=10,
                n_banks=2,
                seed=99,
                sigma_min=0.0,
                sigma_max=0.1,
                sigma_decay=-1.0,
                logging={"default_level": "ERROR"},
            )
            sim.use_role(RnD)
            sim.use_events(*RND_EVENTS)

            prod = sim.get_role("Producer")
            rnd = sim.get_role("RnD")

            ops.assign(prod.labor_productivity, np.full(5, 1.0))
            ops.assign(rnd.rnd_intensity, np.full(5, 0.1))

            evt = sim.get_event("firms_apply_productivity_growth")
            evt.execute(sim)
            z_runs.append(rnd.productivity_increment.copy())

        np.testing.assert_array_equal(z_runs[0], z_runs[1])


class TestFirmsDeductRnDExpenditure:
    """Test the R&D expenditure deduction event."""

    def test_net_profit_reduced(self, small_sim):
        """net_profit *= (1 - sigma)."""
        bor = small_sim.get_role("Borrower")
        rnd = small_sim.get_role("RnD")

        n = small_sim.n_firms
        profit = np.full(n, 100.0)
        sigma = np.full(n, 0.08)
        ops.assign(bor.net_profit, profit.copy())
        ops.assign(rnd.sigma, sigma)

        evt = small_sim.get_event("firms_deduct_rn_d_expenditure")
        evt.execute(small_sim)

        expected = 100.0 * (1.0 - 0.08)  # = 92.0
        np.testing.assert_allclose(bor.net_profit, expected, atol=1e-10)

    def test_zero_sigma_no_net_profit_change(self, small_sim):
        """sigma = 0 => net_profit unchanged."""
        bor = small_sim.get_role("Borrower")
        rnd = small_sim.get_role("RnD")

        n = small_sim.n_firms
        profit = np.full(n, 100.0)
        ops.assign(bor.net_profit, profit.copy())
        ops.assign(rnd.sigma, np.zeros(n))

        evt = small_sim.get_event("firms_deduct_rn_d_expenditure")
        evt.execute(small_sim)

        np.testing.assert_allclose(bor.net_profit, 100.0, atol=1e-10)

    def test_full_sigma_zero_net_profit(self, small_sim):
        """sigma = 1.0 => net_profit = 0."""
        bor = small_sim.get_role("Borrower")
        rnd = small_sim.get_role("RnD")

        n = small_sim.n_firms
        profit = np.full(n, 250.0)
        ops.assign(bor.net_profit, profit.copy())
        ops.assign(rnd.sigma, np.full(n, 1.0))

        evt = small_sim.get_event("firms_deduct_rn_d_expenditure")
        evt.execute(small_sim)

        np.testing.assert_allclose(bor.net_profit, 0.0, atol=1e-10)
