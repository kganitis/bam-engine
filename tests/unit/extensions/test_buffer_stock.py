"""Unit tests for the buffer-stock consumption extension.

Tests cover the three-case MPC formula and spending allocation logic.
"""

from __future__ import annotations

import numpy as np
import pytest

import bamengine as bam
from bamengine import ops


@pytest.fixture
def small_sim():
    """Create a small simulation with BufferStock extension attached."""
    from extensions.buffer_stock import attach_buffer_stock

    sim = bam.Simulation.init(
        n_firms=5,
        n_households=10,
        n_banks=2,
        seed=42,
        buffer_stock_h=1.0,
        logging={"default_level": "ERROR"},
    )
    attach_buffer_stock(sim)
    return sim


class TestBufferStockMPC:
    """Test the three-case MPC formula."""

    def test_normal_case_equilibrium(self, small_sim):
        """At equilibrium (d=0, g=0), c=1.0 (consume all income).

        When income is unchanged and savings equals h * prev_income,
        d=0 and g=0, so c = 1 + (0 - h*0)/(1+0) = 1.0.
        """
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")
        h = small_sim.buffer_stock_h

        n = small_sim.n_households
        income = np.full(n, 100.0)
        prev_income = np.full(n, 100.0)  # g = 0
        savings = np.full(n, h * 100.0)  # d = S/W_{t-1} - h = h - h = 0

        ops.assign(con.income, income)
        ops.assign(con.savings, savings)
        ops.assign(buf.prev_income, prev_income)

        evt = small_sim.get_event("consumers_calc_buffer_stock_propensity")
        evt.execute(small_sim)

        np.testing.assert_allclose(buf.propensity, 1.0, atol=1e-10)

    def test_normal_case_income_growth(self, small_sim):
        """When income grows, c < 1 (save more)."""
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")
        h = small_sim.buffer_stock_h

        n = small_sim.n_households
        prev_income = np.full(n, 100.0)
        income = np.full(n, 150.0)  # g = 0.5 (growth)
        savings = np.full(n, h * 100.0)  # d = 0

        ops.assign(con.income, income)
        ops.assign(con.savings, savings)
        ops.assign(buf.prev_income, prev_income)

        evt = small_sim.get_event("consumers_calc_buffer_stock_propensity")
        evt.execute(small_sim)

        # With d=0 and g=0.5: c = 1 + (0 - h*0.5)/(1.5) = 1 - h/3
        expected = 1.0 - h * 0.5 / 1.5
        np.testing.assert_allclose(buf.propensity, expected, atol=1e-10)
        assert np.all(buf.propensity < 1.0)

    def test_normal_case_income_drop(self, small_sim):
        """When income drops, c > 1 (dissave)."""
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")
        h = small_sim.buffer_stock_h

        n = small_sim.n_households
        prev_income = np.full(n, 100.0)
        income = np.full(n, 50.0)  # g = -0.5 (drop)
        savings = np.full(n, h * 100.0)  # d = 0

        ops.assign(con.income, income)
        ops.assign(con.savings, savings)
        ops.assign(buf.prev_income, prev_income)

        evt = small_sim.get_event("consumers_calc_buffer_stock_propensity")
        evt.execute(small_sim)

        # With d=0 and g=-0.5: c = 1 + (0 - h*(-0.5))/(0.5) = 1 + h
        expected = 1.0 + h * 0.5 / 0.5
        np.testing.assert_allclose(buf.propensity, expected, atol=1e-10)
        assert np.all(buf.propensity > 1.0)

    def test_normal_case_g_clamped(self, small_sim):
        """Income growth rate clamped at -0.99 to avoid singularity."""
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")

        n = small_sim.n_households
        prev_income = np.full(n, 100.0)
        income = np.full(n, 0.001)  # g ~ -1.0, should be clamped to -0.99
        savings = np.full(n, 50.0)

        ops.assign(con.income, income)
        ops.assign(con.savings, savings)
        ops.assign(buf.prev_income, prev_income)

        evt = small_sim.get_event("consumers_calc_buffer_stock_propensity")
        evt.execute(small_sim)

        # Should not produce inf or nan
        assert np.all(np.isfinite(buf.propensity))

    def test_fresh_start_formula(self, small_sim):
        """Just re-employed: c = 1 - h + S/W."""
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")
        h = small_sim.buffer_stock_h

        n = small_sim.n_households
        income = np.full(n, 100.0)
        prev_income = np.zeros(n)  # was unemployed
        savings = np.full(n, 50.0)

        ops.assign(con.income, income)
        ops.assign(con.savings, savings)
        ops.assign(buf.prev_income, prev_income)

        evt = small_sim.get_event("consumers_calc_buffer_stock_propensity")
        evt.execute(small_sim)

        expected = 1.0 - h + savings / income
        np.testing.assert_allclose(buf.propensity, expected, atol=1e-10)

    def test_unemployed_case(self, small_sim):
        """Unemployed: c = 1/h (gradual savings drawdown)."""
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")

        n = small_sim.n_households
        income = np.zeros(n)  # unemployed
        prev_income = np.full(n, 100.0)
        savings = np.full(n, 200.0)

        ops.assign(con.income, income)
        ops.assign(con.savings, savings)
        ops.assign(buf.prev_income, prev_income)

        evt = small_sim.get_event("consumers_calc_buffer_stock_propensity")
        evt.execute(small_sim)

        h = small_sim.buffer_stock_h
        np.testing.assert_allclose(buf.propensity, 1.0 / h, atol=1e-10)

    def test_c_non_negative(self, small_sim):
        """MPC is clipped to >= 0."""
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")

        n = small_sim.n_households
        # Scenario that would produce c < 0: very large d (oversaving)
        # with income growth. d = S/W_{t-1} - h; if S >> h*W_{t-1}, d is large
        # but c = 1 + (d - h*g)/(1+g) should still be >= 0 for moderate d.
        # Force a scenario: fresh start with negative savings-income balance
        # c = 1 - h + S/W. If h=1.0 and S=0, W=100: c = 0. If S<0 impossible.
        # Actually c >= 0 is the floor. Let's test with very high h.
        # h=10, fresh start: c = 1 - 10 + 50/100 = -8.5 -> clamped to 0
        small_sim.extra_params["buffer_stock_h"] = 10.0

        income = np.full(n, 100.0)
        prev_income = np.zeros(n)  # fresh start
        savings = np.full(n, 50.0)

        ops.assign(con.income, income)
        ops.assign(con.savings, savings)
        ops.assign(buf.prev_income, prev_income)

        evt = small_sim.get_event("consumers_calc_buffer_stock_propensity")
        evt.execute(small_sim)

        assert np.all(buf.propensity >= 0.0)

    def test_mixed_cases(self, small_sim):
        """Verify correct case selection with mixed household states."""
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")
        h = small_sim.buffer_stock_h

        n = small_sim.n_households
        income = np.zeros(n)
        prev_income = np.zeros(n)
        savings = np.full(n, 50.0)

        # Set up different cases
        # Household 0: normal (both incomes positive)
        income[0] = 100.0
        prev_income[0] = 80.0
        # Household 1: fresh start (prev=0, income>0)
        income[1] = 100.0
        prev_income[1] = 0.0
        # Household 2: unemployed (income=0)
        income[2] = 0.0
        prev_income[2] = 100.0

        ops.assign(con.income, income)
        ops.assign(con.savings, savings)
        ops.assign(buf.prev_income, prev_income)

        evt = small_sim.get_event("consumers_calc_buffer_stock_propensity")
        evt.execute(small_sim)

        # Household 0: normal formula
        g = 100.0 / 80.0 - 1.0  # = 0.25
        d = 50.0 / 80.0 - h
        expected_0 = 1.0 + (d - h * g) / (1.0 + g)

        # Household 1: fresh start formula
        expected_1 = 1.0 - h + 50.0 / 100.0

        # Household 2: unemployed (c=1/h)
        expected_2 = 1.0 / h

        np.testing.assert_allclose(buf.propensity[0], expected_0, atol=1e-10)
        np.testing.assert_allclose(buf.propensity[1], expected_1, atol=1e-10)
        np.testing.assert_allclose(buf.propensity[2], expected_2, atol=1e-10)


class TestBufferStockSpending:
    """Test the spending allocation event."""

    def test_employed_income_based(self, small_sim):
        """Employed: budget = c * income."""
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")

        n = small_sim.n_households
        income = np.full(n, 100.0)
        savings = np.full(n, 200.0)
        propensity = np.full(n, 0.8)

        ops.assign(con.income, income)
        ops.assign(con.savings, savings)
        ops.assign(buf.propensity, propensity)
        ops.assign(buf.prev_income, np.zeros(n))

        evt = small_sim.get_event("consumers_decide_buffer_stock_spending")
        evt.execute(small_sim)

        expected_budget = 0.8 * 100.0  # = 80
        np.testing.assert_allclose(con.income_to_spend, expected_budget, atol=1e-10)

    def test_unemployed_from_savings(self, small_sim):
        """Unemployed: budget = c * savings."""
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")

        n = small_sim.n_households
        income = np.zeros(n)  # unemployed
        savings = np.full(n, 200.0)
        propensity = np.full(n, 1.0 / small_sim.buffer_stock_h)  # c=1/h for unemployed

        ops.assign(con.income, income)
        ops.assign(con.savings, savings)
        ops.assign(buf.propensity, propensity)
        ops.assign(buf.prev_income, np.zeros(n))

        evt = small_sim.get_event("consumers_decide_buffer_stock_spending")
        evt.execute(small_sim)

        expected_budget = 1.0 * 200.0  # = 200
        np.testing.assert_allclose(con.income_to_spend, expected_budget, atol=1e-10)
        np.testing.assert_allclose(con.savings, 0.0, atol=1e-10)

    def test_prev_income_stored(self, small_sim):
        """prev_income updated before income zeroed."""
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")

        n = small_sim.n_households
        income = np.full(n, 150.0)

        ops.assign(con.income, income)
        ops.assign(con.savings, np.full(n, 100.0))
        ops.assign(buf.propensity, np.full(n, 0.5))
        ops.assign(buf.prev_income, np.zeros(n))

        evt = small_sim.get_event("consumers_decide_buffer_stock_spending")
        evt.execute(small_sim)

        np.testing.assert_allclose(buf.prev_income, 150.0, atol=1e-10)

    def test_income_zeroed_after_allocation(self, small_sim):
        """Income set to 0 after allocation."""
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")

        n = small_sim.n_households
        ops.assign(con.income, np.full(n, 100.0))
        ops.assign(con.savings, np.full(n, 50.0))
        ops.assign(buf.propensity, np.full(n, 0.5))
        ops.assign(buf.prev_income, np.zeros(n))

        evt = small_sim.get_event("consumers_decide_buffer_stock_spending")
        evt.execute(small_sim)

        np.testing.assert_allclose(con.income, 0.0, atol=1e-10)

    def test_wealth_conservation(self, small_sim):
        """savings + income_to_spend = original savings + original income."""
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")

        n = small_sim.n_households
        income = np.full(n, 100.0)
        savings = np.full(n, 200.0)
        original_wealth = income + savings

        ops.assign(con.income, income)
        ops.assign(con.savings, savings)
        ops.assign(buf.propensity, np.full(n, 0.7))
        ops.assign(buf.prev_income, np.zeros(n))

        evt = small_sim.get_event("consumers_decide_buffer_stock_spending")
        evt.execute(small_sim)

        # income is zeroed, so total = savings + income_to_spend
        actual_wealth = con.savings + con.income_to_spend
        np.testing.assert_allclose(actual_wealth, original_wealth, atol=1e-10)

    def test_budget_capped_at_wealth(self, small_sim):
        """income_to_spend never exceeds savings + income."""
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")

        n = small_sim.n_households
        income = np.full(n, 100.0)
        savings = np.full(n, 50.0)
        # c=2.0 means budget=200 but wealth=150
        propensity = np.full(n, 2.0)

        ops.assign(con.income, income)
        ops.assign(con.savings, savings)
        ops.assign(buf.propensity, propensity)
        ops.assign(buf.prev_income, np.zeros(n))

        evt = small_sim.get_event("consumers_decide_buffer_stock_spending")
        evt.execute(small_sim)

        # Budget should be capped at 150 (savings + income)
        np.testing.assert_allclose(con.income_to_spend, 150.0, atol=1e-10)
        np.testing.assert_allclose(con.savings, 0.0, atol=1e-10)

    def test_savings_non_negative(self, small_sim):
        """Savings floor at 0 after consumption allocation."""
        con = small_sim.get_role("Consumer")
        buf = small_sim.get_role("BufferStock")

        n = small_sim.n_households
        ops.assign(con.income, np.full(n, 100.0))
        ops.assign(con.savings, np.full(n, 50.0))
        ops.assign(buf.propensity, np.full(n, 1.5))  # budget > income
        ops.assign(buf.prev_income, np.zeros(n))

        evt = small_sim.get_event("consumers_decide_buffer_stock_spending")
        evt.execute(small_sim)

        assert np.all(con.savings >= 0.0)
