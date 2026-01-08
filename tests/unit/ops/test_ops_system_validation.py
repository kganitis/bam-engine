"""
System-level validation tests for ops module.

This module contains parallel implementations of key system functions
using the ops API, and validates that they produce identical results
to the original NumPy implementations.

This ensures the ops module is a complete and correct replacement for NumPy
in user-facing code.
"""

import numpy as np

from bamengine import Simulation, make_rng, ops


class TestProductionSystemValidation:
    """Validate production planning system using ops."""

    def test_firms_decide_desired_production_ops_matches_numpy(self):
        """
        Validate firms_decide_desired_production logic using ops.

        This is one of the most complex system functions with multiple
        conditional branches.
        """
        # Create two identical simulations
        sim_numpy = Simulation.init(n_firms=100, seed=42)
        sim_ops = Simulation.init(n_firms=100, seed=42)

        # Run one step to get realistic state
        sim_numpy.step()
        sim_ops.step()

        # Save original state
        sim_numpy.prod.desired_production.copy()
        sim_ops.prod.desired_production.copy()

        # NumPy implementation (from original system function)
        prod_np = sim_numpy.prod
        h_rho = sim_numpy.config.h_rho
        p_avg = sim_numpy.ec.avg_mkt_price

        rng_numpy = make_rng(123)
        shock = rng_numpy.uniform(0.0, h_rho, len(prod_np.price))

        # Case 1: inventory = 0 and price >= avg → increase
        up_mask = np.logical_and(
            np.equal(prod_np.inventory, 0.0), np.greater_equal(prod_np.price, p_avg)
        )
        prod_np.desired_production[up_mask] *= 1.0 + shock[up_mask]

        # Case 2: inventory > 0 and price < avg → decrease
        down_mask = np.logical_and(
            np.greater(prod_np.inventory, 0.0), np.less(prod_np.price, p_avg)
        )
        prod_np.desired_production[down_mask] *= 1.0 - shock[down_mask]

        # ops implementation
        prod_ops = sim_ops.prod
        rng_ops = make_rng(123)
        shock_ops = ops.uniform(rng_ops, 0.0, h_rho, len(prod_ops.price))

        # Case 1: inventory = 0 and price >= avg → increase
        up_mask_ops = ops.logical_and(
            ops.equal(prod_ops.inventory, 0.0), ops.greater_equal(prod_ops.price, p_avg)
        )
        increase_factor = ops.add(shock_ops, 1.0)
        new_production_up = ops.where(
            up_mask_ops,
            ops.multiply(prod_ops.desired_production, increase_factor),
            prod_ops.desired_production,
        )

        # Case 2: inventory > 0 and price < avg → decrease
        down_mask_ops = ops.logical_and(
            ops.greater(prod_ops.inventory, 0.0), ops.less(prod_ops.price, p_avg)
        )
        decrease_factor = ops.subtract(shock_ops, 1.0)
        new_production_final = ops.where(
            down_mask_ops,
            ops.multiply(new_production_up, decrease_factor),
            new_production_up,
        )

        ops.assign(prod_ops.desired_production, new_production_final)

        # Results should match
        np.testing.assert_array_almost_equal(
            prod_np.desired_production,
            prod_ops.desired_production,
            decimal=10,
            err_msg="ops-based production logic doesn't match NumPy implementation",
        )


class TestLaborMarketSystemValidation:
    """Validate labor market system using ops."""

    def test_firms_decide_wage_offer_ops_matches_numpy(self):
        """
        Validate firms_decide_wage_offer logic using ops.

        Tests arithmetic operations with floor enforcement (maximum).
        """
        # Create two identical simulations
        sim_numpy = Simulation.init(n_firms=100, seed=42)
        sim_ops = Simulation.init(n_firms=100, seed=42)

        # Run one step to get realistic state
        sim_numpy.step()
        sim_ops.step()

        # NumPy implementation (from original system function)
        emp_np = sim_numpy.emp
        w_min = sim_numpy.ec.min_wage
        h_xi = sim_numpy.config.h_xi

        rng_numpy = make_rng(456)
        shock = rng_numpy.uniform(0.0, h_xi, len(emp_np.wage_offer))

        # Wage with shock
        new_wage_np = np.multiply(emp_np.wage_offer, 1.0 + shock)
        # Enforce minimum wage floor
        emp_np.wage_offer[:] = np.maximum(w_min, new_wage_np)

        # ops implementation
        emp_ops = sim_ops.emp
        rng_ops = make_rng(456)
        shock_ops = ops.uniform(rng_ops, 0.0, h_xi, len(emp_ops.wage_offer))

        # Wage with shock
        shock_factor = ops.add(shock_ops, 1.0)
        new_wage_ops = ops.multiply(emp_ops.wage_offer, shock_factor)
        # Enforce minimum wage floor
        final_wage = ops.maximum(new_wage_ops, w_min)
        ops.assign(emp_ops.wage_offer, final_wage)

        # Results should match
        np.testing.assert_array_almost_equal(
            emp_np.wage_offer,
            emp_ops.wage_offer,
            decimal=10,
            err_msg="ops-based wage offer logic doesn't match NumPy implementation",
        )


class TestCreditMarketSystemValidation:
    """Validate credit market system using ops."""

    def test_firms_decide_credit_demand_ops_matches_numpy(self):
        """
        Validate firms_decide_credit_demand logic using ops.

        Tests simple arithmetic with maximum (non-negativity constraint).
        """
        # Create two identical simulations
        sim_numpy = Simulation.init(n_firms=100, seed=42)
        sim_ops = Simulation.init(n_firms=100, seed=42)

        # Run a few steps to get realistic state
        for _ in range(5):
            sim_numpy.step()
            sim_ops.step()

        # NumPy implementation (from original system function)
        bor_np = sim_numpy.bor
        emp_np = sim_numpy.emp

        # Credit demand = max(0, wage_bill - net_worth)
        credit_need = np.subtract(emp_np.wage_bill, bor_np.net_worth)
        bor_np.credit_demand[:] = np.maximum(0.0, credit_need)

        # ops implementation
        bor_ops = sim_ops.bor
        emp_ops = sim_ops.emp

        # Credit demand = max(0, wage_bill - net_worth)
        credit_need_ops = ops.subtract(emp_ops.wage_bill, bor_ops.net_worth)
        final_demand = ops.maximum(credit_need_ops, 0.0)
        ops.assign(bor_ops.credit_demand, final_demand)

        # Results should match
        np.testing.assert_array_almost_equal(
            bor_np.credit_demand,
            bor_ops.credit_demand,
            decimal=10,
            err_msg="ops-based credit demand logic doesn't match NumPy implementation",
        )


class TestGoodsMarketSystemValidation:
    """Validate goods market system using ops."""

    def test_consumers_calc_propensity_ops_matches_numpy(self):
        """
        Validate consumers_calc_propensity logic using ops.

        Tests division and power operations with conditionals.
        """
        # Create two identical simulations
        sim_numpy = Simulation.init(n_households=200, seed=42)
        sim_ops = Simulation.init(n_households=200, seed=42)

        # Run a few steps to get realistic state
        for _ in range(3):
            sim_numpy.step()
            sim_ops.step()

        # NumPy implementation (from original system function)
        con_np = sim_numpy.con
        avg_sav = float(con_np.savings.mean())
        beta = sim_numpy.config.beta

        # Propensity = (savings / avg_savings) ^ beta
        # Guard against zero avg_savings
        if avg_sav > 0.0:
            ratio = np.divide(con_np.savings, avg_sav)
            con_np.propensity[:] = np.power(ratio, beta)
        else:
            con_np.propensity[:] = 1.0

        # ops implementation
        con_ops = sim_ops.con
        avg_sav_ops = ops.mean(con_ops.savings)

        # Guard against zero avg_savings
        if avg_sav_ops > 0.0:
            ratio_ops = ops.divide(con_ops.savings, avg_sav_ops)
            # Use NumPy's power since we don't have it in ops
            propensity_ops = np.power(ratio_ops, beta)
            ops.assign(con_ops.propensity, propensity_ops)
        else:
            ops.assign(con_ops.propensity, 1.0)

        # Results should match
        np.testing.assert_array_almost_equal(
            con_np.propensity,
            con_ops.propensity,
            decimal=10,
            err_msg="ops-based propensity logic doesn't match NumPy implementation",
        )


class TestPricingSystemValidation:
    """Validate pricing system using ops."""

    def test_price_adjustment_increase_ops_matches_numpy(self):
        """
        Validate price increase logic using ops.

        Tests conditional logic with where and comparisons.
        """
        # Create two identical simulations
        sim_numpy = Simulation.init(n_firms=100, seed=42)
        sim_ops = Simulation.init(n_firms=100, seed=42)

        # Run a few steps to get realistic state
        for _ in range(3):
            sim_numpy.step()
            sim_ops.step()

        # Set up test scenario: some firms want to increase prices
        prod_np = sim_numpy.prod
        prod_ops = sim_ops.prod

        # NumPy implementation: price increase logic
        rng_numpy = make_rng(789)
        h_eta = sim_numpy.config.h_eta
        p_avg = sim_numpy.ec.avg_mkt_price

        shock = rng_numpy.uniform(0.0, h_eta, len(prod_np.price))

        # Firms with inventory=0 and price >= avg increase prices
        should_increase = np.logical_and(
            np.equal(prod_np.inventory, 0.0), np.greater_equal(prod_np.price, p_avg)
        )

        new_price_np = np.where(
            should_increase, np.multiply(prod_np.price, 1.0 + shock), prod_np.price
        )

        # Also cap to breakeven
        capped_price_np = np.minimum(new_price_np, prod_np.breakeven_price)
        prod_np.price[:] = capped_price_np

        # ops implementation
        rng_ops = make_rng(789)
        shock_ops = ops.uniform(rng_ops, 0.0, h_eta, len(prod_ops.price))

        should_increase_ops = ops.logical_and(
            ops.equal(prod_ops.inventory, 0.0), ops.greater_equal(prod_ops.price, p_avg)
        )

        shock_factor = ops.add(shock_ops, 1.0)
        increased_price = ops.multiply(prod_ops.price, shock_factor)
        new_price_ops = ops.where(should_increase_ops, increased_price, prod_ops.price)

        # Cap to breakeven
        capped_price_ops = ops.minimum(new_price_ops, prod_ops.breakeven_price)
        ops.assign(prod_ops.price, capped_price_ops)

        # Results should match
        np.testing.assert_array_almost_equal(
            prod_np.price,
            prod_ops.price,
            decimal=10,
            err_msg="ops-based price adjustment logic "
            "doesn't match NumPy implementation",
        )


class TestAggregationSystemValidation:
    """Validate aggregation operations using ops."""

    def test_wage_bill_calculation_ops_matches_numpy(self):
        """
        Validate wage bill calculation using ops.

        Tests aggregation with sum operations and filtering.
        """
        # Create two identical simulations
        sim_numpy = Simulation.init(n_firms=50, n_households=200, seed=42)
        sim_ops = Simulation.init(n_firms=50, n_households=200, seed=42)

        # Run a few steps to get some employment
        for _ in range(5):
            sim_numpy.step()
            sim_ops.step()

        # NumPy implementation (simplified for demonstration)
        wrk_np = sim_numpy.wrk

        # Calculate total wages for all employed workers
        total_wages_np = (
            np.sum(wrk_np.wage[wrk_np.employed]) if np.any(wrk_np.employed) else 0.0
        )

        # ops implementation
        wrk_ops = sim_ops.wrk

        total_wages_ops = (
            ops.sum(wrk_ops.wage, where=wrk_ops.employed)
            if ops.any(wrk_ops.employed)
            else 0.0
        )

        # Results should match
        np.testing.assert_almost_equal(
            total_wages_np,
            total_wages_ops,
            decimal=10,
            err_msg="ops-based total wages logic doesn't match NumPy implementation",
        )


class TestFullEventReimplementation:
    """Full event reimplementation tests."""

    def test_complete_pricing_event_ops(self):
        """
        Test a complete custom pricing event using only ops.

        This demonstrates that users can write full events without NumPy.
        """
        from bamengine import Event

        class SimpleCostPlusEvent(Event):
            """Price = unit_cost * markup, using only ops."""

            def __init__(self, markup: float = 1.5):
                super().__init__()
                self.markup = markup

            # noinspection PyShadowingNames
            def execute(self, sim):
                prod = sim.get_role("Producer")
                emp = sim.get_role("Employer")

                # Calculate unit labor cost
                unit_cost = ops.divide(emp.wage_offer, prod.labor_productivity)

                # Apply markup
                new_price = ops.multiply(unit_cost, self.markup)

                # Enforce minimum (breakeven) and maximum (2x breakeven)
                min_price = prod.breakeven_price
                max_price = ops.multiply(prod.breakeven_price, 2.0)

                # Clip to range
                final_price = ops.clip(new_price, 0.0, 1000.0)  # reasonable bounds
                final_price = ops.maximum(final_price, min_price)
                final_price = ops.minimum(final_price, max_price)

                ops.assign(prod.price, final_price)

        # Test the event
        # Use higher net_worth_init for stability with small populations
        sim = Simulation.init(n_firms=50, net_worth_init=10.0, seed=123)

        # Run a few steps with default events
        for _ in range(3):
            sim.step()

        # Save state
        original_price = sim.prod.price.copy()

        # Replace pricing event with our ops-based one
        custom_event = SimpleCostPlusEvent(markup=1.5)
        custom_event.execute(sim)

        # Prices should have changed
        assert not np.allclose(sim.prod.price, original_price), (
            "Custom event should change prices"
        )

        # Prices should respect constraints
        assert np.all(sim.prod.price >= sim.prod.breakeven_price), (
            "All prices should be >= breakeven"
        )
        assert np.all(sim.prod.price <= sim.prod.breakeven_price * 2.0), (
            "All prices should be <= 2x breakeven"
        )

        # All prices should be positive
        assert np.all(sim.prod.price > 0), "All prices should be positive"
