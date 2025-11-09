# tests/unit/test_ops_validation.py
"""
Validation tests for bamengine.ops module.

These tests verify that operations using the ops module produce identical
results to direct NumPy implementations used in system functions.

Strategy: Create parallel implementations using ops, run both, compare results.
"""

import numpy as np

from bamengine import Simulation, ops


def test_getter_methods_work():
    """Verify get_role(), get_event(), get() methods work correctly."""
    sim = Simulation.init(seed=42)

    # Test get_role
    prod = sim.get_role("Producer")
    assert prod is sim.prod

    # Test case-insensitive
    wrk = sim.get_role("WORKER")
    assert wrk is sim.wrk

    # Test get_event
    event = sim.get_event("firms_adjust_price")
    assert event.name == "firms_adjust_price"

    # Test unified get() method
    prod2 = sim.get("Producer")
    assert prod2 is sim.prod

    event2 = sim.get("firms_adjust_price")
    assert event2.name == "firms_adjust_price"


def test_ops_arithmetic_matches_numpy():
    """Verify ops arithmetic operations match NumPy."""
    rng = np.random.default_rng(42)

    a = rng.uniform(1, 100, 50)
    b = rng.uniform(1, 100, 50)

    # Multiplication
    np_result = np.multiply(a, b)
    ops_result = ops.multiply(a, b)
    np.testing.assert_array_almost_equal(ops_result, np_result)

    # Division
    np_result = np.divide(a, np.maximum(b, 1e-10))  # ops.divide is safe
    ops_result = ops.divide(a, b)
    np.testing.assert_array_almost_equal(ops_result, np_result)

    # Addition
    np_result = np.add(a, b)
    ops_result = ops.add(a, b)
    np.testing.assert_array_almost_equal(ops_result, np_result)

    # Subtraction
    np_result = np.subtract(a, b)
    ops_result = ops.subtract(a, b)
    np.testing.assert_array_almost_equal(ops_result, np_result)


def test_ops_conditional_matches_numpy():
    """Verify ops conditional operations match NumPy."""
    rng = np.random.default_rng(42)

    condition = rng.uniform(0, 1, 50) > 0.5
    true_vals = rng.uniform(10, 20, 50)
    false_vals = rng.uniform(1, 5, 50)

    # where
    np_result = np.where(condition, true_vals, false_vals)
    ops_result = ops.where(condition, true_vals, false_vals)
    np.testing.assert_array_almost_equal(ops_result, np_result)


def test_ops_comparisons_match_numpy():
    """Verify ops comparison operations match NumPy."""
    rng = np.random.default_rng(42)

    a = rng.uniform(1, 100, 50)
    b = rng.uniform(1, 100, 50)

    # Greater
    np_result = np.greater(a, b)
    ops_result = ops.greater(a, b)
    np.testing.assert_array_equal(ops_result, np_result)

    # Greater equal
    np_result = np.greater_equal(a, b)
    ops_result = ops.greater_equal(a, b)
    np.testing.assert_array_equal(ops_result, np_result)

    # Less
    np_result = np.less(a, b)
    ops_result = ops.less(a, b)
    np.testing.assert_array_equal(ops_result, np_result)

    # Less equal
    np_result = np.less_equal(a, b)
    ops_result = ops.less_equal(a, b)
    np.testing.assert_array_equal(ops_result, np_result)

    # Equal
    np_result = np.equal(a, b)
    ops_result = ops.equal(a, b)
    np.testing.assert_array_equal(ops_result, np_result)


def test_ops_aggregation_matches_numpy():
    """Verify ops aggregation operations match NumPy."""
    rng = np.random.default_rng(42)

    a = rng.uniform(1, 100, 50)

    # Sum
    np_result = float(np.sum(a))
    ops_result = ops.sum(a)
    np.testing.assert_almost_equal(ops_result, np_result)

    # Mean
    np_result = float(np.mean(a))
    ops_result = ops.mean(a)
    np.testing.assert_almost_equal(ops_result, np_result)


def test_ops_element_wise_matches_numpy():
    """Verify ops element-wise operations match NumPy."""
    rng = np.random.default_rng(42)

    a = rng.uniform(1, 100, 50)
    b = rng.uniform(1, 100, 50)

    # Maximum
    np_result = np.maximum(a, b)
    ops_result = ops.maximum(a, b)
    np.testing.assert_array_almost_equal(ops_result, np_result)

    # Minimum
    np_result = np.minimum(a, b)
    ops_result = ops.minimum(a, b)
    np.testing.assert_array_almost_equal(ops_result, np_result)

    # Clip
    np_result = np.clip(a, 20, 80)
    ops_result = ops.clip(a, 20, 80)
    np.testing.assert_array_almost_equal(ops_result, np_result)


def test_ops_simple_pricing_logic():
    """
    Test simple pricing logic using ops module.

    This validates that a simple markup pricing calculation using ops
    produces consistent results with direct NumPy operations.
    """
    rng = np.random.default_rng(42)

    # Setup test data
    wage = rng.uniform(10, 20, 50)
    productivity = rng.uniform(1, 5, 50)
    markup = 1.5

    # NumPy implementation
    np_unit_cost = np.divide(wage, np.maximum(productivity, 1e-10))
    np_price = np.multiply(np_unit_cost, markup)

    # ops implementation
    ops_unit_cost = ops.divide(wage, productivity)
    ops_price = ops.multiply(ops_unit_cost, markup)

    # Should match
    np.testing.assert_array_almost_equal(ops_price, np_price)


def test_ops_inventory_based_production_logic():
    """
    Test production decision logic using ops module.

    This validates conditional logic similar to firms_decide_desired_production.
    """
    rng = np.random.default_rng(42)

    # Setup test data
    inventory = rng.uniform(0, 50, 50)
    price = rng.uniform(80, 120, 50)
    avg_price = 100.0
    current_production = rng.uniform(10, 30, 50)

    # NumPy implementation: increase production if no inventory and price >= avg
    np_should_increase = np.logical_and(
        np.equal(inventory, 0.0), np.greater_equal(price, avg_price)
    )
    np_new_production = np.where(
        np_should_increase, current_production * 1.1, current_production
    )

    # ops implementation
    ops_should_increase = ops.logical_and(
        ops.equal(inventory, 0.0), ops.greater_equal(price, avg_price)
    )
    ops_new_production = ops.where(
        ops_should_increase, current_production * 1.1, current_production
    )

    # Should match
    np.testing.assert_array_almost_equal(ops_new_production, np_new_production)


def test_ops_credit_demand_logic():
    """
    Test credit demand calculation using ops module.

    This validates logic similar to firms_decide_credit_demand.
    """
    rng = np.random.default_rng(42)

    # Setup test data
    wage_bill = rng.uniform(100, 500, 50)
    net_worth = rng.uniform(-50, 200, 50)

    # NumPy implementation: credit_demand = max(0, wage_bill - net_worth)
    np_credit_demand = np.maximum(0.0, np.subtract(wage_bill, net_worth))

    # ops implementation
    ops_credit_demand = ops.maximum(0.0, ops.subtract(wage_bill, net_worth))

    # Should match
    np.testing.assert_array_almost_equal(ops_credit_demand, np_credit_demand)


def test_ops_wage_calculation_with_floor():
    """
    Test wage calculation with minimum wage floor using ops module.

    This validates logic similar to firms_decide_wage_offer.
    """
    rng = np.random.default_rng(42)

    # Setup test data
    base_wage = rng.uniform(8, 15, 50)
    min_wage = 10.0
    shock = rng.uniform(0.0, 0.1, 50)

    # NumPy implementation: wage = max(min_wage, base_wage * (1 + shock))
    np_wage_with_shock = np.multiply(base_wage, 1.0 + shock)
    np_wage = np.maximum(min_wage, np_wage_with_shock)

    # ops implementation
    ops_wage_with_shock = ops.multiply(base_wage, 1.0 + shock)
    ops_wage = ops.maximum(min_wage, ops_wage_with_shock)

    # Should match
    np.testing.assert_array_almost_equal(ops_wage, np_wage)


def test_ops_multi_condition_select():
    """
    Test multi-condition select logic using ops module.

    This validates logic for categorizing firms into different states.
    """
    rng = np.random.default_rng(42)

    # Setup test data
    inventory = rng.uniform(0, 50, 50)
    price = rng.uniform(80, 120, 50)
    avg_price = 100.0

    # Categorize firms:
    # - High inventory -> state 1
    # - No inventory and high price -> state 2
    # - No inventory and low price -> state 3
    # - Default -> state 0

    # NumPy implementation
    np_cond1 = inventory > 20
    np_cond2 = np.logical_and(inventory == 0, price >= avg_price)
    np_cond3 = np.logical_and(inventory == 0, price < avg_price)
    np_state = np.select([np_cond1, np_cond2, np_cond3], [1.0, 2.0, 3.0], default=0.0)

    # ops implementation
    ops_cond1 = ops.greater(inventory, 20)
    ops_cond2 = ops.logical_and(
        ops.equal(inventory, 0), ops.greater_equal(price, avg_price)
    )
    ops_cond3 = ops.logical_and(ops.equal(inventory, 0), ops.less(price, avg_price))
    ops_state = ops.select(
        [ops_cond1, ops_cond2, ops_cond3], [1.0, 2.0, 3.0], default=0.0
    )

    # Should match
    np.testing.assert_array_almost_equal(ops_state, np_state)


def test_ops_in_place_operations():
    """
    Test that in-place operations work correctly with ops module.

    This validates the out= parameter pattern.
    """
    rng = np.random.default_rng(42)

    # Setup test data
    a = rng.uniform(1, 100, 50)
    b = rng.uniform(1, 100, 50)

    # NumPy implementation (in-place)
    np_result = np.zeros(50)
    np.multiply(a, b, out=np_result)

    # ops implementation (in-place)
    ops_result = ops.zeros(50)
    ops.multiply(a, b, out=ops_result)

    # Should match
    np.testing.assert_array_almost_equal(ops_result, np_result)

    # Test with add
    np_result2 = np.zeros(50)
    np.add(a, b, out=np_result2)

    ops_result2 = ops.zeros(50)
    ops.add(a, b, out=ops_result2)

    np.testing.assert_array_almost_equal(ops_result2, np_result2)
