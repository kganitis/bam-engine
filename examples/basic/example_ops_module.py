"""
==============
BAM Operations
==============

This example provides a comprehensive guide to the ``ops`` module,
which offers NumPy-free array operations for writing custom events.
The ops module makes BAM Engine accessible to economists who may not
be familiar with NumPy.

You'll learn to:

- Use arithmetic operations (add, multiply, divide)
- Perform comparisons and logical operations
- Apply conditional logic (where, select)
- Aggregate data (sum, mean, any, all)
- Create and manipulate arrays
- Use random operations with the simulation RNG
"""

# %%
# Why Use ops Instead of NumPy?
# -----------------------------
#
# The ``ops`` module provides:
#
# 1. **Safe defaults**: Division by zero is handled automatically
# 2. **Consistent naming**: Verb-based names (multiply vs ``*``)
# 3. **In-place operations**: Support for ``out=`` parameter
# 4. **Type hints**: Better IDE support
# 5. **Progressive disclosure**: Beginners don't need NumPy knowledge

import numpy as np

from bamengine import ops

# Example: Safe division (no error on division by zero)
prices = np.array([10.0, 20.0, 0.0, 15.0])
quantities = np.array([2.0, 0.0, 5.0, 3.0])

# NumPy would give infinity or NaN for division by zero
# ops.divide handles this safely
unit_price = ops.divide(prices, quantities)
print("Safe division with ops.divide:")
print(f"  prices:     {prices}")
print(f"  quantities: {quantities}")
print(f"  unit_price: {unit_price}")
print("  (zeros handled safely)")

# %%
# Arithmetic Operations
# ---------------------
#
# Basic math: add, subtract, multiply, divide

a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([10.0, 20.0, 30.0, 40.0])

print("\nArithmetic operations:")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  add(a, b)      = {ops.add(a, b)}")
print(f"  subtract(b, a) = {ops.subtract(b, a)}")
print(f"  multiply(a, 2) = {ops.multiply(a, 2)}")
print(f"  divide(b, a)   = {ops.divide(b, a)}")

# In-place operations with out parameter
result = np.zeros(4)
ops.add(a, b, out=result)
print(
    f"  add(a, b, out=result): result is same object? {result is ops.add(a, b, out=result)}"
)

# %%
# Comparison Operations
# ---------------------
#
# Element-wise comparisons return boolean arrays.

prices = np.array([0.9, 1.0, 1.1, 1.2])
threshold = 1.0

print("\nComparison operations:")
print(f"  prices = {prices}")
print(f"  threshold = {threshold}")
print(f"  equal(prices, threshold):         {ops.equal(prices, threshold)}")
print(f"  less(prices, threshold):          {ops.less(prices, threshold)}")
print(f"  greater(prices, threshold):       {ops.greater(prices, threshold)}")
print(f"  less_equal(prices, threshold):    {ops.less_equal(prices, threshold)}")
print(f"  greater_equal(prices, threshold): {ops.greater_equal(prices, threshold)}")
print(f"  not_equal(prices, threshold):     {ops.not_equal(prices, threshold)}")

# %%
# Logical Operations
# ------------------
#
# Combine boolean conditions.

employed = np.array([True, True, False, True, False])
low_wage = np.array([True, False, True, False, True])

print("\nLogical operations:")
print(f"  employed = {employed}")
print(f"  low_wage = {low_wage}")
print(f"  logical_and(employed, low_wage): {ops.logical_and(employed, low_wage)}")
print(f"  logical_or(employed, low_wage):  {ops.logical_or(employed, low_wage)}")
print(f"  logical_not(employed):           {ops.logical_not(employed)}")

# %%
# Conditional Operations
# ----------------------
#
# Vectorized if-then-else logic.

inventory = np.array([100, 0, 50, 10, 0])
base_price = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

# ops.where: if-then-else
# If inventory > 0, discount by 5%; otherwise premium by 10%
has_inventory = ops.greater(inventory, 0)
new_price = ops.where(
    has_inventory,
    ops.multiply(base_price, 0.95),  # 5% discount
    ops.multiply(base_price, 1.10),  # 10% premium
)

print("\nConditional with ops.where:")
print(f"  inventory:     {inventory}")
print(f"  has_inventory: {has_inventory}")
print(f"  new_price:     {new_price}")
print("  (discount if inventory > 0, premium otherwise)")

# ops.select: multiple conditions (switch/case)
# Pricing based on inventory level
high_stock = ops.greater(inventory, 50)
medium_stock = ops.logical_and(
    ops.greater(inventory, 10), ops.less_equal(inventory, 50)
)
low_stock = ops.logical_and(ops.greater(inventory, 0), ops.less_equal(inventory, 10))

tiered_price = ops.select(
    conditions=[high_stock, medium_stock, low_stock],
    choices=[8.0, 10.0, 12.0],  # Prices for each tier
    default=15.0,  # No inventory
)

print("\nConditional with ops.select:")
print(f"  inventory:    {inventory}")
print(f"  tiered_price: {tiered_price}")
print("  (8 for high stock, 10 medium, 12 low, 15 none)")

# %%
# Element-wise Operations
# -----------------------
#
# Maximum, minimum, and clipping.

proposed_wages = np.array([0.5, 0.8, 1.2, 0.6, 1.5])
min_wage = 0.7
max_wage = 1.3

print("\nElement-wise operations:")
print(f"  proposed_wages: {proposed_wages}")

# Enforce minimum wage
actual_wages = ops.maximum(proposed_wages, min_wage)
print(f"  maximum(wages, {min_wage}): {actual_wages}")

# Enforce maximum wage
capped_wages = ops.minimum(proposed_wages, max_wage)
print(f"  minimum(wages, {max_wage}): {capped_wages}")

# Enforce both (clip)
bounded_wages = ops.clip(proposed_wages, min_wage, max_wage)
print(f"  clip(wages, {min_wage}, {max_wage}): {bounded_wages}")

# %%
# Aggregation Operations
# ----------------------
#
# Sum, mean, any, all with optional masks.

production = np.array([100.0, 0.0, 50.0, 75.0, 0.0])
prices = np.array([10.0, 0.0, 12.0, 11.0, 0.0])
active = ops.greater(production, 0)

print("\nAggregation operations:")
print(f"  production: {production}")
print(f"  active:     {active}")

# Total production
print(f"  sum(production): {ops.sum(production)}")

# Production of active firms only
print(f"  sum(production, where=active): {ops.sum(production, where=active)}")

# Average production
print(f"  mean(production): {ops.mean(production):.2f}")

# Average of active firms
print(f"  mean(production, where=active): {ops.mean(production, where=active):.2f}")

# Check if any firm is inactive
inactive = ops.logical_not(active)
print(f"  any(inactive): {ops.any(inactive)}")

# Check if all firms are active
print(f"  all(active): {ops.all(active)}")

# %%
# Array Creation
# --------------
#
# Create arrays of specific values.

n = 5
print("\nArray creation:")
print(f"  zeros({n}): {ops.zeros(n)}")
print(f"  ones({n}):  {ops.ones(n)}")
print(f"  full({n}, 3.14): {ops.full(n, 3.14)}")
print(f"  empty({n}): (uninitialized values)")

# arange: create evenly spaced values
periods = ops.arange(0, 5, 1)
print(f"  arange(0, 5, 1): {periods}")

# arange with fractional step
time_axis = ops.arange(0, 1, 0.25)
print(f"  arange(0, 1, 0.25): {time_axis}")

# array: create array from Python list (always copies)
data_list = [1.0, 2.0, 3.0, 4.0]
arr = ops.array(data_list)
print(f"\n  array({data_list}): {arr}")

# array always creates a copy, even from existing arrays
original = np.array([10.0, 20.0, 30.0])
copy = ops.array(original)
copy[0] = 999  # Modifying copy
print(f"  array() creates copy: original={original}, copy={copy}")

# asarray: creates view when possible (no copy for compatible arrays)
view = ops.asarray(original)
print(f"  asarray() may create view: shares memory? {np.shares_memory(original, view)}")

# Key difference: array always copies, asarray may reuse memory
# Use array when you need a fresh copy
# Use asarray when you want to avoid unnecessary copies

# %%
# Mathematical Functions
# ----------------------
#
# Logarithm, exponential, and other math operations.

gdp = np.array([100.0, 110.0, 121.0, 133.1])

print("\nMathematical functions:")
print(f"  gdp: {gdp}")

# Natural logarithm (useful for log-scale analysis)
log_gdp = ops.log(gdp)
print(f"  log(gdp): {log_gdp.round(3)}")

# Log differences approximate growth rates
log_diff = np.diff(log_gdp)
print(f"  log differences (≈ growth rates): {log_diff.round(3)}")

# Exponential function (e^x)
# Useful for reversing log transform or modeling exponential decay
growth_rates = np.array([0.0, 0.05, 0.10, -0.05])
print(f"\n  growth_rates: {growth_rates}")
growth_factors = ops.exp(growth_rates)
print(f"  exp(growth_rates): {growth_factors.round(4)}")

# Common pattern: calculate probability-like decays
# sigma = 0.1 * exp(-fragility), where higher fragility reduces investment
fragility = np.array([0.0, 0.5, 1.0, 2.0])
investment_share = ops.multiply(0.1, ops.exp(ops.multiply(-1.0, fragility)))
print(f"\n  fragility: {fragility}")
print(f"  0.1 * exp(-fragility): {investment_share.round(4)}")
print("  (higher fragility -> lower investment)")

# %%
# Utility Operations
# ------------------
#
# Sorting, unique values, counting, membership.

employer_ids = np.array([0, 2, 1, 0, 2, 2, 1, 0])

print("\nUtility operations:")
print(f"  employer_ids: {employer_ids}")

# Unique employers
print(f"  unique(employer_ids): {ops.unique(employer_ids)}")

# Count workers per employer
counts = ops.bincount(employer_ids, minlength=3)
print(f"  bincount (workers per employer): {counts}")

# Check membership
target_employers = np.array([0, 1])
is_target = ops.isin(employer_ids, target_employers)
print(f"  isin(ids, [0,1]): {is_target}")

# Sorting
prices = np.array([30.0, 10.0, 25.0, 15.0])
print(f"\n  prices: {prices}")
print(f"  sort(prices): {ops.sort(prices)}")
print(f"  argsort(prices): {ops.argsort(prices)}")
print("  (argsort gives indices that would sort the array)")

# %%
# In-Place Assignment
# -------------------
#
# Modify arrays in place with ops.assign.

prices = np.array([10.0, 20.0, 30.0])
print("\nIn-place assignment:")
print(f"  Before: {prices}")

# Calculate new prices
new_prices = ops.multiply(prices, 1.1)

# Assign in place (equivalent to prices[:] = new_prices)
ops.assign(prices, new_prices)
print(f"  After ops.assign: {prices}")

# %%
# Random Operations
# -----------------
#
# Generate random numbers using simulation RNG.

import bamengine as bam

# Create simulation to get RNG
sim = bam.Simulation.init(n_firms=100, n_households=500, seed=42)

# Generate uniform random numbers
shocks = ops.uniform(sim.rng, low=-0.1, high=0.1, size=10)
print("\nRandom operations:")
print("  uniform(rng, -0.1, 0.1, size=10):")
print(f"    {shocks.round(4)}")

# Random shocks are reproducible with same seed
sim2 = bam.Simulation.init(n_firms=100, n_households=500, seed=42)
shocks2 = ops.uniform(sim2.rng, low=-0.1, high=0.1, size=10)
print(f"  Same seed produces same values: {np.allclose(shocks, shocks2)}")

# %%
# Practical Example: Custom Pricing Event
# ---------------------------------------
#
# Combine ops operations in a realistic custom event.

from bamengine import event


@event
class CostPlusPricing:
    """Set prices using cost-plus markup strategy.

    This event demonstrates using multiple ops operations together
    in a realistic economic mechanism.
    """

    def execute(self, sim):
        # Get roles
        prod = sim.get_role("Producer")
        emp = sim.get_role("Employer")

        # Parameters
        markup_rate = 0.20  # 20% markup
        price_floor = 0.5  # Minimum price
        max_price_change = 0.10  # Max 10% change per period

        # Calculate unit labor cost
        # (wage / productivity, safe division handles zeros)
        unit_cost = ops.divide(emp.wage_offer, prod.labor_productivity)

        # Apply markup
        target_price = ops.multiply(unit_cost, 1.0 + markup_rate)

        # Enforce price floor
        target_price = ops.maximum(target_price, price_floor)

        # Limit price changes (gradual adjustment)
        price_change = ops.subtract(target_price, prod.price)
        max_increase = ops.multiply(prod.price, max_price_change)
        max_decrease = ops.multiply(prod.price, -max_price_change)

        # Clip change to allowed range
        bounded_change = ops.clip(price_change, max_decrease, max_increase)

        # Apply bounded change
        new_price = ops.add(prod.price, bounded_change)

        # Update prices in place
        ops.assign(prod.price, new_price)


print("\nCostPlusPricing event defined")
print("Operations used:")
print("  - divide (safe division for unit cost)")
print("  - multiply (markup application)")
print("  - maximum (price floor)")
print("  - subtract (price change)")
print("  - clip (bound changes)")
print("  - add (apply change)")
print("  - assign (in-place update)")

# Test the event
sim = bam.Simulation.init(n_firms=50, n_households=250, seed=42)
sim.run(n_periods=5)  # Establish some state

print(f"\nBefore CostPlusPricing: Mean price = {np.mean(sim.prod.price):.3f}")

pricing_event = CostPlusPricing()
pricing_event.execute(sim)

print(f"After CostPlusPricing:  Mean price = {np.mean(sim.prod.price):.3f}")

# %%
# Visualization: Ops in Action
# ----------------------------
#
# Visualize how ops operations affect simulation dynamics.

import matplotlib.pyplot as plt

# Simulate with and without price floor
n_periods = 50

# Standard simulation
sim_standard = bam.Simulation.init(n_firms=100, n_households=500, seed=42)
prices_standard = []
for _ in range(n_periods):
    sim_standard.step()
    prices_standard.append(np.mean(sim_standard.prod.price))

# Simulation with price floor applied each period


@event
class ApplyPriceFloor:
    def execute(self, sim):
        prod = sim.get_role("Producer")
        floor = 0.8
        ops.assign(prod.price, ops.maximum(prod.price, floor))


sim_floor = bam.Simulation.init(n_firms=100, n_households=500, seed=42)
floor_event = ApplyPriceFloor()
prices_floor = []
for _ in range(n_periods):
    sim_floor.step()
    floor_event.execute(sim_floor)
    prices_floor.append(np.mean(sim_floor.prod.price))

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(prices_standard, label="Standard", linewidth=2)
ax.plot(prices_floor, label="With Price Floor (0.8)", linewidth=2)
ax.axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="Floor")
ax.set_xlabel("Period")
ax.set_ylabel("Mean Price")
ax.set_title("Effect of ops.maximum Price Floor")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Operations Reference
# --------------------
#
# Quick reference of all available operations.

reference = """
ARITHMETIC
  add(a, b)           - Element-wise addition
  subtract(a, b)      - Element-wise subtraction
  multiply(a, b)      - Element-wise multiplication
  divide(a, b)        - Safe division (handles zeros)

ASSIGNMENT
  assign(target, val) - In-place assignment (target[:] = val)

COMPARISON
  equal(a, b)         - Element-wise ==
  not_equal(a, b)     - Element-wise !=
  less(a, b)          - Element-wise <
  less_equal(a, b)    - Element-wise <=
  greater(a, b)       - Element-wise >
  greater_equal(a, b) - Element-wise >=

LOGICAL
  logical_and(a, b)   - Element-wise AND
  logical_or(a, b)    - Element-wise OR
  logical_not(a)      - Element-wise NOT

CONDITIONAL
  where(cond, x, y)   - If-then-else
  select(conds, vals) - Multi-condition switch

ELEMENT-WISE
  maximum(a, b)       - Element-wise max
  minimum(a, b)       - Element-wise min
  clip(a, lo, hi)     - Clip to range [lo, hi]

AGGREGATION
  sum(a, axis, where) - Sum elements
  mean(a, axis, where)- Mean of elements
  any(a)              - Any True?
  all(a)              - All True?

ARRAY CREATION
  zeros(n)            - Array of zeros
  ones(n)             - Array of ones
  full(n, val)        - Array of constant value
  empty(n)            - Uninitialized array
  arange(start, stop) - Evenly spaced values
  array(data)         - Create array (always copies)
  asarray(data)       - Convert to array (may share memory)

MATHEMATICAL
  log(a)              - Natural logarithm
  exp(a)              - Exponential (e^x)

UTILITIES
  unique(a)           - Unique sorted values
  bincount(a)         - Count occurrences
  isin(a, vals)       - Membership test
  argsort(a)          - Indices that sort array
  sort(a)             - Sorted copy

RANDOM
  uniform(rng, lo, hi, size) - Uniform random numbers
"""
print(reference)

# %%
# Key Takeaways
# -------------
#
# - Use ``ops`` for NumPy-free array operations
# - ``ops.divide`` handles division by zero safely
# - ``ops.where`` for if-then-else, ``ops.select`` for switch/case
# - ``ops.sum/mean`` support ``where`` parameter for filtered aggregation
# - ``ops.assign`` for in-place array updates
# - ``ops.uniform`` for random numbers (requires ``sim.rng``)
# - ``ops.exp`` for exponential decay patterns (e.g., R&D investment)
# - ``ops.array`` for creating copies, ``ops.asarray`` for memory-efficient views
# - All operations can be combined for complex economic logic
