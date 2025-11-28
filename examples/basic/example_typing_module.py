"""
===========
Type System
===========

This example demonstrates BAM Engine's type aliases for defining custom
roles and components. These types hide NumPy complexity while maintaining
full type safety.

If you're extending BAM Engine with custom roles or events, understanding
these types will make your code cleaner and more maintainable.
"""

# %%
# Introduction to Type Aliases
# ----------------------------
#
# BAM Engine provides four user-friendly type aliases:
#
# - ``Float``: Array of floating-point values (prices, quantities, rates)
# - ``Int``: Array of integer values (counts, periods, durations)
# - ``Bool``: Array of boolean values (flags, conditions)
# - ``Agent``: Array of agent IDs (references to other agents)
#
# These are all NumPy arrays under the hood, but with clear semantic meaning.

import numpy as np

from bamengine import Agent, Bool, Float, Int

# Create arrays using standard NumPy, but type them semantically
prices: Float = np.array([1.0, 1.2, 0.95, 1.1], dtype=np.float64)
counts: Int = np.array([10, 5, 8, 12], dtype=np.int64)
active: Bool = np.array([True, True, False, True], dtype=np.bool_)
employers: Agent = np.array([0, 1, -1, 0], dtype=np.intp)  # -1 = unemployed

print("Type aliases map to NumPy arrays:")
print(f"  Float  -> {prices.dtype}")
print(f"  Int    -> {counts.dtype}")
print(f"  Bool   -> {active.dtype}")
print(f"  Agent  -> {employers.dtype}")

# %%
# Understanding the Underlying Types
# ----------------------------------
#
# Each type alias corresponds to a specific NumPy dtype:

print("\nType mapping:")
print("  Float = NDArray[np.float64]  # 64-bit floating point")
print("  Int   = NDArray[np.int64]    # 64-bit signed integer")
print("  Bool  = NDArray[np.bool_]    # Boolean (True/False)")
print("  Agent = NDArray[np.intp]     # Platform-dependent integer (for indexing)")

# The Agent type uses np.intp for safe array indexing
# This ensures compatibility across 32-bit and 64-bit systems
print(f"\nnp.intp on this system: {np.dtype(np.intp)}")

# %%
# Defining Custom Roles with Types
# --------------------------------
#
# When defining custom roles, use these types to annotate fields.
# This provides IDE autocomplete, type checking, and documentation.

from bamengine import role


@role
class Inventory:
    """Custom inventory tracking role for firms."""

    # Float for quantities and monetary values
    goods_on_hand: Float  # Current stock level
    reorder_point: Float  # When to reorder
    unit_cost: Float  # Cost per unit

    # Int for counts and durations
    days_to_delivery: Int  # Days until next delivery
    order_count: Int  # Number of pending orders

    # Bool for status flags
    needs_reorder: Bool  # Whether stock is low
    is_backordered: Bool  # Whether we're out of stock

    # Agent for relationships to other agents
    supplier_id: Agent  # Which firm supplies us (-1 if none)


print("Inventory role defined with typed fields:")
print(f"  Fields: {list(Inventory.__dataclass_fields__.keys())}")

# %%
# Instantiating Roles with Arrays
# -------------------------------
#
# When creating role instances, pass NumPy arrays of the appropriate type.

# Number of agents with this role
n_firms = 5

# Create arrays for each field
inventory = Inventory(
    goods_on_hand=np.array([100.0, 50.0, 75.0, 200.0, 30.0]),
    reorder_point=np.array([25.0, 20.0, 30.0, 50.0, 15.0]),
    unit_cost=np.array([10.0, 15.0, 12.0, 8.0, 20.0]),
    days_to_delivery=np.array([3, 0, 5, 2, 7], dtype=np.int64),
    order_count=np.array([1, 0, 2, 0, 1], dtype=np.int64),
    needs_reorder=np.array([False, True, False, False, True]),
    is_backordered=np.array([False, True, False, False, False]),
    supplier_id=np.array([2, 0, 4, 1, -1], dtype=np.intp),
)

print("\nInventory instance:")
print(f"  Goods on hand: {inventory.goods_on_hand}")
print(f"  Needs reorder: {inventory.needs_reorder}")
print(f"  Supplier IDs:  {inventory.supplier_id}")

# %%
# When to Use Each Type
# ---------------------
#
# Here's guidance on choosing the right type:

print(
    """
Type Selection Guide
====================

Float (np.float64):
  - Prices, costs, revenues
  - Quantities (production, inventory, consumption)
  - Rates (interest, inflation, unemployment)
  - Ratios and proportions
  - Any continuous numerical value

Int (np.int64):
  - Counts (workers, orders, periods)
  - Durations (contract length, days until event)
  - Discrete quantities (number of items)
  - Period/time indices

Bool (np.bool_):
  - Status flags (employed, bankrupt, active)
  - Conditions (has_inventory, needs_reorder)
  - Masks for filtering (unemployed_mask)
  - Binary states

Agent (np.intp):
  - References to other agents (employer_id, supplier_id)
  - Agent indices for array lookup
  - Use -1 to indicate "no agent" (unemployed, no supplier)
"""
)

# %%
# Type Checking Benefits
# ----------------------
#
# Using type annotations enables:
#
# 1. **IDE Autocomplete**: Your editor knows the types
# 2. **Static Analysis**: mypy can catch type errors
# 3. **Documentation**: Types serve as inline docs
# 4. **Refactoring Safety**: Rename/refactor with confidence

# Example: mypy would catch this error (wrong type)
# inventory.goods_on_hand = "not an array"  # Type error!

# Correct usage
inventory.goods_on_hand = np.array([90.0, 45.0, 70.0, 195.0, 25.0])
print("Updated goods_on_hand:", inventory.goods_on_hand)

# %%
# Working with Agent References
# -----------------------------
#
# The ``Agent`` type is commonly used for relationships between entities.
# The convention is to use -1 to indicate "no reference".

# Example: Workers with employer references
worker_employers: Agent = np.array([0, 0, 1, -1, 2, -1], dtype=np.intp)

# Find employed workers
employed_mask: Bool = worker_employers >= 0
print(f"Worker employers: {worker_employers}")
print(f"Employed mask:    {employed_mask}")
print(f"Employed count:   {np.sum(employed_mask)}")

# Find workers at firm 0
at_firm_0: Bool = worker_employers == 0
print(f"Workers at firm 0: {np.where(at_firm_0)[0]}")

# %%
# Internal vs User-Friendly Types
# -------------------------------
#
# BAM Engine provides two sets of type aliases:
#
# **User-friendly (recommended for custom code):**
# - ``Float``, ``Int``, ``Bool``, ``Agent``
#
# **Internal (used in bamengine source code):**
# - ``Float1D``, ``Int1D``, ``Bool1D``, ``Idx1D``
# - ``Float2D``, ``Int2D``, ``Idx2D`` (for 2D arrays)
#
# Both work identically - use whichever you prefer.

from bamengine.typing import Float1D

# These are equivalent:
prices_v1: Float = np.ones(10)
prices_v2: Float1D = np.ones(10)

print(f"\nFloat and Float1D are equivalent: {Float is Float1D}")

# %%
# Practical Example: Custom Production Role
# -----------------------------------------
#
# Here's a realistic example of a custom role with proper typing.


@role
class AdvancedProducer:
    """Enhanced producer with inventory management and quality control."""

    # Production metrics (Float)
    output: Float  # Units produced this period
    capacity: Float  # Maximum production capacity
    efficiency: Float  # Production efficiency (0-1)
    unit_labor_cost: Float  # Cost per unit of output

    # Inventory (Float)
    inventory: Float  # Current stock
    inventory_target: Float  # Desired stock level

    # Quality metrics (Float)
    defect_rate: Float  # Fraction of defective output (0-1)
    quality_investment: Float  # Spending on quality improvement

    # Workforce (Int)
    n_workers: Int  # Number of employees
    n_skilled_workers: Int  # Subset that are skilled

    # Status flags (Bool)
    is_operating: Bool  # Currently producing
    has_excess_inventory: Bool  # Inventory above target
    needs_workers: Bool  # Has unfilled positions

    # Relationships (Agent)
    primary_bank: Agent  # Main lender (-1 if none)
    main_supplier: Agent  # Key input supplier (-1 if none)


# Instantiate for 3 firms
advanced_prod = AdvancedProducer(
    output=np.array([100.0, 80.0, 120.0]),
    capacity=np.array([150.0, 100.0, 150.0]),
    efficiency=np.array([0.85, 0.75, 0.90]),
    unit_labor_cost=np.array([5.0, 6.0, 4.5]),
    inventory=np.array([50.0, 20.0, 80.0]),
    inventory_target=np.array([40.0, 30.0, 60.0]),
    defect_rate=np.array([0.02, 0.05, 0.01]),
    quality_investment=np.array([10.0, 5.0, 15.0]),
    n_workers=np.array([20, 15, 25], dtype=np.int64),
    n_skilled_workers=np.array([8, 5, 12], dtype=np.int64),
    is_operating=np.array([True, True, True]),
    has_excess_inventory=np.array([True, False, True]),
    needs_workers=np.array([False, True, False]),
    primary_bank=np.array([0, 1, 0], dtype=np.intp),
    main_supplier=np.array([1, -1, 2], dtype=np.intp),
)

print("\nAdvanced Producer role instantiated:")
print(f"  Output:     {advanced_prod.output}")
print(f"  Efficiency: {advanced_prod.efficiency}")
print(f"  Operating:  {advanced_prod.is_operating}")
print(f"  Banks:      {advanced_prod.primary_bank}")

# %%
# Key Takeaways
# -------------
#
# - Use ``Float`` for continuous values (prices, quantities, rates)
# - Use ``Int`` for discrete counts and durations
# - Use ``Bool`` for flags and conditions
# - Use ``Agent`` for references to other agents (-1 = no reference)
# - Type annotations enable IDE support and static analysis
# - Both user-friendly (Float) and internal (Float1D) types work
