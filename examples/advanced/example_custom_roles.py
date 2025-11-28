"""
============
Custom Roles
============

This example demonstrates how to create custom roles (components) to extend
BAM Engine with new agent behaviors and state. Custom roles are essential
for modeling economic phenomena not covered by the built-in roles.

You'll learn to:

- Define roles using the ``@role`` decorator
- Use type annotations for NumPy arrays
- Handle optional fields with defaults
- Register and retrieve custom roles
- Integrate roles with simulations
"""

# %%
# What are Roles?
# ---------------
#
# In BAM Engine's ECS (Entity-Component-System) architecture:
#
# - **Agents** are lightweight entities (just IDs and types)
# - **Roles** are components that hold agent state as NumPy arrays
# - **Events** are systems that operate on roles
#
# Roles bundle related state together. For example, the built-in
# ``Producer`` role holds production-related state (price, inventory, output).

import numpy as np

from bamengine import Agent, Bool, Float, Int, get_role, ops, role

# Check built-in roles
print("Built-in roles:")
for name in ["Producer", "Worker", "Consumer", "Borrower", "Lender", "Employer"]:
    try:
        r = get_role(name)
        print(f"  {name}: {list(r.__dataclass_fields__.keys())[:4]}...")
    except KeyError:
        print(f"  {name}: not registered")

# %%
# Simple Custom Role
# ------------------
#
# The ``@role`` decorator handles everything: dataclass creation,
# Role inheritance, and registration.


@role
class Inventory:
    """Inventory management role for firms.

    Tracks stock levels, reorder points, and delivery status.
    Attach to firms to enable inventory management behavior.
    """

    # Current stock level (Float for continuous quantities)
    stock_level: Float

    # Reorder threshold (when stock falls below this, reorder)
    reorder_point: Float

    # Maximum storage capacity
    max_capacity: Float


print("\nInventory role created!")
print(f"  Fields: {list(Inventory.__dataclass_fields__.keys())}")

# Verify it's registered
inv_role = get_role("Inventory")
print(f"  Registered: {inv_role is Inventory}")

# %%
# Role with Multiple Field Types
# ------------------------------
#
# Roles can use all type aliases: Float, Int, Bool, and Agent.
# Each serves a specific purpose.


@role
class SupplyChain:
    """Supply chain management role for firms.

    Models upstream relationships and delivery logistics.
    """

    # Monetary values and quantities (Float)
    order_value: Float  # Value of pending orders
    unit_cost: Float  # Cost per unit from supplier
    delivery_cost: Float  # Shipping cost

    # Counts and durations (Int)
    pending_orders: Int  # Number of undelivered orders
    days_to_delivery: Int  # Days until next delivery
    contract_length: Int  # Periods remaining on supplier contract

    # Status flags (Bool)
    has_supplier: Bool  # Whether firm has established supplier
    awaiting_delivery: Bool  # Expecting a delivery soon
    is_preferred_customer: Bool  # Gets priority from supplier

    # Agent references (Agent)
    supplier_id: Agent  # ID of supplying firm (-1 if none)
    backup_supplier_id: Agent  # Secondary supplier (-1 if none)


print("\nSupplyChain role created with typed fields:")
print("  Float fields: order_value, unit_cost, delivery_cost")
print("  Int fields: pending_orders, days_to_delivery, contract_length")
print("  Bool fields: has_supplier, awaiting_delivery, is_preferred_customer")
print("  Agent fields: supplier_id, backup_supplier_id")

# %%
# Optional Fields with Defaults
# -----------------------------
#
# Use ``Optional`` and ``field()`` for optional fields that may not
# always be populated.

from dataclasses import field
from typing import Optional


@role
class QualityControl:
    """Quality control role for production firms.

    Tracks defect rates and quality investment. Some fields are
    optional and only populated when quality tracking is enabled.
    """

    # Required fields
    defect_rate: Float  # Fraction of defective output (0-1)
    quality_score: Float  # Overall quality rating (0-100)

    # Optional fields (may be None if feature not used)
    inspection_results: Optional[Float] = field(default=None)
    warranty_claims: Optional[Int] = field(default=None)

    # Optional scratch buffers for computation
    _temp_buffer: Optional[Float] = field(default=None)


print("\nQualityControl role with optional fields:")
print("  Required: defect_rate, quality_score")
print("  Optional: inspection_results, warranty_claims, _temp_buffer")

# %%
# Instantiating Custom Roles
# --------------------------
#
# Create role instances by passing NumPy arrays for each field.

n_firms = 5

# Create Inventory role instance
inventory = Inventory(
    stock_level=np.array([100.0, 50.0, 75.0, 200.0, 30.0]),
    reorder_point=np.array([25.0, 20.0, 30.0, 50.0, 15.0]),
    max_capacity=np.array([500.0, 300.0, 400.0, 600.0, 200.0]),
)

print("\nInventory instance for 5 firms:")
print(f"  Stock levels: {inventory.stock_level}")
print(f"  Reorder points: {inventory.reorder_point}")

# Create SupplyChain instance
supply = SupplyChain(
    order_value=np.array([1000.0, 500.0, 750.0, 2000.0, 300.0]),
    unit_cost=np.array([10.0, 15.0, 12.0, 8.0, 20.0]),
    delivery_cost=np.array([50.0, 40.0, 45.0, 60.0, 35.0]),
    pending_orders=np.array([2, 0, 1, 3, 0], dtype=np.int64),
    days_to_delivery=np.array([3, 0, 5, 2, 0], dtype=np.int64),
    contract_length=np.array([12, 6, 8, 24, 0], dtype=np.int64),
    has_supplier=np.array([True, True, True, True, False]),
    awaiting_delivery=np.array([True, False, True, True, False]),
    is_preferred_customer=np.array([True, False, False, True, False]),
    supplier_id=np.array([2, 3, 4, 1, -1], dtype=np.intp),
    backup_supplier_id=np.array([3, -1, 2, 0, -1], dtype=np.intp),
)

print("\nSupplyChain instance:")
print(f"  Supplier IDs: {supply.supplier_id}")
print(f"  Has supplier: {supply.has_supplier}")

# %%
# Modifying Role State
# --------------------
#
# Role fields are NumPy arrays that can be modified in place.

print("\nModifying inventory state:")
print(f"  Before: {inventory.stock_level}")

# Simulate consumption (reduce stock)
consumption = np.array([20.0, 10.0, 15.0, 40.0, 5.0])
inventory.stock_level -= consumption

print(f"  After consumption: {inventory.stock_level}")

# Check which firms need to reorder
needs_reorder = inventory.stock_level < inventory.reorder_point
print(f"  Needs reorder: {needs_reorder}")
print(f"  Firms needing reorder: {np.where(needs_reorder)[0]}")

# %%
# Role with Methods
# -----------------
#
# Roles can include helper methods for common operations.


@role
class CreditRating:
    """Credit rating role for borrower assessment.

    Tracks creditworthiness metrics and rating history.
    """

    # Credit metrics
    score: Float  # Credit score (0-100)
    default_history: Float  # Historical default rate
    debt_to_income: Float  # Debt-to-income ratio

    # Rating category (encoded as int: 0=AAA, 1=AA, 2=A, etc.)
    rating_category: Int

    def is_investment_grade(self) -> Bool:
        """Check if rating is investment grade (BBB or better)."""
        return self.rating_category <= 3

    def needs_review(self) -> Bool:
        """Check if credit needs manual review."""
        return (self.score < 50) | (self.debt_to_income > 0.5)


# Create instance
credit = CreditRating(
    score=np.array([85.0, 45.0, 72.0, 30.0, 90.0]),
    default_history=np.array([0.0, 0.15, 0.05, 0.3, 0.0]),
    debt_to_income=np.array([0.3, 0.6, 0.4, 0.8, 0.2]),
    rating_category=np.array([1, 5, 3, 6, 0], dtype=np.int64),
)

print("\nCreditRating with methods:")
print(f"  Scores: {credit.score}")
print(f"  Investment grade: {credit.is_investment_grade()}")
print(f"  Needs review: {credit.needs_review()}")

# %%
# Traditional Syntax (Alternative)
# --------------------------------
#
# The ``@role`` decorator is syntactic sugar. You can also use
# explicit inheritance and dataclass decoration.

from dataclasses import dataclass

from bamengine.core import Role


# This is equivalent to using @role
@dataclass(slots=True)
class TraditionalInventory(Role):
    """Inventory role using traditional syntax."""

    stock_level: Float
    reorder_point: Float
    max_capacity: Float


print("\nTraditional syntax role:")
print(f"  Fields: {list(TraditionalInventory.__dataclass_fields__.keys())}")
print(f"  Is subclass of Role: {issubclass(TraditionalInventory, Role)}")

# Both approaches produce equivalent results
# The @role decorator is recommended for cleaner code

# %%
# Visualizing Role Data
# ---------------------
#
# Plot role state over a simulated process.

import matplotlib.pyplot as plt

# Simulate inventory dynamics over 20 periods
n_periods = 20
n_firms = 10

# Initialize
stock = np.full(n_firms, 100.0)
reorder_point = np.full(n_firms, 30.0)
stock_history = [stock.copy()]

# Simulate consumption and reorder
rng = np.random.default_rng(42)
for t in range(n_periods):
    # Random consumption
    consumption = rng.uniform(5, 20, n_firms)
    stock = ops.maximum(ops.subtract(stock, consumption), 0)

    # Reorder if below threshold
    needs_reorder = stock < reorder_point
    stock = ops.where(needs_reorder, ops.add(stock, 80), stock)

    stock_history.append(stock.copy())

stock_history = np.array(stock_history)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot individual firms (faded)
for i in range(n_firms):
    ax.plot(stock_history[:, i], alpha=0.3, linewidth=1)

# Plot mean
ax.plot(stock_history.mean(axis=1), "k-", linewidth=2, label="Mean Stock")

# Reorder point line
ax.axhline(y=30, color="red", linestyle="--", label="Reorder Point")

ax.set_xlabel("Period")
ax.set_ylabel("Stock Level")
ax.set_title("Inventory Dynamics with Reorder Policy")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nSimulated {n_firms} firms over {n_periods} periods")
print(f"Final mean stock: {stock_history[-1].mean():.1f}")

# %%
# Custom Role for R&D Extension
# -----------------------------
#
# A realistic example: R&D and innovation tracking for endogenous growth.


@role
class ResearchDevelopment:
    """R&D and innovation role for endogenous growth modeling.

    Extends the BAM model to include innovation dynamics, productivity
    improvements, and technology spillovers between firms.
    """

    # R&D spending and effort
    rd_budget: Float  # R&D budget this period
    rd_intensity: Float  # R&D as fraction of revenue
    rd_workers: Int  # Number of R&D employees

    # Innovation outcomes
    innovation_success: Bool  # Whether innovation occurred this period
    patent_count: Int  # Cumulative patents held
    tech_level: Float  # Current technology level (affects productivity)

    # Knowledge and spillovers
    absorptive_capacity: Float  # Ability to learn from others (0-1)
    knowledge_stock: Float  # Accumulated R&D knowledge

    # Productivity link
    productivity_boost: Float  # Multiplicative boost from innovation


# Instantiate for 4 firms
rd = ResearchDevelopment(
    rd_budget=np.array([100.0, 50.0, 200.0, 75.0]),
    rd_intensity=np.array([0.05, 0.02, 0.08, 0.03]),
    rd_workers=np.array([5, 2, 10, 3], dtype=np.int64),
    innovation_success=np.array([False, False, True, False]),
    patent_count=np.array([3, 1, 8, 2], dtype=np.int64),
    tech_level=np.array([1.0, 0.9, 1.3, 0.95]),
    absorptive_capacity=np.array([0.7, 0.4, 0.9, 0.5]),
    knowledge_stock=np.array([50.0, 20.0, 100.0, 30.0]),
    productivity_boost=np.array([1.0, 1.0, 1.15, 1.0]),
)

print("\nR&D Role for endogenous growth:")
print(f"  R&D intensity: {rd.rd_intensity}")
print(f"  Tech levels: {rd.tech_level}")
print(f"  Innovation success: {rd.innovation_success}")
print(f"  Firm with highest tech: Firm {np.argmax(rd.tech_level)}")

# %%
# Key Takeaways
# -------------
#
# - Use ``@role`` decorator for clean, automatic role definition
# - Choose appropriate types: Float, Int, Bool, Agent
# - Use Optional and field() for optional/default values
# - Roles can include helper methods
# - Roles are registered automatically and retrievable via ``get_role()``
# - Traditional syntax (explicit @dataclass + Role inheritance) also works
