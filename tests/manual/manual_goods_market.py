# tests/manual/manual_goods_market.py
from __future__ import annotations

import logging

import numpy as np
from numpy.random import default_rng

from bamengine.systems import consumers_calc_propensity, \
    consumers_decide_income_to_spend, consumers_decide_firms_to_visit, \
    consumers_shop_one_round, consumers_finalize_purchases
from helpers.factories import mock_producer, mock_consumer

logging.getLogger("bamengine").setLevel(5)

rng = default_rng(0)
Z = 3
beta = 0.5

# Create producers with varied inventory and prices
# Higher inventory levels to prevent complete sellout
prod = mock_producer(
    n=6,
    production=np.array([15.0, 18.0, 20.0, 22.0, 25.0, 30.0]),
    inventory=np.array([15.0, 18.0, 20.0, 22.0, 25.0, 30.0]),  # 130 total units
    price=np.array([0.6, 0.8, 1.0, 1.2, 1.5, 2.0]),  # Lower prices
    alloc_scratch=False,
)

# Create consumers with more conservative spending patterns
# Reduced consumer count and more varied savings to lower overall demand
con = mock_consumer(
    n=20,
    queue_z=Z,
    # Lower overall income levels
    income=np.array([
        # Low income group (8 consumers)
        0.2, 0.3, 0.4, 0.5, 0.3, 0.6, 0.4, 0.5,
        # Medium income group (8 consumers)
        0.8, 1.0, 1.2, 0.9, 1.1, 1.3, 0.7, 1.0,
        # High income group (4 consumers)
        1.8, 2.0, 1.5, 2.2
    ], dtype=np.float64),

    # Higher savings levels to reduce propensity to spend
    savings=np.array([
        # Low savers (moderate propensity to spend)
        4.0, 5.0, 3.5, 4.5, 6.0, 5.5, 4.8, 3.8,
        # Medium savers (lower propensity)
        10.0, 8.0, 9.5, 11.0, 7.5, 8.8, 9.2, 10.5,
        # High savers (very low propensity to spend)
        18.0, 22.0, 15.0, 25.0
    ], dtype=np.float64),

    # Loyalty distribution across 6 firms
    largest_prod_prev=np.array([
        0, 1, 2, 0, 3, 1, 4, 2, 5, 3,  # First group
        1, 4, 0, 2, 5, 1, 4, 0, 2, 3  # Second group
    ], dtype=np.intp),
)

# Print initial state for analysis
print("=== INITIAL STATE ===")
avg_sav = con.savings.mean()
total_wealth = (con.savings + con.income).sum()
total_inventory_value = (prod.inventory * prod.price).sum()

print(f"Average savings: {avg_sav:.2f}")
print(f"Total consumer wealth: {total_wealth:.2f}")
print(f"Total inventory value: {total_inventory_value:.2f}")
print(f"Wealth to inventory ratio: {total_wealth / total_inventory_value:.2f}")
print()

print("Producer inventory and prices:")
for i in range(len(prod.inventory)):
    value = prod.inventory[i] * prod.price[i]
    print(
        f"  Firm {i}: {prod.inventory[i]:.1f} units @ ${prod.price[i]:.2f} = ${value:.2f} value")
print()

# Run the simulation
print("=== RUNNING SIMULATION ===")
consumers_calc_propensity(con, avg_sav=avg_sav, beta=beta)

print(f"Propensity range: [{con.propensity.min():.3f}, {con.propensity.max():.3f}]")
print(f"Average propensity: {con.propensity.mean():.3f}")
print()

consumers_decide_income_to_spend(con)

print(f"Total spending budget: {con.income_to_spend.sum():.2f}")
print(f"Remaining savings: {con.savings.sum():.2f}")
print()

consumers_decide_firms_to_visit(con, prod, max_Z=Z, rng=rng)

print("=== SHOPPING ROUNDS ===")
for round_num in range(Z):
    print(f"\n--- Round {round_num + 1} ---")
    inventory_before = prod.inventory.sum()
    budget_before = con.income_to_spend.sum()

    consumers_shop_one_round(con, prod, rng)

    inventory_after = prod.inventory.sum()
    budget_after = con.income_to_spend.sum()

    print(f"Inventory: {inventory_before:.2f} → {inventory_after:.2f} "
          f"(sold: {inventory_before - inventory_after:.2f})")
    print(f"Budget: {budget_before:.2f} → {budget_after:.2f} "
          f"(spent: {budget_before - budget_after:.2f})")

consumers_finalize_purchases(con)

print("\n=== FINAL RESULTS ===")
remaining_inventory = prod.inventory.sum()
remaining_inventory_value = (prod.inventory * prod.price).sum()
final_savings = con.savings.sum()

print(f"Remaining inventory: {remaining_inventory:.2f} units")
print(f"Remaining inventory value: ${remaining_inventory_value:.2f}")
print(f"Final consumer savings: ${final_savings:.2f}")
print()

print("Final firm inventory:")
for i in range(len(prod.inventory)):
    remaining_value = prod.inventory[i] * prod.price[i]
    sold_units = prod.production[i] - prod.inventory[i]
    print(f"  Firm {i}: {prod.inventory[i]:.1f} units remaining "
          f"(sold {sold_units:.1f}) = ${remaining_value:.2f} value")

print(f"\nSimulation achieved balance:")
print(f"- Some inventory remains: {remaining_inventory > 0}")
print(f"- Some budget unspent and saved: {final_savings > 0}")
print(
    f"- Market was active: {remaining_inventory < 130 and final_savings < total_wealth}")
