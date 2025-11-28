"""
===============================
BAM Baseline Scenario
===============================

This example reproduces the baseline scenario from section 3.9.1 of the
original BAM model book (Delli Gatti et al., 2011). This scenario demonstrates
the fundamental dynamics of the model with standard parameter values.

The BAM (Bottom-Up Adaptive Macroeconomics) model simulates three types of
agents (firms, households, banks) interacting in three markets (labor, credit,
consumption goods). We visualize four key macroeconomic indicators that
characterize the baseline dynamics: real GDP, unemployment rate, annual
inflation rate, and the productivity to real wage ratio.

This example demonstrates using SimulationResults to collect time series data.
"""

# %%
# Initialize and Run Simulation
# -----------------------------
#
# Create a simulation with 100 firms, 500 households, and 10 banks using
# default parameters that correspond to the baseline scenario.
# We use ``run(collect=...)`` to automatically collect time series data.

import bamengine as bam
from bamengine import ops

sim = bam.Simulation.init(
    n_firms=100, n_households=500, n_banks=10, n_periods=1000, seed=42
)

print("Initialized baseline scenario with:")
print(f"  - {sim.n_firms} firms")
print(f"  - {sim.n_households} households")
print(f"  - {sim.n_banks} banks")

# %%
# Run the simulation and collect data using SimulationResults.
# We collect:
#
# 1. **Real GDP**: Total production across all firms (sum aggregation)
# 2. **Unemployment Rate**: From economy metrics (automatically captured)
# 3. **Annual Inflation Rate**: From economy metrics (automatically captured)
# 4. **Productivity**: Average labor productivity (mean aggregation)
# 5. **Wages**: Per-worker wages (no aggregation, to filter employed only)

results = sim.run(
    collect={
        "roles": ["Producer", "Worker"],
        "variables": {
            "Producer": ["production", "labor_productivity"],
            "Worker": ["wage", "employed"],
        },
        "include_economy": True,
        "aggregate": None,  # Keep full per-agent data for wages
    }
)

print(f"\nSimulation completed: {results.metadata['n_periods']} periods")
print(f"Runtime: {results.metadata['runtime_seconds']:.2f} seconds")

# %%
# Extract Data from Results
# -------------------------
#
# SimulationResults provides easy access to the collected time series data.
# Economy-wide metrics are in ``economy_data``, role data in ``role_data``.

# Economy metrics (automatically captured)
unemployment = results.economy_data["unemployment_rate"]
inflation = results.economy_data["inflation"]
avg_price = results.economy_data["avg_price"]

# Role data - shape is (n_periods, n_agents)
production = results.role_data["Producer"]["production"]  # (1000, 100)
productivity = results.role_data["Producer"]["labor_productivity"]  # (1000, 100)
wages = results.role_data["Worker"]["wage"]  # (1000, 500)
employed = results.role_data["Worker"]["employed"]  # (1000, 500)

# Calculate Real GDP as total production per period
gdp = ops.sum(production, axis=1)  # Sum across all firms

# Calculate average productivity per period
avg_productivity = ops.mean(productivity, axis=1)

# Calculate average wage for EMPLOYED workers only per period
# (unemployed workers have wage=0, which would skew the average)
employed_wages_sum = ops.sum(ops.where(employed, wages, 0.0), axis=1)
employed_count = ops.sum(employed, axis=1)
avg_employed_wage = ops.where(
    ops.greater(employed_count, 0),
    ops.divide(employed_wages_sum, employed_count),
    0.0,
)

# Calculate Productivity / Real Wage Ratio
# Real wage = nominal wage / price level
real_wage = ops.divide(avg_employed_wage, avg_price)
prod_wage_ratio = ops.where(
    ops.greater(real_wage, 0),
    ops.divide(avg_productivity, real_wage),
    0.0,
)

print(f"\nCollected {len(gdp)} periods of data")
print(f"Economy metrics: {list(results.economy_data.keys())}")
print(f"Role data captured: {list(results.role_data.keys())}")

# %%
# Prepare Data for Visualization
# ------------------------------
#
# Apply a burn-in period to focus on steady-state dynamics
# (excluding initial transients).

burn_in = 500  # Exclude first 500 periods

# Apply burn-in and create time axis
periods = ops.arange(burn_in, len(gdp))
log_gdp = ops.log(gdp[burn_in:])
unemployment_pct = unemployment[burn_in:] * 100  # Convert to percentage
inflation_pct = inflation[burn_in:] * 100  # Convert to percentage
prod_wage_ratio_trimmed = prod_wage_ratio[burn_in:]

print(f"Plotting {len(periods)} periods (after {burn_in}-period burn-in)")

# %%
# Visualize Key Economic Indicators
# ---------------------------------
#
# Create a 4-panel figure showing the evolution of macroeconomic indicators
# over time. Each panel shows a different aspect of the economy's dynamics.

import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(12, 16))
fig.suptitle(
    "BAM Model Baseline Scenario Results (Section 3.9.1)", fontsize=16, y=0.995
)

# Panel 1: Log Real GDP
axes[0].plot(periods, log_gdp, linewidth=1.5, color="#2E86AB")
axes[0].set_title("Log Real GDP", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Time (periods)")
axes[0].set_ylabel("Log Output")
axes[0].grid(True, linestyle="--", alpha=0.6)

# Panel 2: Unemployment Rate
axes[1].plot(periods, unemployment_pct, linewidth=1.5, color="#A23B72")
axes[1].set_title("Unemployment Rate (%)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Time (periods)")
axes[1].set_ylabel("Unemployment Rate (%)")
axes[1].grid(True, linestyle="--", alpha=0.6)

# Panel 3: Annual Inflation Rate
axes[2].plot(periods, inflation_pct, linewidth=1.5, color="#F18F01")
axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
axes[2].set_title("Annual Inflation Rate (%)", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Time (periods)")
axes[2].set_ylabel("Inflation Rate (%)")
axes[2].grid(True, linestyle="--", alpha=0.6)

# Panel 4: Productivity / Real Wage Ratio
axes[3].plot(periods, prod_wage_ratio_trimmed, linewidth=1.5, color="#6A994E")
axes[3].set_title("Productivity / Real Wage Ratio", fontsize=12, fontweight="bold")
axes[3].set_xlabel("Time (periods)")
axes[3].set_ylabel("Ratio")
axes[3].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# %%
# Summary Statistics
# ------------------
#
# Display summary statistics for the key indicators (post burn-in period).

print("\n" + "=" * 60)
print("SUMMARY STATISTICS (Post Burn-In)")
print("=" * 60)
print("\nReal GDP (log scale):")
print(f"  Mean:   {log_gdp.mean():.4f}")
print(f"  Std:    {log_gdp.std():.4f}")
print(f"  Min:    {log_gdp.min():.4f}")
print(f"  Max:    {log_gdp.max():.4f}")

print("\nUnemployment Rate:")
print(f"  Mean:   {unemployment_pct.mean():.2f}%")
print(f"  Std:    {unemployment_pct.std():.2f}%")
print(f"  Min:    {unemployment_pct.min():.2f}%")
print(f"  Max:    {unemployment_pct.max():.2f}%")

print("\nAnnual Inflation Rate:")
print(f"  Mean:   {inflation_pct.mean():.2f}%")
print(f"  Std:    {inflation_pct.std():.2f}%")
print(f"  Min:    {inflation_pct.min():.2f}%")
print(f"  Max:    {inflation_pct.max():.2f}%")

print("\nProductivity / Real Wage Ratio:")
print(f"  Mean:   {prod_wage_ratio_trimmed.mean():.4f}")
print(f"  Std:    {prod_wage_ratio_trimmed.std():.4f}")
print(f"  Min:    {prod_wage_ratio_trimmed.min():.4f}")
print(f"  Max:    {prod_wage_ratio_trimmed.max():.4f}")
print("=" * 60)
