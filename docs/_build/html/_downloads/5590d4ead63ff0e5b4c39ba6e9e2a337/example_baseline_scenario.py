"""
===============================
BAM Baseline Scenario (3.9.1)
===============================

This example reproduces the baseline scenario from section 3.9.1 of the
original BAM model book (Delli Gatti et al., 2011). This scenario demonstrates
the fundamental dynamics of the model with standard parameter values.

The BAM (Bottom-Up Adaptive Macroeconomics) model simulates three types of
agents (firms, households, banks) interacting in three markets (labor, credit,
consumption goods). We visualize four key macroeconomic indicators that
characterize the baseline dynamics: real GDP, unemployment rate, annual
inflation rate, and the productivity to real wage ratio.
"""

# %%
# Initialize the Simulation
# --------------------------
#
# Create a simulation with 100 firms, 500 households, and 10 banks using
# default parameters that correspond to the baseline scenario.
# We set a random seed for reproducibility.

import numpy as np
import bamengine as bam

sim = bam.Simulation.init(
    n_firms=100,
    n_households=500,
    n_banks=10,
    n_periods=1000,
    seed=42
)

print(f"Initialized baseline scenario with:")
print(f"  - {sim.n_firms} firms")
print(f"  - {sim.n_households} households")
print(f"  - {sim.n_banks} banks")

# %%
# Run Simulation and Collect Data
# --------------------------------
#
# Run the simulation period-by-period, collecting key economic indicators
# at each step. We collect:
#
# 1. **Real GDP**: Total production across all firms
# 2. **Unemployment Rate**: Fraction of households without jobs
# 3. **Annual Inflation Rate**: Year-over-year change in average price level
# 4. **Productivity/Wage Ratio**: Labor productivity relative to real wages

# Initialize data collection lists
gdp_data = []
unemployment_data = []
inflation_data = []
productivity_wage_ratio_data = []

# Run simulation period by period
for period in range(sim.n_periods):
    # Execute one period
    sim.step()

    # Collect Real GDP (sum of production)
    real_gdp = float(sim.prod.production.sum())
    gdp_data.append(real_gdp)

    # Collect Unemployment Rate
    unemployment_rate = float(sim.ec.unemp_rate_history[-1])
    unemployment_data.append(unemployment_rate)

    # Collect Annual Inflation Rate
    annual_inflation = float(sim.ec.inflation_history[-1])
    inflation_data.append(annual_inflation)

    # Calculate Productivity to Real Wage Ratio
    avg_productivity = float(sim.prod.labor_productivity.mean())
    if sim.ec.avg_mkt_price > 0:
        employed_wages = sim.wrk.wage[sim.wrk.employed]
        avg_nominal_wage = float(employed_wages.mean()) if len(employed_wages) > 0 else 0.0
        avg_real_wage = avg_nominal_wage / sim.ec.avg_mkt_price
        prod_wage_ratio = avg_productivity / avg_real_wage if avg_real_wage > 0 else 0.0
    else:
        prod_wage_ratio = 0.0
    productivity_wage_ratio_data.append(prod_wage_ratio)

print(f"\nSimulation completed: {sim.t} periods")

# %%
# Prepare Data for Visualization
# -------------------------------
#
# Convert collected data to NumPy arrays and apply a burn-in period to
# focus on steady-state dynamics (excluding initial transients).

burn_in = 500  # Exclude first 500 periods

gdp = np.array(gdp_data)
unemployment = np.array(unemployment_data)
inflation = np.array(inflation_data)
prod_wage_ratio = np.array(productivity_wage_ratio_data)

# Apply burn-in and create time axis
periods = np.arange(burn_in, len(gdp))
log_gdp = np.log(gdp[burn_in:])
unemployment_pct = unemployment[burn_in:] * 100  # Convert to percentage
inflation_pct = inflation[burn_in:] * 100  # Convert to percentage
prod_wage_ratio_trimmed = prod_wage_ratio[burn_in:]

print(f"Plotting {len(periods)} periods (after {burn_in}-period burn-in)")

# %%
# Visualize Key Economic Indicators
# ----------------------------------
#
# Create a 4-panel figure showing the evolution of macroeconomic indicators
# over time. Each panel shows a different aspect of the economy's dynamics.

import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(12, 16))
fig.suptitle("BAM Model Baseline Scenario Results (Section 3.9.1)", fontsize=16, y=0.995)

# Panel 1: Log Real GDP
axes[0].plot(periods, log_gdp, linewidth=1.5, color='#2E86AB')
axes[0].set_title("Log Real GDP", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Time (periods)")
axes[0].set_ylabel("Log Output")
axes[0].grid(True, linestyle="--", alpha=0.6)

# Panel 2: Unemployment Rate
axes[1].plot(periods, unemployment_pct, linewidth=1.5, color='#A23B72')
axes[1].set_title("Unemployment Rate (%)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Time (periods)")
axes[1].set_ylabel("Unemployment Rate (%)")
axes[1].grid(True, linestyle="--", alpha=0.6)

# Panel 3: Annual Inflation Rate
axes[2].plot(periods, inflation_pct, linewidth=1.5, color='#F18F01')
axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
axes[2].set_title("Annual Inflation Rate (%)", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Time (periods)")
axes[2].set_ylabel("Inflation Rate (%)")
axes[2].grid(True, linestyle="--", alpha=0.6)

# Panel 4: Productivity / Real Wage Ratio
axes[3].plot(periods, prod_wage_ratio_trimmed, linewidth=1.5, color='#6A994E')
axes[3].set_title("Productivity / Real Wage Ratio", fontsize=12, fontweight='bold')
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

print("\n" + "="*60)
print("SUMMARY STATISTICS (Post Burn-In)")
print("="*60)
print(f"\nReal GDP (log scale):")
print(f"  Mean:   {log_gdp.mean():.4f}")
print(f"  Std:    {log_gdp.std():.4f}")
print(f"  Min:    {log_gdp.min():.4f}")
print(f"  Max:    {log_gdp.max():.4f}")

print(f"\nUnemployment Rate:")
print(f"  Mean:   {unemployment_pct.mean():.2f}%")
print(f"  Std:    {unemployment_pct.std():.2f}%")
print(f"  Min:    {unemployment_pct.min():.2f}%")
print(f"  Max:    {unemployment_pct.max():.2f}%")

print(f"\nAnnual Inflation Rate:")
print(f"  Mean:   {inflation_pct.mean():.2f}%")
print(f"  Std:    {inflation_pct.std():.2f}%")
print(f"  Min:    {inflation_pct.min():.2f}%")
print(f"  Max:    {inflation_pct.max():.2f}%")

print(f"\nProductivity / Real Wage Ratio:")
print(f"  Mean:   {prod_wage_ratio_trimmed.mean():.4f}")
print(f"  Std:    {prod_wage_ratio_trimmed.std():.4f}")
print(f"  Min:    {prod_wage_ratio_trimmed.min():.4f}")
print(f"  Max:    {prod_wage_ratio_trimmed.max():.4f}")
print("="*60)
