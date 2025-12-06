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
    n_firms=100,
    n_households=500,
    n_banks=10,
    n_periods=1000,
    seed=0,
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
        "roles": ["Producer", "Worker", "Employer"],
        "variables": {
            "Producer": ["production", "labor_productivity"],
            "Worker": ["wage", "employed"],
            "Employer": ["n_vacancies"],
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
n_vacancies = results.role_data["Employer"]["n_vacancies"]  # (1000, 100)

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

# %%
# Calculate Metrics for Macroeconomic Curves
# ------------------------------------------
#
# Prepare data for Phillips, Okun, Beveridge curves and firm size distribution.

# Phillips Curve: Wage inflation (period-over-period)
wage_inflation = ops.divide(
    avg_employed_wage[1:] - avg_employed_wage[:-1],
    ops.where(ops.greater(avg_employed_wage[:-1], 0), avg_employed_wage[:-1], 1.0),
)

# Okun Curve: GDP growth rate and unemployment change
gdp_growth = ops.divide(gdp[1:] - gdp[:-1], gdp[:-1])
unemployment_change = unemployment[1:] - unemployment[:-1]

# Beveridge Curve: Vacancy rate
total_vacancies = ops.sum(n_vacancies, axis=1)
vacancy_rate = ops.divide(total_vacancies, sim.n_households)

# Firm size distribution: Production at final period
final_production = production[-1]

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

# Apply burn-in to curve data (note: growth rates lose 1 period due to differencing)
# For Phillips curve: wage_inflation has length n-1, so burn_in-1 aligns with period burn_in
wage_inflation_trimmed = wage_inflation[burn_in - 1 :]
unemployment_phillips = unemployment[burn_in:]

# For Okun curve: align GDP growth with unemployment change
gdp_growth_trimmed = gdp_growth[burn_in - 1 :]
unemployment_change_trimmed = unemployment_change[burn_in - 1 :]

# For Beveridge curve: use full series
vacancy_rate_trimmed = vacancy_rate[burn_in:]
unemployment_beveridge = unemployment[burn_in:]

print(f"Plotting {len(periods)} periods (after {burn_in}-period burn-in)")

# %%
# Visualize Key Economic Indicators
# ---------------------------------
#
# Create a 4x2 figure showing the evolution of macroeconomic indicators
# over time (top row) and macroeconomic curves (bottom row).

import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 2, figsize=(14, 20))
fig.suptitle(
    "BAM Model Baseline Scenario Results (Section 3.9.1)", fontsize=16, y=0.995
)

# Top 2x2: Time series panels
# ---------------------------

# Panel (0,0): Log Real GDP
axes[0, 0].plot(periods, log_gdp, linewidth=1.5, color="#2E86AB")
axes[0, 0].set_title("Log Real GDP", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("Time (periods)")
axes[0, 0].set_ylabel("Log Output")
axes[0, 0].grid(True, linestyle="--", alpha=0.6)

# Panel (0,1): Unemployment Rate
axes[0, 1].plot(periods, unemployment_pct, linewidth=1.5, color="#A23B72")
axes[0, 1].set_title("Unemployment Rate (%)", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("Time (periods)")
axes[0, 1].set_ylabel("Unemployment Rate (%)")
axes[0, 1].grid(True, linestyle="--", alpha=0.6)

# Panel (1,0): Annual Inflation Rate
axes[1, 0].plot(periods, inflation_pct, linewidth=1.5, color="#F18F01")
axes[1, 0].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
axes[1, 0].set_title("Annual Inflation Rate (%)", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("Time (periods)")
axes[1, 0].set_ylabel("Inflation Rate (%)")
axes[1, 0].grid(True, linestyle="--", alpha=0.6)

# Panel (1,1): Productivity / Real Wage Ratio
axes[1, 1].plot(periods, prod_wage_ratio_trimmed, linewidth=1.5, color="#6A994E")
axes[1, 1].set_title("Productivity / Real Wage Ratio", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("Time (periods)")
axes[1, 1].set_ylabel("Ratio")
axes[1, 1].grid(True, linestyle="--", alpha=0.6)

# Bottom 2x2: Macroeconomic curves
# --------------------------------

# Panel (2,0): Phillips Curve
axes[2, 0].scatter(
    unemployment_phillips, wage_inflation_trimmed, s=10, alpha=0.5, color="#2E86AB"
)
axes[2, 0].set_title("Phillips Curve", fontsize=12, fontweight="bold")
axes[2, 0].set_xlabel("Unemployment Rate")
axes[2, 0].set_ylabel("Wage Inflation Rate")
axes[2, 0].grid(True, linestyle="--", alpha=0.6)

# Panel (2,1): Okun Curve
axes[2, 1].scatter(
    unemployment_change_trimmed, gdp_growth_trimmed, s=10, alpha=0.5, color="#A23B72"
)
axes[2, 1].set_title("Okun Curve", fontsize=12, fontweight="bold")
axes[2, 1].set_xlabel("Unemployment Growth Rate")
axes[2, 1].set_ylabel("Output Growth Rate")
axes[2, 1].grid(True, linestyle="--", alpha=0.6)

# Panel (3,0): Beveridge Curve
axes[3, 0].scatter(
    unemployment_beveridge, vacancy_rate_trimmed, s=10, alpha=0.5, color="#F18F01"
)
axes[3, 0].set_title("Beveridge Curve", fontsize=12, fontweight="bold")
axes[3, 0].set_xlabel("Unemployment Rate")
axes[3, 0].set_ylabel("Vacancy Rate")
axes[3, 0].grid(True, linestyle="--", alpha=0.6)

# Panel (3,1): Firm Size Distribution
axes[3, 1].hist(
    final_production, bins=20, edgecolor="black", alpha=0.7, color="#6A994E"
)
axes[3, 1].set_title("Firm Size Distribution", fontsize=12, fontweight="bold")
axes[3, 1].set_xlabel("Production")
axes[3, 1].set_ylabel("Frequency")
axes[3, 1].grid(True, linestyle="--", alpha=0.6)

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

print("\nWage Inflation Rate (period-over-period):")
print(f"  Mean:   {wage_inflation_trimmed.mean():.4f}")
print(f"  Std:    {wage_inflation_trimmed.std():.4f}")
print(f"  Min:    {wage_inflation_trimmed.min():.4f}")
print(f"  Max:    {wage_inflation_trimmed.max():.4f}")

print("\nGDP Growth Rate:")
print(f"  Mean:   {gdp_growth_trimmed.mean():.4f}")
print(f"  Std:    {gdp_growth_trimmed.std():.4f}")
print(f"  Min:    {gdp_growth_trimmed.min():.4f}")
print(f"  Max:    {gdp_growth_trimmed.max():.4f}")

print("\nVacancy Rate:")
print(f"  Mean:   {vacancy_rate_trimmed.mean():.4f}")
print(f"  Std:    {vacancy_rate_trimmed.std():.4f}")
print(f"  Min:    {vacancy_rate_trimmed.min():.4f}")
print(f"  Max:    {vacancy_rate_trimmed.max():.4f}")

print("\nFirm Production (final period):")
print(f"  Mean:   {final_production.mean():.4f}")
print(f"  Std:    {final_production.std():.4f}")
print(f"  Min:    {final_production.min():.4f}")
print(f"  Max:    {final_production.max():.4f}")
print("=" * 60)
