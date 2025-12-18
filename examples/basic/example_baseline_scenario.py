"""
===============================
BAM Baseline Scenario
===============================

This example reproduces the baseline scenario from section 3.9.1 of the
original BAM model book (Delli Gatti et al., 2011). This scenario demonstrates
the fundamental dynamics of the model with standard parameter values.
We visualize eight panels: four time series (real GDP, unemployment rate,
inflation rate, productivity-real wage) and four macroeconomic curves
(Phillips, Okun, Beveridge, and firm size distribution).

Capture Timing Configuration
----------------------------
This example demonstrates configurable capture timing for data collection.
By default, variables are captured at the end of each period. However, for
accurate measurement of employment status and vacancies, we capture them
at specific pipeline events:

- ``EMPLOYED_CAPTURE_EVENT``: When to capture Worker.employed data.
  Set to ``"workers_update_contracts"`` (default) to capture after employment
  contracts are updated, or ``None`` for end-of-period capture.

- ``VACANCIES_CAPTURE_EVENT``: When to capture Employer.n_vacancies data.
  Set to ``"firms_fire_workers"`` (default) to capture after firing decisions,
  or ``None`` for end-of-period capture.

Worker.wage is always captured at ``"workers_receive_wage"`` to ensure
wages are recorded after they are actually paid.
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

# Capture timing configuration (see module docstring for details)
# Set to None to capture at end of period, or specify event name
# EMPLOYED_CAPTURE_EVENT: str | None = "workers_update_contracts"
# VACANCIES_CAPTURE_EVENT: str | None = "firms_fire_workers"
EMPLOYED_CAPTURE_EVENT: str | None = None
VACANCIES_CAPTURE_EVENT: str | None = None

sim = bam.Simulation.init(
    n_firms=100,
    n_households=500,
    n_banks=10,
    n_periods=1000,
    seed=4,
    logging={"default_level": "ERROR"},
)

print("Initialized baseline scenario with:")
print(f"  - {sim.n_firms} firms")
print(f"  - {sim.n_households} households")
print(f"  - {sim.n_banks} banks")

# %%
# Run the simulation and collect data using SimulationResults.
# We collect per-agent data (no aggregation) to compute metrics manually:
#
# - **Production**: Per-firm output, summed to get Real GDP
# - **Wages/Employed**: Per-worker data, filtered to employed workers only
# - **Vacancies**: Per-firm vacancies, summed for Beveridge curve
# - **Economy metrics**: Inflation, avg price (automatic)

# Build capture_timing dict based on configuration
# Worker.wage is always captured after payment; others are configurable
capture_timing: dict[str, str] = {"Worker.wage": "workers_receive_wage"}
if EMPLOYED_CAPTURE_EVENT is not None:
    capture_timing["Worker.employed"] = EMPLOYED_CAPTURE_EVENT
if VACANCIES_CAPTURE_EVENT is not None:
    capture_timing["Employer.n_vacancies"] = VACANCIES_CAPTURE_EVENT

results = sim.run(
    collect={
        "Producer": ["production", "labor_productivity"],
        "Worker": ["wage", "employed"],
        "Employer": ["n_vacancies"],
        "Economy": True,  # Capture economy metrics (inflation, avg_price)
        "aggregate": None,  # Keep full per-agent data for wages
        "capture_timing": capture_timing,
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

import numpy as np

# Economy metrics (automatically captured)
inflation = results.economy_data["inflation"]
avg_price = results.economy_data["avg_price"]

# Role data - shape is (n_periods, n_agents)
production = results.role_data["Producer"]["production"]  # (1000, 100)
labor_productivity = results.role_data["Producer"]["labor_productivity"]  # (1000, 100)
wages = results.role_data["Worker"]["wage"]  # (1000, 500)
employed = results.role_data["Worker"]["employed"]  # (1000, 500)
n_vacancies = results.role_data["Employer"]["n_vacancies"]  # (1000, 100)

# Calculate unemployment rate directly from Worker.employed data
# unemployment = 1 - (employed workers / total workers)
unemployment_raw = 1 - ops.mean(employed.astype(float), axis=1)

# Apply smoothing for time series visualization (4-quarter moving average)
window = 4
kernel = np.ones(window) / window
unemployment_sa_valid = np.convolve(unemployment_raw, kernel, mode="valid")
# Pad the beginning with raw values (not enough history for MA)
unemployment_smoothed = np.concatenate(
    [unemployment_raw[: window - 1], unemployment_sa_valid]
)

# Calculate Real GDP as total production per period
gdp = ops.sum(production, axis=1)  # Sum across all firms

# Calculate aggregate labor productivity as production-weighted mean
# Weight each firm's labor_productivity by its production share
weighted_productivity = ops.sum(ops.multiply(labor_productivity, production), axis=1)
avg_productivity = ops.divide(weighted_productivity, gdp)

# Calculate average wage for EMPLOYED workers only per period
# (unemployed workers have wage=0, which would skew the average)
employed_wages_sum = ops.sum(ops.where(employed, wages, 0.0), axis=1)
employed_count = ops.sum(employed, axis=1)
avg_employed_wage = ops.where(
    ops.greater(employed_count, 0),
    ops.divide(employed_wages_sum, employed_count),
    0.0,
)

# Calculate Productivity - Real Wage
# Real wage = nominal wage / price level
real_wage = ops.divide(avg_employed_wage, avg_price)
prod_wage = ops.where(
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

# Okun Curve: GDP growth rate and unemployment growth rate
# Use raw unemployment for growth rate calculation (scatter plots use raw data)
gdp_growth = ops.divide(gdp[1:] - gdp[:-1], gdp[:-1])
unemployment_growth = ops.divide(
    unemployment_raw[1:] - unemployment_raw[:-1],
    ops.where(ops.greater(unemployment_raw[:-1], 0), unemployment_raw[:-1], 1.0),
)

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

# Use raw log(GDP) for better comparison with reference figures
# ALTERNATIVE: Index GDP to initial period = 100, then take natural log
# gdp_indexed = ops.divide(gdp, gdp[0]) * 100
# log_gdp = ops.log(gdp_indexed[burn_in:])
log_gdp = ops.log(gdp[burn_in:])
inflation_pct = inflation[burn_in:] * 100  # Convert to percentage
prod_wage_trimmed = prod_wage[burn_in:]

# Unemployment for time series: use SMOOTHED version (4-quarter MA)
unemployment_pct = unemployment_smoothed[burn_in:] * 100  # Convert to percentage
# ALTERNATIVE: use RAW unemployment for time series
# unemployment_pct = unemployment_raw[burn_in:] * 100  # Convert to percentage

# Prepare productivity and real wage for co-movement plot
avg_productivity_trimmed = avg_productivity[burn_in:]
real_wage_trimmed = real_wage[burn_in:]

# Apply burn-in to curve data
# For scatter plots, use RAW unemployment (no smoothing) for Phillips/Okun/Beveridge
# For Phillips curve: wage_inflation has length n-1, so burn_in-1 aligns with period burn_in
wage_inflation_trimmed = wage_inflation[burn_in - 1 :]
unemployment_phillips = unemployment_raw[burn_in:]  # RAW for scatter plot

# For Okun curve: align GDP growth with unemployment growth (both use raw)
gdp_growth_trimmed = gdp_growth[burn_in - 1 :]
unemployment_growth_trimmed = unemployment_growth[burn_in - 1 :]

# For Beveridge curve (raw unemployment)
vacancy_rate_trimmed = vacancy_rate[burn_in:]
unemployment_beveridge = unemployment_raw[burn_in:]  # RAW for scatter plot

# Calculate correlations
phillips_corr = np.corrcoef(unemployment_phillips, wage_inflation_trimmed)[0, 1]
okun_corr = np.corrcoef(unemployment_growth_trimmed, gdp_growth_trimmed)[0, 1]
beveridge_corr = np.corrcoef(unemployment_beveridge, vacancy_rate_trimmed)[0, 1]

print(f"Plotting {len(periods)} periods (after {burn_in}-period burn-in)")

# %%
# Visualize Key Economic Indicators
# ---------------------------------
#
# Create a 4x2 figure: time series in the top two rows (GDP, unemployment,
# inflation, productivity-real wage) and macroeconomic curves in the bottom
# two rows (Phillips, Okun, Beveridge, firm size distribution).

import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 2, figsize=(14, 20))
fig.suptitle(
    "BAM Model Baseline Scenario Results (Section 3.9.1)", fontsize=16, y=0.995
)

# Top 2x2: Time series panels
# ---------------------------

# Panel (0,0): Log Real GDP
axes[0, 0].plot(periods, log_gdp, linewidth=1.5, color="#2E86AB")
axes[0, 0].axhline(5.40, color="green", linestyle="--", alpha=0.5, label="Target min")
axes[0, 0].axhline(5.50, color="green", linestyle="--", alpha=0.5, label="Target max")
axes[0, 0].set_title("Log Real GDP", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("Time (periods)")
axes[0, 0].set_ylabel("Log Output")
axes[0, 0].grid(True, linestyle="--", alpha=0.6)
axes[0, 0].legend(fontsize=8)

# Panel (0,1): Unemployment Rate
axes[0, 1].plot(periods, unemployment_pct, linewidth=1.5, color="#A23B72")
axes[0, 1].axhline(2, color="green", linestyle="--", alpha=0.5, label="Target min (2%)")
axes[0, 1].axhline(
    12, color="green", linestyle="--", alpha=0.5, label="Target max (12%)"
)
axes[0, 1].set_title("Unemployment Rate (%)", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("Time (periods)")
axes[0, 1].set_ylabel("Unemployment Rate (%)")
axes[0, 1].grid(True, linestyle="--", alpha=0.6)
axes[0, 0].legend(fontsize=8)

# Panel (1,0): Annual Inflation Rate
axes[1, 0].plot(periods, inflation_pct, linewidth=1.5, color="#F18F01")
axes[1, 0].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
axes[1, 0].axhline(
    -5, color="green", linestyle="--", alpha=0.5, label="Target min (-5%)"
)
axes[1, 0].axhline(
    10, color="green", linestyle="--", alpha=0.5, label="Target max (10%)"
)
axes[1, 0].set_title("Annual Inflation Rate (%)", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("Time (periods)")
axes[1, 0].set_ylabel("Inflation Rate (%)")
axes[1, 0].grid(True, linestyle="--", alpha=0.6)
axes[0, 0].legend(fontsize=8)

# Panel (1,1): Productivity and Real Wage Co-movement (two-line plot)
# This demonstrates that income shares stay roughly constant as both grow
ax_prod = axes[1, 1]
ax_prod.plot(
    periods,
    avg_productivity_trimmed,
    linewidth=1.5,
    color="#E74C3C",
    label="Productivity",
)
ax_prod.plot(
    periods, real_wage_trimmed, linewidth=1.5, color="#6A994E", label="Real Wage"
)
axes[1, 1].axhline(0.325, color="green", linestyle="--", alpha=0.5, label="Target min")
axes[1, 1].axhline(0.375, color="green", linestyle="--", alpha=0.5, label="Target max")
ax_prod.set_title(
    "Productivity & Real Wage Co-movement", fontsize=12, fontweight="bold"
)
ax_prod.set_xlabel("Time (periods)")
ax_prod.set_ylabel("Level")
ax_prod.legend(loc="upper left")
ax_prod.grid(True, linestyle="--", alpha=0.6)

# Bottom 2x2: Macroeconomic curves
# --------------------------------

# Panel (2,0): Phillips Curve
axes[2, 0].scatter(
    unemployment_phillips, wage_inflation_trimmed, s=10, alpha=0.5, color="#2E86AB"
)
# Add reference lines: our results and target
# Slope = r * (std_y / std_x)
x_mean, y_mean = np.mean(unemployment_phillips), np.mean(wage_inflation_trimmed)
x_std, y_std = np.std(unemployment_phillips), np.std(wage_inflation_trimmed)
if x_std > 0:
    x_range = np.array([unemployment_phillips.min(), unemployment_phillips.max()])
    # Our results regression line
    actual_slope = phillips_corr * (y_std / x_std)
    y_actual = y_mean + actual_slope * (x_range - x_mean)
    axes[2, 0].plot(
        x_range,
        y_actual,
        color="#2E86AB",
        linewidth=2,
        alpha=0.8,
        label=f"Actual (r={phillips_corr:.2f})",
    )
    # Target reference line (r = -0.10)
    target_slope = -0.10 * (y_std / x_std)
    y_target = y_mean + target_slope * (x_range - x_mean)
    axes[2, 0].plot(
        x_range, y_target, "g--", linewidth=2, alpha=0.7, label="Target (r=-0.10)"
    )
axes[2, 0].set_title("Phillips Curve", fontsize=12, fontweight="bold")
axes[2, 0].set_xlabel("Unemployment Rate")
axes[2, 0].set_ylabel("Wage Inflation Rate")
axes[2, 0].grid(True, linestyle="--", alpha=0.6)
axes[2, 0].legend(fontsize=8, loc="upper right")

# Panel (2,1): Okun Curve
axes[2, 1].scatter(
    unemployment_growth_trimmed, gdp_growth_trimmed, s=10, alpha=0.5, color="#A23B72"
)
# Add reference lines: our results and target
x_mean, y_mean = np.mean(unemployment_growth_trimmed), np.mean(gdp_growth_trimmed)
x_std, y_std = np.std(unemployment_growth_trimmed), np.std(gdp_growth_trimmed)
if x_std > 0:
    x_range = np.array(
        [unemployment_growth_trimmed.min(), unemployment_growth_trimmed.max()]
    )
    # Our results regression line
    actual_slope = okun_corr * (y_std / x_std)
    y_actual = y_mean + actual_slope * (x_range - x_mean)
    axes[2, 1].plot(
        x_range,
        y_actual,
        color="#A23B72",
        linewidth=2,
        alpha=0.8,
        label=f"Actual (r={okun_corr:.2f})",
    )
    # Target reference line (r = -0.70)
    target_slope = -0.70 * (y_std / x_std)
    y_target = y_mean + target_slope * (x_range - x_mean)
    axes[2, 1].plot(
        x_range, y_target, "g--", linewidth=2, alpha=0.7, label="Target (r=-0.70)"
    )
axes[2, 1].set_title("Okun Curve", fontsize=12, fontweight="bold")
axes[2, 1].set_xlabel("Unemployment Growth Rate")
axes[2, 1].set_ylabel("Output Growth Rate")
axes[2, 1].grid(True, linestyle="--", alpha=0.6)
axes[2, 1].legend(fontsize=8, loc="upper right")

# Panel (3,0): Beveridge Curve
axes[3, 0].scatter(
    unemployment_beveridge, vacancy_rate_trimmed, s=10, alpha=0.5, color="#F18F01"
)
# Add reference lines: our results and target
x_mean, y_mean = np.mean(unemployment_beveridge), np.mean(vacancy_rate_trimmed)
x_std, y_std = np.std(unemployment_beveridge), np.std(vacancy_rate_trimmed)
if x_std > 0:
    x_range = np.array([unemployment_beveridge.min(), unemployment_beveridge.max()])
    # Our results regression line
    actual_slope = beveridge_corr * (y_std / x_std)
    y_actual = y_mean + actual_slope * (x_range - x_mean)
    axes[3, 0].plot(
        x_range,
        y_actual,
        color="#F18F01",
        linewidth=2,
        alpha=0.8,
        label=f"Actual (r={beveridge_corr:.2f})",
    )
    # Target reference line (r = -0.27)
    target_slope = -0.27 * (y_std / x_std)
    y_target = y_mean + target_slope * (x_range - x_mean)
    axes[3, 0].plot(
        x_range, y_target, "g--", linewidth=2, alpha=0.7, label="Target (r=-0.27)"
    )
axes[3, 0].set_title("Beveridge Curve", fontsize=12, fontweight="bold")
axes[3, 0].set_xlabel("Unemployment Rate")
axes[3, 0].set_ylabel("Vacancy Rate")
axes[3, 0].grid(True, linestyle="--", alpha=0.6)
axes[3, 0].legend(fontsize=8, loc="upper right")

# Panel (3,1): Firm Size Distribution
pct_below_3 = np.sum(final_production < 3) / len(final_production) * 100
axes[3, 1].hist(
    final_production, bins=10, edgecolor="black", alpha=0.7, color="#6A994E"
)
# Add vertical line at production=3 (target threshold)
axes[3, 1].axvline(
    x=3,
    color="#A23B72",
    linestyle="--",
    linewidth=3,
    alpha=0.7,
    label="Target threshold (90% below)",
)
axes[3, 1].set_title("Firm Size Distribution", fontsize=12, fontweight="bold")
axes[3, 1].set_xlabel("Production")
axes[3, 1].set_ylabel("Frequency")
axes[3, 1].grid(True, linestyle="--", alpha=0.6)
axes[3, 1].legend(fontsize=8, loc="upper right")
axes[3, 1].text(
    0.98,
    0.60,
    f"{pct_below_3:.0f}% below prod=3\n(Target: 90%)",
    transform=axes[3, 1].transAxes,
    fontsize=9,
    ha="right",
    va="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

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

print("\nProductivity - Real Wage:")
print(f"  Mean:   {prod_wage_trimmed.mean():.4f}")
print(f"  Std:    {prod_wage_trimmed.std():.4f}")
print(f"  Min:    {prod_wage_trimmed.min():.4f}")
print(f"  Max:    {prod_wage_trimmed.max():.4f}")

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
