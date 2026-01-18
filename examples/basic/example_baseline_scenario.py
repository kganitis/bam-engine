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
"""

# %%
# Initialize Simulation
# -----------------------------
#
# Create a simulation using default parameters that correspond to the baseline scenario.
# We use ``run(collect=...)`` to automatically collect time series data.

import bamengine as bam
from bamengine import ops

N_HOUSEHOLDS = 3000

sim = bam.Simulation.init(
    n_firms=300,
    n_households=N_HOUSEHOLDS,
    n_banks=10,
    n_periods=1000,
    seed=0,
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
# - **Economy metrics**: Inflation, avg price

results = sim.run(
    collect={
        "Producer": ["production", "labor_productivity"],
        "Worker": ["wage", "employed"],
        "Employer": ["n_vacancies"],
        "Economy": True,  # Capture economy metrics (inflation, avg_price)
        "aggregate": None,  # Keep full per-agent data for wages
        "capture_timing": {
            "Worker.wage": "workers_receive_wage",
            "Worker.employed": "firms_run_production",
            "Producer.production": "firms_run_production",
            "Employer.n_vacancies": "firms_decide_vacancies",
        },
    }
)

print(f"\nSimulation completed: {results.metadata['n_periods']} periods")
print(f"Runtime: {results.metadata['runtime_seconds']:.2f} seconds")

# %%
# Extract and Analyze Results
# -------------------------
#
# SimulationResults provides easy access to the collected time series data.
# Economy-wide metrics are in ``economy_data``, role data in ``role_data``.

import numpy as np

# =============================================================================
# Metric Bounds Configuration
# =============================================================================
# Target bounds for validation and visualization (percentile-based)

BOUNDS: dict[str, dict[str, float]] = {
    "real_wage": {
        "normal_min": 0.30,
        "normal_max": 0.40,
        "extreme_min": 0.25,
        "extreme_max": 0.45,
        "mean_target": 0.33,
    },
    "log_gdp": {
        "normal_min": np.log(N_HOUSEHOLDS * 0.5 * 0.88),
        "normal_max": np.log(N_HOUSEHOLDS * 0.5 * 0.98),
        "extreme_min": np.log(N_HOUSEHOLDS * 0.5 * 0.70),
        "extreme_max": np.log(N_HOUSEHOLDS * 0.5 * 0.99),
        "mean_target": np.log(N_HOUSEHOLDS * 0.5 * 0.95),
    },
    "inflation": {
        "normal_min": -0.05,
        "normal_max": 0.10,
        "extreme_min": -0.10,
        "extreme_max": 0.15,
        "mean_target": 0.05,
    },
    "unemployment": {
        "normal_min": 0.02,
        "normal_max": 0.12,
        "extreme_min": 0.01,
        "extreme_max": 0.30,
        "mean_target": 0.06,
    },
    # Curve correlation targets
    "phillips_corr": {
        "target": -0.10,
    },
    "okun_corr": {
        "target": -0.70,
    },
    "beveridge_corr": {
        "target": -0.27,
    },
    "firm_size": {
        "threshold": 5.0,
        "pct_below_target": 0.90,
    },
}

# Economy metrics (automatically captured)
inflation = results.economy_data["inflation"]
avg_price = results.economy_data["avg_price"]

# Role data - shape is (n_periods, n_agents)
production = results.role_data["Producer"]["production"]  # (1000, 300)
labor_productivity = results.role_data["Producer"]["labor_productivity"]  # (1000, 300)
wages = results.role_data["Worker"]["wage"]  # (1000, 3000)
employed = results.role_data["Worker"]["employed"]  # (1000, 3000)
n_vacancies = results.role_data["Employer"]["n_vacancies"]  # (1000, 300)

# Calculate unemployment rate directly from Worker.employed data
# unemployment = 1 - (employed workers / total workers)
unemployment_raw = 1 - ops.mean(employed.astype(float), axis=1)

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
log_gdp = ops.log(gdp[burn_in:])
inflation_pct = inflation[burn_in:] * 100  # Convert to percentage
prod_wage_trimmed = prod_wage[burn_in:]

# Unemployment for time series
unemployment_pct = unemployment_raw[burn_in:] * 100  # Convert to percentage

# Prepare productivity and real wage for co-movement plot
avg_productivity_trimmed = avg_productivity[burn_in:]
real_wage_trimmed = real_wage[burn_in:]

# Apply burn-in to curve data
# For Phillips curve: wage_inflation has length n-1, so burn_in-1 aligns with period burn_in
wage_inflation_trimmed = wage_inflation[burn_in - 1 :]
unemployment_phillips = unemployment_raw[burn_in:]

# For Okun curve: align GDP growth with unemployment change
gdp_growth_trimmed = gdp_growth[burn_in - 1 :]
unemployment_growth_trimmed = unemployment_growth[burn_in - 1 :]

# For Beveridge curve
vacancy_rate_trimmed = vacancy_rate[burn_in:]
unemployment_beveridge = unemployment_raw[burn_in:]

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
axes[0, 0].plot(periods, log_gdp, linewidth=1, color="#2E86AB")
axes[0, 0].axhline(
    BOUNDS["log_gdp"]["normal_min"], color="green", linestyle="--", alpha=0.5
)
axes[0, 0].axhline(
    BOUNDS["log_gdp"]["normal_max"], color="green", linestyle="--", alpha=0.5
)
axes[0, 0].set_title("Real GDP", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("Log output")
axes[0, 0].set_xlabel("t")
axes[0, 0].grid(True, linestyle="--", alpha=0.3)

# Panel (0,1): Unemployment Rate
axes[0, 1].plot(periods, unemployment_pct, linewidth=1, color="#A23B72")
axes[0, 1].axhline(
    BOUNDS["unemployment"]["normal_min"] * 100,
    color="green",
    linestyle="--",
    alpha=0.5,
)
axes[0, 1].axhline(
    BOUNDS["unemployment"]["normal_max"] * 100,
    color="green",
    linestyle="--",
    alpha=0.5,
)
axes[0, 1].set_title("Unemployment Rate", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("Unemployment Rate (%)")
axes[0, 1].set_xlabel("t")
axes[0, 1].grid(True, linestyle="--", alpha=0.3)
axes[0, 1].set_ylim(bottom=0)

# Panel (1,0): Annual Inflation Rate
axes[1, 0].plot(periods, inflation_pct, linewidth=1, color="#F18F01")
axes[1, 0].axhline(0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
axes[1, 0].axhline(
    BOUNDS["inflation"]["normal_min"] * 100,
    color="green",
    linestyle="--",
    alpha=0.5,
)
axes[1, 0].axhline(
    BOUNDS["inflation"]["normal_max"] * 100,
    color="green",
    linestyle="--",
    alpha=0.5,
)
axes[1, 0].axhline(
    BOUNDS["inflation"]["mean_target"] * 100,
    color="blue",
    linestyle="-.",
    alpha=0.5,
)
axes[1, 0].set_title("Annualized Rate of Inflation", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Yearly inflation rate (%)")
axes[1, 0].set_xlabel("t")  # TODO visualize years (cumulated quarters)
axes[1, 0].grid(True, linestyle="--", alpha=0.3)

# Panel (1,1): Productivity and Real Wage Co-movement (two-line plot)
# This demonstrates that income shares stay roughly constant as both grow
axes[1, 1].plot(
    periods,
    avg_productivity_trimmed,
    linewidth=1,
    color="#E74C3C",
    label="Productivity",
)
axes[1, 1].plot(
    periods, real_wage_trimmed, linewidth=1, color="#6A994E", label="Real Wage"
)
axes[1, 1].axhline(
    BOUNDS["real_wage"]["normal_min"], color="green", linestyle="--", alpha=0.5
)
axes[1, 1].axhline(
    BOUNDS["real_wage"]["normal_max"], color="green", linestyle="--", alpha=0.5
)
axes[1, 1].axhline(
    BOUNDS["real_wage"]["mean_target"],
    color="blue",
    linestyle="-.",
    alpha=0.5,
)
axes[1, 1].set_title("Productivity / Real Wage Ratio", fontsize=12, fontweight="bold")
axes[1, 1].set_ylabel("Productivity - Real Wage")
axes[1, 1].set_xlabel("t")
axes[1, 1].legend(loc="upper left", fontsize=8)
axes[1, 1].grid(True, linestyle="--", alpha=0.3)

# Bottom 2x2: Macroeconomic curves
# --------------------------------

# Panel (2,0): Phillips Curve
axes[2, 0].scatter(
    unemployment_phillips, wage_inflation_trimmed, s=10, alpha=0.5, color="#2E86AB"
)
# Add regression and target lines
x_mean, y_mean = np.mean(unemployment_phillips), np.mean(wage_inflation_trimmed)
x_std, y_std = np.std(unemployment_phillips), np.std(wage_inflation_trimmed)
if x_std > 0:
    x_range = np.array([unemployment_phillips.min(), unemployment_phillips.max()])
    # Actual regression line
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
    # Target line
    target_corr = BOUNDS["phillips_corr"]["target"]
    target_slope = target_corr * (y_std / x_std)
    y_target = y_mean + target_slope * (x_range - x_mean)
    axes[2, 0].plot(
        x_range,
        y_target,
        "g--",
        linewidth=2,
        alpha=0.7,
        label=f"Target (r={target_corr:.2f})",
    )
axes[2, 0].set_title("Phillips Curve", fontsize=12, fontweight="bold")
axes[2, 0].set_xlabel("Unemployment Rate")
axes[2, 0].set_ylabel("Wage Inflation Rate")
axes[2, 0].legend(fontsize=8, loc="upper right")
axes[2, 0].grid(True, linestyle="--", alpha=0.3)

# Panel (2,1): Okun Curve
axes[2, 1].scatter(
    unemployment_growth_trimmed, gdp_growth_trimmed, s=2, alpha=0.5, color="#A23B72"
)
# Add regression and target lines
x_mean, y_mean = np.mean(unemployment_growth_trimmed), np.mean(gdp_growth_trimmed)
x_std, y_std = np.std(unemployment_growth_trimmed), np.std(gdp_growth_trimmed)
if x_std > 0:
    x_range = np.array(
        [unemployment_growth_trimmed.min(), unemployment_growth_trimmed.max()]
    )
    # Actual regression line
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
    # Target line
    target_corr = BOUNDS["okun_corr"]["target"]
    target_slope = target_corr * (y_std / x_std)
    y_target = y_mean + target_slope * (x_range - x_mean)
    axes[2, 1].plot(
        x_range,
        y_target,
        "g--",
        linewidth=2,
        alpha=0.7,
        label=f"Target (r={target_corr:.2f})",
    )
axes[2, 1].set_title("Okun Curve", fontsize=12, fontweight="bold")
axes[2, 1].set_xlabel("Unemployment Growth Rate")
axes[2, 1].set_ylabel("Output Growth Rate")
axes[2, 1].legend(fontsize=8, loc="upper right")
axes[2, 1].grid(True, linestyle="--", alpha=0.3)

# Panel (3,0): Beveridge Curve
axes[3, 0].scatter(
    unemployment_beveridge, vacancy_rate_trimmed, s=10, alpha=0.5, color="#F18F01"
)
# Add regression and target lines
x_mean, y_mean = np.mean(unemployment_beveridge), np.mean(vacancy_rate_trimmed)
x_std, y_std = np.std(unemployment_beveridge), np.std(vacancy_rate_trimmed)
if x_std > 0:
    x_range = np.array([unemployment_beveridge.min(), unemployment_beveridge.max()])
    # Actual regression line
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
    # Target line
    target_corr = BOUNDS["beveridge_corr"]["target"]
    target_slope = target_corr * (y_std / x_std)
    y_target = y_mean + target_slope * (x_range - x_mean)
    axes[3, 0].plot(
        x_range,
        y_target,
        "g--",
        linewidth=2,
        alpha=0.7,
        label=f"Target (r={target_corr:.2f})",
    )
axes[3, 0].set_title("Beveridge Curve", fontsize=12, fontweight="bold")
axes[3, 0].set_xlabel("Unemployment Rate")
axes[3, 0].set_ylabel("Vacancy Rate")
axes[3, 0].legend(fontsize=8, loc="upper right")
axes[3, 0].grid(True, linestyle="--", alpha=0.3)

# Panel (3,1): Firm Size Distribution
threshold = BOUNDS["firm_size"]["threshold"]
pct_below_target = BOUNDS["firm_size"]["pct_below_target"]
pct_below_actual = np.sum(final_production < threshold) / len(final_production)
axes[3, 1].hist(
    final_production, bins=10, edgecolor="black", alpha=0.7, color="#6A994E"
)
axes[3, 1].axvline(
    x=threshold,
    color="#A23B72",
    linestyle="--",
    linewidth=3,
    alpha=0.7,
    label=f"Threshold ({threshold:.0f})",
)
axes[3, 1].set_title("Firm Size Distribution", fontsize=12, fontweight="bold")
axes[3, 1].set_xlabel("Production")
axes[3, 1].set_ylabel("Frequency")
axes[3, 1].legend(fontsize=8, loc="upper right")
axes[3, 1].grid(True, linestyle="--", alpha=0.3)
axes[3, 1].text(
    0.98,
    0.60,
    f"{pct_below_actual * 100:.0f}% below threshold\n(Target: {pct_below_target * 100:.0f}%)",
    transform=axes[3, 1].transAxes,
    fontsize=9,
    ha="right",
    va="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.show()
