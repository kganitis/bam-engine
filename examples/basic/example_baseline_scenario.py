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

sim = bam.Simulation.init(
    n_firms=300,
    n_households=3000,
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
# Run the simulation and compute metrics using the validation module.

from validation.metrics import (
    BASELINE_COLLECT_CONFIG,
    compute_baseline_metrics,
    load_baseline_targets,
)

results = sim.run(collect=BASELINE_COLLECT_CONFIG)

print(f"\nSimulation completed: {results.metadata['n_periods']} periods")
print(f"Runtime: {results.metadata['runtime_seconds']:.2f} seconds")

# %%
# Compute Metrics
# ---------------
#
# Use the shared metrics computation from the validation module.
# This ensures consistency between the example and validation tests.

import numpy as np
from scipy.stats import skew

burn_in = 500
metrics = compute_baseline_metrics(
    sim, results, burn_in=burn_in, firm_size_threshold=5.0
)

print(
    f"\nComputed metrics for {len(metrics.unemployment) - burn_in} periods (after burn-in)"
)

# %%
# Metric Bounds Configuration
# ---------------------------
#
# Load target bounds from validation YAML (single source of truth).

BOUNDS = load_baseline_targets()

# %%
# Prepare Data for Visualization
# ------------------------------
#
# Apply a burn-in period to focus on steady-state dynamics
# (excluding initial transients).

# Time axis for plots
periods = ops.arange(burn_in, len(metrics.unemployment))

# Extract time series after burn-in
log_gdp = metrics.log_gdp[burn_in:]
inflation_pct = metrics.inflation[burn_in:] * 100
unemployment_pct = metrics.unemployment[burn_in:] * 100
real_wage_trimmed = metrics.real_wage[burn_in:]
avg_productivity_trimmed = metrics.avg_productivity[burn_in:]

# Curve data (already aligned in metrics computation)
wage_inflation_trimmed = metrics.wage_inflation[burn_in - 1 :]
unemployment_phillips = metrics.unemployment[burn_in:]
gdp_growth_trimmed = metrics.gdp_growth[burn_in - 1 :]
unemployment_growth_trimmed = metrics.unemployment_growth[burn_in - 1 :]
vacancy_rate_trimmed = metrics.vacancy_rate[burn_in:]
unemployment_beveridge = metrics.unemployment[burn_in:]

# Final period firm production for distribution
final_production = metrics.final_production

# Use pre-computed correlations from metrics
phillips_corr = metrics.phillips_corr
okun_corr = metrics.okun_corr
beveridge_corr = metrics.beveridge_corr

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


# Helper function for statistics annotation
def add_stats_box(ax, data, bounds_key, is_pct=False):
    """Add statistics annotation box to axis."""
    b = BOUNDS[bounds_key]
    scale = 100 if is_pct else 1
    actual_mean = np.mean(data)
    actual_std = np.std(data)
    target_mean = b["mean_target"] * scale
    normal_min = b["normal_min"] * scale
    normal_max = b["normal_max"] * scale
    in_bounds = np.sum((data >= normal_min) & (data <= normal_max)) / len(data)

    if is_pct:
        stats_text = f"μ = {actual_mean:.1f}% (target: {target_mean:.1f}%)\nσ = {actual_std:.1f}%\n{in_bounds * 100:.0f}% in bounds"
    else:
        stats_text = f"μ = {actual_mean:.2f} (target: {target_mean:.2f})\nσ = {actual_std:.3f}\n{in_bounds * 100:.0f}% in bounds"

    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def add_corr_stats_box(ax, actual_corr, bounds_key):
    """Add correlation statistics box to curve axis."""
    b = BOUNDS[bounds_key]
    corr_min, corr_max = b["min"], b["max"]
    in_range = corr_min <= actual_corr <= corr_max
    status = "PASS" if in_range else "WARN"
    color = "lightgreen" if in_range else "lightyellow"

    stats_text = f"r = {actual_corr:.2f}\nRange: [{corr_min:.2f}, {corr_max:.2f}]\nStatus: {status}"
    ax.text(
        0.02,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor=color, alpha=0.7),
    )


# Top 2x2: Time series panels
# ---------------------------

# Panel (0,0): Log Real GDP
ax = axes[0, 0]
# Extreme bounds (shaded red zones)
ax.axhspan(
    BOUNDS["log_gdp"]["extreme_min"],
    BOUNDS["log_gdp"]["normal_min"],
    alpha=0.1,
    color="red",
    label="Extreme zone",
)
ax.axhspan(
    BOUNDS["log_gdp"]["normal_max"],
    BOUNDS["log_gdp"]["extreme_max"],
    alpha=0.1,
    color="red",
)
# Data
ax.plot(periods, log_gdp, linewidth=1, color="#2E86AB", label="Log GDP")
# Normal bounds
ax.axhline(
    BOUNDS["log_gdp"]["normal_min"],
    color="green",
    linestyle="--",
    alpha=0.5,
    label="Normal bounds",
)
ax.axhline(BOUNDS["log_gdp"]["normal_max"], color="green", linestyle="--", alpha=0.5)
# Mean target
ax.axhline(
    BOUNDS["log_gdp"]["mean_target"],
    color="blue",
    linestyle="-.",
    alpha=0.5,
    label="Mean target",
)
ax.set_title("Real GDP", fontsize=12, fontweight="bold")
ax.set_ylabel("Log output")
ax.set_xlabel("t")
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="upper left", fontsize=7)
# Stats box at lower right to avoid legend overlap
b = BOUNDS["log_gdp"]
actual_mean = np.mean(log_gdp)
actual_std = np.std(log_gdp)
in_bounds = np.sum((log_gdp >= b["normal_min"]) & (log_gdp <= b["normal_max"])) / len(
    log_gdp
)
ax.text(
    0.98,
    0.03,
    f"μ = {actual_mean:.2f} (target: {b['mean_target']:.2f})\nσ = {actual_std:.3f}\n{in_bounds * 100:.0f}% in bounds",
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

# Panel (0,1): Unemployment Rate
ax = axes[0, 1]
# Extreme bounds (shaded red zones)
ax.axhspan(
    BOUNDS["unemployment"]["extreme_min"] * 100,
    BOUNDS["unemployment"]["normal_min"] * 100,
    alpha=0.1,
    color="red",
)
ax.axhspan(
    BOUNDS["unemployment"]["normal_max"] * 100,
    BOUNDS["unemployment"]["extreme_max"] * 100,
    alpha=0.1,
    color="red",
)
# Data
ax.plot(periods, unemployment_pct, linewidth=1, color="#A23B72")
# Normal bounds
ax.axhline(
    BOUNDS["unemployment"]["normal_min"] * 100, color="green", linestyle="--", alpha=0.5
)
ax.axhline(
    BOUNDS["unemployment"]["normal_max"] * 100, color="green", linestyle="--", alpha=0.5
)
# Mean target
ax.axhline(
    BOUNDS["unemployment"]["mean_target"] * 100, color="blue", linestyle="-.", alpha=0.5
)
ax.set_title("Unemployment Rate", fontsize=12, fontweight="bold")
ax.set_ylabel("Unemployment Rate (%)")
ax.set_xlabel("t")
ax.grid(True, linestyle="--", alpha=0.3)
ax.set_ylim(bottom=0)
add_stats_box(ax, unemployment_pct, "unemployment", is_pct=True)

# Panel (1,0): Annual Inflation Rate
ax = axes[1, 0]
# Extreme bounds (shaded red zones)
ax.axhspan(
    BOUNDS["inflation"]["extreme_min"] * 100,
    BOUNDS["inflation"]["normal_min"] * 100,
    alpha=0.1,
    color="red",
)
ax.axhspan(
    BOUNDS["inflation"]["normal_max"] * 100,
    BOUNDS["inflation"]["extreme_max"] * 100,
    alpha=0.1,
    color="red",
)
# Data
ax.plot(periods, inflation_pct, linewidth=1, color="#F18F01")
ax.axhline(0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
# Normal bounds
ax.axhline(
    BOUNDS["inflation"]["normal_min"] * 100, color="green", linestyle="--", alpha=0.5
)
ax.axhline(
    BOUNDS["inflation"]["normal_max"] * 100, color="green", linestyle="--", alpha=0.5
)
# Mean target
ax.axhline(
    BOUNDS["inflation"]["mean_target"] * 100, color="blue", linestyle="-.", alpha=0.5
)
ax.set_title("Annualized Rate of Inflation", fontsize=12, fontweight="bold")
ax.set_ylabel("Yearly inflation rate (%)")
ax.set_xlabel("t")
ax.grid(True, linestyle="--", alpha=0.3)
add_stats_box(ax, inflation_pct, "inflation", is_pct=True)

# Panel (1,1): Productivity and Real Wage Co-movement (two-line plot)
# This demonstrates that income shares stay roughly constant as both grow
ax = axes[1, 1]
# Extreme bounds (shaded red zones) - for real wage
ax.axhspan(
    BOUNDS["real_wage"]["extreme_min"],
    BOUNDS["real_wage"]["normal_min"],
    alpha=0.1,
    color="red",
)
ax.axhspan(
    BOUNDS["real_wage"]["normal_max"],
    BOUNDS["real_wage"]["extreme_max"],
    alpha=0.1,
    color="red",
)
# Data
ax.plot(
    periods,
    avg_productivity_trimmed,
    linewidth=1,
    color="#E74C3C",
    label="Productivity",
)
ax.plot(periods, real_wage_trimmed, linewidth=1, color="#6A994E", label="Real Wage")
# Normal bounds
ax.axhline(BOUNDS["real_wage"]["normal_min"], color="green", linestyle="--", alpha=0.5)
ax.axhline(BOUNDS["real_wage"]["normal_max"], color="green", linestyle="--", alpha=0.5)
# Mean target
ax.axhline(BOUNDS["real_wage"]["mean_target"], color="blue", linestyle="-.", alpha=0.5)
ax.set_title("Productivity / Real Wage Ratio", fontsize=12, fontweight="bold")
ax.set_ylabel("Productivity - Real Wage")
ax.set_xlabel("t")
ax.legend(loc="lower right", fontsize=7)
ax.grid(True, linestyle="--", alpha=0.3)
# Stats box at upper left
b = BOUNDS["real_wage"]
actual_mean = np.mean(real_wage_trimmed)
actual_std = np.std(real_wage_trimmed)
in_bounds = np.sum(
    (real_wage_trimmed >= b["normal_min"]) & (real_wage_trimmed <= b["normal_max"])
) / len(real_wage_trimmed)
ax.text(
    0.02,
    0.97,
    f"μ = {actual_mean:.2f} (target: {b['mean_target']:.2f})\nσ = {actual_std:.3f}\n{in_bounds * 100:.0f}% in bounds",
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

# Bottom 2x2: Macroeconomic curves
# --------------------------------

# Panel (2,0): Phillips Curve
ax = axes[2, 0]
ax.scatter(
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
    ax.plot(
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
    ax.plot(
        x_range,
        y_target,
        "g--",
        linewidth=2,
        alpha=0.7,
        label=f"Target (r={target_corr:.2f})",
    )
ax.set_title("Phillips Curve", fontsize=12, fontweight="bold")
ax.set_xlabel("Unemployment Rate")
ax.set_ylabel("Wage Inflation Rate")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, linestyle="--", alpha=0.3)
# Stats box at lower left
b = BOUNDS["phillips_corr"]
corr_min, corr_max = b["min"], b["max"]
in_range = corr_min <= phillips_corr <= corr_max
status = "PASS" if in_range else "WARN"
color = "lightgreen" if in_range else "lightyellow"
ax.text(
    0.02,
    0.03,
    f"r = {phillips_corr:.2f}\nRange: [{corr_min:.2f}, {corr_max:.2f}]\nStatus: {status}",
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment="bottom",
    horizontalalignment="left",
    bbox=dict(boxstyle="round", facecolor=color, alpha=0.7),
)

# Panel (2,1): Okun Curve
ax = axes[2, 1]
ax.scatter(
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
    ax.plot(
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
    ax.plot(
        x_range,
        y_target,
        "g--",
        linewidth=2,
        alpha=0.7,
        label=f"Target (r={target_corr:.2f})",
    )
ax.set_title("Okun Curve", fontsize=12, fontweight="bold")
ax.set_xlabel("Unemployment Growth Rate")
ax.set_ylabel("Output Growth Rate")
ax.legend(fontsize=8, loc="lower left")
ax.grid(True, linestyle="--", alpha=0.3)
# Stats box at upper right
b = BOUNDS["okun_corr"]
corr_min, corr_max = b["min"], b["max"]
in_range = corr_min <= okun_corr <= corr_max
status = "PASS" if in_range else "WARN"
color = "lightgreen" if in_range else "lightyellow"
ax.text(
    0.98,
    0.97,
    f"r = {okun_corr:.2f}\nRange: [{corr_min:.2f}, {corr_max:.2f}]\nStatus: {status}",
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor=color, alpha=0.7),
)

# Panel (3,0): Beveridge Curve
ax = axes[3, 0]
ax.scatter(
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
    ax.plot(
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
    ax.plot(
        x_range,
        y_target,
        "g--",
        linewidth=2,
        alpha=0.7,
        label=f"Target (r={target_corr:.2f})",
    )
ax.set_title("Beveridge Curve", fontsize=12, fontweight="bold")
ax.set_xlabel("Unemployment Rate")
ax.set_ylabel("Vacancy Rate")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.3)
add_corr_stats_box(ax, beveridge_corr, "beveridge_corr")

# Panel (3,1): Firm Size Distribution
ax = axes[3, 1]
threshold = BOUNDS["firm_size"]["threshold"]
pct_below_target = BOUNDS["firm_size"]["pct_below_target"]
pct_below_actual = np.sum(final_production < threshold) / len(final_production)
skewness_actual = skew(final_production)
skewness_min = BOUNDS["firm_size"]["skewness_min"]
skewness_max = BOUNDS["firm_size"]["skewness_max"]
skewness_in_range = skewness_min <= skewness_actual <= skewness_max
ax.hist(final_production, bins=10, edgecolor="black", alpha=0.7, color="#6A994E")
ax.axvline(
    x=threshold,
    color="#A23B72",
    linestyle="--",
    linewidth=3,
    alpha=0.7,
    label=f"Threshold ({threshold:.0f})",
)
ax.set_title("Firm Size Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Production")
ax.set_ylabel("Frequency")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.3)
# Stats box with skewness (below legend at upper right)
skew_status = "PASS" if skewness_in_range else "WARN"
skew_color = "lightgreen" if skewness_in_range else "lightyellow"
ax.text(
    0.98,
    0.70,
    f"{pct_below_actual * 100:.0f}% below threshold\n(Target: {pct_below_target * 100:.0f}%)\n"
    f"Skew: {skewness_actual:.2f} [{skewness_min:.1f}, {skewness_max:.1f}]\n"
    f"Status: {skew_status}",
    transform=ax.transAxes,
    fontsize=8,
    ha="right",
    va="top",
    bbox=dict(boxstyle="round", facecolor=skew_color, alpha=0.7),
)

plt.tight_layout()
plt.show()
