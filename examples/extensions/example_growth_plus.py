"""
=============================
Growth+ Model Extension
=============================

This example implements the Growth+ extension from chapter 3.8 of Macroeconomics from
the Bottom-up, demonstrating endogenous productivity growth based on R&D investment.

Key Equations
-------------

**Productivity Evolution (Equation 3.15):**

.. math::

    \\alpha_{t+1} = \\alpha_t + z_t

Where :math:`z_t \\sim \\text{Exponential}(\\mu)` represents the productivity
increment drawn from an exponential distribution with scale parameter :math:`\\mu`.

**R&D Intensity (expected productivity gain):**

.. math::

    \\mu = \\sigma \\cdot \\frac{\\pi}{p \\cdot Y}

Where:
- :math:`\\sigma` = R&D share of profits (varies with fragility)
- :math:`\\pi` = net profit (positive only)
- :math:`p` = firm's selling price
- :math:`Y` = production quantity

**R&D Share (parameterized):**

.. math::

    \\sigma = \\sigma_{min} + (\\sigma_{max} - \\sigma_{min}) \\cdot \\exp(k \\cdot \\text{fragility})

Where:
- :math:`\\sigma_{min}` = R&D share for poorest firms (default: 0.0)
- :math:`\\sigma_{max}` = R&D share for richest firms (default: 0.1)
- :math:`k` = decay rate (default: -1.0, negative means higher fragility → lower R&D)
- fragility = W/A (wage_bill / net_worth)

Firms with higher financial fragility invest less in R&D.

**Net Worth Evolution (Equation 3.16):**

.. math::

    A_t = A_{t-1} + (1-\\sigma)(1-\\delta)\\pi_{t-1}

Where :math:`\\delta` is the dividend payout ratio.

This example demonstrates:

- Defining custom roles with the ``@role`` decorator
- Creating custom events with the ``@event`` decorator
- Using pipeline hooks via ``@event(after=...)`` for automatic event positioning
- Attaching custom roles to simulations via ``sim.use_role()``
- Passing extension parameters to ``Simulation.init()``
- Accessing extension parameters directly as ``sim.param_name``
- Collecting custom role data in simulation results
- Using ``results.get_array()`` for easy data access
"""

# %%
# Import Dependencies
# -------------------
#
# We import BAM Engine and the decorators needed to define custom components.

import bamengine as bam
from bamengine import Float, event, ops, role

# %%
# Define Custom Role: RnD
# -----------------------
#
# The RnD role tracks R&D-related state for each firm. This extends firms
# with productivity growth capabilities.


@role
class RnD:
    """R&D state for Growth+ extension.

    Tracks R&D investment decisions and productivity increments for firms.

    Parameters
    ----------
    sigma : Float
        R&D share of profits (0.0 to 0.1). Higher values mean more
        investment in R&D. Decreases with financial fragility.
    rnd_intensity : Float
        Expected productivity gain (mu). Scale parameter for the
        exponential distribution from which actual gains are drawn.
    productivity_increment : Float
        Actual productivity increment (z) drawn each period.
        Added to labor_productivity.
    fragility : Float
        Financial fragility metric (W/A = wage_bill / net_worth).
        High fragility leads to lower R&D investment.
    """

    sigma: Float
    rnd_intensity: Float
    productivity_increment: Float
    fragility: Float


print(f"Custom {RnD.name} role defined!")

# %%
# Define Custom Events
# --------------------
#
# We define three events that implement the Growth+ mechanism:
#
# 1. ``FirmsComputeRnDIntensity``: Calculate R&D share and intensity
# 2. ``FirmsApplyProductivityGrowth``: Draw and apply productivity increments
# 3. ``FirmsDeductRnDExpenditure``: Adjust retained profits for R&D spending


@event(
    name="firms_compute_rnd_intensity",  # optional custom name
    after="firms_pay_dividends",  # hook to insert after existing event
)
class FirmsComputeRnDIntensity:
    """Compute R&D share and intensity for firms.

    Calculates:
    - fragility = wage_bill / net_worth
    - sigma = sigma_min + (sigma_max - sigma_min) * exp(sigma_decay * fragility)
    - mu = sigma * net_profit / (price * production)

    Requires extension parameters: sigma_min, sigma_max, sigma_decay
    Firms with non-positive profits have sigma = 0 (no R&D).

    Note: This event is automatically inserted after 'firms_pay_dividends'
    via the ``@event(after=...)`` hook.
    """

    def execute(self, sim: bam.Simulation) -> None:
        """Execute R&D intensity computation."""
        bor = sim.get_role("Borrower")
        prod = sim.get_role("Producer")
        emp = sim.get_role("Employer")
        rnd = sim.get_role("RnD")

        # Access extension parameters directly via sim.param_name
        sigma_min = sim.sigma_min
        sigma_max = sim.sigma_max
        sigma_decay = sim.sigma_decay

        # Calculate fragility = W/A (wage_bill / net_worth)
        # Use safe division with small epsilon to avoid division by zero
        eps = 1e-10
        safe_net_worth = ops.where(ops.greater(bor.net_worth, eps), bor.net_worth, eps)
        fragility = ops.divide(emp.wage_bill, safe_net_worth)

        # Store fragility
        ops.assign(rnd.fragility, fragility)

        # Calculate sigma = sigma_min + (sigma_max - sigma_min) * exp(sigma_decay * fragility)
        decay_factor = ops.exp(ops.multiply(sigma_decay, fragility))
        sigma_range = sigma_max - sigma_min
        sigma = ops.add(sigma_min, ops.multiply(sigma_range, decay_factor))

        # Set sigma = 0 for firms with non-positive net profit
        sigma = ops.where(ops.greater(bor.net_profit, 0.0), sigma, 0.0)
        ops.assign(rnd.sigma, sigma)

        # Calculate mu = sigma * net_profit / (price * production)
        # This is the expected productivity gain (scale parameter for exponential)
        revenue = ops.multiply(prod.price, prod.production)
        safe_revenue = ops.where(ops.greater(revenue, eps), revenue, eps)
        mu = ops.divide(ops.multiply(sigma, bor.net_profit), safe_revenue)

        # Clamp mu to reasonable range
        mu = ops.where(ops.greater(mu, 0.0), mu, 0.0)
        ops.assign(rnd.rnd_intensity, mu)


@event(after="firms_compute_rnd_intensity")
class FirmsApplyProductivityGrowth:
    """Apply productivity growth based on R&D.

    For firms with positive R&D intensity (mu > 0):
    - Draw z from Exponential(scale=mu)
    - Update: labor_productivity += z

    This implements equation 3.15 from Macroeconomics from the Bottom-up.

    Note: This event is automatically inserted after 'firms_compute_rnd_intensity'
    via the ``@event(after=...)`` hook.
    """

    def execute(self, sim: bam.Simulation) -> None:
        """Execute productivity growth."""
        prod = sim.get_role("Producer")
        rnd = sim.get_role("RnD")

        # Draw productivity increments from exponential distribution
        # z ~ Exponential(scale=mu), where E[z] = mu
        # Only for firms with mu > 0
        n_firms = sim.n_firms
        mu = rnd.rnd_intensity

        # Draw from exponential - use sim.rng for reproducibility
        # For firms with mu=0, we set z=0
        z = ops.zeros(n_firms)
        active = ops.greater(mu, 0.0)
        if ops.any(active):
            # Draw from exponential with scale=mu for active firms
            # Note: sim.rng.exponential is proper RNG usage
            z[active] = sim.rng.exponential(scale=mu[active])

        # Store the increment
        ops.assign(rnd.productivity_increment, z)

        # Apply to labor productivity: alpha_{t+1} = alpha_t + z
        new_productivity = ops.add(prod.labor_productivity, z)
        ops.assign(prod.labor_productivity, new_productivity)


@event(after="firms_apply_productivity_growth")
class FirmsDeductRnDExpenditure:
    """Adjust retained profits for R&D expenditure.

    Modifies retained profit calculation:
    - new_retained = old_retained * (1 - sigma)

    This implements the (1-sigma) factor in equation 3.16,
    ensuring retained profits account for R&D spending.

    Note: This event is automatically inserted after 'firms_apply_productivity_growth'
    via the ``@event(after=...)`` hook.
    """

    def execute(self, sim: bam.Simulation) -> None:
        """Execute R&D expenditure deduction."""
        bor = sim.get_role("Borrower")
        rnd = sim.get_role("RnD")

        # Adjust retained profit: multiply by (1 - sigma)
        # This captures the R&D expenditure before profit retention
        one_minus_sigma = ops.subtract(1.0, rnd.sigma)
        new_retained = ops.multiply(bor.retained_profit, one_minus_sigma)
        ops.assign(bor.retained_profit, new_retained)


print("Custom events defined:")
print(f"  - {FirmsComputeRnDIntensity.name}")
print(f"  - {FirmsApplyProductivityGrowth.name}")
print(f"  - {FirmsDeductRnDExpenditure.name}")

# %%
# Initialize Simulation
# ---------------------
#
# Create a baseline simulation and attach the custom RnD role using ``use_role()``.

sim = bam.Simulation.init(
    n_firms=100,
    n_households=500,
    n_banks=10,
    n_periods=1000,
    seed=1,
    logging={"default_level": "WARNING"},
    # Growth+ R&D parameters (extension parameters)
    sigma_min=0.0,  # R&D share for poorest (highest fragility) firms
    sigma_max=0.1,  # R&D share for richest (lowest fragility) firms
    sigma_decay=-1.0,  # Decay rate: negative means higher fragility → lower sigma
)

# Attach custom RnD role using use_role() - automatically initializes with zeros
rnd = sim.use_role(RnD)

print("\nGrowth+ simulation initialized:")
print(f"  - {sim.n_firms} firms")
print(f"  - {sim.n_households} households")
print(f"  - {sim.n_banks} banks")
print(f"  - Custom RnD role attached: {rnd is not None}")
print(
    f"  - Extension params: sigma_min={sim.sigma_min}, sigma_max={sim.sigma_max}, "
    f"sigma_decay={sim.sigma_decay}"
)

# Verify role is accessible via get_role() too
assert sim.get_role("RnD") is rnd

# %%
# Run Growth+ Simulation
# ----------------------
#
# Run the simulation for 1000 periods, collecting both standard and custom role data.

print("\nRunning Growth+ simulation...")

results = sim.run(
    progress=True,
    collect={
        "Producer": ["production", "labor_productivity"],
        "Worker": ["wage", "employed"],
        "Employer": ["n_vacancies", "current_labor"],
        "Economy": True,  # Capture economy metrics (inflation, avg_price)
        "aggregate": None,  # Keep full per-agent data for wages
        "capture_timing": {
            "Worker.wage": "workers_receive_wage",
            "Worker.employed": "firms_run_production",
            "Producer.production": "firms_run_production",
            "Producer.labor_productivity": "firms_apply_productivity_growth",
            "Employer.n_vacancies": "firms_decide_vacancies",
        },
    },
)

print(f"Simulation completed: {results.metadata['n_periods']} periods")
print(f"Runtime: {results.metadata['runtime_seconds']:.2f} seconds")

# %%
# Extract and Analyze Results
# ---------------------------
#
# Extract relevant data arrays and compute key metrics: GDP, unemployment,
# productivity, real wages, and productivity/wage ratio. Also prepare data for
# macroeconomic curves.

import numpy as np

# =============================================================================
# Metric Bounds Configuration
# =============================================================================
# Target bounds for validation and visualization (percentile-based)

# Growth+ scenario parameters
N_HOUSEHOLDS = 3000
LABOR_PRODUCTIVITY = 0.5  # Default from config

BOUNDS: dict[str, dict[str, float]] = {
    "real_wage": {
        "normal_min": 0.30,
        "normal_max": 0.40,
        "extreme_min": 0.25,
        "extreme_max": 0.45,
        "mean_target": 0.33,
    },
    "log_gdp": {
        "normal_min": np.log(N_HOUSEHOLDS * LABOR_PRODUCTIVITY * 0.88),
        "normal_max": np.log(N_HOUSEHOLDS * LABOR_PRODUCTIVITY * 0.98),
        "extreme_min": np.log(N_HOUSEHOLDS * LABOR_PRODUCTIVITY * 0.70),
        "extreme_max": np.log(N_HOUSEHOLDS * LABOR_PRODUCTIVITY * 0.99),
        "mean_target": np.log(N_HOUSEHOLDS * LABOR_PRODUCTIVITY * 0.95),
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

# Get role data using get_array() - cleaner than navigating nested dicts
production = results.get_array("Producer", "production")
labor_productivity = results.get_array("Producer", "labor_productivity")
wages = results.get_array("Worker", "wage")
employed = results.get_array("Worker", "employed")
n_vacancies = results.get_array("Employer", "n_vacancies")
current_labor = results.get_array("Employer", "current_labor")

# Economy data
inflation = results.get_array("Economy", "inflation")
avg_price = results.get_array("Economy", "avg_price")

# Calculate unemployment rate directly from Worker.employed data
# unemployment = 1 - (employed workers / total workers)
unemployment_raw = 1 - ops.mean(employed.astype(float), axis=1)

# Calculate aggregates (axis=1 for per-period mean/sum across agents)
gdp = ops.sum(production, axis=1)

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

# Calculate Productivity / Real Wage
# Real wage = nominal wage / price level
real_wage = ops.divide(avg_employed_wage, avg_price)
prod_wage = ops.where(
    ops.greater(real_wage, 0),
    ops.divide(avg_productivity, real_wage),
    0.0,
)

# Calculate productivity growth rate
prod_growth = ops.divide(
    avg_productivity[1:] - avg_productivity[:-1],
    ops.where(avg_productivity[:-1] > 0, avg_productivity[:-1], 1.0),
)

print("\nGrowth+ Results Summary:")
print(f"  Initial avg productivity: {avg_productivity[0]:.4f}")
print(f"  Final avg productivity: {avg_productivity[-1]:.4f}")
print(
    f"  Productivity growth: {(avg_productivity[-1] / avg_productivity[0] - 1) * 100:.1f}%"
)
print(
    f"  Mean productivity growth rate: {float(ops.mean(prod_growth)) * 100:.2f}% per period"
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

# For Okun curve: align GDP growth with unemployment growth
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
# Visualization
# -------------
#
# Create a 4x2 figure: time series in the top two rows (GDP, unemployment,
# inflation, productivity-real wage) and macroeconomic curves in the bottom
# two rows (Phillips, Okun, Beveridge, firm size distribution).

import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 2, figsize=(14, 20))
fig.suptitle(
    "Growth+ Model Results (Section 3.8) - Endogenous Productivity Growth",
    fontsize=16,
    y=0.995,
)

# Top 2x2: Time series panels
# ---------------------------

# Panel (0,0): Log Real GDP
axes[0, 0].plot(periods, log_gdp, linewidth=1.5, color="#2E86AB")
# axes[0, 0].axhline(
#     BOUNDS["log_gdp"]["normal_min"], color="green", linestyle="--", alpha=0.5
# )
# axes[0, 0].axhline(
#     BOUNDS["log_gdp"]["normal_max"], color="green", linestyle="--", alpha=0.5
# )
axes[0, 0].set_title("Real GDP", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("Log Output")
axes[0, 0].set_xlabel("t")
axes[0, 0].grid(True, linestyle="--", alpha=0.3)

# Panel (0,1): Unemployment Rate
axes[0, 1].plot(periods, unemployment_pct, linewidth=1.5, color="#A23B72")
# axes[0, 1].axhline(
#     BOUNDS["unemployment"]["normal_min"] * 100,
#     color="green",
#     linestyle="--",
#     alpha=0.5,
# )
# axes[0, 1].axhline(
#     BOUNDS["unemployment"]["normal_max"] * 100,
#     color="green",
#     linestyle="--",
#     alpha=0.5,
# )
axes[0, 1].set_title("Unemployment Rate", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("Unemployment Rate (%)")
axes[0, 1].set_xlabel("t")
axes[0, 1].grid(True, linestyle="--", alpha=0.3)
axes[0, 1].set_ylim(bottom=0)

# Panel (1,0): Annual Inflation Rate
axes[1, 0].plot(periods, inflation_pct, linewidth=1, color="#F18F01")
axes[1, 0].axhline(0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
# axes[1, 0].axhline(
#     BOUNDS["inflation"]["normal_min"] * 100,
#     color="green",
#     linestyle="--",
#     alpha=0.5,
# )
# axes[1, 0].axhline(
#     BOUNDS["inflation"]["normal_max"] * 100,
#     color="green",
#     linestyle="--",
#     alpha=0.5,
# )
# axes[1, 0].axhline(
#     BOUNDS["inflation"]["mean_target"] * 100,
#     color="blue",
#     linestyle="-.",
#     alpha=0.5,
# )
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
    color="#6A994E",
    label="Productivity",
)
axes[1, 1].plot(
    periods,
    real_wage_trimmed,
    linewidth=1,
    linestyle="--",
    color="#6A994E",
    label="Real Wage",
)
# axes[1, 1].axhline(
#     BOUNDS["real_wage"]["normal_min"], color="green", linestyle="--", alpha=0.5
# )
# axes[1, 1].axhline(
#     BOUNDS["real_wage"]["normal_max"], color="green", linestyle="--", alpha=0.5
# )
# axes[1, 1].axhline(
#     BOUNDS["real_wage"]["mean_target"],
#     color="blue",
#     linestyle="-.",
#     alpha=0.5,
# )
axes[1, 1].set_title(
    "Productivity & Real Wage Co-movement", fontsize=12, fontweight="bold"
)
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
# axes[3, 1].axvline(
#     x=threshold,
#     color="#A23B72",
#     linestyle="--",
#     linewidth=3,
#     alpha=0.7,
#     label=f"Threshold ({threshold:.0f})",
# )
axes[3, 1].set_title("Firm Size Distribution", fontsize=12, fontweight="bold")
axes[3, 1].set_xlabel("Production")
axes[3, 1].set_ylabel("Frequency")
axes[3, 1].grid(True, linestyle="--", alpha=0.3)
# axes[3, 1].text(
#     0.98,
#     0.60,
#     f"{pct_below_actual * 100:.0f}% below threshold\n(Target: {pct_below_target * 100:.0f}%)",
#     transform=axes[3, 1].transAxes,
#     fontsize=9,
#     ha="right",
#     va="top",
#     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
# )

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
print("\nProductivity (Growth+ Key Metric):")
print(f"  Start (period {burn_in}): {avg_productivity[burn_in]:.4f}")
print(f"  End (period {len(avg_productivity) - 1}): {avg_productivity[-1]:.4f}")
print(
    f"  Total growth: {(avg_productivity[-1] / avg_productivity[burn_in] - 1) * 100:.1f}%"
)
print(
    f"  Mean growth rate: {float(ops.mean(prod_growth[burn_in - 1 :])) * 100:.4f}% per period"
)

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
print("\nGrowth+ extension demonstrates endogenous productivity")
print("growth through R&D investment by financially healthy firms.")
print("=" * 60)
