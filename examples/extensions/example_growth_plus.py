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
# Visualization Function
# ----------------------
#
# The visualization is encapsulated in a function so it can be called
# explicitly. This prevents plots from appearing during import.


def visualize_growth_plus_results(metrics, bounds, burn_in=500):
    """Create visualization plots for Growth+ scenario.

    Call this function explicitly after running the simulation to display plots.
    This approach prevents plots from appearing during module import.

    Parameters
    ----------
    metrics : GrowthPlusMetrics
        Computed metrics from the simulation.
    bounds : dict
        Target bounds from validation YAML.
    burn_in : int
        Number of burn-in periods (already applied to metrics).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import skew

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

    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    fig.suptitle(
        "Growth+ Model Results (Section 3.8) - Endogenous Productivity Growth",
        fontsize=16,
        y=0.995,
    )

    # Helper function for statistics annotation
    def add_stats_box(ax, data, bounds_key, is_pct=False):
        """Add statistics annotation box to axis."""
        b = bounds[bounds_key]
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
        b = bounds[bounds_key]
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

    # Panel (0,0): Log Real GDP (Growing in Growth+ scenario)
    ax = axes[0, 0]
    # Extreme bounds (shaded red zones)
    ax.axhspan(
        bounds["log_gdp"]["extreme_min"],
        bounds["log_gdp"]["normal_min"],
        alpha=0.1,
        color="red",
        label="Extreme zone",
    )
    ax.axhspan(
        bounds["log_gdp"]["normal_max"],
        bounds["log_gdp"]["extreme_max"],
        alpha=0.1,
        color="red",
    )
    # Data
    ax.plot(periods, log_gdp, linewidth=1, color="#2E86AB", label="Log GDP")
    # Normal bounds
    ax.axhline(
        bounds["log_gdp"]["normal_min"],
        color="green",
        linestyle="--",
        alpha=0.5,
        label="Normal bounds",
    )
    ax.axhline(
        bounds["log_gdp"]["normal_max"], color="green", linestyle="--", alpha=0.5
    )
    # Mean target
    ax.axhline(
        bounds["log_gdp"]["mean_target"],
        color="blue",
        linestyle="-.",
        alpha=0.5,
        label="Mean target",
    )
    ax.set_title("Real GDP (Growing)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Log output")
    ax.set_xlabel("t")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=7)
    # Stats box at lower right to avoid legend overlap
    b = bounds["log_gdp"]
    actual_mean = np.mean(log_gdp)
    actual_std = np.std(log_gdp)
    in_bounds = np.sum(
        (log_gdp >= b["normal_min"]) & (log_gdp <= b["normal_max"])
    ) / len(log_gdp)
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
        bounds["unemployment"]["extreme_min"] * 100,
        bounds["unemployment"]["normal_min"] * 100,
        alpha=0.1,
        color="red",
    )
    ax.axhspan(
        bounds["unemployment"]["normal_max"] * 100,
        bounds["unemployment"]["extreme_max"] * 100,
        alpha=0.1,
        color="red",
    )
    # Data
    ax.plot(periods, unemployment_pct, linewidth=1, color="#A23B72")
    # Normal bounds
    ax.axhline(
        bounds["unemployment"]["normal_min"] * 100,
        color="green",
        linestyle="--",
        alpha=0.5,
    )
    ax.axhline(
        bounds["unemployment"]["normal_max"] * 100,
        color="green",
        linestyle="--",
        alpha=0.5,
    )
    # Mean target
    ax.axhline(
        bounds["unemployment"]["mean_target"] * 100,
        color="blue",
        linestyle="-.",
        alpha=0.5,
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
        bounds["inflation"]["extreme_min"] * 100,
        bounds["inflation"]["normal_min"] * 100,
        alpha=0.1,
        color="red",
    )
    ax.axhspan(
        bounds["inflation"]["normal_max"] * 100,
        bounds["inflation"]["extreme_max"] * 100,
        alpha=0.1,
        color="red",
    )
    # Data
    ax.plot(periods, inflation_pct, linewidth=1, color="#F18F01")
    ax.axhline(0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
    # Normal bounds
    ax.axhline(
        bounds["inflation"]["normal_min"] * 100,
        color="green",
        linestyle="--",
        alpha=0.5,
    )
    ax.axhline(
        bounds["inflation"]["normal_max"] * 100,
        color="green",
        linestyle="--",
        alpha=0.5,
    )
    # Mean target
    ax.axhline(
        bounds["inflation"]["mean_target"] * 100,
        color="blue",
        linestyle="-.",
        alpha=0.5,
    )
    ax.set_title("Annualized Rate of Inflation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Yearly inflation rate (%)")
    ax.set_xlabel("t")
    ax.grid(True, linestyle="--", alpha=0.3)
    add_stats_box(ax, inflation_pct, "inflation", is_pct=True)

    # Panel (1,1): Productivity and Real Wage Co-movement (two-line plot)
    # Both grow over time in Growth+ scenario - this is figure (d) in Section 3.8
    ax = axes[1, 1]
    # Data: Two separate growing lines
    ax.plot(
        periods,
        avg_productivity_trimmed,
        linewidth=1,
        color="#E74C3C",
        label="Productivity",
    )
    ax.plot(periods, real_wage_trimmed, linewidth=1, color="#6A994E", label="Real Wage")
    ax.set_title("Productivity & Real Wage Co-movement", fontsize=12, fontweight="bold")
    ax.set_ylabel("Value")
    ax.set_xlabel("t")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.3)
    # Add growth stats box at upper left
    growth_text = (
        f"Productivity: {metrics.initial_productivity:.2f} -> {metrics.final_productivity:.2f}\n"
        f"Growth: {metrics.total_productivity_growth * 100:.0f}%\n"
        f"Real Wage: {metrics.real_wage_initial:.2f} -> {metrics.real_wage_final:.2f}\n"
        f"Growth: {metrics.total_real_wage_growth * 100:.0f}%"
    )
    ax.text(
        0.02,
        0.97,
        growth_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Bottom 2x2: Macroeconomic curves
    # --------------------------------

    # Panel (2,0): Phillips Curve (stronger in Growth+: -0.19)
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
        # Target line (Phillips target is -0.19 for Growth+)
        target_corr = bounds["phillips_corr"]["target"]
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
    ax.set_title("Phillips Curve (Target: -0.19)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Unemployment Rate")
    ax.set_ylabel("Wage Inflation Rate")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    # Stats box at lower left
    b = bounds["phillips_corr"]
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
        target_corr = bounds["okun_corr"]["target"]
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
    b = bounds["okun_corr"]
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
        target_corr = bounds["beveridge_corr"]["target"]
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

    # Panel (3,1): Firm Size Distribution (larger due to productivity growth)
    ax = axes[3, 1]
    threshold = bounds["firm_size"]["threshold"]
    pct_below_target = bounds["firm_size"]["pct_below_target"]
    pct_below_actual = np.sum(final_production < threshold) / len(final_production)
    skewness_actual = skew(final_production)
    skewness_min = bounds["firm_size"]["skewness_min"]
    skewness_max = bounds["firm_size"]["skewness_max"]
    skewness_in_range = skewness_min <= skewness_actual <= skewness_max
    ax.hist(final_production, bins=15, edgecolor="black", alpha=0.7, color="#6A994E")
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
    # Stats box with skewness (upper right below legend to avoid overlap)
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


# %%
# Main Execution Guard
# --------------------
#
# The following code only runs when the script is executed directly,
# not when the RnD role is imported by other modules (e.g., validation tests).

if __name__ == "__main__":
    # %%
    # Initialize Simulation
    # ---------------------
    #
    # Create a simulation and attach the custom RnD role using ``use_role()``.
    # Population sizes match the book exactly (100 firms, 500 households, 10 banks).

    # Calibrated defaults (Combined Score = 0.7946, 100% pass rate)
    sim = bam.Simulation.init(
        n_firms=100,
        n_households=500,
        n_banks=10,
        n_periods=1000,
        seed=2,
        logging={"default_level": "ERROR"},
        new_firm_size_factor=0.5,
        new_firm_production_factor=0.5,
        new_firm_wage_factor=0.5,
        new_firm_price_markup=1.5,
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

    # %%
    # Run Growth+ Simulation
    # ----------------------
    #
    # Run the simulation for 1000 periods, using the standard collection config.

    from validation.metrics import (
        GROWTH_PLUS_COLLECT_CONFIG,
        compute_growth_plus_metrics,
        load_growth_plus_targets,
    )

    results = sim.run(collect=GROWTH_PLUS_COLLECT_CONFIG)

    print(f"\nSimulation completed: {results.metadata['n_periods']} periods")
    print(f"Runtime: {results.metadata['runtime_seconds']:.2f} seconds")

    # %%
    # Compute Metrics
    # ---------------
    #
    # Use the shared metrics computation from the validation module.
    # This ensures consistency between the example and validation tests.

    burn_in = 500
    metrics = compute_growth_plus_metrics(
        sim, results, burn_in=burn_in, firm_size_threshold=150.0
    )

    print(
        f"\nComputed metrics for {len(metrics.unemployment) - burn_in} periods (after burn-in)"
    )
    print(f"  Initial productivity: {metrics.initial_productivity:.4f}")
    print(f"  Final productivity: {metrics.final_productivity:.4f}")
    print(
        f"  Total productivity growth: {metrics.total_productivity_growth * 100:.1f}%"
    )

    # %%
    # Load Target Bounds
    # ------------------
    #
    # Load target bounds from validation YAML (single source of truth).

    BOUNDS = load_growth_plus_targets()

    # %%
    # Visualize Results
    # -----------------
    #
    # Call the visualization function to display the plots.

    visualize_growth_plus_results(metrics, BOUNDS, burn_in=burn_in)
