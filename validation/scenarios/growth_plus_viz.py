"""Growth+ scenario visualization.

This module provides detailed visualization for the Growth+ scenario from
section 3.9.2 of Delli Gatti et al. (2011). The visualization includes
target bounds, statistical annotations, and validation status indicators.

This module contains only visualization functions. For running the scenario,
use the main growth_plus.py module:

    python -m validation.scenarios.growth_plus

Or programmatically:
    from validation.scenarios.growth_plus import run_scenario
    result = run_scenario(seed=42, show_plot=True)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import skew

from bamengine import ops
from validation.scenarios.growth_plus import GrowthPlusMetrics

_OUTPUT_DIR = Path(__file__).parent / "output" / "growth-plus"

_PANEL_NAMES = [
    "1_gdp",
    "2_unemployment",
    "3_inflation",
    "4_productivity_wage",
    "5_phillips",
    "6_okun",
    "7_beveridge",
    "8_firm_size",
]

_FINANCIAL_PANEL_NAMES = [
    "9_output_growth_dist",
    "10_networth_growth_dist",
    "11_real_interest",
    "12_bankruptcies",
    "13_fragility",
    "14_price_ratio",
    "15_price_dispersion",
    "16_equity_sales_dispersion",
]


def _save_panels(fig, axes, output_dir, panel_names, dpi=150):
    """Save each subplot as an individual image and a combined figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for ax, name in zip(axes.flat, panel_names, strict=True):
        extent = (
            ax.get_tightbbox(renderer)
            .transformed(fig.dpi_scale_trans.inverted())
            .padded(0.15)
        )
        fig.savefig(output_dir / f"{name}.png", bbox_inches=extent, dpi=dpi)

    fig.savefig(output_dir / "combined.png", bbox_inches="tight", dpi=dpi)
    print(f"Saved {len(panel_names)} panels + combined figure to {output_dir}/")


def _shade_beyond_extreme(ax, extreme_min, extreme_max, axis="y"):
    """Shade areas beyond extreme bounds darker than the transition zone.

    Creates a visual hierarchy:
    - Normal zone (white): within normal bounds
    - Transition zone (alpha=0.1 red): between extreme and normal bounds
    - Beyond extreme (alpha=0.25 red): outside extreme bounds — clearly dangerous

    Must be called AFTER all data is plotted so axis limits are set by the data.
    """
    alpha = 0.25
    color = "red"
    if axis == "y":
        ymin, ymax = ax.get_ylim()
        if ymin < extreme_min:
            ax.axhspan(ymin, extreme_min, alpha=alpha, color=color, zorder=0)
        if ymax > extreme_max:
            ax.axhspan(extreme_max, ymax, alpha=alpha, color=color, zorder=0)
        ax.set_ylim(ymin, ymax)
    else:
        xmin, xmax = ax.get_xlim()
        if xmin < extreme_min:
            ax.axvspan(xmin, extreme_min, alpha=alpha, color=color, zorder=0)
        if xmax > extreme_max:
            ax.axvspan(extreme_max, xmax, alpha=alpha, color=color, zorder=0)
        ax.set_xlim(xmin, xmax)


def _add_cv_cyclicality_box(ax, mean, cv, cyclicality_corr, label="pro-cyc"):
    """Add stats box showing pre-computed mean, CV, and cyclicality correlation."""
    ax.text(
        0.02,
        0.97,
        f"mu = {mean:.3f}\nCV = {cv:.3f}\n{label} r = {cyclicality_corr:.2f}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def visualize_growth_plus_results(
    metrics: GrowthPlusMetrics,
    bounds: dict,
    burn_in: int = 500,
) -> None:
    """Create visualization plots for Growth+ scenario with bounds and targets.

    Produces an 8-panel figure showing time series and macroeconomic curves
    with validation bounds, target lines, and statistical annotations.

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

    # Recession mask for shading
    recession_mask = metrics.recession_mask[burn_in:]

    print(f"Plotting {len(periods)} periods (after {burn_in}-period burn-in)")

    def add_recession_bands_local(ax, periods, mask):
        """Shade recession periods on plot."""
        if not np.any(mask):
            return
        in_recession = False
        start_idx = 0
        for i, is_rec in enumerate(mask):
            if is_rec and not in_recession:
                start_idx = i
                in_recession = True
            elif not is_rec and in_recession:
                ax.axvspan(periods[start_idx], periods[i - 1], alpha=0.2, color="gray")
                in_recession = False
        if in_recession:
            ax.axvspan(periods[start_idx], periods[-1], alpha=0.2, color="gray")

    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    fig.suptitle(
        "Growth+ Model Results (Section 3.9.2) - Endogenous Productivity Growth",
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
            stats_text = f"mu = {actual_mean:.1f}% (target: {target_mean:.1f}%)\nsigma = {actual_std:.1f}%\n{in_bounds * 100:.0f}% in bounds"
        else:
            stats_text = f"mu = {actual_mean:.2f} (target: {target_mean:.2f})\nsigma = {actual_std:.3f}\n{in_bounds * 100:.0f}% in bounds"

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

        stats_text = (
            f"r = {actual_corr:.2f}\n"
            f"Range: [{corr_min:.2f}, {corr_max:.2f}]\n"
            f"Status: {status}"
        )
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
    # Recession bands (shaded gray)
    add_recession_bands_local(ax, periods, recession_mask)
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
        f"mu = {actual_mean:.2f} (target: {b['mean_target']:.2f})\nsigma = {actual_std:.3f}\n{in_bounds * 100:.0f}% in bounds",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    _shade_beyond_extreme(ax, b["extreme_min"], b["extreme_max"])

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
    _shade_beyond_extreme(
        ax,
        bounds["unemployment"]["extreme_min"] * 100,
        bounds["unemployment"]["extreme_max"] * 100,
    )

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
    _shade_beyond_extreme(
        ax,
        bounds["inflation"]["extreme_min"] * 100,
        bounds["inflation"]["extreme_max"] * 100,
    )

    # Panel (1,1): Productivity and Real Wage Co-movement (two-line plot)
    # Both grow over time in Growth+ scenario - this is figure (d) in Section 3.9.2
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
    _save_panels(fig, axes, _OUTPUT_DIR, _PANEL_NAMES)
    plt.show()


def visualize_financial_dynamics(
    metrics: GrowthPlusMetrics,
    bounds: dict,
    burn_in: int = 500,
) -> None:
    """Create financial dynamics visualization (Figures 3.6 and 3.7).

    Produces an 8-panel figure showing growth distributions, real interest rate,
    bankruptcies, financial fragility, price ratio, and dispersion metrics
    with recession bands overlaid.

    Parameters
    ----------
    metrics : GrowthPlusMetrics
        Computed metrics from the simulation.
    bounds : dict
        Target bounds from validation YAML.
    burn_in : int
        Number of burn-in periods.
    """
    import matplotlib.pyplot as plt

    periods = np.arange(burn_in, len(metrics.unemployment))
    recession_mask = metrics.recession_mask[burn_in:]
    fin_bounds = bounds.get("financial_dynamics", {})

    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    fig.suptitle(
        "Growth+ Financial Dynamics (Figures 3.6 & 3.7)",
        fontsize=16,
        y=0.995,
    )

    def add_recession_bands(ax, periods, recession_mask):
        """Shade recession periods on plot."""
        if not np.any(recession_mask):
            return
        in_recession = False
        start_idx = 0
        for i, is_rec in enumerate(recession_mask):
            if is_rec and not in_recession:
                start_idx = i
                in_recession = True
            elif not is_rec and in_recession:
                ax.axvspan(periods[start_idx], periods[i - 1], alpha=0.2, color="gray")
                in_recession = False
        if in_recession:
            ax.axvspan(periods[start_idx], periods[-1], alpha=0.2, color="gray")

    # Figure 3.6a: Output Growth Rate Distribution (Log-Rank Plot)
    ax = axes[0, 0]
    output_growth = metrics.output_growth_rates
    output_growth_filtered = output_growth[np.isfinite(output_growth)]
    if len(output_growth_filtered) > 0:
        # Get target bounds from YAML
        ogd_bounds = fin_bounds.get("output_growth_distribution", {}).get("targets", {})

        # Add tiered range bands (before plotting data)
        tight_min = ogd_bounds.get("tight_range_min", -0.05)
        tight_max = ogd_bounds.get("tight_range_max", 0.05)
        normal_min = ogd_bounds.get("normal_range_min", -0.10)
        normal_max = ogd_bounds.get("normal_range_max", 0.10)
        extreme_min = ogd_bounds.get("extreme_range_min", -0.15)
        extreme_max = ogd_bounds.get("extreme_range_max", 0.15)

        # Extreme zones only (red shading)
        ax.axvspan(extreme_min, normal_min, alpha=0.15, color="red", zorder=0)
        ax.axvspan(normal_max, extreme_max, alpha=0.15, color="red", zorder=0)

        # Normal bounds (dashed orange lines)
        ax.axvline(normal_min, color="orange", linestyle="--", alpha=0.7, zorder=1)
        ax.axvline(normal_max, color="orange", linestyle="--", alpha=0.7, zorder=1)

        # Tight bounds (dashed green lines)
        ax.axvline(tight_min, color="green", linestyle="--", alpha=0.7, zorder=1)
        ax.axvline(tight_max, color="green", linestyle="--", alpha=0.7, zorder=1)

        # Separate negative and positive growth rates
        negative_growth = output_growth_filtered[output_growth_filtered < 0]
        positive_growth = output_growth_filtered[output_growth_filtered >= 0]

        # Sort and compute ranks for negative (ascending: most negative to zero)
        neg_sorted = np.sort(negative_growth)
        neg_ranks = np.arange(1, len(neg_sorted) + 1)

        # Sort and compute ranks for positive (descending: zero to most positive)
        pos_sorted = np.sort(positive_growth)[::-1]
        pos_ranks = np.arange(1, len(pos_sorted) + 1)

        # Plot as scatter with log-scale Y-axis
        if len(neg_sorted) > 0:
            ax.scatter(neg_sorted, neg_ranks, s=10, alpha=0.7, color="#2E86AB")
        if len(pos_sorted) > 0:
            ax.scatter(pos_sorted, pos_ranks, s=10, alpha=0.7, color="#E74C3C")
        ax.set_yscale("log")

        # Use pre-computed metrics instead of recomputing
        n_total = len(output_growth_filtered)
        pct_tight = metrics.output_growth_pct_within_tight
        pct_normal = metrics.output_growth_pct_within_normal
        pct_outliers = metrics.output_growth_pct_outliers

        target_tight = ogd_bounds.get("pct_within_tight_target", 0.95)
        target_normal = ogd_bounds.get("pct_within_normal_target", 0.99)
        max_outlier = ogd_bounds.get("max_outlier_pct", 0.02)

        status_color = (
            "lightgreen"
            if (pct_tight >= target_tight * 0.9 and pct_outliers <= max_outlier * 2)
            else "lightyellow"
        )
        ax.text(
            0.02,
            0.97,
            f"N = {n_total}\n"
            f"Tight: {pct_tight * 100:.1f}% (target: {target_tight * 100:.0f}%)\n"
            f"Normal: {pct_normal * 100:.1f}% (target: {target_normal * 100:.0f}%)\n"
            f"Outliers: {pct_outliers * 100:.1f}% (max: {max_outlier * 100:.0f}%)\n"
            f"Tent R\u00b2: {metrics.output_growth_tent_r2:.3f}\n"
            f"Pos growth: {metrics.output_growth_positive_frac * 100:.1f}%",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=status_color, alpha=0.7),
        )
    ax.set_title("Output Growth Rate Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Output growth rate")
    ax.set_ylabel("Log-rank")
    ax.grid(True, linestyle="--", alpha=0.3)
    _shade_beyond_extreme(ax, extreme_min, extreme_max, axis="x")

    # Figure 3.6b: Net Worth Growth Rate Distribution (Log-Rank Plot)
    ax = axes[0, 1]
    networth_growth = metrics.networth_growth_rates
    networth_growth_filtered = networth_growth[np.isfinite(networth_growth)]
    if len(networth_growth_filtered) > 0:
        # Get target bounds from YAML
        nwd_bounds = fin_bounds.get("networth_growth_distribution", {}).get(
            "targets", {}
        )

        # Add tiered range bands (before plotting data)
        tight_min = nwd_bounds.get("tight_range_min", -0.05)
        tight_max = nwd_bounds.get("tight_range_max", 0.05)
        normal_min = nwd_bounds.get("normal_range_min", -0.10)
        normal_max = nwd_bounds.get("normal_range_max", 0.10)
        extreme_min = nwd_bounds.get("extreme_range_min", -0.50)
        extreme_max = nwd_bounds.get("extreme_range_max", 0.20)

        # Extreme zones only (red shading)
        ax.axvspan(extreme_min, normal_min, alpha=0.15, color="red", zorder=0)
        ax.axvspan(normal_max, extreme_max, alpha=0.15, color="red", zorder=0)

        # Normal bounds (dashed orange lines)
        ax.axvline(normal_min, color="orange", linestyle="--", alpha=0.7, zorder=1)
        ax.axvline(normal_max, color="orange", linestyle="--", alpha=0.7, zorder=1)

        # Tight bounds (dashed green lines)
        ax.axvline(tight_min, color="green", linestyle="--", alpha=0.7, zorder=1)
        ax.axvline(tight_max, color="green", linestyle="--", alpha=0.7, zorder=1)

        # Separate negative and positive growth rates
        neg_nw = networth_growth_filtered[networth_growth_filtered < 0]
        pos_nw = networth_growth_filtered[networth_growth_filtered >= 0]

        # Sort and compute ranks
        neg_sorted = np.sort(neg_nw)
        neg_ranks = np.arange(1, len(neg_sorted) + 1)
        pos_sorted = np.sort(pos_nw)[::-1]
        pos_ranks = np.arange(1, len(pos_sorted) + 1)

        # Plot as scatter with log-scale Y-axis
        if len(neg_sorted) > 0:
            ax.scatter(neg_sorted, neg_ranks, s=10, alpha=0.7, color="#2E86AB")
        if len(pos_sorted) > 0:
            ax.scatter(pos_sorted, pos_ranks, s=10, alpha=0.7, color="#E74C3C")
        ax.set_yscale("log")

        # Use pre-computed metrics instead of recomputing
        n_total = len(networth_growth_filtered)
        pct_tight = metrics.networth_growth_pct_within_tight
        pct_normal = metrics.networth_growth_pct_within_normal
        pct_outliers = metrics.networth_growth_pct_outliers

        target_tight = nwd_bounds.get("pct_within_tight_target", 0.75)
        target_normal = nwd_bounds.get("pct_within_normal_target", 0.90)
        max_outlier = nwd_bounds.get("max_outlier_pct", 0.05)

        status_color = (
            "lightgreen"
            if (pct_tight >= target_tight * 0.8 and pct_outliers <= max_outlier * 2)
            else "lightyellow"
        )
        ax.text(
            0.02,
            0.97,
            f"N = {n_total}\n"
            f"Tight: {pct_tight * 100:.1f}% (target: {target_tight * 100:.0f}%)\n"
            f"Normal: {pct_normal * 100:.1f}% (target: {target_normal * 100:.0f}%)\n"
            f"Outliers: {pct_outliers * 100:.1f}% (max: {max_outlier * 100:.0f}%)\n"
            f"Tent R\u00b2: {metrics.networth_growth_tent_r2:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=status_color, alpha=0.7),
        )
    ax.set_title(
        "Firms' Asset Growth Rate Distribution", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Firms' asset growth rate")
    ax.set_ylabel("Log-rank")
    ax.grid(True, linestyle="--", alpha=0.3)
    _shade_beyond_extreme(ax, extreme_min, extreme_max, axis="x")

    # Figure 3.6c: Real Interest Rate
    ax = axes[1, 0]
    real_interest = metrics.real_interest_rate[burn_in:]
    rir_bounds = fin_bounds.get("real_interest_rate", {}).get("targets", {})

    # Get bounds from targets (with defaults matching book figure)
    extreme_min = rir_bounds.get("extreme_min", -0.05)
    extreme_max = rir_bounds.get("extreme_max", 0.15)
    normal_min = rir_bounds.get("normal_min", 0.00)
    normal_max = rir_bounds.get("normal_max", 0.10)

    # Extreme bounds (shaded red zones)
    ax.axhspan(
        extreme_min * 100,
        normal_min * 100,
        alpha=0.1,
        color="red",
    )
    ax.axhspan(
        normal_max * 100,
        extreme_max * 100,
        alpha=0.1,
        color="red",
    )

    # Plot data
    ax.plot(periods, real_interest * 100, linewidth=1, color="#F18F01")
    ax.axhline(0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

    # Normal bounds (green dashed lines)
    ax.axhline(
        normal_min * 100,
        color="green",
        linestyle="--",
        alpha=0.5,
    )
    ax.axhline(
        normal_max * 100,
        color="green",
        linestyle="--",
        alpha=0.5,
    )

    # Mean target line
    if "mean_target" in rir_bounds:
        ax.axhline(
            rir_bounds["mean_target"] * 100,
            color="blue",
            linestyle="-.",
            alpha=0.5,
            label=f"Target: {rir_bounds['mean_target'] * 100:.1f}%",
        )
    ax.set_title("Real Interest Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Real Interest Rate (%)")
    ax.set_xlabel("t")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Stats box with mean, std, and % in bounds
    actual_mean = np.mean(real_interest)
    actual_std = np.std(real_interest)
    in_bounds = np.sum(
        (real_interest >= normal_min) & (real_interest <= normal_max)
    ) / len(real_interest)
    ax.text(
        0.02,
        0.97,
        f"mu = {actual_mean * 100:.2f}%\nsigma = {actual_std * 100:.2f}%\n{in_bounds * 100:.0f}% in bounds",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    _shade_beyond_extreme(ax, extreme_min * 100, extreme_max * 100)

    # Figure 3.6d: Number of Bankruptcies
    ax = axes[1, 1]
    bankruptcies = metrics.n_firm_bankruptcies[burn_in:]
    ax.plot(periods, bankruptcies, linewidth=1, color="#E74C3C")
    bkr_bounds = fin_bounds.get("bankruptcies", {}).get("targets", {})
    if "mean_target" in bkr_bounds:
        ax.axhline(
            bkr_bounds["mean_target"],
            color="blue",
            linestyle="-.",
            alpha=0.5,
            label=f"Target: {bkr_bounds['mean_target']:.1f}",
        )
    ax.set_title("Firm Bankruptcies per Period", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Bankruptcies")
    ax.set_xlabel("t")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.text(
        0.02,
        0.97,
        f"mean = {np.mean(bankruptcies):.2f}\nmax = {np.max(bankruptcies)}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Figure 3.7a: Financial Fragility
    ax = axes[2, 0]
    fragility = metrics.avg_financial_fragility[burn_in:]
    frag_bounds = fin_bounds.get("financial_fragility", {}).get("targets", {})

    # Get bounds from targets (with defaults matching book figure)
    extreme_min = frag_bounds.get("extreme_min", 0.02)
    extreme_max = frag_bounds.get("extreme_max", 0.15)
    normal_min = frag_bounds.get("normal_min", 0.05)
    normal_max = frag_bounds.get("normal_max", 0.12)

    # Add recession bands (shaded gray)
    add_recession_bands(ax, periods, recession_mask)

    # Extreme bounds (shaded red zones)
    ax.axhspan(
        extreme_min,
        normal_min,
        alpha=0.1,
        color="red",
    )
    ax.axhspan(
        normal_max,
        extreme_max,
        alpha=0.1,
        color="red",
    )

    # Plot data
    ax.plot(periods, fragility, linewidth=1, color="#6A994E")

    # Normal bounds (green dashed lines)
    ax.axhline(
        normal_min,
        color="green",
        linestyle="--",
        alpha=0.5,
    )
    ax.axhline(
        normal_max,
        color="green",
        linestyle="--",
        alpha=0.5,
    )

    # Mean target line
    if "mean_target" in frag_bounds:
        ax.axhline(
            frag_bounds["mean_target"],
            color="blue",
            linestyle="-.",
            alpha=0.5,
            label=f"Target: {frag_bounds['mean_target']:.2f}",
        )
    ax.set_title("Financial Fragility", fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg(Wage Bill / Net Worth)")
    ax.set_xlabel("t")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    _add_cv_cyclicality_box(
        ax,
        metrics.avg_fragility_mean,
        metrics.financial_fragility_cv,
        metrics.fragility_gdp_correlation,
    )
    _shade_beyond_extreme(ax, extreme_min, extreme_max)

    # Figure 3.7b: Price Ratio (Market Price / Clearing Price)
    ax = axes[2, 1]
    price_ratio = metrics.price_ratio[burn_in:]
    pr_bounds = fin_bounds.get("price_ratio", {}).get("targets", {})

    # Get bounds from targets
    pr_extreme_min = pr_bounds.get("extreme_min", 1.05)
    pr_extreme_max = pr_bounds.get("extreme_max", 1.75)
    pr_normal_min = pr_bounds.get("normal_min", 1.15)
    pr_normal_max = pr_bounds.get("normal_max", 1.55)

    # Recession bands
    add_recession_bands(ax, periods, recession_mask)

    # Extreme bounds (shaded red zones)
    ax.axhspan(pr_extreme_min, pr_normal_min, alpha=0.1, color="red")
    ax.axhspan(pr_normal_max, pr_extreme_max, alpha=0.1, color="red")

    # Plot data
    ax.plot(periods, price_ratio, linewidth=1, color="#2E86AB")
    ax.axhline(1.0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

    # Normal bounds (green dashed lines)
    ax.axhline(pr_normal_min, color="green", linestyle="--", alpha=0.5)
    ax.axhline(pr_normal_max, color="green", linestyle="--", alpha=0.5)

    # Mean target line
    if "mean_target" in pr_bounds:
        ax.axhline(
            pr_bounds["mean_target"],
            color="blue",
            linestyle="-.",
            alpha=0.5,
            label=f"Target: {pr_bounds['mean_target']:.2f}",
        )
    ax.set_title("Price Ratio (P / P*)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Market Price / Clearing Price")
    ax.set_xlabel("t")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    _add_cv_cyclicality_box(
        ax,
        metrics.price_ratio_mean,
        metrics.price_ratio_cv,
        metrics.price_ratio_gdp_correlation,
        label="counter-cyc",
    )
    _shade_beyond_extreme(ax, pr_extreme_min, pr_extreme_max)

    # Figure 3.7c: Price Dispersion
    ax = axes[3, 0]
    price_disp = metrics.price_dispersion[burn_in:]
    pd_bounds = fin_bounds.get("price_dispersion", {}).get("targets", {})

    # Get bounds from targets
    pd_extreme_min = pd_bounds.get("extreme_min", 0.15)
    pd_extreme_max = pd_bounds.get("extreme_max", 0.40)
    pd_normal_min = pd_bounds.get("normal_min", 0.20)
    pd_normal_max = pd_bounds.get("normal_max", 0.35)

    # Recession bands
    add_recession_bands(ax, periods, recession_mask)

    # Extreme bounds (shaded red zones)
    ax.axhspan(pd_extreme_min, pd_normal_min, alpha=0.1, color="red")
    ax.axhspan(pd_normal_max, pd_extreme_max, alpha=0.1, color="red")

    # Plot data
    ax.plot(periods, price_disp, linewidth=1, color="#A23B72")

    # Normal bounds (green dashed lines)
    ax.axhline(pd_normal_min, color="green", linestyle="--", alpha=0.5)
    ax.axhline(pd_normal_max, color="green", linestyle="--", alpha=0.5)

    # Mean target line
    if "mean_target" in pd_bounds:
        ax.axhline(
            pd_bounds["mean_target"],
            color="blue",
            linestyle="-.",
            alpha=0.5,
            label=f"Target: {pd_bounds['mean_target']:.2f}",
        )
    ax.set_title("Price Dispersion (CV)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_xlabel("t")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    _add_cv_cyclicality_box(
        ax,
        metrics.price_dispersion_mean,
        metrics.price_dispersion_cv,
        metrics.price_dispersion_gdp_correlation,
    )
    _shade_beyond_extreme(ax, pd_extreme_min, pd_extreme_max)

    # Figure 3.7d: Equity and Sales Dispersion
    ax = axes[3, 1]
    equity_disp = metrics.equity_dispersion[burn_in:]
    sales_disp = metrics.sales_dispersion[burn_in:]
    eq_bounds = fin_bounds.get("equity_dispersion", {}).get("targets", {})
    sl_bounds = fin_bounds.get("sales_dispersion", {}).get("targets", {})

    # Combined bounds (use wider sales range to cover both series)
    es_extreme_min = min(
        eq_bounds.get("extreme_min", 0.3), sl_bounds.get("extreme_min", 0.3)
    )
    es_extreme_max = max(
        eq_bounds.get("extreme_max", 3.0), sl_bounds.get("extreme_max", 3.5)
    )
    es_normal_min = min(
        eq_bounds.get("normal_min", 0.6), sl_bounds.get("normal_min", 0.5)
    )
    es_normal_max = max(
        eq_bounds.get("normal_max", 2.5), sl_bounds.get("normal_max", 3.0)
    )

    # Recession bands
    add_recession_bands(ax, periods, recession_mask)

    # Extreme bounds (shaded red zones)
    ax.axhspan(es_extreme_min, es_normal_min, alpha=0.1, color="red")
    ax.axhspan(es_normal_max, es_extreme_max, alpha=0.1, color="red")

    # Plot data
    ax.plot(periods, equity_disp, linewidth=1, color="#2E86AB", label="Equity")
    ax.plot(periods, sales_disp, linewidth=1, color="#E74C3C", label="Sales")

    # Normal bounds (green dashed lines)
    ax.axhline(es_normal_min, color="green", linestyle="--", alpha=0.5)
    ax.axhline(es_normal_max, color="green", linestyle="--", alpha=0.5)

    ax.set_title("Equity & Sales Dispersion", fontsize=12, fontweight="bold")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_xlabel("t")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Stats box with both series (using pre-computed metrics)
    ax.text(
        0.02,
        0.97,
        f"Equity: mu={metrics.equity_dispersion_mean:.2f} CV={metrics.equity_dispersion_cv:.3f} pro-cyc r={metrics.equity_dispersion_gdp_correlation:.2f}\n"
        f"Sales:  mu={metrics.sales_dispersion_mean:.2f} CV={metrics.sales_dispersion_cv:.3f} pro-cyc r={metrics.sales_dispersion_gdp_correlation:.2f}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    _shade_beyond_extreme(ax, es_extreme_min, es_extreme_max)

    plt.tight_layout()
    _save_panels(fig, axes, _OUTPUT_DIR, _FINANCIAL_PANEL_NAMES)
    plt.show()

    # Print Minsky classification summary
    print("\nMinsky Classification (Avg over Post-Burn-in Periods):")
    print(f"  Hedge:       {metrics.minsky_hedge_pct * 100:5.1f}% (target: ~67%)")
    print(f"  Speculative: {metrics.minsky_speculative_pct * 100:5.1f}% (target: ~23%)")
    print(f"  Ponzi:       {metrics.minsky_ponzi_pct * 100:5.1f}% (target: ~10%)")
    print("\nRecession Statistics:")
    print(f"  Number of recessions: {metrics.n_recessions}")
    print(f"  Avg recession length: {metrics.avg_recession_length:.1f} periods")
