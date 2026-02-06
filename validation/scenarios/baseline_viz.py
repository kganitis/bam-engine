"""Baseline scenario visualization.

This module provides detailed visualization for the baseline scenario from
section 3.9.1 of Delli Gatti et al. (2011). The visualization includes
target bounds, statistical annotations, and validation status indicators.

This module contains only visualization functions. For running the scenario,
use the main baseline.py module:

    python -m validation.scenarios.baseline

Or programmatically:
    from validation.scenarios.baseline import run_scenario
    result = run_scenario(seed=42, show_plot=True)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import skew

from bamengine import ops
from validation.scenarios.baseline import BaselineMetrics

_OUTPUT_DIR = Path(__file__).parent / "output" / "baseline"

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


def visualize_baseline_results(
    metrics: BaselineMetrics,
    bounds: dict,
    burn_in: int = 500,
) -> None:
    """Create visualization plots for baseline scenario with bounds and targets.

    Produces an 8-panel figure showing time series and macroeconomic curves
    with validation bounds, target lines, and statistical annotations.

    Parameters
    ----------
    metrics : BaselineMetrics
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
        "BAM Model Baseline Scenario Results (Section 3.9.1)", fontsize=16, y=0.995
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

    # Panel (0,0): Log Real GDP
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
    ax.set_title("Real GDP", fontsize=12, fontweight="bold")
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
    # NOTE: Unlike other panels, inflation shows ALL periods (no burn-in) with x-axis in years,
    # matching Figure 3.2(c) in Delli Gatti et al. (2011)
    ax = axes[1, 0]
    # Full inflation series (all periods, no burn-in)
    inflation_full_pct = metrics.inflation * 100
    years = ops.arange(0, len(metrics.inflation)) / 4  # Convert quarters to years
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
    # Data - plot all periods
    ax.plot(years, inflation_full_pct, linewidth=1, color="#F18F01")
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
    ax.set_xlabel("Years (cumulated quarters)")
    ax.grid(True, linestyle="--", alpha=0.3)
    # Stats box uses full-period data (matching the plot and validation metrics)
    add_stats_box(ax, inflation_full_pct, "inflation", is_pct=True)

    # Panel (1,1): Productivity and Real Wage Co-movement (two-line plot)
    ax = axes[1, 1]
    # Extreme bounds (shaded red zones) - for real wage
    ax.axhspan(
        bounds["real_wage"]["extreme_min"],
        bounds["real_wage"]["normal_min"],
        alpha=0.1,
        color="red",
    )
    ax.axhspan(
        bounds["real_wage"]["normal_max"],
        bounds["real_wage"]["extreme_max"],
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
    ax.axhline(
        bounds["real_wage"]["normal_min"], color="green", linestyle="--", alpha=0.5
    )
    ax.axhline(
        bounds["real_wage"]["normal_max"], color="green", linestyle="--", alpha=0.5
    )
    # Mean target
    ax.axhline(
        bounds["real_wage"]["mean_target"], color="blue", linestyle="-.", alpha=0.5
    )
    ax.set_title("Productivity / Real Wage Ratio", fontsize=12, fontweight="bold")
    ax.set_ylabel("Productivity - Real Wage")
    ax.set_xlabel("t")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.3)
    # Stats box at upper left
    b = bounds["real_wage"]
    actual_mean = np.mean(real_wage_trimmed)
    actual_std = np.std(real_wage_trimmed)
    in_bounds = np.sum(
        (real_wage_trimmed >= b["normal_min"]) & (real_wage_trimmed <= b["normal_max"])
    ) / len(real_wage_trimmed)
    ax.text(
        0.02,
        0.97,
        f"mu = {actual_mean:.2f} (target: {b['mean_target']:.2f})\nsigma = {actual_std:.3f}\n{in_bounds * 100:.0f}% in bounds",
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
    ax.set_title("Phillips Curve", fontsize=12, fontweight="bold")
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

    # Panel (3,1): Firm Size Distribution
    ax = axes[3, 1]
    threshold = bounds["firm_size"]["threshold"]
    pct_below_target = bounds["firm_size"]["pct_below_target"]
    pct_below_actual = np.sum(final_production < threshold) / len(final_production)
    skewness_actual = skew(final_production)
    skewness_min = bounds["firm_size"]["skewness_min"]
    skewness_max = bounds["firm_size"]["skewness_max"]
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
    _save_panels(fig, axes, _OUTPUT_DIR, _PANEL_NAMES)
    plt.show()
