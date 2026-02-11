"""Buffer-stock consumption scenario visualization.

This module provides visualization for the buffer-stock scenario from
Section 3.9.4 of Delli Gatti et al. (2011). The key visualization is
a complementary CDF (CCDF) of household wealth on log-log axes,
replicating Figure 3.8 from the book.

This module contains only visualization functions. For running the scenario,
use the buffer_stock package:

    python -m validation.scenarios.buffer_stock

Or programmatically:
    from validation.scenarios.buffer_stock import run_scenario
    result = run_scenario(seed=42, show_plot=True)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

from bamengine import ops
from validation.scenarios.buffer_stock import BufferStockMetrics
from validation.scoring import STATUS_COLORS, check_range

_OUTPUT_DIR = Path(__file__).parent / "output"

_PANEL_NAMES = [
    "1_gdp",
    "2_unemployment",
    "3_inflation",
    "4_real_wage",
    "5_ccdf_wealth",
    "6_mpc_distribution",
    "7_phillips",
    "8_okun",
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


def visualize_buffer_stock_results(
    metrics: BufferStockMetrics,
    bounds: dict,
    burn_in: int = 500,
) -> None:
    """Create visualization plots for buffer-stock scenario.

    Produces an 8-panel figure:
    - Row 1: Log GDP, Unemployment rate
    - Row 2: Inflation rate, Real wage
    - Row 3: CCDF of wealth (Figure 3.8), MPC distribution
    - Row 4: Phillips curve, Okun's law

    Parameters
    ----------
    metrics : BufferStockMetrics
        Computed metrics from the simulation.
    bounds : dict
        Target bounds from validation YAML.
    burn_in : int
        Number of burn-in periods.
    """
    import matplotlib.pyplot as plt

    periods = ops.arange(burn_in, len(metrics.unemployment))

    log_gdp = metrics.log_gdp[burn_in:]
    unemployment_pct = metrics.unemployment[burn_in:] * 100
    inflation_pct = metrics.inflation[burn_in:] * 100
    real_wage_trimmed = metrics.real_wage[burn_in:]

    # Curve data
    wage_inflation_trimmed = metrics.wage_inflation[burn_in - 1 :]
    unemployment_phillips = metrics.unemployment[burn_in:]
    gdp_growth_trimmed = metrics.gdp_growth[burn_in - 1 :]
    unemployment_growth_trimmed = metrics.unemployment_growth[burn_in - 1 :]

    print(f"Plotting {len(periods)} periods (after {burn_in}-period burn-in)")

    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    fig.suptitle(
        "Buffer-Stock Consumption Results (Section 3.9.4)",
        fontsize=16,
        y=0.995,
    )

    # -------------------------------------------------------------------
    # Panel (0,0): Log Real GDP
    # -------------------------------------------------------------------
    ax = axes[0, 0]
    b = bounds.get("log_gdp", {})
    if "extreme_min" in b and "normal_min" in b:
        ax.axhspan(b["extreme_min"], b["normal_min"], alpha=0.1, color="red")
    if "normal_max" in b and "extreme_max" in b:
        ax.axhspan(b["normal_max"], b["extreme_max"], alpha=0.1, color="red")
    ax.plot(periods, log_gdp, linewidth=1, color="#2E86AB")
    if "normal_min" in b:
        ax.axhline(b["normal_min"], color="green", linestyle="--", alpha=0.5)
    if "normal_max" in b:
        ax.axhline(b["normal_max"], color="green", linestyle="--", alpha=0.5)
    if "mean_target" in b:
        ax.axhline(b["mean_target"], color="blue", linestyle="-.", alpha=0.5)
    ax.set_title("Real GDP", fontsize=12, fontweight="bold")
    ax.set_ylabel("Log output")
    ax.set_xlabel("t")
    ax.grid(True, linestyle="--", alpha=0.3)

    # -------------------------------------------------------------------
    # Panel (0,1): Unemployment Rate
    # -------------------------------------------------------------------
    ax = axes[0, 1]
    b = bounds.get("unemployment", {})
    if "extreme_min" in b and "normal_min" in b:
        ax.axhspan(
            b["extreme_min"] * 100, b["normal_min"] * 100, alpha=0.1, color="red"
        )
    if "normal_max" in b and "extreme_max" in b:
        ax.axhspan(
            b["normal_max"] * 100, b["extreme_max"] * 100, alpha=0.1, color="red"
        )
    ax.plot(periods, unemployment_pct, linewidth=1, color="#A23B72")
    if "normal_min" in b:
        ax.axhline(b["normal_min"] * 100, color="green", linestyle="--", alpha=0.5)
    if "normal_max" in b:
        ax.axhline(b["normal_max"] * 100, color="green", linestyle="--", alpha=0.5)
    ax.set_title("Unemployment Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("t")
    ax.grid(True, linestyle="--", alpha=0.3)

    # -------------------------------------------------------------------
    # Panel (1,0): Inflation Rate
    # -------------------------------------------------------------------
    ax = axes[1, 0]
    b = bounds.get("inflation", {})
    if "extreme_min" in b and "normal_min" in b:
        ax.axhspan(
            b["extreme_min"] * 100, b["normal_min"] * 100, alpha=0.1, color="red"
        )
    if "normal_max" in b and "extreme_max" in b:
        ax.axhspan(
            b["normal_max"] * 100, b["extreme_max"] * 100, alpha=0.1, color="red"
        )
    ax.plot(periods, inflation_pct, linewidth=1, color="#F18F01")
    if "normal_min" in b:
        ax.axhline(b["normal_min"] * 100, color="green", linestyle="--", alpha=0.5)
    if "normal_max" in b:
        ax.axhline(b["normal_max"] * 100, color="green", linestyle="--", alpha=0.5)
    ax.set_title("Inflation Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("t")
    ax.grid(True, linestyle="--", alpha=0.3)

    # -------------------------------------------------------------------
    # Panel (1,1): Real Wage
    # -------------------------------------------------------------------
    ax = axes[1, 1]
    b = bounds.get("real_wage", {})
    ax.plot(periods, real_wage_trimmed, linewidth=1, color="#2E86AB")
    if "normal_min" in b:
        ax.axhline(b["normal_min"], color="green", linestyle="--", alpha=0.5)
    if "normal_max" in b:
        ax.axhline(b["normal_max"], color="green", linestyle="--", alpha=0.5)
    ax.set_title("Real Wage", fontsize=12, fontweight="bold")
    ax.set_ylabel("Real wage (w/p)")
    ax.set_xlabel("t")
    ax.grid(True, linestyle="--", alpha=0.3)

    # -------------------------------------------------------------------
    # Panel (2,0): CCDF of Wealth (Figure 3.8 replica)
    # -------------------------------------------------------------------
    ax = axes[2, 0]
    savings = metrics.final_savings
    positive_savings = savings[savings > 0]

    if len(positive_savings) > 10:
        sorted_wealth = np.sort(positive_savings)
        n = len(sorted_wealth)
        ccdf = 1.0 - np.arange(1, n + 1) / n

        # Empirical CCDF as circles
        ax.scatter(
            sorted_wealth,
            ccdf,
            s=12,
            facecolors="none",
            edgecolors="black",
            linewidths=0.5,
            label="Simulated",
            zorder=5,
        )

        # Fitted distribution lines
        x_range = np.linspace(sorted_wealth[0], sorted_wealth[-1], 500)

        if metrics.sm_params:
            try:
                sm_ccdf = 1.0 - stats.burr12.cdf(x_range, *metrics.sm_params)
                ax.plot(
                    x_range,
                    sm_ccdf,
                    color="#0000FE",
                    linestyle="-",
                    linewidth=1.5,
                    label=f"SM (R²={metrics.sm_ccdf_r2:.3f})",
                )
            except Exception:
                pass

        if metrics.dagum_params:
            try:
                dagum_ccdf = 1.0 - stats.mielke.cdf(x_range, *metrics.dagum_params)
                ax.plot(
                    x_range,
                    dagum_ccdf,
                    color="#FE00FE",
                    linestyle="-",
                    linewidth=1.5,
                    label=f"D (R²={metrics.dagum_ccdf_r2:.3f})",
                )
            except Exception:
                pass

        if metrics.gb2_params:
            try:
                gb2_ccdf = 1.0 - stats.betaprime.cdf(x_range, *metrics.gb2_params)
                ax.plot(
                    x_range,
                    gb2_ccdf,
                    color="#00FEFE",
                    linestyle="-",
                    linewidth=1.5,
                    label=f"GB2 (R²={metrics.gb2_ccdf_r2:.3f})",
                )
            except Exception:
                pass

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="lower left")

    ax.set_title("Wealth CCDF (Figure 3.8)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Household wealth (savings)")
    ax.set_ylabel("P(X > x)")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Stats box
    ax.text(
        0.98,
        0.97,
        f"Gini = {metrics.wealth_gini:.3f}\n"
        f"Skew = {metrics.wealth_skewness:.2f}\n"
        f"Best: {metrics.best_fit} (R²={metrics.best_r2:.3f})",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # -------------------------------------------------------------------
    # Panel (2,1): MPC Distribution
    # -------------------------------------------------------------------
    ax = axes[2, 1]
    ax.hist(
        metrics.final_propensity,
        bins=50,
        density=True,
        alpha=0.7,
        color="#2E86AB",
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_title("MPC Distribution (Final Period)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Marginal Propensity to Consume")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.text(
        0.98,
        0.97,
        f"Mean MPC = {metrics.mean_mpc:.3f}\n"
        f"Std MPC = {metrics.std_mpc:.3f}\n"
        f"% dissaving = {metrics.pct_dissaving * 100:.1f}%",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # -------------------------------------------------------------------
    # Panel (3,0): Phillips Curve
    # -------------------------------------------------------------------
    ax = axes[3, 0]
    ax.scatter(
        unemployment_phillips * 100,
        wage_inflation_trimmed * 100,
        s=8,
        alpha=0.5,
        color="#A23B72",
    )
    ax.set_title("Phillips Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("Unemployment rate (%)")
    ax.set_ylabel("Wage inflation (%)")
    ax.grid(True, linestyle="--", alpha=0.3)

    b = bounds.get("phillips_corr", {})
    if b:
        corr_min = b.get("min", -1)
        corr_max = b.get("max", 0)
        status = check_range(metrics.phillips_corr, corr_min, corr_max)
        color = STATUS_COLORS[status]
        ax.text(
            0.02,
            0.97,
            f"r = {metrics.phillips_corr:.2f}\n"
            f"Range: [{corr_min:.2f}, {corr_max:.2f}]\n"
            f"Status: {status}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.7),
        )

    # -------------------------------------------------------------------
    # Panel (3,1): Okun's Law
    # -------------------------------------------------------------------
    ax = axes[3, 1]
    from validation.scenarios._utils import filter_outliers_iqr

    unemp_filt, gdp_filt = filter_outliers_iqr(
        unemployment_growth_trimmed, gdp_growth_trimmed
    )
    ax.scatter(
        unemp_filt * 100,
        gdp_filt * 100,
        s=8,
        alpha=0.5,
        color="#F18F01",
    )
    ax.set_title("Okun's Law", fontsize=12, fontweight="bold")
    ax.set_xlabel("Unemployment growth (%)")
    ax.set_ylabel("GDP growth (%)")
    ax.grid(True, linestyle="--", alpha=0.3)

    b = bounds.get("okun_corr", {})
    if b:
        corr_min = b.get("min", -1)
        corr_max = b.get("max", 0)
        status = check_range(metrics.okun_corr, corr_min, corr_max)
        color = STATUS_COLORS[status]
        ax.text(
            0.02,
            0.97,
            f"r = {metrics.okun_corr:.2f}\n"
            f"Range: [{corr_min:.2f}, {corr_max:.2f}]\n"
            f"Status: {status}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.7),
        )

    plt.tight_layout()
    _save_panels(fig, axes, _OUTPUT_DIR, _PANEL_NAMES)
    plt.show()
