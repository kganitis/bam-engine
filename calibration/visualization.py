"""
Visualization
=============

8-panel visualization matching example_baseline_scenario.py format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import bamengine as bam
from bamengine import ops

from .config import FIXED_PARAMS, apply_config_offsets
from .runner import apply_unemployment_smoothing
from .scoring import SCORE_TARGETS, compute_all_scores

# Check for matplotlib
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def visualize_configuration(
    params: dict[str, Any],
    seed: int = 0,
    n_periods: int = 1000,
    burn_in: int = 500,
    title: str = "BAM Calibration Results",
    save_path: str | Path | None = None,
) -> None:
    """
    Generate 8-panel visualization matching baseline scenario format.

    Layout (4 rows x 2 columns):
    - (0,0) Log Real GDP time series
    - (0,1) Unemployment Rate time series
    - (1,0) Inflation Rate time series
    - (1,1) Productivity & Real Wage co-movement (two lines)
    - (2,0) Phillips Curve scatter with regression
    - (2,1) Okun Curve scatter with regression
    - (3,0) Beveridge Curve scatter with regression
    - (3,1) Firm Size Distribution histogram

    Parameters
    ----------
    params : dict
        Parameter configuration to visualize.
    seed : int
        Random seed.
    n_periods : int
        Simulation periods.
    burn_in : int
        Burn-in periods to exclude.
    title : str
        Figure title.
    save_path : str or Path, optional
        Path to save figure. If None, displays interactively.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping visualization")
        return

    print(f"\nRunning visualization for configuration (seed={seed})...")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Build full config
    config = {**FIXED_PARAMS, **params}
    config["seed"] = seed
    config["n_periods"] = n_periods
    config["logging"] = {"default_level": "ERROR"}

    # Extract capture timing params before applying offsets
    employed_capture_event = config.pop(
        "employed_capture_event", "workers_update_contracts"
    )
    vacancies_capture_event = config.pop(
        "vacancies_capture_event", "firms_fire_workers"
    )

    config = apply_config_offsets(config)

    sim = bam.Simulation.init(**config)

    # Build capture_timing dict (matching baseline scenario)
    capture_timing: dict[str, str] = {"Worker.wage": "workers_receive_wage"}
    if employed_capture_event is not None:
        capture_timing["Worker.employed"] = employed_capture_event
    if vacancies_capture_event is not None:
        capture_timing["Employer.n_vacancies"] = vacancies_capture_event

    results = sim.run(
        collect={
            "Producer": ["production", "labor_productivity"],
            "Worker": ["wage", "employed"],
            "Employer": ["n_vacancies"],
            "Economy": True,
            "aggregate": None,
            "capture_timing": capture_timing,
        }
    )

    # Extract data (matching baseline scenario)
    inflation = np.array(results.economy_data["inflation"])
    avg_price = np.array(results.economy_data["avg_price"])

    production = np.array(results.role_data["Producer"]["production"])
    labor_productivity = np.array(results.role_data["Producer"]["labor_productivity"])
    wages = np.array(results.role_data["Worker"]["wage"])
    employed = np.array(results.role_data["Worker"]["employed"])
    n_vacancies = np.array(results.role_data["Employer"]["n_vacancies"])

    # Calculate unemployment (matching baseline)
    unemployment_raw = 1 - ops.mean(employed.astype(float), axis=1)
    unemployment = apply_unemployment_smoothing(unemployment_raw)

    # GDP (matching baseline)
    gdp = ops.sum(production, axis=1)

    # Production-weighted productivity (matching baseline)
    weighted_productivity = ops.sum(
        ops.multiply(labor_productivity, production), axis=1
    )
    avg_productivity = ops.divide(weighted_productivity, gdp)

    # Employed wages only (matching baseline)
    employed_wages_sum = ops.sum(ops.where(employed, wages, 0.0), axis=1)
    employed_count = ops.sum(employed, axis=1)
    avg_employed_wage = ops.where(
        ops.greater(employed_count, 0),
        ops.divide(employed_wages_sum, employed_count),
        0.0,
    )

    # Real wage (matching baseline)
    real_wage = ops.divide(avg_employed_wage, avg_price)

    # Vacancies
    total_vacancies = ops.sum(n_vacancies, axis=1)

    # Final production distribution
    final_production = production[-1]

    # Compute scores
    destroyed = sim.ec.destroyed if hasattr(sim.ec, "destroyed") else False
    scores = compute_all_scores(
        unemployment=unemployment,
        unemployment_raw=unemployment_raw,
        inflation=inflation,
        gdp=gdp,
        avg_productivity=avg_productivity,
        avg_employed_wage=avg_employed_wage,
        avg_price=avg_price,
        real_wage=real_wage,
        total_vacancies=total_vacancies,
        n_households=config["n_households"],
        final_production=final_production,
        burn_in=burn_in,
        destroyed=destroyed,
    )

    # Prepare visualization data
    # Index GDP to period 0 (matching baseline)
    gdp_indexed = ops.divide(gdp, gdp[0]) * 100
    log_gdp = ops.log(gdp_indexed)

    # Wage inflation
    wage_inflation = ops.divide(
        avg_employed_wage[1:] - avg_employed_wage[:-1],
        ops.where(ops.greater(avg_employed_wage[:-1], 0), avg_employed_wage[:-1], 1.0),
    )

    # GDP growth
    gdp_growth = ops.divide(gdp[1:] - gdp[:-1], gdp[:-1])
    unemployment_growth = ops.divide(
        unemployment_raw[1:] - unemployment_raw[:-1],
        ops.where(ops.greater(unemployment_raw[:-1], 0), unemployment_raw[:-1], 1.0),
    )

    # Vacancy rate
    vacancy_rate = ops.divide(total_vacancies, sim.n_households)

    # Apply burn-in
    periods = ops.arange(burn_in, len(gdp))
    log_gdp_trimmed = log_gdp[burn_in:]
    unemployment_pct = unemployment[burn_in:] * 100
    inflation_pct = inflation[burn_in:] * 100
    avg_productivity_trimmed = avg_productivity[burn_in:]
    real_wage_trimmed = real_wage[burn_in:]

    # For curves: use RAW unemployment
    unemployment_phillips = unemployment_raw[burn_in:]
    wage_inflation_trimmed = wage_inflation[burn_in - 1 :]
    gdp_growth_trimmed = gdp_growth[burn_in - 1 :]
    unemployment_growth_trimmed = unemployment_growth[burn_in - 1 :]
    vacancy_rate_trimmed = vacancy_rate[burn_in:]
    unemployment_beveridge = unemployment_raw[burn_in:]

    # Calculate correlations
    phillips_corr = np.corrcoef(unemployment_phillips, wage_inflation_trimmed)[0, 1]
    okun_corr = np.corrcoef(unemployment_growth_trimmed, gdp_growth_trimmed)[0, 1]
    beveridge_corr = np.corrcoef(unemployment_beveridge, vacancy_rate_trimmed)[0, 1]

    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    fig.suptitle(f"{title} (Score: {scores['total']:.2f})", fontsize=16, y=0.995)

    # Colors (matching baseline)
    colors = {
        "gdp": "#2E86AB",
        "unemployment": "#A23B72",
        "inflation": "#F18F01",
        "productivity": "#E74C3C",
        "real_wage": "#6A994E",
    }

    # Get target values
    targets = SCORE_TARGETS

    # Panel (0,0): Log Real GDP
    axes[0, 0].plot(periods, log_gdp_trimmed, linewidth=1.5, color=colors["gdp"])
    axes[0, 0].axhline(
        targets["log_gdp_min"],
        color="green",
        linestyle="--",
        alpha=0.5,
        label="Target min",
    )
    axes[0, 0].axhline(
        targets["log_gdp_max"],
        color="green",
        linestyle="--",
        alpha=0.5,
        label="Target max",
    )
    axes[0, 0].set_title("Log Real GDP", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Time (periods)")
    axes[0, 0].set_ylabel("Log Output")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].legend(fontsize=8)

    # Panel (0,1): Unemployment Rate
    axes[0, 1].plot(
        periods, unemployment_pct, linewidth=1.5, color=colors["unemployment"]
    )
    axes[0, 1].axhline(
        targets["unemployment_range_min"] * 100,
        color="green",
        linestyle="--",
        alpha=0.5,
        label="Target min",
    )
    axes[0, 1].axhline(
        targets["unemployment_range_max"] * 100,
        color="green",
        linestyle="--",
        alpha=0.5,
        label="Target max",
    )
    axes[0, 1].set_title("Unemployment Rate (%)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Time (periods)")
    axes[0, 1].set_ylabel("Unemployment Rate (%)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    axes[0, 1].legend(fontsize=8)

    # Panel (1,0): Annual Inflation Rate
    axes[1, 0].plot(periods, inflation_pct, linewidth=1.5, color=colors["inflation"])
    axes[1, 0].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[1, 0].axhline(
        targets["inflation_mean_min"] * 100,
        color="green",
        linestyle="--",
        alpha=0.5,
        label="Target min",
    )
    axes[1, 0].axhline(
        targets["inflation_mean_max"] * 100,
        color="green",
        linestyle="--",
        alpha=0.5,
        label="Target max",
    )
    axes[1, 0].set_title("Annual Inflation Rate (%)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Time (periods)")
    axes[1, 0].set_ylabel("Inflation Rate (%)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].legend(fontsize=8)

    # Panel (1,1): Productivity & Real Wage Co-movement (two lines)
    axes[1, 1].plot(
        periods,
        avg_productivity_trimmed,
        linewidth=1.5,
        color=colors["productivity"],
        label="Productivity",
    )
    axes[1, 1].plot(
        periods,
        real_wage_trimmed,
        linewidth=1.5,
        color=colors["real_wage"],
        label="Real Wage",
    )
    axes[1, 1].axhline(
        targets["real_wage_min"],
        color="green",
        linestyle="--",
        alpha=0.5,
        label="Real wage target min",
    )
    axes[1, 1].axhline(
        targets["real_wage_max"],
        color="green",
        linestyle="--",
        alpha=0.5,
        label="Real wage target max",
    )
    axes[1, 1].set_title(
        "Productivity & Real Wage Co-movement", fontsize=12, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Time (periods)")
    axes[1, 1].set_ylabel("Level")
    axes[1, 1].legend(loc="upper left", fontsize=8)
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    # Panel (2,0): Phillips Curve
    axes[2, 0].scatter(
        unemployment_phillips,
        wage_inflation_trimmed,
        s=10,
        alpha=0.5,
        color=colors["gdp"],
    )
    _add_regression_line(
        axes[2, 0],
        unemployment_phillips,
        wage_inflation_trimmed,
        phillips_corr,
        targets["phillips_target"],
        colors["gdp"],
    )
    axes[2, 0].set_title("Phillips Curve", fontsize=12, fontweight="bold")
    axes[2, 0].set_xlabel("Unemployment Rate")
    axes[2, 0].set_ylabel("Wage Inflation Rate")
    axes[2, 0].grid(True, linestyle="--", alpha=0.6)
    axes[2, 0].legend(fontsize=8, loc="upper right")

    # Panel (2,1): Okun Curve
    axes[2, 1].scatter(
        unemployment_growth_trimmed,
        gdp_growth_trimmed,
        s=10,
        alpha=0.5,
        color=colors["unemployment"],
    )
    _add_regression_line(
        axes[2, 1],
        unemployment_growth_trimmed,
        gdp_growth_trimmed,
        okun_corr,
        targets["okun_threshold"],
        colors["unemployment"],
    )
    axes[2, 1].set_title("Okun Curve", fontsize=12, fontweight="bold")
    axes[2, 1].set_xlabel("Unemployment Growth Rate")
    axes[2, 1].set_ylabel("Output Growth Rate")
    axes[2, 1].grid(True, linestyle="--", alpha=0.6)
    axes[2, 1].legend(fontsize=8, loc="upper right")

    # Panel (3,0): Beveridge Curve
    axes[3, 0].scatter(
        unemployment_beveridge,
        vacancy_rate_trimmed,
        s=10,
        alpha=0.5,
        color=colors["inflation"],
    )
    _add_regression_line(
        axes[3, 0],
        unemployment_beveridge,
        vacancy_rate_trimmed,
        beveridge_corr,
        targets["beveridge_target"],
        colors["inflation"],
    )
    axes[3, 0].set_title("Beveridge Curve", fontsize=12, fontweight="bold")
    axes[3, 0].set_xlabel("Unemployment Rate")
    axes[3, 0].set_ylabel("Vacancy Rate")
    axes[3, 0].grid(True, linestyle="--", alpha=0.6)
    axes[3, 0].legend(fontsize=8, loc="upper right")

    # Panel (3,1): Firm Size Distribution
    pct_below_3 = np.sum(final_production < 3) / len(final_production) * 100
    axes[3, 1].hist(
        final_production,
        bins=20,
        edgecolor="black",
        alpha=0.7,
        color=colors["real_wage"],
    )
    axes[3, 1].axvline(
        x=3,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Target threshold",
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

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    # Print summary statistics
    _print_summary_stats(
        log_gdp_trimmed,
        unemployment_pct,
        inflation_pct,
        real_wage_trimmed,
        avg_productivity_trimmed,
        phillips_corr,
        okun_corr,
        beveridge_corr,
        pct_below_3,
        scores,
    )


def _add_regression_line(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    actual_corr: float,
    target_corr: float,
    color: str,
):
    """Add regression lines (actual and target) to scatter plot."""
    x_mean, y_mean = np.mean(x), np.mean(y)
    x_std, y_std = np.std(x), np.std(y)

    if x_std > 0:
        x_range = np.array([x.min(), x.max()])

        # Actual regression line
        actual_slope = actual_corr * (y_std / x_std)
        y_actual = y_mean + actual_slope * (x_range - x_mean)
        ax.plot(
            x_range,
            y_actual,
            color=color,
            linewidth=2,
            alpha=0.8,
            label=f"Actual (r={actual_corr:.2f})",
        )

        # Target regression line
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


def _print_summary_stats(
    log_gdp: np.ndarray,
    unemployment_pct: np.ndarray,
    inflation_pct: np.ndarray,
    real_wage: np.ndarray,
    avg_productivity: np.ndarray,
    phillips_corr: float,
    okun_corr: float,
    beveridge_corr: float,
    pct_below_3: float,
    scores: dict,
):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (Post Burn-In)")
    print("=" * 60)

    print("\nLog GDP:")
    print(f"  Mean: {np.mean(log_gdp):.4f}")
    print(f"  Std:  {np.std(log_gdp):.4f}")

    print("\nUnemployment Rate:")
    print(f"  Mean: {np.mean(unemployment_pct):.2f}%")
    print(f"  Std:  {np.std(unemployment_pct):.2f}%")
    print(
        f"  Range: [{np.min(unemployment_pct):.2f}%, {np.max(unemployment_pct):.2f}%]"
    )

    print("\nInflation Rate:")
    print(f"  Mean: {np.mean(inflation_pct):.2f}%")
    print(f"  Positive: {np.mean(inflation_pct > 0) * 100:.0f}%")

    print("\nReal Wage:")
    print(f"  Mean: {np.mean(real_wage):.4f}")
    print(f"  Std:  {np.std(real_wage):.4f}")

    print("\nProductivity:")
    print(f"  Mean: {np.mean(avg_productivity):.4f}")
    print(f"  Std:  {np.std(avg_productivity):.4f}")

    print("\nCurve Correlations vs Targets:")
    print(
        f"  Phillips:  r={phillips_corr:+.2f} (target: {SCORE_TARGETS['phillips_target']:.2f})"
    )
    print(
        f"  Okun:      r={okun_corr:+.2f} (target: < {SCORE_TARGETS['okun_threshold']:.2f})"
    )
    print(
        f"  Beveridge: r={beveridge_corr:+.2f} (target: {SCORE_TARGETS['beveridge_target']:.2f})"
    )

    print(f"\nFirm Size: {pct_below_3:.0f}% below production 3 (target: 90%)")
    print(f"\nTotal Score: {scores['total']:.2f}")
    print("=" * 60)
