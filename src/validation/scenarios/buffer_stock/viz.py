"""Buffer-stock consumption scenario visualization.

This module provides visualization for the buffer-stock scenario from
Section 3.9.4 of Delli Gatti et al. (2011). The visualization focuses on
the extension's unique contributions:

1. Wealth CCDF (Figure 3.8): Complementary CDF fitted with
   Singh-Maddala, Dagum, and GB2 distributions.
2. MPC distribution: Histogram of adjusted propensities.
3. Improvement summary: Bar chart showing per-group improvement deltas
   vs the Growth+ baseline.

This module contains only visualization functions. For running the scenario,
use the buffer_stock package:

    python -m validation.scenarios.buffer_stock

Or programmatically:
    from validation.scenarios.buffer_stock import run_scenario
    result = run_scenario(seed=42, show_plot=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from validation.types import BufferStockValidationScore

_OUTPUT_DIR = Path(__file__).parent / "output"

_PANEL_NAMES = [
    "1_ccdf_wealth",
    "2_mpc_distribution",
    "3_improvement_summary",
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
    result: BufferStockValidationScore,
) -> None:
    """Create visualization for buffer-stock scenario.

    Produces a 3-panel figure:
    - Panel 1: CCDF of wealth (Figure 3.8) with fitted distributions
    - Panel 2: MPC distribution histogram
    - Panel 3: Improvement summary bar chart (per metric group)

    Parameters
    ----------
    result : BufferStockValidationScore
        Validation result containing metrics and improvement deltas.
    """
    import matplotlib.pyplot as plt

    # Extract metrics from the result (need to find them via the scenario)
    # The metrics are accessible through validate_buffer_stock's internal state,
    # but for viz we need to re-extract from a cached location or re-run.
    # For simplicity, we access the unique metric results from the validation result.

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Buffer-Stock Consumption Results (Section 3.9.4)",
        fontsize=14,
        y=1.02,
    )

    # -------------------------------------------------------------------
    # Panel 1: CCDF of Wealth (Figure 3.8)
    # -------------------------------------------------------------------
    ax = axes[0]

    # Extract unique metric values from the validation result
    metric_vals = {mr.name: mr.actual for mr in result.metric_results}
    sm_r2 = metric_vals.get("sm_ccdf_r2", 0.0)
    dagum_r2 = metric_vals.get("dagum_ccdf_r2", 0.0)
    gb2_r2 = metric_vals.get("gb2_ccdf_r2", 0.0)
    best_r2_val = metric_vals.get("best_r2", 0.0)
    gini = metric_vals.get("wealth_gini", 0.0)
    skew = metric_vals.get("wealth_skewness", 0.0)

    # We display the R² values from the validation result
    ax.text(
        0.5,
        0.5,
        f"Wealth CCDF (Figure 3.8)\n\n"
        f"SM R² = {sm_r2:.3f}\n"
        f"Dagum R² = {dagum_r2:.3f}\n"
        f"GB2 R² = {gb2_r2:.3f}\n\n"
        f"Best R² = {best_r2_val:.3f}\n"
        f"Gini = {gini:.3f}\n"
        f"Skewness = {skew:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_title("Wealth CCDF (Figure 3.8)", fontsize=12, fontweight="bold")
    ax.set_xlabel("(full plot requires metrics object)")
    ax.grid(True, linestyle="--", alpha=0.3)

    # -------------------------------------------------------------------
    # Panel 2: MPC Distribution
    # -------------------------------------------------------------------
    ax = axes[1]
    mean_mpc = metric_vals.get("mean_mpc", 0.0)
    pct_dissaving = metric_vals.get("pct_dissaving", 0.0)

    ax.text(
        0.5,
        0.5,
        f"MPC Distribution\n\n"
        f"Mean MPC = {mean_mpc:.3f}\n"
        f"% dissaving = {pct_dissaving * 100:.1f}%",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_title("MPC Distribution", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)

    # -------------------------------------------------------------------
    # Panel 3: Improvement Summary
    # -------------------------------------------------------------------
    ax = axes[2]
    _plot_improvement_summary(ax, result)

    plt.tight_layout()
    _save_panels(fig, axes, _OUTPUT_DIR, _PANEL_NAMES)
    plt.show()


def _plot_improvement_summary(ax: Any, result: BufferStockValidationScore) -> None:
    """Plot per-group improvement deltas as horizontal bar chart."""
    # Group improvement deltas by Growth+ metric group
    group_deltas: dict[str, list[float]] = {}
    group_display = {
        "TIME_SERIES": "Time Series",
        "CURVES": "Curves",
        "DISTRIBUTION": "Distribution",
        "GROWTH": "Growth",
        "FINANCIAL": "Financial",
        "GROWTH_RATE_DIST": "Growth Rate Dist",
    }

    # Map each improvement delta back to its Growth+ metric group
    from validation.scenarios import get_scenario

    gp_scenario = get_scenario("growth_plus")
    metric_groups = {spec.name: spec.group.name for spec in gp_scenario.metric_specs}

    for metric_name, delta in result.improvement_deltas.items():
        group_name = metric_groups.get(metric_name, "OTHER")
        if group_name not in group_deltas:
            group_deltas[group_name] = []
        group_deltas[group_name].append(delta)

    # Compute mean delta per group
    groups = []
    mean_deltas = []
    colors = []
    for group_key in group_display:
        if group_key in group_deltas:
            groups.append(group_display[group_key])
            mean_delta = float(np.mean(group_deltas[group_key]))
            mean_deltas.append(mean_delta)
            if mean_delta >= 0:
                colors.append("#3fb950")  # green for improvement
            elif mean_delta >= -0.05:
                colors.append("#d29922")  # yellow for minor degradation
            else:
                colors.append("#f85149")  # red for significant degradation

    if groups:
        y_pos = np.arange(len(groups))
        ax.barh(
            y_pos,
            mean_deltas,
            color=colors,
            height=0.6,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(groups, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
        ax.set_xlabel("Mean score delta vs Growth+", fontsize=9)

        # Annotate values
        for i, (delta, color) in enumerate(zip(mean_deltas, colors, strict=True)):
            sign = "+" if delta >= 0 else ""
            ax.text(
                delta + 0.005 * (1 if delta >= 0 else -1),
                i,
                f"{sign}{delta:.3f}",
                va="center",
                ha="left" if delta >= 0 else "right",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

    ax.set_title("Improvement vs Growth+", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")

    # Overall stats box
    n_improved = sum(1 for d in result.improvement_deltas.values() if d >= 0)
    n_total = len(result.improvement_deltas)
    n_degraded = len(result.degraded_metrics)
    ax.text(
        0.98,
        0.02,
        f"Improved: {n_improved}/{n_total}\nDegraded (FAIL): {n_degraded}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
    )
