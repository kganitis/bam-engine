"""Visualization for robustness analysis.

Generates co-movement plots (Figure 3.9 from the book), impulse-response
function comparisons, sensitivity analysis summary plots, and structural
experiment visualizations (Section 3.10.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from validation.robustness.internal_validity import (
    COMOVEMENT_VARIABLES,
    InternalValidityResult,
)
from validation.robustness.sensitivity import ExperimentResult

if TYPE_CHECKING:
    from validation.robustness.structural import (
        EntryExperimentResult,
        PAExperimentResult,
    )

_OUTPUT_DIR = Path(__file__).parent / "output"

# Display names for co-movement variables
_VARIABLE_TITLES = {
    "unemployment": "Unemployment",
    "productivity": "Productivity",
    "price_index": "Price index",
    "interest_rate": "Real interest rate",
    "real_wage": "Real wage",
}

# Panel labels matching the book (a)-(e)
_PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)"]

_PANEL_NAMES = [
    "3_9a_unemployment",
    "3_9b_productivity",
    "3_9c_price-index",
    "3_9d_real-interest-rate",
    "3_9e_real-wage",
]


def _save_panels(fig, axes_flat, output_dir, panel_names, dpi=150):
    """Save each subplot as individual image plus combined figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for ax, name in zip(axes_flat, panel_names, strict=True):
        extent = (
            ax.get_tightbbox(renderer)
            .transformed(fig.dpi_scale_trans.inverted())
            .padded(0.15)
        )
        fig.savefig(output_dir / f"{name}.png", bbox_inches=extent, dpi=dpi)

    fig.savefig(output_dir / "3_9_comovements.png", bbox_inches="tight", dpi=dpi)
    print(f"Saved {len(panel_names)} panels + combined figure to {output_dir}/")


def plot_comovements(
    result: InternalValidityResult,
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Plot Figure 3.9: co-movements at leads and lags.

    Creates a 3x2 grid (5 panels + 1 empty) showing cross-correlations
    between HP-filtered GDP and five macroeconomic variables at lags
    -4 to +4. Baseline run shown as '+', cross-simulation mean as 'o'.

    Parameters
    ----------
    result : InternalValidityResult
        Result from :func:`run_internal_validity`.
    output_dir : Path or None
        Directory for saving figures. Uses default if None.
    show : bool
        Whether to call plt.show().
    """
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = _OUTPUT_DIR

    max_lag = (len(next(iter(result.baseline_comovements.values()))) - 1) // 2
    lags = np.arange(-max_lag, max_lag + 1)

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle(
        "Co-movements at leads and lags (Section 3.10, Figure 3.9)\n"
        f"{result.n_seeds} seeds, {result.n_periods} periods "
        f"({result.n_collapsed} collapsed)",
        fontsize=13,
        y=0.98,
    )

    axes_flat = axes.flat

    for i, var in enumerate(COMOVEMENT_VARIABLES):
        ax = axes_flat[i]
        title = _VARIABLE_TITLES[var]

        baseline = result.baseline_comovements[var]
        mean = result.mean_comovements[var]

        # Plot baseline as '+' and mean as 'o' (matching book)
        ax.plot(
            lags,
            baseline,
            "+",
            markersize=10,
            color="blue",
            label="basic",
            markeredgewidth=1.5,
        )
        ax.plot(
            lags,
            mean,
            "o",
            markersize=6,
            color="blue",
            label="average",
            markerfacecolor="blue",
        )

        # Reference line at y=0
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="-")
        # ±0.2 acyclicality band (book convention)
        ax.axhline(y=0.2, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(y=-0.2, color="gray", linewidth=0.5, linestyle=":")

        ax.set_title(f"{_PANEL_LABELS[i]}\n{title}", fontsize=11)
        ax.set_xlabel("Lead/lag (quarters)")
        ax.set_xticks(lags)
        ax.grid(False)
        ax.legend(fontsize=8, loc="best")

    # Hide the 6th subplot (3x2 grid, 5 variables)
    axes_flat[5].set_visible(False)

    plt.tight_layout()
    _save_panels(fig, list(axes_flat)[:5], output_dir, _PANEL_NAMES)

    if show:
        plt.show()


def plot_irf(
    result: InternalValidityResult,
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Plot impulse-response function comparison.

    Shows the baseline AR(2) IRF and the cross-simulation average AR(1) IRF.

    Parameters
    ----------
    result : InternalValidityResult
        Result from :func:`run_internal_validity`.
    output_dir : Path or None
        Directory for saving figures.
    show : bool
        Whether to call plt.show().
    """
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = _OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find baseline seed's IRF
    baseline = next(
        (a for a in result.seed_analyses if a.seed == 0 and not a.collapsed),
        result.seed_analyses[0] if result.seed_analyses else None,
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    if baseline and not baseline.collapsed:
        periods = np.arange(len(baseline.irf))
        ax.plot(
            periods,
            baseline.irf,
            "b--",
            linewidth=1.5,
            label=f"Baseline AR({baseline.ar_order}) (R²={baseline.ar_r_squared:.2f})",
        )

    if len(result.mean_irf) > 0:
        periods = np.arange(len(result.mean_irf))
        ax.plot(
            periods,
            result.mean_irf,
            "b-",
            linewidth=2,
            label=f"Mean AR({result.mean_ar_order}) "
            f"(R²={result.mean_ar_r_squared:.2f})",
        )

    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_title(
        "GDP cyclical component: impulse-response function",
        fontsize=12,
    )
    ax.set_xlabel("Periods after shock")
    ax.set_ylabel("Response")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "irf_comparison.png", bbox_inches="tight", dpi=150)
    print(f"Saved IRF comparison to {output_dir}/irf_comparison.png")

    if show:
        plt.show()


def plot_sensitivity_comovements(
    exp_result: ExperimentResult,
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Plot co-movement comparison for a sensitivity experiment.

    Shows baseline vs extreme parameter values to illustrate how
    co-movement structure changes with parameter variation.

    Parameters
    ----------
    exp_result : ExperimentResult
        Result for one experiment from sensitivity analysis.
    output_dir : Path or None
        Directory for saving figures.
    show : bool
        Whether to call plt.show().
    """
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = _OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp = exp_result.experiment
    vrs = exp_result.value_results
    n_values = len(vrs)

    # Select which values to plot (baseline + first + last at minimum)
    if n_values <= 4:
        plot_indices = list(range(n_values))
    else:
        # Baseline + first + middle + last
        plot_indices = sorted(
            set([0, exp_result.baseline_idx, n_values // 2, n_values - 1])
        )

    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_indices)))
    max_lag = (len(next(iter(vrs[0].mean_comovements.values()))) - 1) // 2
    lags = np.arange(-max_lag, max_lag + 1)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle(
        f"Sensitivity: {exp.description}\nCo-movements at leads and lags",
        fontsize=12,
        y=0.98,
    )

    for i, var in enumerate(COMOVEMENT_VARIABLES):
        ax = axes.flat[i]
        title = _VARIABLE_TITLES[var]

        for j, idx in enumerate(plot_indices):
            vr = vrs[idx]
            marker = "s" if idx == exp_result.baseline_idx else "o"
            linewidth = 2.0 if idx == exp_result.baseline_idx else 1.0
            ax.plot(
                lags,
                vr.mean_comovements[var],
                marker=marker,
                markersize=5,
                color=colors[j],
                linewidth=linewidth,
                label=vr.label,
            )

        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.axhline(y=0.2, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(y=-0.2, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(f"{_PANEL_LABELS[i]} {title}", fontsize=10)
        ax.set_xlabel("Lead/lag")
        ax.set_xticks(lags)
        ax.legend(fontsize=7, loc="best")

    axes.flat[5].set_visible(False)
    plt.tight_layout()

    fname = f"sensitivity_{exp.name}_comovements.png"
    fig.savefig(output_dir / fname, bbox_inches="tight", dpi=150)
    print(f"Saved {fname} to {output_dir}/")

    if show:
        plt.show()


# ─── Section 3.10.2: Structural Experiment Plots ─────────────────────────


def plot_pa_gdp_comparison(
    pa_result: PAExperimentResult,
    output_dir: Path | None = None,
    show: bool = True,
    seed: int = 0,
) -> None:
    """Plot GDP time series comparison: PA on vs PA off (Figure 3.10).

    Runs two quick single-seed simulations to produce overlaid GDP
    time series showing how volatility drops when PA is disabled.

    Parameters
    ----------
    pa_result : PAExperimentResult
        Result from :func:`run_pa_experiment`.
    output_dir : Path or None
        Directory for saving figures.
    show : bool
        Whether to call plt.show().
    seed : int
        Seed for the comparison simulations.
    """
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = _OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use log_gdp from the PA-off seed analysis and baseline for comparison
    log_gdp_off = pa_result.pa_off_validity.seed_analyses[seed].log_gdp
    if pa_result.baseline_validity is not None:
        log_gdp_on = pa_result.baseline_validity.seed_analyses[seed].log_gdp
    else:
        raise ValueError(
            "PA GDP comparison requires baseline; re-run with include_baseline=True"
        )

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    periods = np.arange(len(log_gdp_on))
    ax.plot(periods, log_gdp_on, "b-", linewidth=1, alpha=0.8, label="PA on")
    ax.plot(
        periods[: len(log_gdp_off)],
        log_gdp_off,
        "r-",
        linewidth=1,
        alpha=0.8,
        label="PA off",
    )
    ax.set_title(
        "Log GDP: Preferential Attachment on vs off (Section 3.10.2, Figure 3.10)",
        fontsize=12,
    )
    ax.set_xlabel("Period")
    ax.set_ylabel("Log GDP")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = "pa_gdp_comparison.png"
    fig.savefig(output_dir / fname, bbox_inches="tight", dpi=150)
    print(f"Saved {fname} to {output_dir}/")

    if show:
        plt.show()


def plot_pa_comovements(
    pa_result: PAExperimentResult,
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Plot co-movement comparison: baseline vs PA-off.

    3x2 grid with both conditions overlaid. Shows price index and wages
    shifting to lagging/acyclical when PA is disabled.

    Parameters
    ----------
    pa_result : PAExperimentResult
        Result from :func:`run_pa_experiment`.
    output_dir : Path or None
        Directory for saving figures.
    show : bool
        Whether to call plt.show().
    """
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = _OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pa_off = pa_result.pa_off_validity
    baseline = pa_result.baseline_validity

    max_lag = (len(next(iter(pa_off.mean_comovements.values()))) - 1) // 2
    lags = np.arange(-max_lag, max_lag + 1)

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle(
        "Co-movements: PA on (baseline) vs PA off\n"
        f"{pa_off.n_seeds} seeds, {pa_off.n_periods} periods",
        fontsize=13,
        y=0.98,
    )

    for i, var in enumerate(COMOVEMENT_VARIABLES):
        ax = axes.flat[i]
        title = _VARIABLE_TITLES[var]

        # PA off (always available)
        ax.plot(
            lags,
            pa_off.mean_comovements[var],
            "ro-",
            markersize=5,
            linewidth=1.5,
            label="PA off (mean)",
        )

        # Baseline (if available)
        if baseline is not None:
            ax.plot(
                lags,
                baseline.mean_comovements[var],
                "b+-",
                markersize=8,
                linewidth=1.5,
                label="PA on (mean)",
                markeredgewidth=1.5,
            )

        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.axhline(y=0.2, color="gray", linewidth=0.5, linestyle=":")
        ax.axhline(y=-0.2, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(f"{_PANEL_LABELS[i]} {title}", fontsize=10)
        ax.set_xlabel("Lead/lag (quarters)")
        ax.set_xticks(lags)
        ax.legend(fontsize=8, loc="best")

    axes.flat[5].set_visible(False)
    plt.tight_layout()

    fname = "pa_comovements_comparison.png"
    fig.savefig(output_dir / fname, bbox_inches="tight", dpi=150)
    print(f"Saved {fname} to {output_dir}/")

    if show:
        plt.show()


def plot_entry_comparison(
    entry_result: EntryExperimentResult,
    output_dir: Path | None = None,
    show: bool = True,
) -> None:
    """Plot entry neutrality results across tax rates.

    Shows unemployment, GDP growth volatility, and collapse rate across
    profit tax rates. Monotonic degradation confirms entry mechanism
    does not artificially drive recovery.

    Parameters
    ----------
    entry_result : EntryExperimentResult
        Result from :func:`run_entry_experiment`.
    output_dir : Path or None
        Directory for saving figures.
    show : bool
        Whether to call plt.show().
    """
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = _OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_result = entry_result.tax_sweep.experiments["entry_neutrality"]
    vrs = exp_result.value_results
    labels = [vr.label for vr in vrs]

    # Extract statistics
    unemp = [
        vr.stats.get("unemployment_mean", {}).get("mean", float("nan")) for vr in vrs
    ]
    gdp_vol = [
        vr.stats.get("gdp_growth_std", {}).get("mean", float("nan")) for vr in vrs
    ]
    collapse_rates = [vr.collapse_rate for vr in vrs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Entry Neutrality: Impact of Profit Taxation (Section 3.10.2)",
        fontsize=13,
    )

    x = np.arange(len(labels))

    # Unemployment
    axes[0].bar(x, [u * 100 for u in unemp], color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Unemployment (%)")
    axes[0].set_title("Mean Unemployment")

    # GDP growth volatility
    axes[1].bar(x, [v * 100 for v in gdp_vol], color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("GDP Growth Volatility (%)")
    axes[1].set_title("GDP Volatility")

    # Collapse rate
    axes[2].bar(x, [c * 100 for c in collapse_rates], color="gray")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("Collapse Rate (%)")
    axes[2].set_title("Collapse Rate")

    plt.tight_layout()

    fname = "entry_neutrality_comparison.png"
    fig.savefig(output_dir / fname, bbox_inches="tight", dpi=150)
    print(f"Saved {fname} to {output_dir}/")

    if show:
        plt.show()
