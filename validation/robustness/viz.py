"""Visualization for robustness analysis.

Generates co-movement plots (Figure 3.9 from the book), impulse-response
function comparisons, and sensitivity analysis summary plots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from validation.robustness.internal_validity import (
    COMOVEMENT_VARIABLES,
    InternalValidityResult,
)
from validation.robustness.sensitivity import ExperimentResult

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
