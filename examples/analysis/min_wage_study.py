"""
=============================
Minimum Wage Dynamics Study
=============================

This script analyzes minimum wage dynamics during BAM simulations:

1. How many firms per period offer exactly the minimum wage
2. How many firms are very close to it (within 1%, 5%, 10%)
3. The level of minimum wage over time

The analysis runs 10 simulations with different seeds and averages
the metrics to show robust patterns with confidence bands.

Output
------
Generates 5 visualization plots saved to examples/analysis/output/:

- min_wage_level.png - Min wage trajectory over time
- firms_at_min_wage.png - % of firms at/near min wage (stacked area)
- wage_distribution_snapshots.png - Histograms at t=100, 500, 1000
- wage_spread_ratio.png - Mean wage / min wage over time
- min_wage_vs_unemployment.png - Correlation scatter plot
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bamengine as bam

# =============================================================================
# Configuration
# =============================================================================

# Recommended calibration config
CONFIG = {
    "price_init_offset": 0.5,
    "min_wage_ratio": 0.5,
    "contract_poisson_mean": 10,
    "net_worth_init": 1.0,
    "new_firm_price_markup": 1.25,
    "new_firm_production_factor": 0.9,
    "new_firm_wage_factor": 0.5,
    "equity_base_init": 5.0,
    "new_firm_size_factor": 0.5,
    "n_periods": 1000,
    "logging": {"default_level": "ERROR"},
}

N_SEEDS = 10
OUTPUT_DIR = Path(__file__).parent / "output"


# =============================================================================
# Data Collection
# =============================================================================


def collect_single_seed(seed: int) -> dict[str, np.ndarray]:
    """
    Run simulation for one seed and collect min wage metrics each period.

    Parameters
    ----------
    seed : int
        Random seed for this simulation run.

    Returns
    -------
    dict
        Dictionary of numpy arrays with per-period metrics.
    """
    config = {**CONFIG, "seed": seed}
    sim = bam.Simulation.init(**config)

    n_periods = config["n_periods"]
    n_firms = sim.n_firms

    # Pre-allocate arrays
    data = {
        "min_wage": np.zeros(n_periods),
        "n_at_min_wage": np.zeros(n_periods),
        "n_within_1pct": np.zeros(n_periods),
        "n_within_5pct": np.zeros(n_periods),
        "n_within_10pct": np.zeros(n_periods),
        "pct_at_min_wage": np.zeros(n_periods),
        "pct_within_1pct": np.zeros(n_periods),
        "pct_within_5pct": np.zeros(n_periods),
        "pct_within_10pct": np.zeros(n_periods),
        "wage_offer_mean": np.zeros(n_periods),
        "wage_offer_median": np.zeros(n_periods),
        "wage_offer_std": np.zeros(n_periods),
        "wage_spread_ratio": np.zeros(n_periods),
        "unemployment_rate": np.zeros(n_periods),
    }

    # Also store snapshots for histogram (normalized as ratio to min_wage)
    snapshots = {}

    for t in range(n_periods):
        sim.step()

        min_wage = sim.ec.min_wage
        wage_offers = sim.emp.wage_offer.copy()
        employed = sim.wrk.employed

        # Count firms at exactly minimum wage (float tolerance)
        at_min = np.sum(np.isclose(wage_offers, min_wage, rtol=1e-9))

        # Count firms within X% of minimum wage
        within_1pct = np.sum(wage_offers <= min_wage * 1.01)
        within_5pct = np.sum(wage_offers <= min_wage * 1.05)
        within_10pct = np.sum(wage_offers <= min_wage * 1.10)

        # Unemployment rate
        unemployment = 1 - np.mean(employed)

        # Store
        data["min_wage"][t] = min_wage
        data["n_at_min_wage"][t] = at_min
        data["n_within_1pct"][t] = within_1pct
        data["n_within_5pct"][t] = within_5pct
        data["n_within_10pct"][t] = within_10pct
        data["pct_at_min_wage"][t] = at_min / n_firms * 100
        data["pct_within_1pct"][t] = within_1pct / n_firms * 100
        data["pct_within_5pct"][t] = within_5pct / n_firms * 100
        data["pct_within_10pct"][t] = within_10pct / n_firms * 100
        data["wage_offer_mean"][t] = np.mean(wage_offers)
        data["wage_offer_median"][t] = np.median(wage_offers)
        data["wage_offer_std"][t] = np.std(wage_offers)
        data["wage_spread_ratio"][t] = np.mean(wage_offers) / min_wage
        data["unemployment_rate"][t] = unemployment

        # Store snapshots for histogram - NORMALIZED as ratio to min_wage
        # This allows combining across seeds with different min_wage values
        if t + 1 in [100, 500, 1000]:
            snapshots[t + 1] = wage_offers / min_wage  # Ratio, not absolute

    data["snapshots"] = snapshots
    return data


def run_all_seeds() -> dict[str, np.ndarray]:
    """
    Run simulations for all seeds and compute mean/std across seeds.

    Returns
    -------
    dict
        Dictionary with mean and std arrays for each metric.
    """
    print(f"Running {N_SEEDS} simulations...")

    all_data = []
    all_snapshots = {100: [], 500: [], 1000: []}

    for seed in range(N_SEEDS):
        print(f"  Seed {seed + 1}/{N_SEEDS}...", end=" ", flush=True)
        data = collect_single_seed(seed)
        all_data.append(data)

        # Collect snapshots
        for period in [100, 500, 1000]:
            if period in data["snapshots"]:
                all_snapshots[period].append(data["snapshots"][period])

        print(f"done (final min_wage: {data['min_wage'][-1]:.3f})")

    # Compute mean and std across seeds
    metrics = [k for k in all_data[0] if k != "snapshots"]
    results = {}

    for metric in metrics:
        stacked = np.stack([d[metric] for d in all_data], axis=0)
        results[f"{metric}_mean"] = np.mean(stacked, axis=0)
        results[f"{metric}_std"] = np.std(stacked, axis=0)

    # Store snapshots for histograms
    results["snapshots"] = all_snapshots

    return results


# =============================================================================
# Visualization
# =============================================================================


def plot_min_wage_level(results: dict, output_dir: Path) -> None:
    """Plot 1: Minimum wage level over time."""
    _fig, ax = plt.subplots(figsize=(10, 5))

    periods = np.arange(1, len(results["min_wage_mean"]) + 1)
    mean = results["min_wage_mean"]
    std = results["min_wage_std"]

    ax.plot(periods, mean, color="darkblue", linewidth=1.5, label="Mean")
    ax.fill_between(
        periods,
        mean - std,
        mean + std,
        color="darkblue",
        alpha=0.2,
        label="±1 std",
    )

    ax.set_xlabel("Period")
    ax.set_ylabel("Minimum Wage")
    ax.set_title("Minimum Wage Level Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "min_wage_level.png", dpi=150)
    plt.close()
    print("  Saved: min_wage_level.png")


def plot_firms_at_min_wage(results: dict, output_dir: Path) -> None:
    """Plot 2: Percentage of firms at/near minimum wage (stacked area)."""
    _fig, ax = plt.subplots(figsize=(10, 5))

    periods = np.arange(1, len(results["pct_at_min_wage_mean"]) + 1)

    # Compute bands (non-overlapping)
    at_min = results["pct_at_min_wage_mean"]
    within_1pct_exclusive = results["pct_within_1pct_mean"] - at_min
    within_5pct_exclusive = (
        results["pct_within_5pct_mean"] - results["pct_within_1pct_mean"]
    )
    within_10pct_exclusive = (
        results["pct_within_10pct_mean"] - results["pct_within_5pct_mean"]
    )
    above_10pct = 100 - results["pct_within_10pct_mean"]

    ax.stackplot(
        periods,
        at_min,
        within_1pct_exclusive,
        within_5pct_exclusive,
        within_10pct_exclusive,
        above_10pct,
        labels=[
            "At min wage (exact)",
            "Within 1% above",
            "1-5% above",
            "5-10% above",
            ">10% above",
        ],
        colors=["#d62728", "#ff7f0e", "#ffbb78", "#98df8a", "#2ca02c"],
        alpha=0.8,
    )

    ax.set_xlabel("Period")
    ax.set_ylabel("% of Firms")
    ax.set_title("Distribution of Firm Wage Offers Relative to Minimum Wage")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "firms_at_min_wage.png", dpi=150)
    plt.close()
    print("  Saved: firms_at_min_wage.png")


def plot_wage_distribution_snapshots(results: dict, output_dir: Path) -> None:
    """Plot 3: Wage offer distribution histograms at selected periods.

    Data is normalized as wage_offer / min_wage ratio, so values represent
    how far above the minimum wage floor each firm's offer is.
    """
    _fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    snapshots = results["snapshots"]
    periods = [100, 500, 1000]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, period, color in zip(axes, periods, colors, strict=False):
        if snapshots[period]:
            # Combine all seeds for this period (data is already normalized as ratio)
            all_wage_ratios = np.concatenate(snapshots[period])

            ax.hist(
                all_wage_ratios,
                bins=50,
                color=color,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

            # Vertical line at 1.0 = minimum wage floor
            ax.axvline(
                1.0, color="red", linestyle="--", linewidth=2, label="Min wage (1.0)"
            )

            # Add reference lines for threshold bands
            ax.axvline(1.01, color="orange", linestyle=":", linewidth=1, alpha=0.7)
            ax.axvline(1.05, color="orange", linestyle=":", linewidth=1, alpha=0.7)
            ax.axvline(1.10, color="orange", linestyle=":", linewidth=1, alpha=0.7)

            ax.set_xlabel("Wage Offer / Min Wage")
            ax.set_ylabel("Count")
            ax.set_title(f"Period {period}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0.95)  # Start just below 1.0 to show the floor clearly

    plt.suptitle(
        "Wage Offer Distribution Relative to Minimum Wage", fontsize=12, y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_dir / "wage_distribution_snapshots.png", dpi=150)
    plt.close()
    print("  Saved: wage_distribution_snapshots.png")


def plot_wage_spread_ratio(results: dict, output_dir: Path) -> None:
    """Plot 4: Wage spread ratio (mean wage / min wage) over time."""
    _fig, ax = plt.subplots(figsize=(10, 5))

    periods = np.arange(1, len(results["wage_spread_ratio_mean"]) + 1)
    mean = results["wage_spread_ratio_mean"]
    std = results["wage_spread_ratio_std"]

    ax.plot(periods, mean, color="darkgreen", linewidth=1.5, label="Mean")
    ax.fill_between(
        periods,
        mean - std,
        mean + std,
        color="darkgreen",
        alpha=0.2,
        label="±1 std",
    )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Min wage floor")

    ax.set_xlabel("Period")
    ax.set_ylabel("Mean Wage / Min Wage")
    ax.set_title("Wage Spread Ratio Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "wage_spread_ratio.png", dpi=150)
    plt.close()
    print("  Saved: wage_spread_ratio.png")


def plot_min_wage_vs_unemployment(results: dict, output_dir: Path) -> None:
    """Plot 5: Correlation between % at min wage and unemployment."""
    _fig, ax = plt.subplots(figsize=(8, 6))

    pct_at_min = results["pct_at_min_wage_mean"]
    unemployment = results["unemployment_rate_mean"] * 100  # Convert to percentage

    # Color by time (early = light, late = dark)
    periods = np.arange(len(pct_at_min))
    scatter = ax.scatter(
        pct_at_min,
        unemployment,
        c=periods,
        cmap="viridis",
        alpha=0.6,
        s=20,
    )

    # Add correlation coefficient
    corr = np.corrcoef(pct_at_min, unemployment)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {corr:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Period")

    ax.set_xlabel("% of Firms at Minimum Wage")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.set_title("Minimum Wage Concentration vs Unemployment")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "min_wage_vs_unemployment.png", dpi=150)
    plt.close()
    print("  Saved: min_wage_vs_unemployment.png")


def print_summary(results: dict) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Initial vs Final
    print("\nMinimum Wage:")
    print(f"  Initial: {results['min_wage_mean'][0]:.4f}")
    print(f"  Final:   {results['min_wage_mean'][-1]:.4f}")
    print(
        f"  Growth:  {(results['min_wage_mean'][-1] / results['min_wage_mean'][0] - 1) * 100:.1f}%"
    )

    # Firms at min wage
    print("\nFirms at Exact Min Wage:")
    print(f"  Mean across all periods: {np.mean(results['pct_at_min_wage_mean']):.1f}%")
    print(f"  Max:  {np.max(results['pct_at_min_wage_mean']):.1f}%")
    print(f"  Min:  {np.min(results['pct_at_min_wage_mean']):.1f}%")

    # Firms within 10%
    print("\nFirms Within 10% of Min Wage:")
    print(f"  Mean: {np.mean(results['pct_within_10pct_mean']):.1f}%")

    # Wage spread
    print("\nWage Spread Ratio (mean wage / min wage):")
    print(f"  Initial: {results['wage_spread_ratio_mean'][0]:.3f}")
    print(f"  Final:   {results['wage_spread_ratio_mean'][-1]:.3f}")
    print(f"  Average: {np.mean(results['wage_spread_ratio_mean']):.3f}")

    # Correlation
    corr = np.corrcoef(
        results["pct_at_min_wage_mean"], results["unemployment_rate_mean"]
    )[0, 1]
    print(f"\nCorrelation (% at min wage vs unemployment): {corr:.3f}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run the minimum wage dynamics study."""
    print("=" * 60)
    print("MINIMUM WAGE DYNAMICS STUDY")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Seeds: {N_SEEDS}")
    print(f"  Periods: {CONFIG['n_periods']}")
    print(f"  min_wage_ratio: {CONFIG['min_wage_ratio']}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run simulations
    results = run_all_seeds()

    # Generate plots
    print("\nGenerating plots...")
    plot_min_wage_level(results, OUTPUT_DIR)
    plot_firms_at_min_wage(results, OUTPUT_DIR)
    plot_wage_distribution_snapshots(results, OUTPUT_DIR)
    plot_wage_spread_ratio(results, OUTPUT_DIR)
    plot_min_wage_vs_unemployment(results, OUTPUT_DIR)

    # Print summary
    print_summary(results)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
