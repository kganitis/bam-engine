"""Univariate sensitivity analysis (Section 3.10.1, Part 2).

Varies one parameter at a time while holding all others at baseline
values. For each parameter value, runs multiple simulations with
different seeds and computes the same statistics as the internal
validity analysis.

The five parameter groups from the book are:
    (i)   H — local credit markets
    (ii)  Z — local consumption goods markets
    (iii) M — local labour markets (applications)
    (iv)  theta — employment contracts duration
    (v)   Economy size and composition
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

import bamengine as bam
from validation.robustness.experiments import (
    ALL_EXPERIMENT_NAMES,
    EXPERIMENTS,
    Experiment,
)
from validation.robustness.internal_validity import (
    COMOVEMENT_VARIABLES,
    SeedAnalysis,
    _run_seed,
)
from validation.robustness.stats import impulse_response
from validation.scenarios._utils import adjust_burn_in

# ─── Result Types ───────────────────────────────────────────────────────────


@dataclass
class ValueResult:
    """Aggregated results for one parameter value across multiple seeds."""

    label: str
    config_overrides: dict[str, Any]
    n_seeds: int
    n_collapsed: int

    # Mean co-movements across seeds
    mean_comovements: dict[str, NDArray[np.floating]]
    std_comovements: dict[str, NDArray[np.floating]]

    # Mean AR fit
    mean_ar_coeffs: NDArray[np.floating]
    mean_ar_r_squared: float
    mean_irf: NDArray[np.floating]

    # Cross-seed summary statistics
    stats: dict[str, dict[str, float]] = field(default_factory=dict)

    # Degenerate dynamics count
    n_degenerate: int = 0

    @property
    def collapse_rate(self) -> float:
        return self.n_collapsed / self.n_seeds if self.n_seeds > 0 else 0.0

    @property
    def degenerate_rate(self) -> float:
        return self.n_degenerate / self.n_seeds if self.n_seeds > 0 else 0.0


@dataclass
class ExperimentResult:
    """Results for one sensitivity experiment (all parameter values)."""

    experiment: Experiment
    value_results: list[ValueResult]
    baseline_idx: int  # Index of baseline value in value_results

    @property
    def baseline(self) -> ValueResult:
        return self.value_results[self.baseline_idx]

    def get_stat_table(self, stat_name: str) -> list[tuple[str, float, float]]:
        """Get a statistic across all values as (label, mean, std) tuples."""
        return [
            (
                vr.label,
                vr.stats.get(stat_name, {}).get("mean", float("nan")),
                vr.stats.get(stat_name, {}).get("std", float("nan")),
            )
            for vr in self.value_results
        ]


@dataclass
class SensitivityResult:
    """Full sensitivity analysis result across all experiments."""

    experiments: dict[str, ExperimentResult]
    n_seeds_per_value: int
    n_periods: int
    burn_in: int


# ─── Aggregation Helper ────────────────────────────────────────────────────


def _aggregate_seed_analyses(
    seed_analyses: list[SeedAnalysis],
    irf_periods: int = 20,
) -> ValueResult:
    """Aggregate multiple SeedAnalysis objects into a ValueResult."""
    valid = [a for a in seed_analyses if not a.degenerate]
    n_collapsed = sum(1 for a in seed_analyses if a.collapsed)
    n_degenerate = sum(1 for a in seed_analyses if a.degenerate)

    # Co-movements
    mean_comovements: dict[str, NDArray[np.floating]] = {}
    std_comovements: dict[str, NDArray[np.floating]] = {}

    # Determine expected array length from the first available seed
    any_seed = seed_analyses[0] if seed_analyses else None
    n_lags = len(any_seed.comovements[COMOVEMENT_VARIABLES[0]]) if any_seed else 9

    for var in COMOVEMENT_VARIABLES:
        if valid:
            all_corrs = np.array([a.comovements[var] for a in valid])
            mean_comovements[var] = np.nanmean(all_corrs, axis=0)
            std_comovements[var] = np.nanstd(all_corrs, axis=0)
        else:
            mean_comovements[var] = np.full(n_lags, np.nan)
            std_comovements[var] = np.full(n_lags, np.nan)

    # AR fit (average of individual fits)
    if valid:
        mean_phi1 = np.mean([a.ar_coeffs[1] for a in valid])
        mean_const = np.mean([a.ar_coeffs[0] for a in valid])
        mean_ar_coeffs = np.array([mean_const, mean_phi1])
        mean_ar_r2 = float(np.mean([a.ar_r_squared for a in valid]))
        mean_irf = impulse_response(mean_ar_coeffs, n_periods=irf_periods)
    else:
        mean_ar_coeffs = np.zeros(2)
        mean_ar_r2 = 0.0
        mean_irf = np.zeros(irf_periods)

    # Summary statistics
    stat_fields = [
        "unemployment_mean",
        "unemployment_std",
        "inflation_mean",
        "inflation_std",
        "gdp_growth_mean",
        "gdp_growth_std",
        "real_wage_mean",
        "productivity_mean",
        "phillips_corr",
        "okun_corr",
        "beveridge_corr",
        "firm_size_skewness_sales",
        "firm_size_skewness_net_worth",
    ]

    stats_dict: dict[str, dict[str, float]] = {}
    for attr_name in stat_fields:
        values = [
            getattr(a, attr_name) for a in valid if not np.isnan(getattr(a, attr_name))
        ]
        if values:
            mean_val = float(np.mean(values))
            stats_dict[attr_name] = {
                "mean": mean_val,
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "cv": float(np.std(values) / abs(mean_val))
                if abs(mean_val) > 1e-10
                else 0.0,
            }

    return ValueResult(
        label="",  # Set by caller
        config_overrides={},  # Set by caller
        n_seeds=len(seed_analyses),
        n_collapsed=n_collapsed,
        mean_comovements=mean_comovements,
        std_comovements=std_comovements,
        mean_ar_coeffs=mean_ar_coeffs,
        mean_ar_r_squared=mean_ar_r2,
        mean_irf=mean_irf,
        stats=stats_dict,
        n_degenerate=n_degenerate,
    )


# ─── Main Entry Point ──────────────────────────────────────────────────────


def run_sensitivity_analysis(
    experiments: list[str] | None = None,
    n_seeds: int = 20,
    n_periods: int = 1000,
    burn_in: int = 500,
    n_workers: int = 10,
    max_lag: int = 4,
    ar_order: int = 2,
    irf_periods: int = 20,
    verbose: bool = True,
    setup_hook: Callable[[bam.Simulation], None] | None = None,
    collect_config: dict[str, Any] | None = None,
    **config_overrides: Any,
) -> SensitivityResult:
    """Run univariate sensitivity analysis (Section 3.10.1, Part 2).

    For each experiment, varies one parameter while holding all others
    at baseline values. Runs ``n_seeds`` simulations per parameter value.

    Parameters
    ----------
    experiments : list[str] or None
        Experiment names to run. None means all experiments.
    n_seeds : int
        Number of random seeds per parameter value.
    n_periods : int
        Simulation periods per seed.
    burn_in : int
        Burn-in periods to discard.
    n_workers : int
        Parallel workers for simulation execution.
    max_lag : int
        Maximum lead/lag for cross-correlation.
    ar_order : int
        AR order for GDP cycle fitting.
    irf_periods : int
        Impulse-response function horizon.
    verbose : bool
        Print progress messages.
    setup_hook : callable or None
        Optional function ``(sim) -> None`` called after ``Simulation.init()``
        to attach extension roles, events, and config. Must be a
        module-level function for ``ProcessPoolExecutor`` pickling.
    collect_config : dict or None
        Custom collection configuration. When *None*, uses the default
        ``ROBUSTNESS_COLLECT_CONFIG``.
    **config_overrides
        Additional simulation config overrides applied to all runs.

    Returns
    -------
    SensitivityResult
        Results for all requested experiments.
    """
    burn_in = adjust_burn_in(burn_in, n_periods)

    if experiments is None:
        experiments = ALL_EXPERIMENT_NAMES

    experiment_results: dict[str, ExperimentResult] = {}

    for exp_name in experiments:
        if exp_name not in EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment: {exp_name}. Available: {ALL_EXPERIMENT_NAMES}"
            )

        exp = EXPERIMENTS[exp_name]
        labels = exp.get_labels()
        n_values = len(exp.values)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Experiment: {exp.description}")
            print(
                f"  {n_values} values x {n_seeds} seeds = "
                f"{n_values * n_seeds} simulations"
            )
            print(f"{'=' * 60}")

        value_results: list[ValueResult] = []
        baseline_idx = 0

        for val_idx in range(n_values):
            label = labels[val_idx]
            val_config = exp.get_config(val_idx)

            # Merge with any global overrides
            run_config = {**config_overrides, **val_config}

            if verbose:
                print(f"\n  [{val_idx + 1}/{n_values}] {label}")

            # Check if this is the baseline value
            if exp.values[val_idx] == exp.baseline_value:
                baseline_idx = val_idx

            # Run all seeds (parallel or sequential)
            seed_analyses: list[SeedAnalysis] = []
            seeds = list(range(n_seeds))

            if n_workers == 1:
                for i, seed in enumerate(seeds, 1):
                    analysis = _run_seed(
                        seed,
                        n_periods,
                        burn_in,
                        run_config,
                        max_lag,
                        ar_order,
                        irf_periods,
                        setup_hook,
                        collect_config,
                    )
                    seed_analyses.append(analysis)
                    if verbose and i % 5 == 0:
                        print(f"    {i}/{n_seeds} seeds done")
            else:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(
                            _run_seed,
                            seed,
                            n_periods,
                            burn_in,
                            run_config,
                            max_lag,
                            ar_order,
                            irf_periods,
                            setup_hook,
                            collect_config,
                        ): seed
                        for seed in seeds
                    }
                    for completed, future in enumerate(as_completed(futures), 1):
                        analysis = future.result()
                        seed_analyses.append(analysis)
                        if verbose and completed % 5 == 0:
                            print(f"    {completed}/{n_seeds} seeds done")

            # Sort by seed
            seed_analyses.sort(key=lambda a: a.seed)

            # Aggregate
            vr = _aggregate_seed_analyses(seed_analyses, irf_periods)
            vr.label = label
            vr.config_overrides = val_config

            n_ok = vr.n_seeds - vr.n_collapsed
            if verbose:
                parts = [f"{n_ok} valid", f"{vr.n_collapsed} collapsed"]
                if vr.n_degenerate > 0:
                    parts.append(f"{vr.n_degenerate} degenerate")
                print(f"    Result: {', '.join(parts)}")
                if "unemployment_mean" in vr.stats:
                    u = vr.stats["unemployment_mean"]["mean"]
                    print(f"    Unemployment: {u:.1%}")

            value_results.append(vr)

        experiment_results[exp_name] = ExperimentResult(
            experiment=exp,
            value_results=value_results,
            baseline_idx=baseline_idx,
        )

    return SensitivityResult(
        experiments=experiment_results,
        n_seeds_per_value=n_seeds,
        n_periods=n_periods,
        burn_in=burn_in,
    )
