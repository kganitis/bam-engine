"""
Local Sensitivity Sweep
=======================

Tests alternative parameter values on top configurations from grid search.

This module sweeps ALL parameters including implementation variants, testing
one parameter at a time to find potential improvements.

Usage::

    python -m calibration --local-sweep
"""

from __future__ import annotations

from multiprocessing import Pool
from typing import TYPE_CHECKING

from .progress import ProgressTracker
from .runner import _run_config_worker, run_ensemble

if TYPE_CHECKING:
    from .checkpoint import CheckpointManager


# Parameters to sweep in local sensitivity analysis
# These fill gaps between grid points to find potential improvements.
# Implementation variants (v, savings_init, firing_method, etc.) are NOT included
# because they were already validated in prior sensitivity analysis.
LOCAL_SWEEP_PARAMS: dict[str, list] = {
    # Grid params with alternative values (fill gaps between grid points)
    "price_init_offset": [0.15, 0.2, 0.3, 0.4],
    "min_wage_ratio": [0.52, 0.58, 0.62, 0.68],
    "contract_poisson_mean": [5, 7, 12, 15],
    "net_worth_init": [0.5, 3.0, 7.0, 15.0],
    "new_firm_price_markup": [1.05, 1.15, 1.30],
    "new_firm_production_factor": [0.7, 0.85, 0.95],
    "new_firm_wage_factor": [0.55, 0.65, 0.75, 0.85, 0.95],
    "equity_base_init": [3.0, 7.0, 15.0],
    # Fixed param to sweep (was fixed to 0.8 in FIXED_PARAMS)
    "new_firm_size_factor": [0.5, 0.6, 0.7],
}


def estimate_local_sweep_time(
    n_top_configs: int = 300,
    config_time_seconds: float = 3.0,  # 3 seeds * ~1s per seed
) -> tuple[int, float]:
    """
    Estimate time for local sweep.

    Parameters
    ----------
    n_top_configs : int
        Number of top configurations to sweep.
    config_time_seconds : float
        Estimated time per configuration (including all seeds).

    Returns
    -------
    tuple[int, float]
        (max_variations, estimated_hours)
    """
    total_variations = sum(len(v) for v in LOCAL_SWEEP_PARAMS.values())
    max_sweeps = n_top_configs * total_variations
    total_hours = max_sweeps * config_time_seconds / 3600
    return max_sweeps, total_hours


def print_local_sweep_info(top_k: int = 300):
    """Print information about local sweep configuration."""
    print("\n" + "=" * 70)
    print("LOCAL SENSITIVITY SWEEP CONFIGURATION")
    print("=" * 70)
    print(f"\nParameters to sweep: {len(LOCAL_SWEEP_PARAMS)}")
    print("-" * 70)

    for param, values in LOCAL_SWEEP_PARAMS.items():
        print(f"  {param}: {values} ({len(values)} values)")

    total_variations = sum(len(v) for v in LOCAL_SWEEP_PARAMS.values())
    max_sweeps, hours = estimate_local_sweep_time(top_k)

    print("-" * 70)
    print(f"\nVariations per config: {total_variations}")
    print(f"Top configs to sweep: {top_k}")
    print(f"Maximum sweep configs: {max_sweeps:,}")
    print(f"Estimated max time: {hours:.1f} hours")
    print("(Actual time less due to skipping already-tested values)")
    print("=" * 70)


def run_local_sensitivity_sweep(
    checkpoint: CheckpointManager,
    top_configs: list[dict],
    n_seeds: int = 3,
    n_periods: int = 1000,
    burn_in: int = 500,
    n_workers: int = 1,
    top_k: int = 300,
) -> list[dict]:
    """
    Run local sensitivity sweep on top configurations.

    For each top configuration, tests alternative values of parameters
    one at a time to find potential improvements.

    Parameters
    ----------
    checkpoint : CheckpointManager
        Checkpoint manager for saving results and resuming.
    top_configs : list[dict]
        List of top configurations from grid search.
    n_seeds : int
        Number of seeds per configuration.
    n_periods : int
        Simulation periods.
    burn_in : int
        Burn-in periods.
    n_workers : int
        Number of parallel workers.
    top_k : int
        Number of top configurations to sweep.

    Returns
    -------
    list[dict]
        Updated top configurations after sweep.
    """
    # Check if already completed (skip on resume)
    if checkpoint.data["metadata"].get("local_sweep_completed", False):
        print("\n" + "=" * 70)
        print("LOCAL SENSITIVITY SWEEP (SKIPPED - already completed)")
        print("=" * 70)
        return checkpoint.get_top_configs(100)

    print("\n" + "=" * 70)
    print("LOCAL SENSITIVITY SWEEP")
    print("=" * 70)
    checkpoint.set_metadata("stage", "local_sensitivity_sweep")

    # Print configuration info
    print_local_sweep_info(top_k)

    # Take top k configurations
    configs_to_sweep = top_configs[:top_k]
    print(f"\nSweeping {len(configs_to_sweep)} top configurations")

    # Generate all sweep combinations
    sweep_configs = []
    for base_config in configs_to_sweep:
        base_params = base_config["params"].copy()
        # Test each param's alternative values one at a time
        for param, values in LOCAL_SWEEP_PARAMS.items():
            for value in values:
                # Skip if this is already the value in base config
                if base_params.get(param) == value:
                    continue
                # Create variant
                variant = base_params.copy()
                variant[param] = value
                if not checkpoint.is_evaluated(variant):
                    sweep_configs.append(variant)

    total_sweeps = len(sweep_configs)
    print(f"\nTotal new sweep configurations: {total_sweeps}")

    if not sweep_configs:
        print("All sweep configurations already evaluated!")
        checkpoint.data["metadata"]["local_sweep_completed"] = True
        checkpoint.save()
        return checkpoint.get_top_configs(100)

    # Get initial best score from grid search
    initial_best = float("inf")
    if top_configs:
        initial_best = top_configs[0]["scores"]["total"]
        print(f"Best score from grid search: {initial_best:.2f}")

    print(f"\nStarting local sweep with {n_workers} worker(s)...\n")

    # Create progress tracker
    with ProgressTracker(
        total=total_sweeps,
        stage="Local Sweep",
        update_interval=10,
    ) as progress:
        # Set initial best score
        progress.stats.best_score = initial_best

        if n_workers > 1:
            # Parallel execution
            args_list = [(c, n_seeds, n_periods, burn_in) for c in sweep_configs]

            with Pool(processes=n_workers) as pool:
                for params, scores, seed_totals in pool.imap_unordered(
                    _run_config_worker, args_list
                ):
                    checkpoint.add_result(params, scores, seed_totals)
                    progress.update(score=scores["total"], params=params)

                    # Periodic checkpoint save
                    if checkpoint.should_checkpoint(50):
                        checkpoint.save()
        else:
            # Sequential execution
            for config in sweep_configs:
                scores, seed_totals = run_ensemble(config, n_seeds, n_periods, burn_in)
                checkpoint.add_result(config, scores, seed_totals)
                progress.update(score=scores["total"], params=config)

                # Periodic checkpoint save
                if checkpoint.should_checkpoint(20):
                    checkpoint.save()

    # Mark local sweep as completed so it's skipped on resume
    checkpoint.data["metadata"]["local_sweep_completed"] = True
    checkpoint.save()

    # Print improvement summary
    final_best = checkpoint.get_top_configs(1)[0]["scores"]["total"]
    if final_best < initial_best:
        improvement = initial_best - final_best
        print(
            f"\nScore improvement: {initial_best:.2f} -> {final_best:.2f} ({improvement:.2f})"
        )
    else:
        print("\nNo improvements found over grid search best.")

    return checkpoint.get_top_configs(100)
