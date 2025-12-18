"""
Full Factorial Grid Search
==========================

Exhaustive search over parameter combinations defined in CALIBRATION_PARAM_GRID.

Usage::

    python -m calibration --calibrate

Run sensitivity analysis first to inform grid design::

    python -m calibration --sensitivity
"""

from __future__ import annotations

import itertools
from multiprocessing import Pool
from typing import TYPE_CHECKING

from .config import CALIBRATION_PARAM_GRID
from .progress import ProgressTracker
from .runner import _run_config_worker, run_ensemble

if TYPE_CHECKING:
    from .checkpoint import CheckpointManager


def generate_combinations(grid: dict[str, list]) -> list[dict]:
    """
    Generate all parameter combinations from a grid.

    Parameters
    ----------
    grid : dict[str, list]
        Parameter names mapped to lists of values to test.

    Returns
    -------
    list[dict]
        List of parameter dictionaries, one per combination.
    """
    if not grid:
        return [{}]

    keys = list(grid.keys())
    values = list(grid.values())
    combinations = []

    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo, strict=False)))

    return combinations


def estimate_grid_search_time(
    param_grid: dict[str, list] | None = None,
    n_seeds: int = 3,
    config_time_seconds: float = 1.0,
) -> tuple[int, float]:
    """
    Estimate total time for grid search.

    Parameters
    ----------
    param_grid : dict, optional
        Parameter grid. Defaults to CALIBRATION_PARAM_GRID.
    n_seeds : int
        Seeds per configuration.
    config_time_seconds : float
        Estimated time per single simulation.

    Returns
    -------
    tuple[int, float]
        (total_combinations, estimated_hours)
    """
    if param_grid is None:
        param_grid = CALIBRATION_PARAM_GRID

    total_combos = 1
    for values in param_grid.values():
        total_combos *= len(values)

    total_sims = total_combos * n_seeds
    total_seconds = total_sims * config_time_seconds
    total_hours = total_seconds / 3600

    return total_combos, total_hours


def print_grid_info(param_grid: dict[str, list] | None = None, n_seeds: int = 3):
    """
    Print information about the current grid configuration.

    Parameters
    ----------
    param_grid : dict, optional
        Parameter grid. Defaults to CALIBRATION_PARAM_GRID.
    n_seeds : int
        Seeds per configuration.
    """
    if param_grid is None:
        param_grid = CALIBRATION_PARAM_GRID

    total_combos, total_hours = estimate_grid_search_time(param_grid, n_seeds)

    print("\n" + "=" * 70)
    print("CALIBRATION GRID CONFIGURATION")
    print("=" * 70)
    print(f"\nParameters in grid: {len(param_grid)}")
    print("-" * 70)

    for param, values in param_grid.items():
        print(f"  {param}: {values} ({len(values)} values)")

    print("-" * 70)
    print(f"\nTotal combinations: {total_combos:,}")
    print(f"Seeds per combination: {n_seeds}")
    print(f"Total simulations: {total_combos * n_seeds:,}")
    print(f"Estimated time: {total_hours:.1f} hours")
    print("=" * 70)


def run_grid_search(
    checkpoint: CheckpointManager,
    n_seeds: int = 3,
    n_periods: int = 1000,
    burn_in: int = 500,
    n_workers: int = 1,
    param_grid: dict[str, list] | None = None,
) -> list[dict]:
    """
    Run full factorial grid search over parameter combinations.

    Uses the CALIBRATION_PARAM_GRID from config.py. Edit that file after
    running sensitivity analysis to customize the search space.

    Parameters
    ----------
    checkpoint : CheckpointManager
        Checkpoint manager for saving results and resuming.
    n_seeds : int
        Number of seeds per configuration.
    n_periods : int
        Simulation periods.
    burn_in : int
        Burn-in periods.
    n_workers : int
        Number of parallel workers.
    param_grid : dict, optional
        Custom parameter grid. Defaults to CALIBRATION_PARAM_GRID.

    Returns
    -------
    list[dict]
        Top configurations sorted by score.
    """
    if param_grid is None:
        param_grid = CALIBRATION_PARAM_GRID

    print("\n" + "=" * 70)
    print("FULL FACTORIAL GRID SEARCH")
    print("=" * 70)

    # Print grid info
    print_grid_info(param_grid, n_seeds)

    # Generate all combinations
    all_combinations = generate_combinations(param_grid)
    total_combos = len(all_combinations)

    print("\nConfiguration:")
    print(f"  Seeds per config: {n_seeds}")
    print(f"  Periods: {n_periods}")
    print(f"  Burn-in: {burn_in}")
    print(f"  Workers: {n_workers}")

    # Filter already evaluated (checkpoint resume support)
    remaining = [c for c in all_combinations if not checkpoint.is_evaluated(c)]
    already_done = total_combos - len(remaining)

    print("\nProgress:")
    print(f"  Already evaluated: {already_done}")
    print(f"  Remaining: {len(remaining)}")

    if not remaining:
        print("\nAll configurations already evaluated!")
        return checkpoint.get_top_configs(100)

    # Get best score from checkpoint if resuming
    initial_best = float("inf")
    if already_done > 0:
        top_so_far = checkpoint.get_top_configs(1)
        if top_so_far:
            initial_best = top_so_far[0]["scores"]["total"]
            print(f"  Best score from checkpoint: {initial_best:.2f}")

    print(f"\nStarting grid search with {n_workers} worker(s)...\n")

    # Create progress tracker
    with ProgressTracker(
        total=len(remaining),
        stage="Grid Search",
        update_interval=10,
    ) as progress:
        # Set initial best score
        progress.stats.best_score = initial_best

        if n_workers > 1:
            # Parallel execution
            args_list = [(c, n_seeds, n_periods, burn_in) for c in remaining]

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
            for params in remaining:
                scores, seed_totals = run_ensemble(params, n_seeds, n_periods, burn_in)
                checkpoint.add_result(params, scores, seed_totals)
                progress.update(score=scores["total"], params=params)

                # Periodic checkpoint save
                if checkpoint.should_checkpoint(20):
                    checkpoint.save()

    # Final save
    checkpoint.save()

    # Get and print top configurations
    top_configs = checkpoint.get_top_configs(10)
    print("\n" + "-" * 70)
    print("TOP 10 CONFIGURATIONS")
    print("-" * 70)

    for rank, cfg in enumerate(top_configs, 1):
        print(f"\n#{rank} Score: {cfg['scores']['total']:.2f}")
        print("    Parameters:")
        for key, value in cfg["params"].items():
            print(f"      {key}: {value}")
        # Print key diagnostics
        scores = cfg["scores"]
        print("    Diagnostics:")
        print(f"      Real wage mean: {scores.get('_real_wage_mean', 0):.3f}")
        print(f"      Okun corr: {scores.get('_okun_corr', 0):.3f}")
        print(
            f"      Unemployment mean: {scores.get('_unemployment_mean', 0) * 100:.1f}%"
        )

    return top_configs
