"""
Consistency Analysis
====================

Analyze configuration stability across multiple seeds.

Some configurations may perform well with 3 seeds but show high variance
when tested with more seeds. This module identifies such unstable configurations.

Usage::

    python -m calibration --consistency --top 50 --seeds 10

"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import TYPE_CHECKING

import numpy as np

from .progress import ProgressTracker
from .runner import run_single_simulation

if TYPE_CHECKING:
    from .checkpoint import CheckpointManager


@dataclass
class ConsistencyResult:
    """Results from multi-seed consistency analysis."""

    rank: int
    params: dict
    checkpoint_score: float  # Original score from calibration (3 seeds)
    checkpoint_std: float  # Original std from calibration

    # Multi-seed results
    scores: list[float]
    mean_score: float
    std_score: float
    min_score: float
    max_score: float

    # Derived metrics
    consistency_score: float  # mean + 2*std (lower is better)
    cv: float  # Coefficient of variation (std/mean)

    # Okun correlation
    okun_corrs: list[float]
    mean_okun: float
    std_okun: float

    # Other diagnostics
    collapse_count: int  # Seeds where simulation collapsed


def analyze_config(
    params: dict,
    n_seeds: int = 10,
    n_periods: int = 1000,
    burn_in: int = 500,
) -> dict:
    """
    Run a configuration with multiple seeds and compute statistics.

    Parameters
    ----------
    params : dict
        Configuration parameters.
    n_seeds : int
        Number of seeds to test.
    n_periods : int
        Simulation periods.
    burn_in : int
        Burn-in periods.

    Returns
    -------
    dict
        Statistics across all seeds.
    """
    scores = []
    okun_corrs = []
    collapse_count = 0

    for seed in range(n_seeds):
        result = run_single_simulation(params, seed, n_periods, burn_in)
        scores.append(result["total"])
        okun_corrs.append(result.get("_okun_corr", 0.0))

        # Check for collapse
        if result.get("collapse_penalty", 0) > 0:
            collapse_count += 1

    scores_arr = np.array(scores)
    okun_arr = np.array(okun_corrs)

    mean_score = float(np.mean(scores_arr))
    std_score = float(np.std(scores_arr))

    return {
        "scores": scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "min_score": float(np.min(scores_arr)),
        "max_score": float(np.max(scores_arr)),
        "cv": std_score / mean_score if mean_score > 0 else float("inf"),
        "okun_corrs": okun_corrs,
        "mean_okun": float(np.mean(okun_arr)),
        "std_okun": float(np.std(okun_arr)),
        "collapse_count": collapse_count,
    }


def _analyze_config_worker(
    args: tuple[int, dict, float, float, int, int, int],
) -> tuple[int, dict, float, float, dict]:
    """
    Worker function for parallel consistency analysis.

    Parameters
    ----------
    args : tuple
        (rank, params, checkpoint_score, checkpoint_std, n_seeds, n_periods, burn_in)

    Returns
    -------
    tuple
        (rank, params, checkpoint_score, checkpoint_std, stats)
    """
    rank, params, checkpoint_score, checkpoint_std, n_seeds, n_periods, burn_in = args
    stats = analyze_config(params, n_seeds, n_periods, burn_in)
    return rank, params, checkpoint_score, checkpoint_std, stats


def get_configs_to_analyze(
    checkpoint: CheckpointManager,
    max_score: float = 50.0,
    top_n: int | None = None,
) -> list[dict]:
    """
    Get configurations to analyze from checkpoint.

    Parameters
    ----------
    checkpoint : CheckpointManager
        Checkpoint with calibration results.
    max_score : float
        Maximum score threshold for inclusion.
    top_n : int, optional
        If specified, limit to top N configs regardless of score.

    Returns
    -------
    list[dict]
        List of config dicts with 'params' and 'scores' keys.
    """
    # Get all grid results sorted by score
    grid_results = checkpoint.data.get("grid_results", [])
    if not grid_results:
        return []

    # Sort by total score
    sorted_results = sorted(grid_results, key=lambda x: x["scores"]["total"])

    # Filter by score threshold
    filtered = [r for r in sorted_results if r["scores"]["total"] <= max_score]

    # Apply top_n limit if specified
    if top_n is not None:
        filtered = filtered[:top_n]

    return filtered


def print_consistency_info(
    n_configs: int,
    n_seeds: int,
    max_score: float,
):
    """Print information about consistency analysis configuration."""
    print("\n" + "=" * 70)
    print("CONSISTENCY ANALYSIS CONFIGURATION")
    print("=" * 70)

    print(f"\nConfigurations to analyze: {n_configs}")
    print(f"Seeds per configuration: {n_seeds}")
    print(f"Score threshold: <= {max_score}")
    print(f"Total simulations: {n_configs * n_seeds}")

    print("=" * 70)


def run_consistency_analysis(
    checkpoint: CheckpointManager,
    n_seeds: int = 10,
    n_periods: int = 1000,
    burn_in: int = 500,
    max_score: float = 50.0,
    top_n: int | None = None,
    n_workers: int = 1,
) -> list[ConsistencyResult]:
    """
    Run consistency analysis on top configurations.

    Parameters
    ----------
    checkpoint : CheckpointManager
        Checkpoint with calibration results.
    n_seeds : int
        Number of seeds per configuration.
    n_periods : int
        Simulation periods.
    burn_in : int
        Burn-in periods.
    max_score : float
        Maximum score threshold for inclusion.
    top_n : int, optional
        Limit to top N configs.
    n_workers : int
        Number of parallel workers (currently unused, reserved for future).

    Returns
    -------
    list[ConsistencyResult]
        Results sorted by consistency score.
    """
    # Check if already completed
    if checkpoint.data["metadata"].get("consistency_analysis_completed", False):
        print("\n" + "=" * 70)
        print("CONSISTENCY ANALYSIS (SKIPPED - already completed)")
        print("=" * 70)
        existing = checkpoint.data.get("consistency_results", [])
        if existing:
            print(f"Found {len(existing)} analyzed configurations")
        return _load_results_from_checkpoint(checkpoint)

    print("\n" + "=" * 70)
    print("CONSISTENCY ANALYSIS")
    print("=" * 70)
    checkpoint.set_metadata("stage", "consistency_analysis")

    # Get configs to analyze
    configs = get_configs_to_analyze(checkpoint, max_score=max_score, top_n=top_n)

    if not configs:
        print("\nNo configurations found within score threshold.")
        return []

    print_consistency_info(len(configs), n_seeds, max_score)
    print(f"Workers: {n_workers}")

    results: list[ConsistencyResult] = []

    if n_workers > 1:
        # Parallel mode
        print(f"\nRunning in parallel with {n_workers} workers...")

        # Prepare args for all configs
        work_items = []
        for i, config in enumerate(configs):
            params = config["params"]
            checkpoint_score = config["scores"]["total"]
            checkpoint_std = config["scores"].get("total_std", 0.0)
            work_items.append(
                (
                    i + 1,
                    params,
                    checkpoint_score,
                    checkpoint_std,
                    n_seeds,
                    n_periods,
                    burn_in,
                )
            )

        # Run in parallel
        with Pool(n_workers) as pool:
            for (
                rank,
                params,
                checkpoint_score,
                checkpoint_std,
                stats,
            ) in pool.imap_unordered(_analyze_config_worker, work_items):
                # Compute consistency score (lower is better)
                consistency_score = stats["mean_score"] + 2 * stats["std_score"]

                result = ConsistencyResult(
                    rank=rank,
                    params=params,
                    checkpoint_score=checkpoint_score,
                    checkpoint_std=checkpoint_std,
                    scores=stats["scores"],
                    mean_score=stats["mean_score"],
                    std_score=stats["std_score"],
                    min_score=stats["min_score"],
                    max_score=stats["max_score"],
                    consistency_score=consistency_score,
                    cv=stats["cv"],
                    okun_corrs=stats["okun_corrs"],
                    mean_okun=stats["mean_okun"],
                    std_okun=stats["std_okun"],
                    collapse_count=stats["collapse_count"],
                )
                results.append(result)

                # Print progress
                print(
                    f"  [{len(results)}/{len(configs)}] Rank {rank}: "
                    f"mean={stats['mean_score']:.2f}, std={stats['std_score']:.2f}",
                    flush=True,
                )

                # Save periodically
                if len(results) % 10 == 0:
                    _save_results_to_checkpoint(checkpoint, results, completed=False)
    else:
        # Sequential mode with progress bar
        progress = ProgressTracker(
            total=len(configs),
            stage="Consistency",
            update_interval=1,
        )

        for i, config in enumerate(configs):
            params = config["params"]
            checkpoint_score = config["scores"]["total"]
            checkpoint_std = config["scores"].get("total_std", 0.0)

            # Analyze with multiple seeds
            stats = analyze_config(params, n_seeds, n_periods, burn_in)

            # Compute consistency score (lower is better)
            consistency_score = stats["mean_score"] + 2 * stats["std_score"]

            result = ConsistencyResult(
                rank=i + 1,
                params=params,
                checkpoint_score=checkpoint_score,
                checkpoint_std=checkpoint_std,
                scores=stats["scores"],
                mean_score=stats["mean_score"],
                std_score=stats["std_score"],
                min_score=stats["min_score"],
                max_score=stats["max_score"],
                consistency_score=consistency_score,
                cv=stats["cv"],
                okun_corrs=stats["okun_corrs"],
                mean_okun=stats["mean_okun"],
                std_okun=stats["std_okun"],
                collapse_count=stats["collapse_count"],
            )
            results.append(result)

            progress.update(score=stats["mean_score"], params=params)

            # Save periodically
            if (i + 1) % 10 == 0:
                _save_results_to_checkpoint(checkpoint, results, completed=False)

        progress.close()

    # Sort by consistency score
    results.sort(key=lambda x: x.consistency_score)

    # Save final results
    _save_results_to_checkpoint(checkpoint, results, completed=True)

    # Print summary
    _print_summary(results, n_seeds)

    return results


def _save_results_to_checkpoint(
    checkpoint: CheckpointManager,
    results: list[ConsistencyResult],
    completed: bool,
):
    """Save consistency results to checkpoint."""
    checkpoint.data["consistency_results"] = [
        {
            "rank": r.rank,
            "params": r.params,
            "checkpoint_score": r.checkpoint_score,
            "checkpoint_std": r.checkpoint_std,
            "scores": r.scores,
            "mean_score": r.mean_score,
            "std_score": r.std_score,
            "min_score": r.min_score,
            "max_score": r.max_score,
            "consistency_score": r.consistency_score,
            "cv": r.cv,
            "mean_okun": r.mean_okun,
            "std_okun": r.std_okun,
            "collapse_count": r.collapse_count,
        }
        for r in results
    ]
    checkpoint.data["metadata"]["consistency_analysis_completed"] = completed
    checkpoint.save()


def _load_results_from_checkpoint(
    checkpoint: CheckpointManager,
) -> list[ConsistencyResult]:
    """Load consistency results from checkpoint."""
    data = checkpoint.data.get("consistency_results", [])
    results = []
    for r in data:
        results.append(
            ConsistencyResult(
                rank=r["rank"],
                params=r["params"],
                checkpoint_score=r["checkpoint_score"],
                checkpoint_std=r["checkpoint_std"],
                scores=r["scores"],
                mean_score=r["mean_score"],
                std_score=r["std_score"],
                min_score=r["min_score"],
                max_score=r["max_score"],
                consistency_score=r["consistency_score"],
                cv=r["cv"],
                okun_corrs=[],  # Not stored
                mean_okun=r["mean_okun"],
                std_okun=r["std_okun"],
                collapse_count=r["collapse_count"],
            )
        )
    return results


def _print_summary(results: list[ConsistencyResult], n_seeds: int):
    """Print summary of consistency analysis."""
    print("\n" + "=" * 100)
    print("CONSISTENCY ANALYSIS RESULTS")
    print("=" * 100)

    # Overall statistics
    stable_count = sum(1 for r in results if r.std_score < 5.0)
    unstable_count = sum(1 for r in results if r.std_score >= 10.0)
    collapse_affected = sum(1 for r in results if r.collapse_count > 0)

    print(f"\nTotal configurations analyzed: {len(results)}")
    print(f"Seeds per configuration: {n_seeds}")
    print(f"Stable (std < 5): {stable_count}")
    print(f"Unstable (std >= 10): {unstable_count}")
    print(f"Collapse affected: {collapse_affected}")

    # Table header
    print("\n" + "-" * 100)
    print(
        f"{'Rank':<6} {'Ckpt':<8} {'Mean':<8} {'Std':<8} {'CV%':<8} "
        f"{'Min':<8} {'Max':<8} {'Consist.':<10} {'Okun':<10} {'Coll.':<6}"
    )
    print("-" * 100)

    # Print all results
    for r in results:
        cv_pct = r.cv * 100 if r.cv < float("inf") else 999.9
        print(
            f"{r.rank:<6} {r.checkpoint_score:<8.2f} {r.mean_score:<8.2f} "
            f"{r.std_score:<8.2f} {cv_pct:<8.1f} {r.min_score:<8.2f} "
            f"{r.max_score:<8.2f} {r.consistency_score:<10.2f} "
            f"{r.mean_okun:<+10.3f} {r.collapse_count:<6}"
        )

    # Top 5 most consistent
    print("\n" + "=" * 100)
    print("TOP 5 MOST CONSISTENT CONFIGURATIONS")
    print("=" * 100)

    for i, r in enumerate(results[:5], 1):
        print(f"\n#{i} (Original Rank {r.rank})")
        print(
            f"  Checkpoint Score: {r.checkpoint_score:.2f} +/- {r.checkpoint_std:.2f}"
        )
        print(f"  {n_seeds}-Seed Mean: {r.mean_score:.2f} +/- {r.std_score:.2f}")
        print(f"  Consistency Score: {r.consistency_score:.2f}")
        print(f"  Score Range: [{r.min_score:.2f}, {r.max_score:.2f}]")
        print(f"  CV: {r.cv * 100:.1f}%")
        print(f"  Okun: {r.mean_okun:+.3f} +/- {r.std_okun:.3f}")
        print(f"  Collapses: {r.collapse_count}/{n_seeds}")
        print(f"  Individual scores: {[f'{s:.1f}' for s in r.scores]}")
        print("  Parameters:")
        for k, v in r.params.items():
            print(f"    {k}: {v}")

    # Flag unstable configs
    unstable = [r for r in results if r.std_score >= 10.0 or r.collapse_count >= 2]
    if unstable:
        print("\n" + "=" * 100)
        print("WARNING: UNSTABLE CONFIGURATIONS")
        print("=" * 100)
        print(
            "\nThe following configurations have high variance (std >= 10) or "
            "frequent collapses (>= 2 seeds):"
        )
        for r in unstable[:10]:
            print(
                f"  Rank {r.rank}: mean={r.mean_score:.2f}, "
                f"std={r.std_score:.2f}, collapses={r.collapse_count}"
            )

    print("\n" + "=" * 100)
