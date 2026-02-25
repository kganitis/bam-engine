"""Multi-seed stability testing with tiered evaluation and ranking strategies.

This module handles the stability testing phase of calibration: evaluating
top candidates from screening across multiple seeds with configurable
ranking strategies and tiered pruning.
"""

from __future__ import annotations

import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from calibration.analysis import CalibrationResult, format_eta
from calibration.grid import count_combinations, generate_combinations
from calibration.screening import delete_checkpoint, run_screening, save_checkpoint
from validation import (
    StabilityResult,
    compute_combined_score,
    get_validation_func,
    get_validation_funcs,
)


def _evaluate_single_seed(
    params: dict[str, Any],
    scenario: str,
    seed: int,
    n_periods: int,
) -> tuple[dict[str, Any], int, float, int]:
    """Evaluate a single seed for stability testing. Returns (params, seed, score, n_fail)."""
    validate = get_validation_func(scenario)
    result = validate(seed=seed, n_periods=n_periods, **params)
    return dict(params), seed, result.total_score, result.n_fail


def evaluate_stability(
    params: dict[str, Any],
    scenario: str,
    seeds: list[int],
    n_periods: int,
) -> CalibrationResult:
    """Run multi-seed stability test for full evaluation.

    Parameters
    ----------
    params : dict
        Parameter configuration.
    scenario : str
        Scenario name.
    seeds : list[int]
        List of random seeds to test.
    n_periods : int
        Number of simulation periods.

    Returns
    -------
    CalibrationResult
        Result with stability metrics and combined score.
    """
    _, run_stability_fn, _, _ = get_validation_funcs(scenario)
    stability: StabilityResult = run_stability_fn(
        seeds=seeds, n_periods=n_periods, **params
    )
    combined = compute_combined_score(stability)
    return CalibrationResult(
        params=params,
        single_score=stability.seed_results[0].total_score,
        n_pass=stability.seed_results[0].n_pass,
        n_warn=stability.seed_results[0].n_warn,
        n_fail=stability.seed_results[0].n_fail,
        mean_score=stability.mean_score,
        std_score=stability.std_score,
        pass_rate=stability.pass_rate,
        combined_score=combined,
        stability_result=stability,
        seed_scores=[sr.total_score for sr in stability.seed_results],
    )


def parse_stability_tiers(tiers_str: str) -> list[tuple[int, int]]:
    """Parse stability tiers from CLI string.

    Parameters
    ----------
    tiers_str : str
        Format: "100:10,50:20,10:100" meaning
        (top 100 x 10 seeds, top 50 x 20 seeds, top 10 x 100 seeds)

    Returns
    -------
    list[tuple[int, int]]
        List of (n_configs, total_seeds) tuples.
    """
    tiers = []
    for part in tiers_str.split(","):
        configs, seeds = part.strip().split(":")
        tiers.append((int(configs), int(seeds)))
    return tiers


def _rank_candidates(
    candidates: list[CalibrationResult],
    rank_by: str = "combined",
    k_factor: float = 1.0,
) -> list[CalibrationResult]:
    """Rank candidates by the specified strategy.

    Parameters
    ----------
    candidates : list[CalibrationResult]
        Candidates to rank.
    rank_by : str
        Ranking strategy: "combined", "stability", or "mean".
    k_factor : float
        k in mean - k*std formula for "combined" ranking.

    Returns
    -------
    list[CalibrationResult]
        Sorted candidates (best first).
    """
    if rank_by == "stability":
        # Sort by (pass_rate DESC, n_fail ASC, combined_score DESC)
        candidates.sort(
            key=lambda r: (
                r.pass_rate or 0.0,
                -(r.n_fail or 0),
                r.combined_score or 0.0,
            ),
            reverse=True,
        )
    elif rank_by == "mean":
        candidates.sort(key=lambda r: r.mean_score or 0.0, reverse=True)
    else:
        # "combined": mean * pass_rate * (1 - k * std)
        for c in candidates:
            if c.mean_score is not None and c.std_score is not None:
                pr = c.pass_rate if c.pass_rate is not None else 1.0
                c.combined_score = c.mean_score * pr * (1.0 - k_factor * c.std_score)
        candidates.sort(key=lambda r: r.combined_score or 0.0, reverse=True)

    return candidates


def run_tiered_stability(
    candidates: list[CalibrationResult],
    scenario: str,
    tiers: list[tuple[int, int]],
    n_workers: int = 10,
    n_periods: int = 1000,
    avg_time_per_run: float = 0.0,
    rank_by: str = "combined",
    k_factor: float = 1.0,
) -> list[CalibrationResult]:
    """Run incremental tiered stability testing.

    Each tier runs only NEW seeds (not previously tested ones) and
    accumulates all seed scores for ranking.

    Parameters
    ----------
    candidates : list[CalibrationResult]
        Screening results to stability-test.
    scenario : str
        Scenario name.
    tiers : list[tuple[int, int]]
        List of (n_configs, total_seeds) -- each tier tests the top n_configs
        using enough new seeds to reach total_seeds cumulative.
    n_workers : int
        Parallel workers.
    n_periods : int
        Simulation periods.
    avg_time_per_run : float
        Estimated time per run for ETA.
    rank_by : str
        Ranking strategy: "combined" (mean*(1-k*std)), "stability"
        (pass_rate/n_fail priority), or "mean" (mean_score only).
    k_factor : float
        Configurable k in mean - k*std formula (for "combined" ranking).

    Returns
    -------
    list[CalibrationResult]
        Final results sorted by ranking strategy (best first).
    """
    # Initialize seed_scores from screening (seed 0)
    for c in candidates:
        if c.seed_scores is None:
            c.seed_scores = [c.single_score]

    current = candidates

    for tier_idx, (n_configs, total_seeds) in enumerate(tiers):
        tier_num = tier_idx + 1
        # Take top n_configs (or all if fewer available)
        top = current[:n_configs]

        print(f"\n  Tier {tier_num}: {len(top)} configs x {total_seeds} total seeds")

        # Determine which new seeds each config needs
        for c in top:
            existing_n = len(c.seed_scores or [])
            new_seeds_needed = max(0, total_seeds - existing_n)

            if new_seeds_needed > 0:
                new_seed_ids = list(range(existing_n, total_seeds))
                print(
                    f"    Testing {c.single_score:.3f} config: "
                    f"+{new_seeds_needed} seeds ({existing_n}->{total_seeds})"
                )

                # Run new seeds in parallel
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [
                        executor.submit(
                            _evaluate_single_seed,
                            c.params,
                            scenario,
                            seed,
                            n_periods,
                        )
                        for seed in new_seed_ids
                    ]
                    new_scores: list[float] = []
                    new_fails: list[int] = []
                    for i, future in enumerate(as_completed(futures)):
                        _, _seed, score, n_fail = future.result()
                        new_scores.append(score)
                        new_fails.append(n_fail)
                        done = i + 1
                        remaining = new_seeds_needed - done
                        eta = format_eta(remaining, avg_time_per_run, n_workers)
                        print(
                            f"    Tier {tier_num}: "
                            f"Testing {done}/{new_seeds_needed} "
                            f"({100 * done / new_seeds_needed:.0f}%) "
                            f"| {remaining} remaining | ETA: {eta}"
                        )

                c.seed_scores = (c.seed_scores or []) + new_scores
                c.seed_fails = (c.seed_fails or [c.n_fail]) + new_fails

            # Update aggregate metrics
            scores = c.seed_scores or []
            if scores:
                c.mean_score = statistics.mean(scores)
                c.std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
                n_passed = sum(1 for nf in (c.seed_fails or []) if nf == 0)
                c.pass_rate = n_passed / len(c.seed_fails) if c.seed_fails else None
                pr = c.pass_rate if c.pass_rate is not None else 1.0
                c.combined_score = c.mean_score * pr * (1.0 - k_factor * c.std_score)

        # Rank by chosen strategy
        top = _rank_candidates(top, rank_by=rank_by, k_factor=k_factor)
        current = top

        # Print tier results
        print(f"\n  Tier {tier_num} results (top 5):")
        for i, r in enumerate(current[:5]):
            print(
                f"    #{i + 1}: combined={r.combined_score:.4f} "
                f"mean={r.mean_score:.3f} +/- {r.std_score:.3f} "
                f"({len(r.seed_scores or [])} seeds)"
            )

        # Checkpoint after each tier
        save_checkpoint(current, scenario, "stability")

    delete_checkpoint(scenario, "stability")
    return current


# =============================================================================
# Focused calibration (orchestrates screening + stability)
# =============================================================================


def run_focused_calibration(
    grid: dict[str, list[Any]],
    fixed_params: dict[str, Any],
    scenario: str = "baseline",
    n_workers: int = 10,
    n_periods: int = 1000,
    stability_tiers: list[tuple[int, int]] | None = None,
    avg_time_per_run: float = 0.0,
    resume: bool = False,
    rank_by: str = "combined",
    k_factor: float = 1.0,
) -> list[CalibrationResult]:
    """Run calibration on focused grid with fixed params.

    Parameters
    ----------
    grid : dict
        Parameter grid to search (from build_focused_grid).
    fixed_params : dict
        Fixed parameter values (from build_focused_grid).
    scenario : str
        Scenario name.
    n_workers : int
        Number of parallel workers.
    n_periods : int
        Number of simulation periods.
    stability_tiers : list[tuple[int, int]], optional
        Tiered stability config. Defaults to [(100, 10), (50, 20), (10, 100)].
    avg_time_per_run : float
        Average time per simulation run (from sensitivity).
    resume : bool
        If True, resume from checkpoint.
    rank_by : str
        Ranking strategy for stability testing.
    k_factor : float
        k in mean - k*std formula.

    Returns
    -------
    list[CalibrationResult]
        Results sorted by ranking strategy (best first).
    """
    if stability_tiers is None:
        stability_tiers = [(100, 10), (50, 20), (10, 100)]

    total = count_combinations(grid)
    print(f"\n[{scenario}] Focused Grid Search: {total} combinations")
    print(f"Fixed params: {fixed_params}")

    # Generate all combinations (merging fixed params)
    combinations = list(generate_combinations(grid, fixed=fixed_params))

    # Screening phase
    print(f"\nScreening {total} combinations with {n_workers} workers...")
    screening_results = run_screening(
        combinations,
        scenario,
        n_workers=n_workers,
        n_periods=n_periods,
        avg_time_per_run=avg_time_per_run,
        resume=resume,
    )

    # Stability testing phase
    print("\nTiered stability testing...")
    return run_tiered_stability(
        screening_results,
        scenario,
        tiers=stability_tiers,
        n_workers=n_workers,
        n_periods=n_periods,
        avg_time_per_run=avg_time_per_run,
        rank_by=rank_by,
        k_factor=k_factor,
    )
